# lib/toy/io/json_builder.rb — Toy::Json::Builder, absorbed from
# spinel_kit 0.1.x when upstream retired the codec in 0.3.0 (spin's
# bundled stdlib json supersedes it there; toy keeps the streaming
# builder its event/checkpoint writers are shaped around). Mirror of
# tep's absorption (tep#217). toy#107 deps leg.
# Toy::Json::Builder -- an incremental, ordered JSON-object builder.
#
# WHY A SEPARATE FILE/CLASS. This is toy's hand-rolled builder; the codec
# (Toy::Json, encode/decode) is tep's. They are split so a consumer
# compiles only the surface it uses: Spinel has no tree-shaking, so an
# uncalled method is still compiled and degrades its params, emitting benign
# but gate-tripping `emitting 0` warnings. A builder-only consumer requires
# THIS file and never pays for the decoder walkers; a codec-only consumer
# requires spinel_kit/json and never pays for the builder.
#
# To stay self-contained, Builder carries its OWN escape/quote/hex2 rather
# than calling Toy::Json.* (which would drag the whole codec, decoders
# included, into a builder-only compile). These are byte-identical to the
# codec's; the small duplication buys surface isolation. The same-named
# methods across the two classes are safe -- the Spinel name-collision bug
# that once made that dangerous is fixed (see docs/spinel-discipline.md).
#
# USAGE (mutating appenders, NOT method-chaining -- chaining is a Spinel
# poly-degradation risk):
#
#   j = Toy::Json::Builder.new
#   j.add_str("kind", "run_start")
#   j.add_num("t", now)                  # int OR float via .to_s
#   host = Toy::Json::Builder.new
#   host.add_str("name", host_name)
#   j.add_obj("host", host)              # nest a sub-builder
#   j.add_raw("lr", "0.001")             # already-encoded JSON literal
#   ev = j.dump                          # "{...}"
#
# NUMERIC NOTE: if a single compiled program passes BOTH Integer and Float to
# `add_num`'s `value`, Spinel unifies the param to Float and an Integer
# renders as "N.0". For hardcoded numeric literals where byte-exact output
# matters, use `add_raw` with an already-encoded string instead of relying on
# `value.to_s`.
module Toy
  class Json
    class Builder
      def initialize
        @buf   = "{"
        @first = true
      end

      # Append `"key":"escaped-value"`.
      def add_str(key, value)
        comma
        @buf = @buf + Builder.quote(key) + ":" + Builder.quote(value)
      end

      # Append `"key":<number>` -- `value.to_s` covers Integer ("5") and
      # Float ("1.5"). For hardcoded literals prefer add_raw.
      def add_num(key, value)
        comma
        @buf = @buf + Builder.quote(key) + ":" + value.to_s
      end

      # Append `"key":true|false`.
      def add_bool(key, value)
        comma
        @buf = @buf + Builder.quote(key) + ":" + (value ? "true" : "false")
      end

      # Append `"key":<already-encoded JSON>` -- for arrays / numeric literals.
      def add_raw(key, raw)
        comma
        @buf = @buf + Builder.quote(key) + ":" + raw
      end

      # Append `"key":<nested object>` from another Builder.
      def add_obj(key, child)
        comma
        @buf = @buf + Builder.quote(key) + ":" + child.dump
      end

      # Close the object and return the JSON string.
      def dump
        @buf + "}"
      end

      # Emit a separator before every entry except the first.
      def comma
        if @first
          @first = false
        else
          @buf = @buf + ","
        end
      end

      # ---- self-contained string escaping (byte-identical to
      #      Toy::Json.escape/quote/hex2; kept local so a builder-only
      #      compile never pulls in the codec) ----

      # Wrap a string in JSON quotes, escaping its body.
      def self.quote(s)
        "\"" + Builder.escape(s) + "\""
      end

      # Escape a string for inclusion inside a JSON string literal (no
      # surrounding quotes). Handles ", \, and the JSON control-char escapes
      # (\b \f \n \r \t); other control bytes go through \u00XX. ASCII-clean
      # input passes through unchanged.
      def self.escape(s)
        out = ""
        i = 0
        n = s.length
        while i < n
          c = s[i]
          if c == "\""
            out = out + "\\\""
          elsif c == "\\"
            out = out + "\\\\"
          elsif c == "\n"
            out = out + "\\n"
          elsif c == "\r"
            out = out + "\\r"
          elsif c == "\t"
            out = out + "\\t"
          elsif c == "\b"
            out = out + "\\b"
          elsif c == "\f"
            out = out + "\\f"
          elsif c < " "
            b = c.getbyte(0)
            out = out + "\\u00" + Builder.hex2(b)
          else
            out = out + c
          end
          i += 1
        end
        out
      end

      # Two-digit lowercase hex of a byte (0..255).
      def self.hex2(n)
        hex = "0123456789abcdef"
        out = ""
        out = out + hex[(n / 16) % 16, 1]
        out = out + hex[n % 16, 1]
        out
      end
    end
  end
end
