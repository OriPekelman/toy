# lib/toy/io/toy_json.rb — Toy::Json, a tiny ordered JSON-object builder.
#
# WHY THIS EXISTS. The tep-free runners (`toy train` / `eval` / the dev taps)
# and the training examples emit `toy/v1` instrumentation events as JSON. They
# used to hand-concatenate the literal:
#
#   rs  = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\""
#   rs += ",\"t\":" + now.to_s
#   rs += ",\"run_id\":\"" + rid + "\""        # <-- rid never escaped (latent bug)
#   ...
#
# which is unreadable, error-prone (a missing comma / stray quote is silent),
# and skips JSON string-escaping entirely. We can't reach for `Tep::Json` here:
# `train`/`eval` are deliberately tep-free (see docs/gating.md), and Tep::Json
# has no float encoder and no nested-object helper anyway. So this is a small
# self-contained builder with NO FFI — pure string ops, compiles under Spinel,
# and needs no CUDA/Metal mirror.
#
# SPINEL NAMING DISCIPLINE (load-bearing — do not "tidy" the names). Spinel does
# whole-program inference keyed partly on method- and parameter-NAMES (landmines
# #12/#16 in feedback_spinel_type_inference_landmines). A builder method named
# `num(key, value)` whose `value` is numeric-poly silently widens every other
# `value` parameter in the program to poly — which corrupted the warm-start
# training compute even with Toy::Json merely required and never called. So every
# method and parameter here carries a `j_`/`tj` prefix to stay type-isolated, and
# we deliberately do NOT override `to_s` (it merges across the whole program).
# Verified: with these names, requiring + using Toy::Json leaves the train gate
# byte-identical.
#
# USAGE (mutating appenders, NOT method-chaining — chaining is a Spinel
# poly-degradation risk):
#
#   j = Toy::Json.new
#   j.j_str("kind", "run_start")
#   j.j_str("schema", "toy/v1")
#   j.j_num("t", TinyNN.tnn_events_now_seconds)    # int OR float via .to_s
#   host = Toy::Json.new
#   host.j_str("name", host_name)
#   j.j_obj("host", host)                          # nest a sub-builder
#   j.j_raw("lr", "0.001")                          # already-encoded JSON literal
#   ev = j.j_dump                                   # "{...}"
#
# Output is compact (no spaces), keys in insertion order — byte-identical to the
# old hand-built strings for ASCII values with no special characters, so the
# structural events gates (prep/{train,serve_events}_gate.rb) stay green.
module Toy
  class Json
    def initialize
      @j_buf   = "{"
      @j_first = true
    end

    # Append `"jk":"escaped-jv"`.
    def j_str(jk, jv)
      j_comma
      @j_buf = @j_buf + Toy::Json.tj_quote(jk) + ":" + Toy::Json.tj_quote(jv)
    end

    # Append `"jk":<number>` — `jv.to_s` covers Integer ("5") and Float ("1.5").
    # JSON-numeric (no quotes). For hardcoded literals prefer j_raw (so Spinel's
    # Float#to_s formatting can never drift the bytes).
    def j_num(jk, jv)
      j_comma
      @j_buf = @j_buf + Toy::Json.tj_quote(jk) + ":" + jv.to_s
    end

    # Append `"jk":true|false`.
    def j_bool(jk, jv)
      j_comma
      @j_buf = @j_buf + Toy::Json.tj_quote(jk) + ":" + (jv ? "true" : "false")
    end

    # Append `"jk":<already-encoded JSON>` — for arrays / numeric literals.
    def j_raw(jk, jraw)
      j_comma
      @j_buf = @j_buf + Toy::Json.tj_quote(jk) + ":" + jraw
    end

    # Append `"jk":<nested object>` from another Toy::Json builder.
    def j_obj(jk, jchild)
      j_comma
      @j_buf = @j_buf + Toy::Json.tj_quote(jk) + ":" + jchild.j_dump
    end

    # Close the object and return the JSON string.
    def j_dump
      @j_buf + "}"
    end

    # Emit a separator before every entry except the first.
    def j_comma
      if @j_first
        @j_first = false
      else
        @j_buf = @j_buf + ","
      end
    end

    # ---- string escaping (mirrors Tep::Json.escape; proven correct) ----

    # Wrap a string in JSON quotes, escaping its body.
    def self.tj_quote(tjs)
      "\"" + Toy::Json.tj_escape(tjs) + "\""
    end

    # Escape a string for inclusion inside a JSON string literal (no
    # surrounding quotes). Handles ", \, and the JSON control-char escapes
    # (\b \f \n \r \t); other control bytes go through \u00XX. ASCII-clean
    # input passes through unchanged → byte-identical to the old literals.
    def self.tj_escape(tjs)
      tjout = ""
      tji = 0
      tjn = tjs.length
      while tji < tjn
        tjc = tjs[tji]
        if tjc == "\""
          tjout = tjout + "\\\""
        elsif tjc == "\\"
          tjout = tjout + "\\\\"
        elsif tjc == "\n"
          tjout = tjout + "\\n"
        elsif tjc == "\r"
          tjout = tjout + "\\r"
        elsif tjc == "\t"
          tjout = tjout + "\\t"
        elsif tjc == "\b"
          tjout = tjout + "\\b"
        elsif tjc == "\f"
          tjout = tjout + "\\f"
        elsif tjc < " "
          tjb = tjc.getbyte(0)
          tjout = tjout + "\\u00" + Toy::Json.tj_hex2(tjb)
        else
          tjout = tjout + tjc
        end
        tji = tji + 1
      end
      tjout
    end

    # Two-digit lowercase hex of a byte (0..255).
    def self.tj_hex2(tjnb)
      tjhex = "0123456789abcdef"
      tjo = ""
      tjo = tjo + tjhex[(tjnb / 16) % 16, 1]
      tjo = tjo + tjhex[tjnb % 16, 1]
      tjo
    end
  end
end
