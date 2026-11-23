#!/usr/bin/env ruby
# prep/patch_vendored_tep.rb — carry a fix to the vendored spinel_kit
# that is not in a released gem yet.
#
# The byte-length, stock-Ruby-guard, seed-descriptor and scheduler-fairness
# edits that used to be here shipped in tep 0.11.8 - 0.11.11 (tep#272,
# tep#276, tep#283, tep#291) and were dropped as the Gemfile moved up to
# `~> 0.11.12` (0.11.11 itself loses a connection out of a burst, tep#294).
# One remains:
#
#   jsonu    the JSON decoder tep and toy share (SpinelKit::Json, vendored
#            twice) turned every \\uXXXX escape above ASCII into "?": a
#            client that escapes non-ASCII, as Python's json.dumps does by
#            default, had "été" reach the model as "?t?" (toy#215). The
#            escape is now decoded to UTF-8, surrogate pairs included.
#
# Run by `make vendor-tep` after the vendor step; vendor/ is gitignored, so
# without this every re-vendor silently loses the fixes. Each edit is exact
# text: already fixed -> left alone; original text -> replaced; neither ->
# exit 1 (spinel_kit changed under us — re-check and update or delete this file).
#
# DELETE THIS FILE and its vendor-tep line once the Gemfile resolves to
# a spinel_kit release that reports "already fixed" for every edit
# (toy#215).

ROOT = File.expand_path("..", __dir__)
LIB  = File.join(ROOT, "vendor", "spinel", "tep", "lib")

JSONU_FROM = "          elsif esc == \"u\"\n            # \\u00XX -> map the two-digit hex back to a byte. Wider\n            # codepoints (U+0100+ or surrogate pairs) aren't decoded; the\n            # byte we emit is the low byte of the codepoint, which\n            # round-trips ASCII at minimum.\n            if pos + 5 < s.length\n              h1 = Json.hex_nibble(s[pos + 4])\n              h2 = Json.hex_nibble(s[pos + 5])\n              if h1 >= 0 && h2 >= 0\n                # rebuild the byte and push it -- spinel strings are\n                # byte-blobs, so this works for ASCII; for non-ASCII the\n                # original encoder would have used a passthrough byte\n                # anyway.\n                b = h1 * 16 + h2\n                out = out + Json.byte_to_chr(b)\n                pos += 6\n                next\n              end\n            end\n            out = out + \"?\"\n            pos += 2\n            next\n"
JSONU_TO = "          elsif esc == \"u\"\n            # \\uXXXX -> the UTF-8 bytes of the code point; a high surrogate\n            # followed by a \\uDC00..DFFF escape is one code point.\n            cp = Json.hex4(s, pos + 2)\n            if cp < 0\n              out = out + \"?\"\n              pos += 2\n              next\n            end\n            pos += 6\n            if cp >= 0xD800 && cp <= 0xDBFF && pos + 1 < s.length && s[pos] == \"\\\\\" && s[pos + 1] == \"u\"\n              lo = Json.hex4(s, pos + 2)\n              if lo >= 0xDC00 && lo <= 0xDFFF\n                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)\n                pos += 6\n              end\n            end\n            out = out + Json.cp_to_utf8(cp)\n            next\n"
JSONU_HELPERS = "    # Four hex digits at s[at, 4] as an integer; -1 when they are not there.\n    def self.hex4(s, at)\n      if at + 3 >= s.length\n        return -1\n      end\n      v = 0\n      i = 0\n      while i < 4\n        h = Json.hex_nibble(s[at + i])\n        if h < 0\n          return -1\n        end\n        v = v * 16 + h\n        i += 1\n      end\n      v\n    end\n\n    # A code point as UTF-8. A surrogate on its own is U+FFFD.\n    def self.cp_to_utf8(cp)\n      if cp < 0x80\n        return cp.chr\n      end\n      if cp >= 0xD800 && cp <= 0xDFFF\n        cp = 0xFFFD\n      end\n      if cp < 0x800\n        return ((0xC0 | (cp >> 6)).chr + (0x80 | (cp & 0x3F)).chr).force_encoding(\"UTF-8\")\n      end\n      if cp < 0x10000\n        return ((0xE0 | (cp >> 12)).chr + (0x80 | ((cp >> 6) & 0x3F)).chr +\n                (0x80 | (cp & 0x3F)).chr).force_encoding(\"UTF-8\")\n      end\n      ((0xF0 | (cp >> 18)).chr + (0x80 | ((cp >> 12) & 0x3F)).chr +\n       (0x80 | ((cp >> 6) & 0x3F)).chr + (0x80 | (cp & 0x3F)).chr).force_encoding(\"UTF-8\")\n    end\n\n    def self.hex_nibble(c)\n"

EDITS = [
  ["jsonu", "spinel_kit/json_decoder.rb",
   JSONU_FROM,
   JSONU_TO],
  ["jsonu", "spinel_kit/json_decoder.rb",
   "    def self.hex_nibble(c)\n",
   JSONU_HELPERS],
  ["jsonu", "../../spinel_kit/lib/spinel_kit/json_decoder.rb",
   JSONU_FROM,
   JSONU_TO],
  ["jsonu", "../../spinel_kit/lib/spinel_kit/json_decoder.rb",
   "    def self.hex_nibble(c)\n",
   JSONU_HELPERS],
].freeze

applied = 0
present = 0
EDITS.each do |name, rel, from, to|
  path = File.join(LIB, rel)
  unless File.file?(path)
    warn "patch_vendored_tep: #{path} is missing (run `make vendor-tep`)"
    exit 1
  end
  src = File.read(path)
  if src.include?(to)
    present += 1
  elsif src.scan(from).length == 1
    File.write(path, src.sub(from) { to })
    applied += 1
  else
    warn "patch_vendored_tep: [#{name}] #{rel}: neither the original nor the fixed text " \
         "is there exactly once — tep changed; re-check this edit:\n    #{from.strip}"
    exit 1
  end
end
puts "  patch_vendored_tep: #{applied} edit(s) applied, #{present} already fixed (of #{EDITS.length})"
