#!/usr/bin/env ruby
# prep/patch_vendored_tep.rb — carry tep fixes that are on tep `main` but
# not in the 0.11.x gem that `make vendor-tep` pulls (asked for in tep#276).
#
#   guard    tep.rb's stock-Ruby guard fires inside every compiled app at
#            Spinel 2026.10.25, which defines RUBY_ENGINE ("spinel"): the
#            server exits at boot (tep#272). Fixed in 0.11.8; kept so a
#            lock still at 0.11.7 works.
#   reqbody  the request body is read until its CHARACTER count reaches
#            Content-Length (bytes): any request with a non-ASCII byte in
#            the body stalls until the client gives up (toy#209).
#   reslen   Content-Length of a response is its character count: a reply
#            with non-ASCII text is cut short (toy#209).
#
#   seedfd   tep.rb's boot-time type seeds build a Stream and two WebSocket
#            drivers on file descriptor 0 and write test frames to it (8
#            sendto(0, ...) calls per process start). On a pipe, a file or
#            a terminal they fail with ENOTSOCK and nothing shows. When
#            stdin IS a socket they land in it, and once an unread socket
#            fills, the next start blocks forever before the server's
#            first line (the gate-pack-cuda "never became ready" timeout).
#            The seeds now use descriptor -1.
#
#   jsonu    the JSON decoder tep and toy share (SpinelKit::Json, vendored
#            twice) turned every \\uXXXX escape above ASCII into "?": a
#            client that escapes non-ASCII, as Python's json.dumps does by
#            default, had "été" reach the model as "?t?" (toy#215). The
#            escape is now decoded to UTF-8, surrogate pairs included.
#
#   fairtick Tep::Scheduler.tick resumes the ready fiber with the smallest
#            wake_at and breaks ties by the lowest index. wake_at is in
#            whole seconds, so a fiber that yields with pause(0) ties with
#            every other ready fiber and, having the lowest index, is
#            resumed again and again until the second changes: the others
#            (a connection with a request waiting, a stream with tokens to
#            send) wait up to a second. Ties now go round-robin, starting
#            after the fiber resumed last (toy#217: several requests
#            decoding as one batch yield after every step; tep#291).
#
# Run by `make vendor-tep` after the vendor step; vendor/ is gitignored, so
# without this every re-vendor silently loses the fixes. Each edit is exact
# text: already fixed -> left alone; original text -> replaced; neither ->
# exit 1 (tep changed under us — re-check and update or delete this file).
#
# DELETE THIS FILE and its vendor-tep line once the Gemfile resolves to a
# tep release that reports "already fixed" for every edit.

ROOT = File.expand_path("..", __dir__)
LIB  = File.join(ROOT, "vendor", "spinel", "tep", "lib")

JSONU_FROM = "          elsif esc == \"u\"\n            # \\u00XX -> map the two-digit hex back to a byte. Wider\n            # codepoints (U+0100+ or surrogate pairs) aren't decoded; the\n            # byte we emit is the low byte of the codepoint, which\n            # round-trips ASCII at minimum.\n            if pos + 5 < s.length\n              h1 = Json.hex_nibble(s[pos + 4])\n              h2 = Json.hex_nibble(s[pos + 5])\n              if h1 >= 0 && h2 >= 0\n                # rebuild the byte and push it -- spinel strings are\n                # byte-blobs, so this works for ASCII; for non-ASCII the\n                # original encoder would have used a passthrough byte\n                # anyway.\n                b = h1 * 16 + h2\n                out = out + Json.byte_to_chr(b)\n                pos += 6\n                next\n              end\n            end\n            out = out + \"?\"\n            pos += 2\n            next\n"
JSONU_TO = "          elsif esc == \"u\"\n            # \\uXXXX -> the UTF-8 bytes of the code point; a high surrogate\n            # followed by a \\uDC00..DFFF escape is one code point.\n            cp = Json.hex4(s, pos + 2)\n            if cp < 0\n              out = out + \"?\"\n              pos += 2\n              next\n            end\n            pos += 6\n            if cp >= 0xD800 && cp <= 0xDBFF && pos + 1 < s.length && s[pos] == \"\\\\\" && s[pos + 1] == \"u\"\n              lo = Json.hex4(s, pos + 2)\n              if lo >= 0xDC00 && lo <= 0xDFFF\n                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)\n                pos += 6\n              end\n            end\n            out = out + Json.cp_to_utf8(cp)\n            next\n"
JSONU_HELPERS = "    # Four hex digits at s[at, 4] as an integer; -1 when they are not there.\n    def self.hex4(s, at)\n      if at + 3 >= s.length\n        return -1\n      end\n      v = 0\n      i = 0\n      while i < 4\n        h = Json.hex_nibble(s[at + i])\n        if h < 0\n          return -1\n        end\n        v = v * 16 + h\n        i += 1\n      end\n      v\n    end\n\n    # A code point as UTF-8. A surrogate on its own is U+FFFD.\n    def self.cp_to_utf8(cp)\n      if cp < 0x80\n        return cp.chr\n      end\n      if cp >= 0xD800 && cp <= 0xDFFF\n        cp = 0xFFFD\n      end\n      if cp < 0x800\n        return ((0xC0 | (cp >> 6)).chr + (0x80 | (cp & 0x3F)).chr).force_encoding(\"UTF-8\")\n      end\n      if cp < 0x10000\n        return ((0xE0 | (cp >> 12)).chr + (0x80 | ((cp >> 6) & 0x3F)).chr +\n                (0x80 | (cp & 0x3F)).chr).force_encoding(\"UTF-8\")\n      end\n      ((0xF0 | (cp >> 18)).chr + (0x80 | ((cp >> 12) & 0x3F)).chr +\n       (0x80 | ((cp >> 6) & 0x3F)).chr + (0x80 | (cp & 0x3F)).chr).force_encoding(\"UTF-8\")\n    end\n\n    def self.hex_nibble(c)\n"

EDITS = [
  ["guard", "tep.rb",
   "if defined?(RUBY_ENGINE)\n",
   "if defined?(RUBY_ENGINE) && RUBY_ENGINE != \"spinel\"\n"],
  ["reqbody", "tep/request.rb",
   "      already = @raw_body.length\n",
   "      already = @raw_body.bytesize\n"],
  ["reqbody", "tep/request.rb",
   "      while @raw_body.length < cl\n",
   "      while @raw_body.bytesize < cl\n"],
  ["reqbody", "tep/request.rb",
   "cl - @raw_body.length)\n        if chunk.length == 0",
   "cl - @raw_body.bytesize)\n        if chunk.bytesize == 0"],
  ["reslen", "tep/server.rb",
   "res.headers[\"Content-Length\"] = res.body.length.to_s",
   "res.headers[\"Content-Length\"] = res.body.bytesize.to_s"],
  ["reslen", "tep/server.rb",
   "\"Content-Length: \" + body.length.to_s",
   "\"Content-Length: \" + body.bytesize.to_s"],
  ["reslen", "tep/server_scheduled.rb",
   "\"Content-Length: \" + res.body.length.to_s",
   "\"Content-Length: \" + res.body.bytesize.to_s"],
  ["reslen", "tep/server_scheduled.rb",
   "\"Content-Length: \" + msg.length.to_s",
   "\"Content-Length: \" + msg.bytesize.to_s"],
  ["seedfd", "tep.rb",
   "  _tep_seed_stream = Stream.new(0)\n",
   "  _tep_seed_stream = Stream.new(-1)\n"],
  ["seedfd", "tep.rb",
   "_tep_seed_res.start_websocket(\"\", Tep::WebSocket::Driver.new(0))",
   "_tep_seed_res.start_websocket(\"\", Tep::WebSocket::Driver.new(-1))"],
  ["seedfd", "tep.rb",
   "_tep_seed_llm_stream = Tep::Stream.new(0)",
   "_tep_seed_llm_stream = Tep::Stream.new(-1)"],
  ["seedfd", "tep.rb",
   "_tep_seed_ws_drv = Tep::WebSocket::Driver.new(0)",
   "_tep_seed_ws_drv = Tep::WebSocket::Driver.new(-1)"],
  ["fairtick", "tep/app.rb",
   "    attr_accessor :sched_fibers, :sched_wake_at, :sched_current\n",
   "    attr_accessor :sched_fibers, :sched_wake_at, :sched_current, :sched_last\n"],
  ["fairtick", "tep/app.rb",
   "      @sched_current  = -1               # currently-running fiber idx\n",
   "      @sched_last     = -1               # fiber idx resumed last (round-robin ties)\n" \
   "      @sched_current  = -1               # currently-running fiber idx\n"],
  ["fairtick", "tep/scheduler.rb",
   "        if Tep::APP.sched_fibers[i].f.alive? && Tep::APP.sched_wake_at[i] <= now\n" \
   "          if best < 0 || Tep::APP.sched_wake_at[i] < Tep::APP.sched_wake_at[best]\n            best = i\n",
   "        if Tep::APP.sched_fibers[i].f.alive? && Tep::APP.sched_wake_at[i] <= now\n" \
   "          if best < 0 || Tep::APP.sched_wake_at[i] < Tep::APP.sched_wake_at[best] ||\n" \
   "             (Tep::APP.sched_wake_at[i] == Tep::APP.sched_wake_at[best] &&\n" \
   "              best <= Tep::APP.sched_last && i > Tep::APP.sched_last)\n            best = i\n"],
  ["fairtick", "tep/scheduler.rb",
   "      Tep::APP.sched_current = best\n      Tep::APP.sched_wake_at[best] = -1\n",
   "      Tep::APP.sched_current = best\n      Tep::APP.sched_last    = best\n      Tep::APP.sched_wake_at[best] = -1\n"],
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
