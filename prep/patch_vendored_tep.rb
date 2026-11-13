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
# Run by `make vendor-tep` after the vendor step; vendor/ is gitignored, so
# without this every re-vendor silently loses the fixes. Each edit is exact
# text: already fixed -> left alone; original text -> replaced; neither ->
# exit 1 (tep changed under us — re-check and update or delete this file).
#
# DELETE THIS FILE and its vendor-tep line once the Gemfile resolves to a
# tep release that reports "already fixed" for every edit.

ROOT = File.expand_path("..", __dir__)
LIB  = File.join(ROOT, "vendor", "spinel", "tep", "lib")

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
