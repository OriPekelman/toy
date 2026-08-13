#!/usr/bin/env ruby
# frozen_string_literal: true
# prep/fetch_text.rb — materialise a real byte-level TEXT corpus as a pack
# the toy#165 autoencoder lane can read (`toy train ae --text data/ae_names`).
#
# ── WHY THIS EXISTS, AND WHY THE CORPUS IS AN AXIS ──
#
# P1a asks where a d-dim per-token latent stops being decodable under
# noise. That margin is set by how tightly the per-position head must pack
# codepoints into d dimensions, so the number of DISTINCT SYMBOLS the head
# actually has to separate is not a detail — it is the second independent
# variable (tao#22). To first order the nearest-neighbour spacing goes as
# N^(1/d), so a `go` obtained on a 27-symbol corpus would be a statement
# about 27 symbols and not about text.
#
# toy had NO raw-text corpus at all before this script: `data/*.txt` is
# logits dumps and pre-tokenized integer ids. So the corpus had to be
# fetched, and once it is fetched it costs nothing to fetch three at
# different alphabet sizes and make N an axis instead of a confound.
#
#   names        27 distinct bytes  (makemore names list, a-z + newline)
#   shakespeare  65 distinct bytes  (tinyshakespeare, ordinary English)
#   udhr        201 distinct bytes  (UDHR in 388 languages, UTF-8)
#
# Those are MEASURED counts on the fetched bytes, not nominal vocab sizes.
# The runner recomputes the alphabet over the windows it actually draws
# and refuses a pack whose meta disagrees, so nothing here is taken on
# trust downstream.
#
# Plain MRI Ruby (NOT Spinel-compiled) — same status as every other prep/
# converter. Writes to data/, which is gitignored for regenerable
# artefacts.
#
#   ruby prep/fetch_text.rb --corpus names        [--out data/ae_names]
#   ruby prep/fetch_text.rb --corpus shakespeare
#   ruby prep/fetch_text.rb --corpus udhr
#   ruby prep/fetch_text.rb --all
#
# PACK FORMAT (read by AeTask#load_pack!):
#   <prefix>.meta.i32   [n_tokens, alphabet_size]
#   <prefix>.tok.i32    n_tokens byte ids, 0..255, in corpus order
#   <prefix>.json       provenance (source, sha256, distinct, entropy) —
#                       for the write-up; the runner does not read it.
#
# DETERMINISM. Every source is pinned by SHA-256 and the script REFUSES a
# mismatch rather than proceeding with different bytes under the same
# name. `udhr` is a zip of 388 files: the members are concatenated in
# SORTED name order, so the pack is byte-identical on any machine.

require "open-uri"
require "fileutils"
require "digest"
require "tmpdir"
require "json"
require "shellwords"

ROOT = File.expand_path("..", __dir__)

# The pinned sources. `sha256` is of the DOWNLOADED artefact (the zip for
# udhr, the text itself otherwise) — verified before anything is written.
CORPORA = {
  "names" => {
    "url"    => "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
    "sha256" => "0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d",
    "kind"   => "text",
    "note"   => "makemore names list; lowercase a-z + newline",
  },
  "shakespeare" => {
    "url"    => "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "sha256" => "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    "kind"   => "text",
    "note"   => "tinyshakespeare; ordinary English prose/verse",
  },
  "udhr" => {
    "url"    => "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip",
    "sha256" => "0796c314b09a930c989c6f9d93d226af9af13feccd88496e196c743dd266c7f3",
    "kind"   => "zip",
    "note"   => "UDHR in 388 languages, UTF-8; the multilingual high-N anchor",
  },
}.freeze

corpus = nil
out    = nil
do_all = false
allow_sha_update = false
i = 0
while i < ARGV.length
  case ARGV[i]
  when "--corpus" then i += 1; corpus = ARGV[i]
  when "--out"    then i += 1; out    = ARGV[i]
  when "--all"    then do_all = true
  # Only for the first run against a source whose pin is not yet known.
  # It PRINTS the observed digest so the constant above can be updated;
  # it never silently accepts a changed source in normal use.
  when "--print-sha" then allow_sha_update = true
  else abort "fetch_text: unknown argument #{ARGV[i].inspect}"
  end
  i += 1
end

if !do_all && corpus.nil?
  abort "fetch_text: pass --corpus #{CORPORA.keys.join('|')} or --all"
end
if corpus && !CORPORA.key?(corpus)
  abort "fetch_text: unknown corpus #{corpus.inspect} (#{CORPORA.keys.join('|')})"
end
if do_all && out
  abort "fetch_text: --out names ONE output prefix and --all writes three; " \
        "run them separately if you need custom paths"
end

def download(url)
  warn "fetch_text: GET #{url}"
  URI.parse(url).open("rb", &:read)
rescue OpenURI::HTTPError => e
  abort "fetch_text: #{url} -> #{e.message}"
end

# Concatenate the .txt members of a zip in SORTED name order. Sorted, so
# two machines produce the same bytes; `unzip -p` streams one member at a
# time, which keeps this to stdlib plus a tool that is always present.
def unzip_text(zip_path)
  listing = `unzip -Z1 #{zip_path.shellescape}`
  abort "fetch_text: cannot list #{zip_path} (is unzip installed?)" unless $?.success?
  members = listing.split("\n").select { |m| m.end_with?(".txt") }.sort
  abort "fetch_text: no .txt members in #{zip_path}" if members.empty?
  warn "fetch_text: concatenating #{members.length} members"
  buf = +""
  members.each do |m|
    part = `unzip -p #{zip_path.shellescape} #{m.shellescape}`
    abort "fetch_text: cannot extract #{m}" unless $?.success?
    buf << part
  end
  buf
end

def build(name, spec, out_prefix, allow_sha_update)
  raw = download(spec["url"])
  got = Digest::SHA256.hexdigest(raw)
  if got != spec["sha256"]
    if allow_sha_update
      warn "fetch_text: #{name} observed sha256=#{got} (pin says #{spec['sha256']})"
    else
      abort "fetch_text: #{name} SOURCE CHANGED — sha256=#{got}, pinned " \
            "#{spec['sha256']}. Refusing: a corpus that silently changed " \
            "would relabel every cell measured against it. Re-run with " \
            "--print-sha, check the diff is intended, then update CORPORA."
    end
  end

  bytes =
    if spec["kind"] == "zip"
      Dir.mktmpdir do |dir|
        zp = File.join(dir, "src.zip")
        File.binwrite(zp, raw)
        unzip_text(zp)
      end
    else
      raw
    end
  bytes = bytes.b

  counts = Array.new(256, 0)
  bytes.each_byte { |b| counts[b] += 1 }
  distinct = counts.count { |c| c > 0 }
  n = bytes.bytesize
  entropy = counts.sum { |c| c.zero? ? 0.0 : (-(c.to_f / n) * Math.log2(c.to_f / n)) }
  top = counts.each_with_index.max_by { |c, _| c }

  FileUtils.mkdir_p(File.dirname(out_prefix))
  File.binwrite(out_prefix + ".meta.i32", [n, distinct].pack("l<*"))
  File.binwrite(out_prefix + ".tok.i32",  bytes.unpack("C*").pack("l<*"))
  File.write(out_prefix + ".json", JSON.pretty_generate(
    "corpus"        => name,
    "source"        => spec["url"],
    "source_sha256" => got,
    "note"          => spec["note"],
    "n_tokens"      => n,
    "alphabet"      => distinct,
    "entropy_bits"  => entropy.round(6),
    "unigram_floor" => (top[0].to_f / n).round(6),
    "floor_id"      => top[1],
  ) + "\n")

  warn format("fetch_text: %-12s n=%d distinct=%d H=%.3f floor=%.4f -> %s.{meta,tok}.i32",
              name, n, distinct, entropy, top[0].to_f / n, out_prefix)
end

targets = do_all ? CORPORA.keys : [corpus]
targets.each do |name|
  prefix = out || File.join(ROOT, "data", "ae_" + name)
  build(name, CORPORA[name], prefix, allow_sha_update)
end
