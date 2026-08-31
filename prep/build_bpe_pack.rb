#!/usr/bin/env ruby
# prep/build_bpe_pack.rb — a bytelm pack tokenised with GPT-2 BPE, at a
# chosen merge count (toy#184 / N4).
#
# WHY. The whole DFA programme has run on character and AST-symbol
# alphabets. A real subword vocabulary is the claim most likely to be
# doubted from outside, and D5 makes it falsifiable: DFA should stay above a
# frozen body at BPE classes, because those classes carry real information
# rather than planted noise. This emits the packs that test it.
#
# REPRODUCIBLE FROM SOURCE, which is why it lives here rather than in a
# scratch directory: same doctrine as prep/research/build_ast_pack.py. The
# merge count, the tokeniser tables and the corpus sha are all recorded in
# the sidecar, so a pack that anchors a result can be rebuilt from it.
#
#   ruby prep/build_bpe_pack.rb --text data/ae_shakespeare.txt \
#                               --merges 4000 --out data/bpe_shak_m4000
#   ruby prep/build_bpe_pack.rb ... --vocab-from data/bpe_shak_m4000
#
# ── THE CEILING, READ THIS BEFORE SWEEPING ──────────────────────────────
# Toy::AeTask::VOCAB_MAX is 4096 and the loader REFUSES anything above it:
# the labels are a dense one-hot Mat, so the bound is what a per-step upload
# carries. This builder will happily emit a 16k pack — nothing here enforces
# the ceiling, because a pack is a file and the ceiling belongs to the lane
# that loads it — but only rungs at or below 4096 CLASSES can currently be
# trained on. `--merges 8000` and above therefore produce packs that the
# bytelm lane will reject at load. That is an engine decision with a real
# cost (at context 64, a 16384-class one-hot label Mat is ~4.2 MB per step),
# not an oversight to route around here.
#
# ── SHARED VOCABULARY ───────────────────────────────────────────────────
# A ladder read against a common baseline, or a pretrain/adapt pair, MUST be
# emitted against a shared vocabulary or the same class id means different
# things in different packs. `--vocab-from <prefix>` reuses another pack's
# id assignment and FAILS LOUD on any token the source vocabulary does not
# contain, naming them. That is D5's lesson carried over rather than
# rediscovered: a 4k rung's ids are NOT a prefix of a 16k rung's unless it
# is built that way.
require "json"
require "digest"

def die(msg)
  warn "build_bpe_pack: #{msg}"
  exit 1
end

# ---- args ----------------------------------------------------------------
opts = { merges: nil, text: nil, out: nil, vocab_from: nil,
         tables: "data/gpt2-bpe" }
i = 0
while i < ARGV.length
  case ARGV[i]
  when "--text"       then opts[:text]       = ARGV[i += 1]
  when "--merges"     then opts[:merges]     = Integer(ARGV[i += 1])
  when "--out"        then opts[:out]        = ARGV[i += 1]
  when "--vocab-from" then opts[:vocab_from] = ARGV[i += 1]
  when "--tables"     then opts[:tables]     = ARGV[i += 1]
  else die("unknown flag #{ARGV[i].inspect}")
  end
  i += 1
end
die("--text, --merges and --out are all required") unless opts[:text] && opts[:merges] && opts[:out]
die("--merges must be >= 1, got #{opts[:merges]}") if opts[:merges] < 1
die("no such corpus: #{opts[:text]}") unless File.file?(opts[:text])

%w[vocab merges bytechars].each do |t|
  p_ = "#{opts[:tables]}-#{t}.tsv"
  die("missing tokeniser table #{p_} — run prep/dump_bpe.py") unless File.file?(p_)
end

# ---- tokeniser tables ----------------------------------------------------
# bytechars: GPT-2's byte<->printable-codepoint map. BPE runs over these
# stand-ins, never over raw bytes, so that every byte has a representable
# character and merges cannot straddle an encoding boundary.
byte2char = {}
File.foreach("#{opts[:tables]}-bytechars.tsv") do |ln|
  b, c = ln.chomp("\n").split("\t", 2)
  byte2char[b.to_i] = c
end
die("bytechars table is not 256 entries (got #{byte2char.size})") unless byte2char.size == 256

# merges: rank-ordered pairs. Taking the FIRST n is what makes merge count
# the swept axis — the tables are a superset and the prefix is the rung.
ranks = {}
nm = 0
File.foreach("#{opts[:tables]}-merges.tsv") do |ln|
  break if nm >= opts[:merges]
  r, l, rr = ln.chomp("\n").split("\t", 3)
  next if l.nil? || rr.nil?
  ranks[[l, rr]] = r.to_i
  nm += 1
end
die("asked for #{opts[:merges]} merges, table only had #{nm}") if nm < opts[:merges]

# ---- pre-tokenise --------------------------------------------------------
# GPT-2's own split. Without it this would not be GPT-2 BPE — merges would
# run across word boundaries and the ids would not correspond to the shipped
# vocabulary at all.
PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/

raw = File.binread(opts[:text])
sha = Digest::SHA256.hexdigest(raw)
text = raw.force_encoding("UTF-8")
text = raw.force_encoding("BINARY").encode("UTF-8", invalid: :replace, undef: :replace) unless text.valid_encoding?

def bpe(word, ranks)
  syms = word.chars
  return syms if syms.size < 2
  loop do
    best = nil
    bi = nil
    j = 0
    while j < syms.size - 1
      r = ranks[[syms[j], syms[j + 1]]]
      if r && (best.nil? || r < best)
        best = r
        bi = j
      end
      j += 1
    end
    break if bi.nil?
    syms = syms[0...bi] + [syms[bi] + syms[bi + 1]] + syms[(bi + 2)..]
  end
  syms
end

cache = {}
pieces = []
text.scan(PAT) do |m|
  enc = m.bytes.map { |b| byte2char[b] }.join
  pieces.concat(cache[enc] ||= bpe(enc, ranks))
end
die("corpus produced no tokens") if pieces.empty?

# ---- id assignment -------------------------------------------------------
# CONTIGUOUS by construction. `meta.alphabet` is the count of DISTINCT
# values actually present, not the vocabulary size — the convention
# ae_shakespeare already follows (it declares 65 while its ids run past
# 100), and the one the loader checks.
if opts[:vocab_from]
  src = "#{opts[:vocab_from]}.json"
  die("--vocab-from #{opts[:vocab_from]}: no #{src}") unless File.file?(src)
  meta = JSON.parse(File.read(src))
  vocab = meta["token_ids"]
  die("--vocab-from #{opts[:vocab_from]}: sidecar has no token_ids map") unless vocab.is_a?(Hash)
  missing = pieces.uniq.reject { |t| vocab.key?(t) }
  unless missing.empty?
    die("this corpus uses #{missing.size} token(s) absent from " \
        "#{opts[:vocab_from]}'s vocabulary: #{missing.first(8).inspect}" \
        "#{missing.size > 8 ? ' …' : ''}\n" \
        "  Sharing a head across packs requires a shared vocabulary — a pack " \
        "built without these tokens assigns their ids to something else, so " \
        "the same class id would mean different things in the two packs. " \
        "Rebuild the source pack over the union of both corpora.")
  end
else
  vocab = {}
  pieces.each { |t| vocab[t] = vocab.size unless vocab.key?(t) }
end

toks = pieces.map { |t| vocab[t] }
distinct = toks.uniq.size
maxid = toks.max

# ---- entropy -------------------------------------------------------------
counts = Hash.new(0)
toks.each { |t| counts[t] += 1 }
n = toks.size.to_f
ent = -counts.values.sum { |c| (c / n) * Math.log2(c / n) }
floor_id, floor_c = counts.max_by { |_, c| c }

# ---- write ---------------------------------------------------------------
File.binwrite("#{opts[:out]}.meta.i32", [toks.size, distinct].pack("l<*"))
File.binwrite("#{opts[:out]}.tok.i32",  toks.pack("l<*"))
File.write("#{opts[:out]}.json", JSON.pretty_generate(
  "corpus"        => File.basename(opts[:text]),
  "source_sha256" => sha,
  "tokeniser"     => "gpt2-bpe",
  "tokeniser_tables" => opts[:tables],
  "merges"        => opts[:merges],
  "vocab_from"    => opts[:vocab_from],
  "n_tokens"      => toks.size,
  "alphabet"      => distinct,
  "max_id"        => maxid,
  "entropy_bits"  => ent.round(6),
  "unigram_floor" => (floor_c / n).round(6),
  "floor_id"      => floor_id,
  # The id map is what makes --vocab-from possible at all. It is the pack's
  # contract with any sibling that must share its head.
  "token_ids"     => vocab
) + "\n")

puts "wrote #{opts[:out]}.{meta,tok,json}"
puts "  merges=#{opts[:merges]} n_tokens=#{toks.size} alphabet=#{distinct} " \
     "max_id=#{maxid} entropy=#{ent.round(4)} bits"
if distinct > 4096
  puts "  NOTE: alphabet #{distinct} exceeds Toy::AeTask::VOCAB_MAX (4096), so the"
  puts "  bytelm lane will REFUSE this pack at load. The pack is valid; the"
  puts "  ceiling is a per-step one-hot upload cost on the lane side (toy#184)."
end
