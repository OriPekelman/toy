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
# ── SHARED VOCABULARY, AND WHY A LADDER NEEDS --rungs ───────────────────
# A ladder read against a common baseline, or a pretrain/adapt pair, MUST be
# emitted against a shared vocabulary or the same class id means different
# things in different packs. `--vocab-from <prefix>` reuses another pack's
# id assignment and FAILS LOUD on any token the source vocabulary does not
# contain, naming them.
#
# BUT BPE RUNGS DO NOT NEST, and this is stronger than D5's caution rather
# than the same shape. D5's worry was a held-out corpus using node types the
# pretrain corpus lacked — a subset relation that holds if you build the
# bigger side first. Here, on ONE corpus, NEITHER rung's vocabulary contains
# the other's:
#
#   at 4000 merges a word tokenises as "rou" + "nd"
#   at 8000 merges that pair is itself merged, giving "round"
#
# so "rou" exists ONLY in the smaller rung. Measured on tinyshakespeare:
# 4k-from-8k is missing 71 tokens, 8k-from-4k is missing 2000. Building the
# largest rung first does NOT give a superset — that was the obvious guess
# and it is false.
#
# So a shared head across a ladder needs ONE id assignment covering every
# token ANY rung produces: tokenise per rung, union the tokens, assign ids
# once, emit every rung against them.
#
# THAT IS NOT IMPLEMENTED HERE, deliberately. VOCAB_MAX is 4096 and only the
# 4k rung is loadable, so a 4k/8k/16k ladder cannot be trained today and a
# flag to build one could not be exercised — an unexercised builder for an
# untrainable ladder is a liability, not a feature. When the ceiling moves,
# this is the shape to add, and the union must be over the RUNGS (one
# corpus, several merge counts), not over corpora as D5 needed.
#
# Until then `--vocab-from` covers the case that IS reachable: two corpora
# at the SAME merge count sharing a head.
#
# ── WITH --vocab-from, ids MAY BE SPARSE, AND `alphabet` IS NOT THE HEAD ─
# Without it, ids are contiguous 0..alphabet-1 by construction. WITH it the
# ids come from the source pack's map, so a corpus that uses only some of
# those tokens gets GAPS: `alphabet` counts distinct values PRESENT, while
# the head must cover `max_id + 1`.
#
# The two coincide only when the borrowing corpus happens to use a
# first-appearance prefix of the source's tokens — which is exactly what a
# leading slice of the same text does, and is why a naive test of this path
# looks contiguous and proves nothing about the general case.
#
# `max_id` is in the sidecar for this reason. Size the lane's head from
# max_id + 1, never from alphabet, on any pack built with --vocab-from.
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

# --text takes EITHER a plain file or an existing character-level pack
# prefix. The repo ships its corpora packed, not as .txt — ae_shakespeare
# exists only as .meta/.tok/.json — and the natural first rung is exactly
# that corpus, so that a BPE ladder can be read directly against every
# character-level number the programme already has on the same text.
#
# A character pack's ids ARE raw byte values (ae_shakespeare declares
# alphabet 65 while its ids run past 100, and floor_id 32 is ASCII space),
# so reconstructing the bytes is an unpack, not a decode.
if File.file?(opts[:text])
  src_kind = "file"
elsif File.file?("#{opts[:text]}.tok.i32")
  src_kind = "pack"
else
  die("no such corpus: #{opts[:text]} (looked for the file, and for " \
      "#{opts[:text]}.tok.i32 as a character pack)")
end

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

if src_kind == "pack"
  ids = File.binread("#{opts[:text]}.tok.i32").unpack("l<*")
  bad = ids.reject { |v| v >= 0 && v <= 255 }
  unless bad.empty?
    die("#{opts[:text]} is not a CHARACTER pack — #{bad.size} id(s) fall " \
        "outside 0..255 (e.g. #{bad.first(4).inspect}). BPE runs over bytes, " \
        "so a pack whose ids are already subword tokens cannot be re-tokenised.")
  end
  raw = ids.pack("C*")
else
  raw = File.binread(opts[:text])
end
sha = Digest::SHA256.hexdigest(raw)
text = raw.dup.force_encoding("UTF-8")
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
  "corpus_kind"   => src_kind,
  "source_sha256" => sha,
  "tokeniser"     => "gpt2-bpe",
  "tokeniser_tables" => opts[:tables],
  "merges"        => opts[:merges],
  "vocab_from"    => opts[:vocab_from],
  "n_tokens"      => toks.size,
  "alphabet"      => distinct,
  "max_id"        => maxid,
  "entropy_bits"  => ent.round(6),
  # ── READ THIS BEFORE COMPARING TO A CHARACTER-LEVEL NUMBER ──────────
  # The lane reports bpb PER TOKEN. A BPE token is several characters, so
  # a BPE bpb and a character bpb on the SAME text are not on the same
  # scale and must not be compared directly. On tinyshakespeare at 4000
  # merges the raw figures are 11.94 (BPE) against 4.43 (character) —
  # which reads as a catastrophic regression and is not one: divide by
  # chars_per_token and it is 4.54 against 4.43.
  #
  # N4 exists to read BPE rungs against the character-level numbers this
  # programme already has on this text, so the conversion is recorded
  # here rather than left to be rederived (or forgotten) per comparison.
  "n_source_bytes"   => raw.bytesize,
  "chars_per_token"  => (raw.bytesize / toks.size.to_f).round(6),
  "bpb_per_char_from_per_token" => "divide the lane's bpb by chars_per_token",
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
