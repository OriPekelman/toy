#!/usr/bin/env ruby
# prep/remap_alphabet.rb — toy#170 spec P4 follow-up: the CONTROLLED
# alphabet remap.
#
# P4 varied the alphabet by varying the CORPUS, so alphabet moved
# together with difficulty, redundancy and language count. This holds
# ONE corpus fixed (shakespeare, 65 symbols) and moves only the active
# output rank.
#
# ── INFLATION (the clean control) ──
#
# Each of the 65 symbols is assigned m private codes, and every
# occurrence picks one of its m codes UNIFORMLY AT RANDOM. So:
#
#   * the active output rank becomes 65*m,
#   * the learnable structure is IDENTICAL — the code still determines
#     the symbol, and the symbol sequence is untouched,
#   * exactly log2(m) bits of IRREDUCIBLE noise are added per token,
#     and they are added to EVERY ARM EQUALLY.
#
# That last property is what makes this a control: an ideal predictor's
# loss rises by log2(m) on every arm, so the DIFFERENCE between two arms
# (frozen - dfa, frozen - bp) is unchanged in theory. If DFA's absolute
# bits recovered falls as m grows, the output dimension caused it, with
# no difficulty confound available to explain it away.
#
#   m=2 -> alphabet 130
#   m=3 -> alphabet 195   <- sits next to udhr's 201, which is the
#                            direct test of P4's language confound:
#                            same rank, vastly different difficulty.
#
# ── DEFLATION (the low-end probe) ──
#
# The P4 low end (names, 27) came from a DIFFERENT corpus, so its drop
# could be corpus shape rather than alphabet. `a27` merges shakespeare's
# own tail into one OTHER class to hit rank 27 on the same text. This
# one is NOT entropy-preserving — merging destroys information and makes
# the task easier — so it is read against its own bp/frozen anchors, not
# against the inflation arms.
#
# Codes stay under 256 (the nominal head) in every variant.
require "json"
require "digest"

SRC = "data/ae_shakespeare"
SEED = 20260817   # fixed: the packs must be reproducible byte-for-byte

def read_pack(prefix)
  meta = File.binread(prefix + ".meta.i32").unpack("l<*")
  toks = File.binread(prefix + ".tok.i32").unpack("l<*")
  raise "meta/tok disagree: #{meta[0]} vs #{toks.size}" unless meta[0] == toks.size
  [toks, meta[1]]
end

# DENSIFY — relabel the used codes to 0..k-1 in ascending order.
#
# This is what lets the NOMINAL head narrow to k. Under `--vocab k` the
# head, the embedding and the DFA feedback matrix B are all k wide, and a
# token id >= k has no row: it would index off the end of the one-hot
# label. So a pack destined for the nominal axis must use every code
# below its own alphabet.
#
# It is a no-op wherever the remap already produced contiguous codes
# (deflate, and inflate x2 here), and the sha256 in the .json is the
# check that it was: a pack whose bytes move has invalidated every cell
# already measured against it.
def densify(toks)
  used = toks.uniq.sort
  return toks if used.first == 0 && used.last == used.size - 1
  m = {}
  used.each_with_index { |c, i| m[c] = i }
  toks.map { |t| m[t] }
end

# toy#170 (P6) — the ceiling is AeTask::VOCAB_MAX, and it is about the
# DENSE one-hot label upload, not the file format. Kept in sync by hand
# because prep/ must not require the lib (see the lib-vs-example rule);
# the runner refuses anything past it, so a drift here fails loud there
# rather than producing a pack nothing can load.
MAX_CODE = Integer(ENV["REMAP_MAX_CODE"] || "4096")

def write_pack(prefix, toks, note)
  toks = densify(toks)
  distinct = toks.uniq.size
  if toks.max >= MAX_CODE
    raise "#{prefix}: code #{toks.max} >= #{MAX_CODE} — past what a dense " \
          "one-hot label upload carries; raise AeTask::VOCAB_MAX first, and " \
          "read its comment on why 50k is out of scope by design"
  end
  raise "#{prefix}: not dense (max #{toks.max}, distinct #{distinct})" if toks.max != distinct - 1
  counts = Hash.new(0); toks.each { |t| counts[t] += 1 }
  n = toks.size.to_f
  ent = -counts.values.sum { |c| (c / n) * Math.log2(c / n) }
  File.binwrite(prefix + ".meta.i32", [toks.size, distinct].pack("l<*"))
  File.binwrite(prefix + ".tok.i32",  toks.pack("l<*"))
  File.write(prefix + ".json", JSON.pretty_generate(
    "source"   => SRC,
    "remap"    => note,
    "seed"     => SEED,
    "n_tokens" => toks.size,
    "distinct" => distinct,
    "entropy"  => ent,
    "sha256"   => Digest::SHA256.hexdigest(toks.pack("l<*"))))
  puts "  %-22s n=%d alphabet=%-4d entropy=%.4f b" % [prefix, toks.size, distinct, ent]
  ent
end

toks, alpha = read_pack(SRC)
puts "source: #{SRC} n=#{toks.size} alphabet=#{alpha}"

# Rank symbols by frequency, descending — deterministic, so the packs
# do not depend on hash iteration order.
counts = Hash.new(0); toks.each { |t| counts[t] += 1 }
ranked = counts.sort_by { |sym, c| [-c, sym] }.map(&:first)
rank_of = {}; ranked.each_with_index { |s, i| rank_of[s] = i }

base_ent = nil
n = toks.size.to_f
base_ent = -counts.values.sum { |c| (c / n) * Math.log2(c / n) }
puts "base entropy = %.4f b/token" % base_ent
puts

# ── inflation: m private codes per symbol, chosen uniformly at random ──
#
# m=1 is the DEGENERATE case and it earns its place: it is a pure
# relabeling of shakespeare into dense ids, so it adds log2(1)=0 bits and
# is the SAME TASK as the anchor up to a permutation of the embedding and
# head rows. Two things fall out of it. It gives the nominal-head axis a
# dense 65 point (the raw pack uses byte ids, max 122, so `--vocab 65`
# cannot be run on it). And running it at head 256 must reproduce
# ae_shakespeare's numbers within seed noise — which is the only end-to-
# end check that this remap pipeline does not itself move the result.
# toy#170 (P6) — the ladder. m=1/2/3 were P5 and are UNCHANGED by adding
# rungs above them: the stream is seeded per m (`SEED + m`) and densify is
# deterministic, so each pack's sha depends only on its own m. m=6 is the
# noise-linearity pre-check that gates the rest; 8/16/40 are the ladder
# proper, topping out near what the dense labels carry.
#
# Note the rungs land BELOW 65*m: a symbol occurring fewer than m times
# cannot use all m of its codes, so the realised rank is what the pack
# reports, never what the arithmetic predicts. The runs read it from
# provenance for exactly this reason.
REMAP_MS = (ENV["REMAP_MS"] || "1 2 3 6 8 16 40").split.map { |x| Integer(x) }
REMAP_MS.each do |m|
  rng = Random.new(SEED + m)
  out = toks.map { |t| rank_of[t] * m + rng.rand(m) }
  ent = write_pack("data/ae_shak_a#{out.uniq.size}", out,
                   "inflate x#{m}: each symbol -> #{m} private codes, uniform random per occurrence")
  # The whole design rests on this: the added entropy must be log2(m)
  # and nothing else. If it is not, the variant is not a control and
  # every arm difference measured on it is confounded.
  want = base_ent + Math.log2(m)
  if (ent - want).abs > 0.01
    abort "  REFUSED: inflation x#{m} added %.4f bits, expected log2(%d)=%.4f — " \
          "this is not an entropy-preserving control" % [ent - base_ent, m, Math.log2(m)]
  end
  puts "  ok: added %.4f bits, expected %.4f (log2 %d)" % [ent - base_ent, Math.log2(m), m]
end
puts

# ── deflation: keep the 26 most frequent symbols, merge the tail ──
KEEP = 26
merged = toks.map { |t| r = rank_of[t]; r < KEEP ? r : KEEP }
write_pack("data/ae_shak_a27", merged,
           "deflate: top #{KEEP} symbols kept, tail merged into one OTHER class")
puts "  (NOT entropy-preserving by construction — merging destroys information;"
puts "   read against its own bp/frozen anchors, not against the inflation arms)"
