#!/usr/bin/env ruby
# prep/build_oracle_map.rb — a pooling map pack for GTX_DFA_ADAPT=oracle.
#
# WHY THIS EXISTS. The oracle path (toy#176 / D1) reads its pooling P from a
# two-file map pack, and no such pack was committed. So the path could be
# gated on its CONTRACT — every guard refusing for its own reason — but not
# on its BEHAVIOUR, and toy#183's oracle variants had nothing to run against.
# A path carrying this programme's most striking result (perfect routing
# performing exactly like never training the body) deserves better than a
# fixture that lives only in someone's scratch directory.
#
# THE FORMAT, from gtx_engine.rb's init_p_from_map!:
#   <prefix>.meta.i32   [n_codes, n_base]
#   <prefix>.tok.i32    n_codes row ids — code c maps to base symbol map[c]
#
# P is then the indicator matrix of that pooling, row-normalised, so it is
# exactly orthonormal and adapt_p!(apply=0) measures it unchanged.
#
# The engine requires: n_base <= GTX_DFA_RANK, n_codes <= head classes, and
# EVERY base row non-empty. The last one is the easy mistake — a map with an
# unused base symbol is refused, which is why `modulo` is the default here:
# it fills every row by construction for any n_codes >= n_base.
#
#   ruby prep/build_oracle_map.rb --codes 256 --base 65 \
#                                 --out data/oracle_map_c256_b65
#
# `modulo` is deliberately NOT a semantically meaningful pooling — it is a
# FIXTURE, and a gate that asserts the oracle's mechanics wants a map whose
# structure is trivially predictable, not one that smuggles in a hypothesis
# about which codes belong together. A research pooling (by byte class, by
# frequency, by an inflation design) belongs in the research repo and should
# say so in its own provenance.
require "json"

def die(m)
  warn "build_oracle_map: #{m}"
  exit 1
end

opts = { codes: nil, base: nil, out: nil, strategy: "modulo" }
i = 0
while i < ARGV.length
  case ARGV[i]
  when "--codes"    then opts[:codes]    = Integer(ARGV[i += 1])
  when "--base"     then opts[:base]     = Integer(ARGV[i += 1])
  when "--out"      then opts[:out]      = ARGV[i += 1]
  when "--strategy" then opts[:strategy] = ARGV[i += 1]
  else die("unknown flag #{ARGV[i].inspect}")
  end
  i += 1
end
die("--codes, --base and --out are all required") unless opts[:codes] && opts[:base] && opts[:out]
die("--codes must be >= 1") if opts[:codes] < 1
die("--base must be >= 1") if opts[:base] < 1
# The engine refuses an empty base row, so refuse it here too rather than
# emit a pack that is guaranteed to be rejected at load.
die("--base #{opts[:base]} exceeds --codes #{opts[:codes]} — every base row " \
    "must receive at least one code, and the engine refuses a map that " \
    "leaves one empty") if opts[:base] > opts[:codes]

map = case opts[:strategy]
      when "modulo" then (0...opts[:codes]).map { |c| c % opts[:base] }
      when "block"
        # Contiguous blocks. Sizes differ by at most one, so no row is empty.
        per = opts[:codes].to_f / opts[:base]
        (0...opts[:codes]).map { |c| [(c / per).floor, opts[:base] - 1].min }
      else die("--strategy #{opts[:strategy].inspect} unsupported (modulo|block)")
      end

counts = Array.new(opts[:base], 0)
map.each { |b| counts[b] += 1 }
empty = counts.each_index.select { |i2| counts[i2].zero? }
die("strategy #{opts[:strategy]} left #{empty.size} base row(s) empty " \
    "(#{empty.first(5).inspect}) — the engine refuses such a map") unless empty.empty?

File.binwrite("#{opts[:out]}.meta.i32", [opts[:codes], opts[:base]].pack("l<*"))
File.binwrite("#{opts[:out]}.tok.i32",  map.pack("l<*"))
File.write("#{opts[:out]}.json", JSON.pretty_generate(
  "kind"      => "oracle-pooling-map",
  "n_codes"   => opts[:codes],
  "n_base"    => opts[:base],
  "strategy"  => opts[:strategy],
  "codes_per_base" => { "min" => counts.min, "max" => counts.max },
  "note"      => "FIXTURE for gating GTX_DFA_ADAPT=oracle (toy#176/#183). " \
                 "The pooling is arbitrary by design: a gate wants predictable " \
                 "structure, not a hypothesis about which codes belong together.",
  "requires"  => "GTX_DFA_RANK >= #{opts[:base]}; head classes >= #{opts[:codes]}"
) + "\n")

puts "wrote #{opts[:out]}.{meta,tok,json}"
puts "  codes=#{opts[:codes]} base=#{opts[:base]} strategy=#{opts[:strategy]} " \
     "codes_per_base=#{counts.min}..#{counts.max}"
