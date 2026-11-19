#!/usr/bin/env ruby
# prep/tokenizer_pre_gate.rb — toy#215 regression gate.
#
# The BPE pre-tokenizer must split text the way the model's own tokenizer
# does. Before toy#215 it knew only ASCII letters and digits, so "été" was
# cut at each accent before BPE ran and came out as more tokens than
# llama.cpp or HuggingFace give; it also ignored tokenizer.ggml.pre, so
# every model got one rule set.
#
# Every expected id list is HuggingFace `tokenizers` output, not something
# toy produced: the SmolLM2-135M tokenizer.json with its pre_tokenizer
# replaced by the one each llama.cpp name stands for (and, for qwen2, the
# NFC normalizer Qwen's tokenizer.json has). The fixture's vocabulary is
# the same in every leg; only tokenizer.ggml.pre is relabelled.
#
# Legs (fixture: data/smollm2-135m-tok.gguf):
#   llama-bpe, qwen2, smollm, gpt-2
#             each rule set → HF ids, on accented words, digits, runs of
#             spaces, newlines, no-break spaces, Greek/Cyrillic, decomposed
#             accents. Three strings tell the four rule sets apart:
#             "end.\n\nNext" (llama-bpe, qwen2 | smollm, gpt-2), "x  1"
#             (smollm | the rest), decomposed "été déjà" (qwen2 | the rest)
#   distinct  the legs above really ran different rules (the id lines of
#             those three strings differ as the reference's do)
#   absent    no tokenizer.ggml.pre → the llama-bpe rules
#   unknown   an unrecognised name is reported and takes the llama-bpe rules
#   qwen      (only with TOY_GATE_QWEN_GGUF=<a Qwen2.5 GGUF with tokenizer>)
#             the issue's own strings → llama.cpp's ids ("été" is 38783)
#
# The SmolLM2 vocabulary has no multi-digit token, so the legs cannot see
# the difference between digit groupings (1..3, one by one, whole runs);
# the qwen leg can ("2026" → four ids).
#
#   ruby prep/tokenizer_pre_gate.rb     # exit 0 pass, 1 fail, 2 unusable

require "open3"
require "tmpdir"

ROOT    = File.expand_path("..", __dir__)
SMOKE   = ENV["TOY_GATE_TOKENIZE"] || File.join(ROOT, "prep", "smokes", "smoke_tokenize")
FIXTURE = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
RELABEL = File.join(ROOT, "prep", "gguf_relabel.rb")
QWEN    = ENV["TOY_GATE_QWEN_GGUF"] || ""

unless File.file?(FIXTURE)
  warn "tokenizer_pre_gate: missing fixture #{FIXTURE} (never skipped silently)"
  exit 2
end
unless ENV["TOY_GATE_TOKENIZE"]
  _o, st = Open3.capture2e("make", "-C", ROOT, "prep/smokes/smoke_tokenize")
  unless st.success?
    warn "tokenizer_pre_gate: could not build prep/smokes/smoke_tokenize"
    exit 2
  end
end

TEXTS = [
  "été",
  "naïve cœur, Noël où ?",
  "à l'été 2026",
  "le 14 juillet 1789 à 20h30",
  "a  b   c ",
  "x\n\n y \n",
  "« poésie » : 12 €",
  "été déjà",
  "I'm sure WE'LL see",
  "Ελληνικά и кириллица",
  "tab\t\tend  ",
  "end.\n\nNext",
  "x  1",
].freeze
NFD, NEWLINES, SPACES = 7, 11, 12   # indexes of the three telling strings

COMMON = [
  %w[19380 2756],
  %w[3546 46494 265 34493 315 28 2838 26547 92 263 45586 9148],
  %w[16739 303 23 19380 2756 216 34 32 34 38],
  %w[290 216 33 36 4871 30537 216 33 39 40 41 25782 216 34 32 88 35 32],
  %w[81 216 278 256 265 216],
  %w[104 1116 329 3717],
  %w[39969 844 17842 519 17228 15442 42 216 33 34 321 124 14195],
  nil,
  %w[57 5248 2090 26587 23 16421 963],
  %w[153 239 17162 17162 25723 11517 12001 21118 26112 29972 31338 7872 46879 12681 35420 32233 6511],
  %w[15999 197 197 486 256],
  nil,
  nil,
].freeze
DECOMPOSED = %w[85 151 219 676 151 219 367 151 219 10824 151 218].freeze
COMPOSED   = %w[19380 2756 32564 90 16739].freeze
WANT = {
  "llama-bpe" => [DECOMPOSED, %w[486 30 1116 8693], %w[104 216 216 33]],
  "qwen2"     => [COMPOSED,   %w[486 30 1116 8693], %w[104 216 216 33]],
  "smollm"    => [DECOMPOSED, %w[486 30 198 198 8693], %w[104 256 33]],
  "gpt-2"     => [DECOMPOSED, %w[486 30 198 198 8693], %w[104 216 216 33]],
}.freeze

def want_for(kind)
  w = COMMON.dup
  w[NFD], w[NEWLINES], w[SPACES] = WANT.fetch(kind)
  w.map { |ids| ids.join(" ") }
end

# → [id lines, pre kind, whole output, ok?]
def tokenize_all(gguf, texts)
  Dir.mktmpdir("toy215") do |dir|
    f = File.join(dir, "texts.txt")
    File.write(f, texts.map { |t| t.gsub("\n", "\\n") }.join("\n") + "\n")
    out, st = Open3.capture2e({ "GGUF" => gguf, "TEXT_FILE" => f, "PLAIN" => "1" }, SMOKE)
    lines = out.lines.select { |l| l.start_with?("ids:") }.map { |l| l.sub("ids:", "").strip }
    return [lines, out[/^pre: (\d+)/, 1].to_i, out, st.success?]
  end
end

def relabel(dir, name, *args)
  path = File.join(dir, "#{name}.gguf")
  o, st = Open3.capture2e("ruby", RELABEL, FIXTURE, path, "--prefix", "llama", *args)
  st.success? ? path : (warn("tokenizer_pre_gate: relabel failed: #{o.lines.last}"); exit 2)
end

failures = []
got_by_kind = {}

Dir.mktmpdir("toy215") do |dir|
  WANT.each_key do |kind|
    gguf = relabel(dir, kind, "--set-str", "tokenizer.ggml.pre=#{kind}")
    got, _k, _out, ok = tokenize_all(gguf, TEXTS)
    got_by_kind[kind] = got
    want = want_for(kind)
    if !ok || got.length != TEXTS.length
      failures << "#{kind}: the tokenizer printed #{got.length} id lines for #{TEXTS.length} texts"
      next
    end
    bad = (0...TEXTS.length).reject { |i| got[i] == want[i] }
    bad.each do |i|
      failures << "#{kind}: #{TEXTS[i].inspect}\n      want: #{want[i]}\n      got:  #{got[i]}"
    end
    puts "  ok: #{kind} — #{TEXTS.length} strings match the HF ids" if bad.empty?
  end

  # ---- distinct ----
  if got_by_kind.values.all? { |g| g.length == TEXTS.length }
    sig = got_by_kind.transform_values { |g| [g[NFD], g[NEWLINES], g[SPACES]] }
    if sig.values.uniq.length == WANT.length
      puts "  ok: distinct — the four names give four different id sets on the three telling strings"
    else
      failures << "distinct: only #{sig.values.uniq.length} different rule sets ran for #{WANT.length} names"
    end
  end

  # ---- absent ----
  bare = relabel(dir, "absent", "--drop", "tokenizer.ggml.pre")
  got, kind, _out, ok = tokenize_all(bare, TEXTS)
  if ok && kind == 0 && got == want_for("llama-bpe")
    puts "  ok: absent — no tokenizer.ggml.pre takes the llama-bpe rules"
  else
    failures << "absent: a GGUF without tokenizer.ggml.pre did not encode as llama-bpe (kind #{kind})"
  end

  # ---- unknown ----
  odd = relabel(dir, "unknown", "--set-str", "tokenizer.ggml.pre=no-such-pre")
  got, kind, out, ok = tokenize_all(odd, TEXTS)
  if !ok || got != want_for("llama-bpe")
    failures << "unknown: an unrecognised name did not encode as llama-bpe"
  elsif out !~ /WARN: tokenizer: unknown pre-tokenizer "no-such-pre"/
    failures << "unknown: an unrecognised name was not reported"
  else
    puts "  ok: unknown — an unrecognised name is reported and takes the llama-bpe rules"
  end
end

# ---- qwen (optional) ----
if QWEN == ""
  puts "  --: qwen — not run (set TOY_GATE_QWEN_GGUF to a Qwen2.5 GGUF with a tokenizer)"
elsif !File.file?(QWEN)
  failures << "qwen: TOY_GATE_QWEN_GGUF=#{QWEN} is not a file"
else
  # llama.cpp /tokenize on the same vocabulary (toy#215), and HF for the rest.
  cases = [
    ["été", %w[38783]],
    ["naïve cœur", %w[3376 37572 586 138295]],
    ["à l'été", %w[6362 326 6 38783]],
    ["été", %w[38783]],
    ["Noël où 2026  x", %w[2753 73453 38623 220 17 15 17 21 220 856]],
  ]
  got, kind, _out, ok = tokenize_all(QWEN, cases.map(&:first))
  bad = cases.each_index.reject { |i| got[i] == cases[i][1].join(" ") }
  if !ok || kind != 1
    failures << "qwen: #{File.basename(QWEN)} did not select the qwen2 rules (kind #{kind}); is its tokenizer.ggml.pre \"qwen2\"?"
  elsif bad.empty?
    puts "  ok: qwen — #{cases.length} strings match the reference ids on #{File.basename(QWEN)}"
  else
    bad.each { |i| failures << "qwen: #{cases[i][0].inspect}\n      want: #{cases[i][1].join(' ')}\n      got:  #{got[i]}" }
  end
end

if failures.empty?
  puts "GATE PASS [tokenizer-pre]: each pre-tokenizer name splits text as its reference tokenizer does"
  exit 0
end
failures.each { |f| puts "  FAIL: #{f}" }
puts "GATE FAIL [tokenizer-pre]: #{failures.length} failure(s)"
exit 1
