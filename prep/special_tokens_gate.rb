#!/usr/bin/env ruby
# prep/special_tokens_gate.rb — toy#208 regression gate.
#
# A special token written out in prompt text ("<|im_start|>user\n…") must
# become its ONE id, as it does in HuggingFace's tokenizer and in
# llama.cpp. Before toy#208 the tokenizer ran the whole string through the
# pre-tokenizer + BPE, so "<|im_start|>" came out as `<`, `|`, `im`,
# `_start`, `|`, `>` — a ChatML prompt the model never saw, and wrong
# output with exit 0.
#
# Every expected id list below is HuggingFace `tokenizers` output for the
# model's own tokenizer.json (encode(text, add_special_tokens=False)),
# not something toy produced: SmolLM2-135M for the fixture legs,
# Qwen2.5-7B-Instruct for the optional leg.
#
# Legs (fixture: data/smollm2-135m-tok.gguf, 17 specials):
#   chatml    ChatML strings → HF ids, specials as single ids
#   edges     special at the start / end / back to back / touching
#             non-ASCII text / a near-miss ("<|im_start>") that must NOT match
#   ordinary  text with no specials is the ids it always was
#   plain     encode_plain still BPE-splits a special (the opt-out works,
#             and proves the special pass is what changes the ids)
#   untyped   the same GGUF with tokenizer.ggml.token_type removed registers
#             0 specials and encodes as before — no token_type, no change
#   infer     toy-infer on a ChatML --prompt reports the HF token count
#   qwen      (only with TOY_GATE_QWEN_GGUF=<a Qwen2.5 GGUF with tokenizer>)
#             the issue's own string → 9 ids, 151644/151645 single
#
#   ruby prep/special_tokens_gate.rb     # exit 0 pass, 1 fail, 2 unusable

require "open3"
require "tmpdir"

ROOT    = File.expand_path("..", __dir__)
SMOKE   = ENV["TOY_GATE_TOKENIZE"] || File.join(ROOT, "prep", "smokes", "smoke_tokenize")
RUNNER  = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, "libexec", "toy-infer")
FIXTURE = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
RELABEL = File.join(ROOT, "prep", "gguf_relabel.rb")
QWEN    = ENV["TOY_GATE_QWEN_GGUF"] || ""

unless File.file?(FIXTURE)
  warn "special_tokens_gate: missing fixture #{FIXTURE} (never skipped silently)"
  exit 2
end
[["prep/smokes/smoke_tokenize", "TOY_GATE_TOKENIZE"], ["libexec/toy-infer", "TOY_GATE_RUNNER"]].each do |target, env|
  next if ENV[env]
  _o, st = Open3.capture2e("make", "-C", ROOT, target)
  unless st.success?
    warn "special_tokens_gate: could not build #{target}"
    exit 2
  end
end

# → [ids string, specials count, ok?]
def tokenize(gguf, text, plain = false)
  env = { "GGUF" => gguf, "TEXT" => text }
  env["PLAIN"] = "1" if plain
  out, st = Open3.capture2e(env, SMOKE)
  ids = out[/^ids:(.*)$/, 1].to_s.strip
  [ids, out[/^specials: (\d+)/, 1].to_i, st.success?]
end

failures = []

def check(failures, leg, gguf, cases)
  bad = 0
  cases.each do |text, want|
    got, _n, ok = tokenize(gguf, text)
    next if ok && got == want.join(" ")
    bad += 1
    failures << "#{leg}: #{text.inspect}\n      want: #{want.join(' ')}\n      got:  #{got}"
  end
  puts "  ok: #{leg} — #{cases.length} strings match the HF ids" if bad.zero?
end

# ---- chatml ----
CHATML = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
CHATML_IDS = [1, 4093, 198, 19556, 2, 198, 1, 520, 9531, 198]
check(failures, "chatml", FIXTURE, [
  [CHATML, CHATML_IDS],
  ["<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n",
   [1, 4093, 198, 30904, 24583, 2, 198, 1, 520, 9531, 198]],
  ["<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
   [1, 9690, 198, 6077, 5453, 30, 2, 198, 1, 4093, 198, 1780, 314, 216, 34, 27, 34, 47, 2, 198, 1, 520, 9531, 198]],
])

# ---- edges ----
check(failures, "edges", FIXTURE, [
  ["<|endoftext|>", [0]],
  ["<|im_start|><|im_end|>", [1, 2]],
  ["a<|im_end|>b <|im_end|> c", [81, 2, 82, 216, 2, 265]],
  ["café<|im_end|>été", [83, 1939, 2756, 2, 19380, 2756]],
  ["x <|im_start> y <tool_call>z</tool_call>",
   [104, 2067, 108, 306, 79, 3738, 46, 329, 2067, 19324, 79, 4350, 46, 106, 9617, 19324, 79, 4350, 46]],
])

# ---- ordinary ----
check(failures, "ordinary", FIXTURE, [
  ["Once upon a time", [6403, 1980, 253, 655]],
  ["Hello, world! I am a 1", [19556, 28, 905, 17, 339, 744, 253, 216, 33]],
])

# ---- plain ----
p_ids, n_special, p_ok = tokenize(FIXTURE, CHATML, true)
p_list = p_ids.split.map(&:to_i)
if !p_ok || n_special != 17
  failures << "plain: expected 17 registered specials, got #{n_special}"
elsif p_list.include?(1) || p_list.include?(2)
  failures << "plain: encode_plain produced a special id (#{p_ids})"
elsif p_list.length <= CHATML_IDS.length
  failures << "plain: encode_plain did not BPE-split the specials (#{p_list.length} ids)"
else
  puts "  ok: plain — encode_plain gives #{p_list.length} pieces and no special id (encode gives #{CHATML_IDS.length})"
end

# ---- untyped ----
Dir.mktmpdir("toy208") do |dir|
  bare = File.join(dir, "untyped.gguf")
  o, st = Open3.capture2e("ruby", RELABEL, FIXTURE, bare, "--prefix", "llama",
                          "--drop", "tokenizer.ggml.token_type")
  if !st.success?
    failures << "untyped: could not manufacture the fixture: #{o.lines.last.to_s.strip}"
  else
    u_ids, u_n, u_ok = tokenize(bare, CHATML)
    o_ids, _n, o_ok  = tokenize(bare, "Once upon a time")
    if !u_ok || !o_ok
      failures << "untyped: tokenizer did not load without token_type"
    elsif u_n != 0
      failures << "untyped: #{u_n} specials registered with no token_type array"
    elsif u_ids != p_ids
      failures << "untyped: ids differ from the plain encoding\n      plain:   #{p_ids}\n      untyped: #{u_ids}"
    elsif o_ids != "6403 1980 253 655"
      failures << "untyped: ordinary text changed (#{o_ids})"
    else
      puts "  ok: untyped — no token_type → 0 specials, ids are the plain encoding"
    end
  end
end

# ---- infer ----
i_out, i_st = Open3.capture2e({ "GGUF" => FIXTURE, "PROMPT" => CHATML, "N_NEW" => "2" }, RUNNER)
n_tok = i_out[/→ (\d+) tokens/, 1].to_i
if !i_st.success?
  failures << "infer: toy-infer failed on a ChatML prompt (#{i_out.lines.last.to_s.strip})"
elsif n_tok != CHATML_IDS.length
  failures << "infer: toy-infer saw #{n_tok} prompt tokens, HF gives #{CHATML_IDS.length}"
else
  puts "  ok: infer — toy-infer sees #{n_tok} prompt tokens for the ChatML string"
end

# ---- qwen (optional) ----
if QWEN.empty?
  puts "  not run: qwen — set TOY_GATE_QWEN_GGUF to a Qwen2.5 GGUF with an embedded tokenizer"
elsif !File.file?(QWEN)
  failures << "qwen: TOY_GATE_QWEN_GGUF=#{QWEN} does not exist"
else
  check(failures, "qwen", QWEN, [
    ["<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n",
     [151644, 872, 198, 81581, 151645, 198, 151644, 77091, 198]],
    ["x <|im_start> y <tool_call>z</tool_call>",
     [87, 82639, 318, 4906, 29, 379, 220, 151657, 89, 151658]],
    ["Hello, world! I am a 1", [9707, 11, 1879, 0, 358, 1079, 264, 220, 16]],
  ])
end

if failures.empty?
  puts "GATE PASS [special-tokens]: special tokens in prompt text encode to their single ids (HF-identical), ordinary text and untyped GGUFs are unchanged"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [special-tokens]: #{failures.length} failure(s)"
exit 1
