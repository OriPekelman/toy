require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_seq_engine"
require_relative "../lib/toy/dev/toy_describe_flow"
require_relative "../lib/toy/train/toy_drift_grad"

GGUF    = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
CONTEXT = (ENV["CONTEXT"] || "16").to_i

puts "m1"
cfg = SmolLM2ConfigLoader.read(GGUF)
puts "m2"
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "m3"
gguf = TinyNN.tnn_gguf_load(GGUF)
puts "m4"
seq = Toy::LLM::Engine::LlamaSeqEngine.new
puts "m5"
seq.realize_for_mmap(gguf, cfg, CONTEXT, flags.untied, flags.qkv_bias)
puts "m6 (realize done)"

n_drift = ToyDriftGrad.params(seq.sess).length
puts "m7 (drift params=" + n_drift.to_s + ")"
n_card = ToyDescribeFlow.param_count(seq.sess)
puts "m8 (card param_count=" + n_card.to_s + ")"

puts "m9 (about to call card())"
card = ToyDescribeFlow.card(seq.sess)
puts "m10 (card returned)"

# Cross-check counts surface in the Card as hypers (P1 workaround
# for the step_bind static-init landmine).
puts "card.hypers:"
i = 0
while i < card.hypers.length
  h = card.hypers[i]
  puts "  " + h.key + " = " + h.value
  i = i + 1
end

puts "card.inputs.count = " + card.inputs.length.to_s
puts "card.steps.count  = " + card.steps.length.to_s

ret_var = ""
k = 0
while k < card.steps.length
  s = card.steps[k]
  if s.kind == "return"
    ret_var = s.var
  end
  k = k + 1
end
puts "card.return       = " + ret_var

# Verify Card renders without crashing (downstream contract: any
# derived Card must survive render_pseudocode).
text = card.render_pseudocode
puts "render_pseudocode ok (" + text.length.to_s + " chars)"

# ── P1.3 verification gates ──
# Fail loud (exit 1 + warn) if any invariant breaks. Per memory
# feedback_never_mask_fail_loud.
failures = [""]; failures.pop

if n_drift != n_card
  failures.push("ToyDriftGrad.params count (" + n_drift.to_s +
                ") != ToyDescribeFlow.param_count (" + n_card.to_s + ")")
end

n_nodes_hyper = "-1"
n_leafs_hyper = "-1"
k = 0
while k < card.hypers.length
  h = card.hypers[k]
  if h.key == "nodes.compute"
    n_nodes_hyper = h.value
  end
  if h.key == "leafs.total"
    n_leafs_hyper = h.value
  end
  k = k + 1
end
if n_nodes_hyper == "-1"
  failures.push("nodes.compute hyper not present on derived Card")
elsif n_nodes_hyper.to_i <= 0
  failures.push("nodes.compute = " + n_nodes_hyper + " (expected > 0; graph walk yielded no compute nodes)")
end
if n_leafs_hyper == "-1"
  failures.push("leafs.total hyper not present on derived Card")
elsif n_leafs_hyper.to_i <= 0
  failures.push("leafs.total = " + n_leafs_hyper + " (expected > 0; graph walk found no weight/input leaves)")
end

if text.length <= 0
  failures.push("Card#render_pseudocode returned empty")
end

if card.steps.length <= 0
  failures.push("Card has no steps (expected at least the return step)")
end

if failures.length > 0
  STDERR.puts "FAIL — Card derivation invariants violated:"
  k = 0
  while k < failures.length
    STDERR.puts "  - " + failures[k]
    k = k + 1
  end
  exit 1
end

puts "OK (P1.3 gates passed)"
