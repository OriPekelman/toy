# prep/smokes/lora_node_dump.rb — spinel-dev#33 first-diverging-op harness.
#
# Runs ONE LoRA training step (step 0, is_first) on the EXACT gate config
# (smoke_recipe_lora.rb ground truth: smollm2-135m base, RANK=8, seed 42,
# init_scale 0.01, TOKENS=[12092,4845,253,1429], TARGET_ID=99), then walks
# EVERY graph node and prints a per-node checksum:
#
#   NODE <i> op=<OP> ne=<n> [ne0 ne1 ne2 ne3] sum=<s> sumabs=<a> v0=<..> vN=<..> name=<..>
#
# Compile with BOTH toolchains and diff:
#   make prep/smokes/lora_node_dump
#   SPINEL_DIR=~/sites/spinel-union-pin make -B prep/smokes/lora_node_dump && ./prep/smokes/lora_node_dump > /tmp/pin.txt
#   SPINEL_DIR=/srv/data/scratch/spinel-wt-master make -B prep/smokes/lora_node_dump && ./prep/smokes/lora_node_dump > /tmp/master.txt
#   diff <(grep '^NODE' /tmp/pin.txt) <(grep '^NODE' /tmp/master.txt) | head
# The FIRST differing NODE line = the first diverging op (master vs union).
# step-1 loss differs (pin 9.236 vs master 10.291), so the divergence is a
# forward op at/before the CE loss. CPU-only.

require_relative "../../lib/toy"
require_relative "../../lib/toy/models/toy_smollm2"
require_relative "../../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../../lib/toy/llm/engine/llama_seq_engine"
require_relative "../../lib/toy/llm/adamw"
require_relative "../../lib/toy/llm/recipes/lora"

GGUF  = ENV["GGUF"]  || "data/smollm2-135m-native.gguf"
RANK  = (ENV["RANK"]  || "8").to_i
CAP   = (ENV["CAP"]   || "524288").to_i   # skip-checksum nodes larger than this

TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

if !File.exist?(GGUF)
  puts "lora_node_dump: cannot find " + GGUF
  exit 1
end

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
gguf  = TinyNN.tnn_gguf_load(GGUF)

recipe = Toy::LLM::Recipes::LoRA.new
opts = Toy::LLM::RecipeOptions.new
opts.t_seq      = TOKENS.length
opts.untied     = flags.untied ? true : false
opts.qkv_bias   = flags.qkv_bias ? true : false
opts.seed       = 42
opts.init_scale = 0.01
recipe.realize!(gguf, cfg, RANK, opts)

m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0
while i < TOKENS.length * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

adamw = Toy::AdamW.for_lora
adamw.lr = (ENV["LR"] || "0.001").to_f
positions = [0, 1, 2, 3]

m_hp = adamw.hp_for_step(0)
loss = recipe.step!(TOKENS, positions, m_labels, m_hp, true)
puts "step 1: loss=" + loss.to_s

sess = recipe.lora_cache.sess

# spinel-dev#33 decisive check: the gather INDICES feeding NODE 0 (GET_ROWS).
# These MUST be the TOKENS [12092, 4845, 253, 1429]. If master shows other
# values, the token-id index buffer is corrupted (#1449 aliasing family) and
# the wrong embedding rows are gathered → wrong forward → flat loss.
t_ids = recipe.lora_cache.t_seq_token_ids
TinyNN.tnn_download(sess, t_ids)
ne_ids = TinyNN.tnn_tensor_nelements(t_ids)
puts "token_ids ne=" + ne_ids.to_s
j = 0
while j < ne_ids
  puts "  token_id[" + j.to_s + "]=" + TinyNN.tnn_scratch_get_i32(sess, j).to_s
  j = j + 1
end

nn   = TinyNN.tnn_graph_n_nodes(sess)
puts "graph_n_nodes=" + nn.to_s

# Reused download buffer (big enough for the logits; nodes over CAP skip).
buf = Mat.new(1, CAP)

idx = 0
while idx < nn
  node = TinyNN.tnn_graph_node(sess, idx)
  op   = TinyNN.tnn_tensor_op_name(node)
  nm   = TinyNN.tnn_tensor_name(node)
  ne   = TinyNN.tnn_tensor_nelements(node)
  n0   = TinyNN.tnn_tensor_ne0(node)
  n1   = TinyNN.tnn_tensor_ne1(node)
  n2   = TinyNN.tnn_tensor_ne2(node)
  n3   = TinyNN.tnn_tensor_ne3(node)

  if ne > 0 && ne <= CAP
    TinyNN.tnn_download(sess, node)
    TinyNN.tnn_download_to_f64_array(sess, node, buf.flat, ne)
    s   = 0.0
    a   = 0.0
    k   = 0
    while k < ne
      v = buf.flat[k]
      s = s + v
      av = v < 0.0 ? (0.0 - v) : v
      a = a + av
      k = k + 1
    end
    v0   = buf.flat[0]
    vlst = buf.flat[ne - 1]
    puts "NODE " + idx.to_s + " op=" + op + " ne=" + ne.to_s +
         " [" + n0.to_s + " " + n1.to_s + " " + n2.to_s + " " + n3.to_s + "]" +
         " sum=" + s.to_s + " sumabs=" + a.to_s +
         " v0=" + v0.to_s + " vN=" + vlst.to_s + " name=" + nm
  else
    puts "NODE " + idx.to_s + " op=" + op + " ne=" + ne.to_s +
         " [" + n0.to_s + " " + n1.to_s + " " + n2.to_s + " " + n3.to_s + "]" +
         " (skipped: ne>" + CAP.to_s + ") name=" + nm
  end
  idx = idx + 1
end
puts "DONE"
