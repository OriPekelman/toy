# examples/smoke_compute_surface.rb — proves the toy#42 full-API require
# (lib/toy/compute.rb) compiles under Spinel and yields a working compute
# surface from ONE require.
#
# The BUILD step is itself the combined-surface gate: Spinel compiles every
# file lib/toy/compute.rb pulls, so a clean build proves all three engines +
# recipes + loaders co-compile in one program (the runners each load only one
# engine — this is the only place the full surface is compiled together). The
# RUN then realizes a tiny LlamaSeqEngine to prove the surface is live.
#
#   make examples/smoke_compute_surface && ./examples/smoke_compute_surface
#
# Expected last line: "compute-surface: ok".

require_relative "../lib/toy/compute"

cfg = Toy::SmolLM2Config.new(627, 64, 4, 4, 128, 2, 16, 10000.0, 1.0e-5)

# One require gave us the whole composition API. Instantiate each engine class
# (proves all three co-compiled + are addressable) and realize the Llama one
# (proves the L1 primitives / L2 block / L3 arch the engine pulls are live).
llama = Toy::LLM::Engine::LlamaSeqEngine.new
vit   = Toy::LLM::Engine::ViTTinyEngine.new
gpt2  = Toy::LLM::Engine::GPT2SeqEngine.new

llama.realize_for_random_init(cfg, 16, 1, 0, false, false, 0, 1.0)
ok = llama.sess != TinyNN.tnn_null_ptr

puts "compute-surface: engines=[llama,vit,gpt2] realized, sess_present=" + ok.to_s
if ok
  puts "compute-surface: ok"
else
  puts "compute-surface: FAIL (realize produced null session)"
end
