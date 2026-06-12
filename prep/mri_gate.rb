# prep/mri_gate.rb — toy#71 Stage A gate: the MRI dev-run entrypoint.
#
#   make gate-mri        # plain `ruby` — NO Spinel, NO SPINEL_DIR, NO build
#
# Proves `require "toy/mri"` gives plain CRuby the whole compute surface:
#
#   1. LOAD  — the full compute require chain loads under MRI (no
#      NoMethodError from the ffi_lib/ffi_func/ffi_cflags intrinsics or
#      anything else), and the declaration registry is populated.
#   2. PURE  — the pure-Ruby teaching surface genuinely works: configs,
#      RecipeOptions, a real TransformerLM forward + Trainer steps on
#      the Mat path (losses move), engine CONSTRUCTION.
#   3. NATIVE — crossing the native boundary fails LOUD with the named
#      Toy::MRI::NativeCallError (never a bare NoMethodError, never a
#      silent wrong answer), at both a direct TinyNN call and an
#      engine realize.
#
# Structural + behavioral, not byte-exact: MRI float printing is the
# Ruby-libm arm (docs/gating.md "cross-platform"); numeric byte-equality
# vs the Spinel binaries is Stage B's differential gate.

lib = File.expand_path("../lib", __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

def gate_fail(msg)
  warn "GATE FAIL [mri]: #{msg}"
  exit 1
end

# ── 1. LOAD ──────────────────────────────────────────────────────────
begin
  require "toy/mri"
rescue StandardError, NoMethodError => e
  gate_fail "require \"toy/mri\" raised #{e.class}: #{e.message}"
end

gate_fail "Toy::MRI.declarations is empty" if Toy::MRI.declarations.empty?
n_decl = Toy::MRI.declarations.values.sum(&:size)
mods   = Toy::MRI.declarations.keys.map(&:to_s).sort
gate_fail "TinyNN declarations missing" unless mods.include?("TinyNN")
puts "load ok: #{n_decl} ffi_func declarations recorded across #{mods.join(", ")}"

# ── 2. PURE-RUBY SURFACE ─────────────────────────────────────────────
srand(0)

cfg = Toy::SmolLM2Config.tiny
gate_fail "SmolLM2Config.tiny wrong shape" unless
  cfg.vocab == 627 && cfg.d_model == 64 && cfg.n_layers == 2

opts = Toy::LLM::RecipeOptions.new
gate_fail "RecipeOptions.new failed" unless opts.is_a?(Toy::LLM::RecipeOptions)

# Engine CONSTRUCTION is pure (ivars seed with typed-NULL handles);
# only realize crosses the native boundary (checked in part 3).
engine = Toy::Device.llama_engine
gate_fail "engine construction failed" unless
  engine.is_a?(Toy::LLM::Engine::LlamaSeqEngine)

# A real forward + training loop on the Mat path (the teaching stack):
# tiny shape, 5 steps, assert the loss MOVES DOWN (behavioral, not
# byte-pinned — see header).
require "toy/train/toy_trainer"
lm  = TransformerLM.new(64, 32, 64, 2, 2, 16)
tr  = Toy::Trainer.new(lm)
seq = (1..12).to_a

logits = lm.forward(seq)
gate_fail "forward shape: got #{logits.nrows}x#{logits.ncols}" unless
  logits.nrows == 12 && logits.ncols == 64

losses = 5.times.map { tr.step!(seq) }
gate_fail "loss not finite: #{losses.inspect}" unless
  losses.all? { |l| l.finite? }
gate_fail "loss did not decrease over 5 steps: #{losses.inspect}" unless
  losses.last < losses.first
puts "pure ok: cfg/options/engine construct; 5 Trainer steps " \
     "loss #{losses.first.round(4)} -> #{losses.last.round(4)}"

# ── 3. NATIVE BOUNDARY FAILS LOUD + NAMED ────────────────────────────
expect_native_error = lambda do |what, &blk|
  begin
    blk.call
    gate_fail "#{what} did NOT raise"
  rescue Toy::MRI::NativeCallError => e
    unless e.message.include?("requires the Spinel-compiled binary") &&
           e.message.include?("toy#71") &&
           e.message =~ /native call `tnn_\w+`/
      gate_fail "#{what} raised with unexpected message: #{e.message}"
    end
    e.message[/`(tnn_\w+)`/, 1]
  rescue => e
    gate_fail "#{what} raised #{e.class} (want Toy::MRI::NativeCallError): #{e.message}"
  end
end

name1 = expect_native_error.call("direct TinyNN.tnn_session_new(0)") do
  TinyNN.tnn_session_new(0)
end
name2 = expect_native_error.call("engine.realize_for_random_init") do
  engine.realize_for_random_init(cfg, 32, 1, 0, false, false, 0, 1.0)
end
puts "native ok: `#{name1}` and `#{name2}` raise the named NativeCallError"

puts "mri-gate: ok"
