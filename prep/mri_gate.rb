# prep/mri_gate.rb — toy#71 gate: the MRI dev-run entrypoint, BOTH arms.
#
#   make gate-mri        # plain `ruby` — NO Spinel, NO SPINEL_DIR, NO build
#
# Orchestrates two subprocess legs (each `require "toy/mri"` exactly once,
# so the two arms never share a process):
#
#   --stub    Stage A (forced TOY_MRI_NATIVE=0):
#             1. LOAD  — the full compute require chain loads under MRI and
#                the declaration registry is populated.
#             2. PURE  — the pure-Ruby teaching surface genuinely works:
#                configs, RecipeOptions, TransformerLM forward + Trainer
#                steps on the Mat path, engine CONSTRUCTION.
#             3. NATIVE — crossing the native boundary fails LOUD with the
#                named Toy::MRI::NativeCallError.
#
#   --native  Stage B (the CRuby oracle; needs `make libtinynn_shared`):
#             4. BIND  — the Fiddle arm is live; a real session opens on
#                the cpu backend.
#             5. DIFFERENTIAL TRAIN — the from-scratch gate shape (the
#                exact lib/toy/run/train.rb compute: SmolLM2Config.mha
#                627/64/4/128/2/32, donor_d_in=128, untied, seed 0,
#                constant AdamW hp, first 32-token window of the PINNED
#                prep/fixtures/ts_seqs_gate.bin) trained for 5 steps under
#                MRI+Fiddle MUST reproduce prep/fixtures/train_baseline.txt
#                — the same recorded curve the Spinel-compiled runner is
#                gated on (prep/train_gate.rb). Floats compared BIT-EXACT
#                (pack("G")); the formatted lines are additionally compared
#                byte-wise and any string-only divergence is REPORTED (not
#                failed — the documented MRI-Float#to_s-vs-Spinel-printing
#                boundary; docs/gating.md "cross-platform").
#             6. KV DECODE — greedy generate on data/smollm2-135m-f32.gguf
#                (MODEL-GATED dev artifact: loud SKIP when absent, like
#                gate-full-finetune) MUST byte-equal the recorded `ids:`
#                line in prep/fixtures/infer_baseline.txt.
#             Plus an informative perf line (steps/s, decode tok/s).
#
# When tinynn/libtinynn_ggml_shared.so is absent the native leg SKIPs
# LOUDLY (exit 0) so dev boxes without a ggml build stay green; set
# MRI_GATE_STRICT=1 to turn that skip into a failure (CI-strict mode).
#
# WHY THIS IS THE ORACLE (spinel-dev#6 phase 1): leg 5/6 run the SAME toy
# program over the SAME C library as the Spinel binaries — only the Ruby
# runtime differs. Bit-equality here means any future divergence between
# MRI and a Spinel build isolates a Spinel codegen bug by construction.

require "open3"

ROOT       = File.expand_path("..", __dir__)
NATIVE_LIB = File.join(ROOT, "tinynn", "libtinynn_ggml_shared.so")

def gate_fail(msg)
  warn "GATE FAIL [mri]: #{msg}"
  exit 1
end

# ── Orchestrator (no toy require here) ───────────────────────────────
if ARGV.empty?
  stub_out, stub_st = Open3.capture2e(
    { "TOY_MRI_NATIVE" => "0" }, RbConfig.ruby, __FILE__, "--stub"
  )
  print stub_out
  gate_fail "stub leg failed" unless stub_st.success?

  if File.file?(NATIVE_LIB)
    nat_out, nat_st = Open3.capture2e(RbConfig.ruby, __FILE__, "--native")
    print nat_out
    gate_fail "native leg failed" unless nat_st.success?
  elsif ENV["MRI_GATE_STRICT"] == "1"
    gate_fail "native leg REQUIRED (MRI_GATE_STRICT=1) but " \
              "#{NATIVE_LIB} is missing — run `make libtinynn_shared`"
  else
    puts "mri-gate: NATIVE LEG SKIPPED — #{NATIVE_LIB} missing; " \
         "`make libtinynn_shared` builds it (MRI_GATE_STRICT=1 makes this a failure)"
  end
  puts "mri-gate: ok"
  exit 0
end

lib = File.join(ROOT, "lib")
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

case ARGV[0]

# ── The Stage A leg: stub arm ────────────────────────────────────────
when "--stub"
  ENV["TOY_MRI_NATIVE"] = "0"   # belt-and-braces for direct invocation

  # 1. LOAD
  begin
    require "toy/mri"
  rescue StandardError, NoMethodError => e
    gate_fail "require \"toy/mri\" raised #{e.class}: #{e.message}"
  end
  gate_fail "native arm bound despite TOY_MRI_NATIVE=0" if Toy::MRI.native?
  gate_fail "Toy::MRI.declarations is empty" if Toy::MRI.declarations.empty?
  n_decl = Toy::MRI.declarations.values.sum(&:size)
  mods   = Toy::MRI.declarations.keys.map(&:to_s).sort
  gate_fail "TinyNN declarations missing" unless mods.include?("TinyNN")
  puts "load ok: #{n_decl} ffi_func declarations recorded across #{mods.join(", ")}"

  # 2. PURE-RUBY SURFACE
  srand(0)
  cfg = Toy::SmolLM2Config.tiny
  gate_fail "SmolLM2Config.tiny wrong shape" unless
    cfg.vocab == 627 && cfg.d_model == 64 && cfg.n_layers == 2
  opts = Toy::LLM::RecipeOptions.new
  gate_fail "RecipeOptions.new failed" unless opts.is_a?(Toy::LLM::RecipeOptions)
  engine = Toy::Device.llama_engine
  gate_fail "engine construction failed" unless
    engine.is_a?(Toy::LLM::Engine::LlamaSeqEngine)

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

  # 3. NATIVE BOUNDARY FAILS LOUD + NAMED
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

# ── The Stage B leg: native arm (the CRuby oracle) ───────────────────
when "--native"
  gate_fail "native leg invoked but #{NATIVE_LIB} is missing" unless
    File.file?(NATIVE_LIB)
  gate_fail "native leg needs TOY_MRI_NATIVE unset/non-0" if
    ENV["TOY_MRI_NATIVE"] == "0"

  begin
    require "toy/mri"
  rescue StandardError, NoMethodError => e
    gate_fail "require \"toy/mri\" (native arm) raised #{e.class}: #{e.message}"
  end
  gate_fail "Fiddle arm did not bind (Toy::MRI.native? false)" unless
    Toy::MRI.native?

  # 4. BIND — a real session round-trip on the cpu backend.
  sess = TinyNN.tnn_session_new(0)
  gate_fail "tnn_session_new returned #{sess.inspect}" unless
    sess.is_a?(Integer) && sess != 0
  backend = TinyNN.tnn_backend_name(sess)
  gate_fail "backend #{backend.inspect}, want \"cpu\"" unless backend == "cpu"
  TinyNN.tnn_session_free(sess)
  puts "bind ok: Fiddle arm live (#{File.basename(NATIVE_LIB)}, backend cpu)"

  # 5. DIFFERENTIAL TRAIN — vs the recorded Spinel gate curve.
  baseline_path = File.join(ROOT, "prep", "fixtures", "train_baseline.txt")
  expected = File.readlines(baseline_path).filter_map do |line|
    l = line.chomp
    next if l.empty? || l.start_with?("#")
    m = l.match(/\Astep (\d+): loss=(.+)\z/)
    gate_fail "unparseable baseline line: #{l.inspect}" unless m
    [l, Float(m[2])]
  end
  gate_fail "train_baseline.txt held no step lines" if expected.empty?

  cfg = Toy::SmolLM2Config.mha(627, 64, 4, 128, 2, 32, 10000.0, 1.0e-5)
  cfg.donor_d_in = 128
  opts = Toy::LLM::RecipeOptions.new
  opts.t_seq  = 32
  opts.untied = true
  opts.seed   = 0
  recipe = Toy::LLM::Recipes::FromScratch.new
  recipe.realize!(cfg, opts)

  # PINNED corpus (cross-platform gate rule): its first 32-token window
  # equals data/ts_seqs.txt line 1, the runner's from-scratch input.
  seq_ids   = ToyCorpusLoader.read_seq(
    File.join(ROOT, "prep", "fixtures", "ts_seqs_gate.bin"), 0, 32)
  positions = (0...32).to_a
  m_labels  = Toy::Labels.next_token(seq_ids, 627, 32, 1)
  m_hp      = Toy::AdamW.for_from_scratch.hp(0)

  t0 = Time.now
  got = expected.each_index.map do |step|
    loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step.zero?)
    ["step #{step + 1}: loss=#{loss}", loss]
  end
  train_secs = Time.now - t0

  string_drift = []
  expected.zip(got).each_with_index do |((want_line, want_f), (got_line, got_f)), i|
    unless [want_f].pack("G") == [got_f].pack("G")
      gate_fail "differential train step #{i + 1}: MRI+Fiddle loss #{got_f} " \
                "(bits #{[got_f].pack("G").unpack1("H*")}) != Spinel gate curve " \
                "#{want_f} (bits #{[want_f].pack("G").unpack1("H*")})"
    end
    string_drift << [want_line, got_line] if want_line != got_line
  end
  if string_drift.empty?
    puts "differential ok: #{expected.size} train losses BIT-EXACT vs " \
         "train_baseline.txt (formatted lines byte-equal too)"
  else
    # Values bit-equal, formatting differs: the documented fallback arm.
    string_drift.each do |want, gotl|
      puts "  format-only drift: spinel=#{want.inspect} mri=#{gotl.inspect}"
    end
    puts "differential ok: #{expected.size} train losses BIT-EXACT vs " \
         "train_baseline.txt (#{string_drift.size} format-only drifts above)"
  end

  # 6. KV DECODE — vs the recorded infer baseline (model-gated).
  gguf = File.join(ROOT, "data", "smollm2-135m-f32.gguf")
  if File.exist?(gguf)
    infer_baseline = File.join(ROOT, "prep", "fixtures", "infer_baseline.txt")
    want_line = nil
    File.foreach(infer_baseline) do |line|
      next if line.strip.empty? || line.start_with?("#")
      key, val = line.chomp.split("\t", 2)
      want_line = val if key == "smollm2-135m-f32.gguf"
    end
    gate_fail "no smollm2-135m-f32.gguf entry in infer_baseline.txt" unless want_line

    require "toy/models/transformer_lm"
    arch = Arch.load_or_fail(gguf, "mri_gate")
    lm = ToyLM.new(arch, :cpu)
    lm.load(gguf)
    prompt_ids = [6403, 1980, 253, 655, 28]   # infer_gate's PROMPT_IDS
    t1 = Time.now
    out_ids = lm.generate(prompt_ids, 8)
    decode_secs = Time.now - t1
    got_line = "ids: " + out_ids.join(" ")
    gate_fail "KV decode ids diverge:\n  want #{want_line.inspect}\n  got  #{got_line.inspect}" unless
      got_line == want_line
    puts "kv-decode ok: greedy ids byte-equal infer_baseline.txt " \
         "(smollm2-135m-f32, prompt 5 ids + 8 new)"
    puts format("perf (informative): %.1f train steps/s; %.1f decode tok/s — MRI+Fiddle, CPU",
                expected.size / train_secs, 8 / decode_secs)
  else
    puts "kv-decode SKIPPED — #{gguf} missing (gitignored dev model, " \
         "same gating as gate-full-finetune); train differential above still binding"
    puts format("perf (informative): %.1f train steps/s — MRI+Fiddle, CPU",
                expected.size / train_secs)
  end

else
  gate_fail "unknown argument #{ARGV[0].inspect} (use --stub / --native / no args)"
end
