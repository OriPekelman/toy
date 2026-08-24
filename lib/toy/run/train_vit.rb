# lib/toy/run/train_vit.rb — Spinel-compiled ViT-Tiny from-scratch TRAINING
# compute runner (→ libexec/toy-train-vit).
#
# This is the lib-side home of `toy train vit-tiny`'s compute. The CRuby CLI
# shell (lib/toy/core/cli/train.rb) cannot compute in-process (every
# ffi_lib-bearing lib crashes under MRI), so it locates the toy root, builds
# this runner, creates runs/<id>/, and shells out via Open3 with a CONTROLLED
# ENV — the same CRuby→runner COMPUTE BRIDGE that train.rb uses.
#
# SEPARATE BINARY (landmine #16): ViTTinyConfig must NOT share a Spinel
# compilation unit with SmolLM2Config (cfg type-merge miscompile). Hence this
# distinct source/binary, and it never requires toy_gguf_writer/toy_gguf_fuse
# — their cfg.vocab/cfg.d_ff style getters poly-collide with ViTTinyConfig
# (07_train_vit_tiny.rb:22-26). The ViT cache has no GGUF writer + there is no
# ViT infer consumer yet, so NO checkpoint is written here (#169 follow-up).
#
# REAL DATA, NOT SYNTHETIC: this trains random-init on the COMMITTED/pinned
# corpus data/vit_smoke/{images.bin,labels.bin} via ToyImageLoader (mirrors
# 07_train_vit_tiny.rb's data path verbatim), NOT a synthetic-fill smoke. The
# 224/16/196/10 timm shape below is the shape data/vit_smoke matches.
#
# DETERMINISM: random init is self-contained (upload_random_init! is
# Box-Muller xorshift64 seeded by SEED — no donor). All loss math is
# ggml-internal (RMSNorm + SiLU + cross-entropy, no Ruby libm), so the printed
# "step N: loss=" curve is byte-for-byte reproducible AND cross-platform
# portable — STRICT byte-exact, the same class as the llama train gate
# (prep/train_gate.rb's vit-tiny case vs prep/fixtures/train_vit_baseline.txt).
#
# OUTPUT (byte-exact line the CLI + gate parse):
#   "step <N>: loss=<float>"   one per step, to STDOUT (NO lr/idx/label
# decoration — that matches the llama gate line format). events.jsonl goes to
# runs/<id>/ (file-side, never stdout).
#
# Spinel hygiene (landmine #16): hand-built String-concat JSON (no #{}
# interpolation); ENV reads are TOP-LEVEL constants (a constant assigned
# inside a conditional arm reads back empty at runtime); no Struct.new.

require_relative "../io/json_builder"
require_relative "../dev/toy_describe_flow"
require_relative "../io/toy_events"
require_relative "../models/toy_vit"
require_relative "../llm/engine/vit_tiny_engine"
require_relative "../io/toy_image_loader"
require_relative "../train/toy_lr_schedule"
require_relative "../train/toy_drift_grad"
require_relative "../llm/recipes/vit_tiny"
require_relative "../llm/adamw"

# ENV reads — TOP-LEVEL constants (Spinel constant-in-conditional caveat).
STEPS       = (ENV["STEPS"]      || "5").to_i
SEED        = (ENV["SEED"]       || "0").to_i
IMG_DIR     = (ENV["IMG_DIR"]    || "data/vit_smoke")
# The run-directory contract. TOY_RUN_DIR is canonical; RUN_DIR is
# the compatibility fallback — the framework's own contract should not
# be named after a client repo. Length-checked, not truthiness-checked:
# "" is truthy in Ruby.
RUN_DIR_NEW = ENV["TOY_RUN_DIR"] || ""
RUN_DIR     = RUN_DIR_NEW.length > 0 ? RUN_DIR_NEW : (ENV["TAO_RUN_DIR"] || "")
TOY_RUN_ID  = (ENV["TOY_RUN_ID"]  || "vit-tiny")
EVENTS      = RUN_DIR.length > 0 ? (RUN_DIR + "/events.jsonl") : ""

# Gate-fixed timm ViT-Tiny SHAPE — hardcoded (NOT env/flags). This is the
# shape data/vit_smoke matches (224/16/196/10); the 16x16 ENV defaults in
# 07_train_vit_tiny.rb's header are the REJECTED synthetic shape.
IMAGE_SIZE  = 224
PATCH_SIZE  = 16
NUM_CHAN    = 3
D_MODEL     = 192
N_HEADS     = 3
D_FF        = 768
N_LAYERS    = 12
NUM_CLASSES = 10
LN_EPS      = 1.0e-5
N_IMAGES    = 1

# LR schedule PINNED to 07's defaults (confirmed produce the recorded
# baseline curve). With WARMUP=10 > STEPS=5, every step is on the linear
# warmup ramp.
LR_MAX = 0.003
LR_MIN = 0.0001
WARMUP = 10

# d_head = 192/3 = 64 derived inside ViTTinyConfig.
cfg = ViTTinyConfig.new(IMAGE_SIZE, PATCH_SIZE, NUM_CHAN, D_MODEL,
                        N_HEADS, D_FF, N_LAYERS, NUM_CLASSES, LN_EPS)

# Named realize options (toy#64): the ViT path consumes seed +
# init_scale only (init_scale keeps the canonical 1.0 default).
opts = Toy::LLM::RecipeOptions.new
opts.seed = SEED

recipe = Toy::LLM::Recipes::VitTiny.new
recipe.realize!(cfg, opts)
# tao#flow-json-emit (#25): self-describing run bundle, parallel to events.jsonl.
ToyDescribeFlow.emit_flow_json(RUN_DIR, recipe.vt_cache.sess)

# Pre-allocate buffers used every step (07_train_vit_tiny.rb:223-237).
n_patches  = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE)  # 196
patch_flat = NUM_CHAN * PATCH_SIZE * PATCH_SIZE                     # 768
record_f   = patch_flat * n_patches                                # 150528 (=602112/4)

m_image  = Mat.new(patch_flat, n_patches)   # [768, 196]; upload_row_major matches t_image input_2d(196,768)
m_labels = Mat.new(1, NUM_CLASSES)          # [1, 10]
cls_idx  = [0]
# NAMED AdamW. Defaults (beta2=0.95, bias_correct=false) → slots5/6 =
# constant betas, byte-identical to the historical inline hp. lr is set
# per step from the cosine schedule (vit never reads m_hp before the
# per-step assignment), and the hp Mat is rebuilt per step below — the
# original implicit flat[0]=0.0-until-set is reproduced by Mat.new's
# zero-init inside hp(). Single-class labels stay inline (NOT shift-by-one).
adamw_vit = Toy::AdamW.for_from_scratch

images_path = IMG_DIR + "/images.bin"
labels_path = IMG_DIR + "/labels.bin"

# FAIL LOUD on a missing/truncated corpus. ToyImageLoader.read_image
# zero-fills a short read (returning Array.new(record_f, 0.0)) and only puts a
# warning — so a missing corpus would silently train on zeros and reproduce the
# baseline curve, masking the failure (never-mask rule). The vit-tiny gate
# corpus is COMMITTED at data/vit_smoke/ (un-gitignored). Probe the first image
# record with the loader's OWN primitive (tnn_read_f32_file returns the actual
# count read) and abort non-zero if missing/short — NO synthetic zero-fill.
# (Use `puts`, not STDERR.puts: STDERR is a no-op in the Spinel runner; the
# from-scratch runner emits its errors via `puts` too, and the CRuby CLI
# capture2e merges stdout+stderr and prints the tail on a non-zero exit.)
if !File.exist?(images_path) || !File.exist?(labels_path)
  puts "toy-train-vit: corpus missing under " + IMG_DIR +
       " (need images.bin + labels.bin; the vit-tiny gate corpus is committed at data/vit_smoke/)."
  exit 1
end
probe_buf = Array.new(record_f, 0.0)
probe_got = TinyNN.tnn_read_f32_file(images_path, 0, record_f, probe_buf)
if probe_got != record_f
  puts "toy-train-vit: corpus short read " + probe_got.to_s + "/" + record_f.to_s +
       " from " + images_path + " — refusing to train on a zero-filled read."
  exit 1
end

# git provenance read pure-Ruby from .git/HEAD (07:140-167).

# --- run_start event (FILE only; arch=vit). ---
if EVENTS.length > 0
  rc = TinyNN.tnn_events_open(EVENTS)
  if rc == 0
    rid = TOY_RUN_ID.length > 0 ? TOY_RUN_ID : "anonymous"
    rs = Toy::Json::Builder.new
    rs.add_str("kind", "run_start")
    rs.add_str("schema", "toy/v1")
    rs.add_num("t", TinyNN.tnn_events_now_seconds)
    rs.add_str("started_at", TinyNN.tnn_events_iso8601_now)
    rs.add_str("run_id", rid)
    rs.add_str("phase", "train")
    rs.add_str("name", "vit-tiny")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.vt_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "vit")
    model.add_str("name", "vit-tiny")
    model.add_num("image_size",  cfg.image_size)
    model.add_num("patch_size",  cfg.patch_size)
    model.add_num("d_model",     cfg.d_model)
    model.add_num("n_layers",    cfg.n_layers)
    model.add_num("n_heads",     cfg.n_heads)
    model.add_num("d_ff",        cfg.d_ff)
    model.add_num("num_classes", cfg.num_classes)
    rs.add_obj("model", model)
    config = Toy::Json::Builder.new
    config.add_str("image_dir", IMG_DIR)
    config.add_num("n_images",  N_IMAGES)
    config.add_num("steps",     STEPS)
    config.add_num("seed",      SEED)
    config.add_str("init",      "scratch")
    rs.add_obj("config", config)
    schedule = Toy::Json::Builder.new
    schedule.add_num("lr_max",  LR_MAX)
    schedule.add_num("lr_min",  LR_MIN)
    schedule.add_num("warmup",  WARMUP)
    schedule.add_num("n_steps", STEPS)
    rs.add_obj("schedule", schedule)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# --- Training loop (0-indexed; cosine LR + REAL data from data/vit_smoke). ---
final_loss = 0.0
step = 0
while step < STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
  lr = ToyLR.cosine(step, STEPS, LR_MAX, LR_MIN, WARMUP)
  adamw_vit.lr = lr
  m_hp = adamw_vit.hp(step)   # bias_correct=false → slots5/6=betas

  # Single-image memorisation: idx=0 every step (N_IMAGES=1, deterministic).
  idx = step % N_IMAGES
  patches = ToyImageLoader.read_image(images_path, idx, record_f)
  label   = ToyImageLoader.read_label(labels_path, idx)
  if label < 0 || label >= NUM_CLASSES
    puts "skip step " + step.to_s + " (bad label=" + label.to_s + ")"
    step = step + 1
    next
  end

  # Copy patches into the pre-allocated image Mat.
  i = 0
  while i < record_f
    m_image.flat[i] = patches[i]
    i = i + 1
  end
  # One-hot label.
  j = 0
  while j < NUM_CLASSES
    m_labels.flat[j] = (j == label) ? 1.0 : 0.0
    j = j + 1
  end

  loss = recipe.step!(m_image, cls_idx, m_labels, m_hp, step == 0)
  final_loss = loss
  # The byte-gated line — to STDOUT, NO decoration (llama gate line format).
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNN.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es = Toy::Json::Builder.new
    es.add_str("kind",  "step")
    es.add_str("phase", "train")
    es.add_num("t",       TinyNN.tnn_events_now_seconds)
    es.add_num("step",    step + 1)
    es.add_num("loss",    loss)
    es.add_num("lr",      lr)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end
  step = step + 1
end

# --- run_end (FILE only). NO checkpoint write: the ViT cache has no GGUF
# writer + ToyGGUFWriter/Fuser poly-collide with ViTTinyConfig (#169
# follow-up). Document the deferral in a note field (events, NOT stdout). ---
if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_num("final_loss", final_loss)
  re.add_str("checkpoint", "deferred-169")
  re.add_str("note",       "ViT cache has no GGUF writer; checkpoint is the #169 follow-up")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
