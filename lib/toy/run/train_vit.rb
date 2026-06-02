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

require_relative "../../toy_vit"
require_relative "../../vit_tiny_forward_ffi"
require_relative "../../toy_image_loader"
require_relative "../../toy_lr_schedule"
require_relative "../../toy_drift_grad"
require_relative "../llm/recipes/vit_tiny"
require_relative "../llm/adamw"

# ENV reads — TOP-LEVEL constants (Spinel constant-in-conditional caveat).
STEPS       = (ENV["STEPS"]      || "5").to_i
SEED        = (ENV["SEED"]       || "0").to_i
IMG_DIR     = (ENV["IMG_DIR"]    || "data/vit_smoke")
TAO_RUN_DIR = (ENV["TAO_RUN_DIR"] || "")
TOY_RUN_ID  = (ENV["TOY_RUN_ID"]  || "vit-tiny")
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

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

recipe = Toy::LLM::Recipes::VitTiny.new
recipe.realize!(cfg, SEED, 1.0)

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
adamw_vit = Toy::AdamW.new

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
git_sha    = "unknown"
git_branch = "unknown"
if File.exist?(".git/HEAD")
  head = File.read(".git/HEAD")
  if head.length > 0 && head[head.length - 1...head.length] == "\n"
    head = head[0...head.length - 1]
  end
  if head.length > 5 && head[0...5] == "ref: "
    ref_rel = head[5...head.length]
    pp = ref_rel.split("/")
    if pp.length >= 3
      git_branch = pp[pp.length - 1]
    end
    ref_path = ".git/" + ref_rel
    if File.exist?(ref_path)
      sha = File.read(ref_path)
      if sha.length >= 40
        git_sha = sha[0...40]
      end
    end
  else
    if head.length >= 40
      git_sha    = head[0...40]
      git_branch = "HEAD"
    end
  end
end

# --- run_start event (FILE only; arch=vit). ---
if EVENTS.length > 0
  rc = TinyNN.tnn_events_open(EVENTS)
  if rc == 0
    rid = TOY_RUN_ID.length > 0 ? TOY_RUN_ID : "anonymous"
    rs  = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\""
    rs = rs + ",\"t\":" + TinyNN.tnn_events_now_seconds.to_s
    rs = rs + ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\""
    rs = rs + ",\"run_id\":\"" + rid + "\""
    rs = rs + ",\"phase\":\"train\""
    rs = rs + ",\"name\":\"vit-tiny\""
    rs = rs + ",\"host\":{\"name\":\""   + TinyNN.tnn_provenance_host_name + "\""
    rs = rs + ",\"os\":\""               + TinyNN.tnn_provenance_host_os   + "\""
    rs = rs + ",\"arch\":\""             + TinyNN.tnn_provenance_host_arch + "\"}"
    rs = rs + ",\"backend\":{\"kind\":\"" + TinyNN.tnn_backend_name(recipe.vt_cache.sess) + "\"}"
    rs = rs + ",\"git\":{\"sha\":\""     + git_sha    + "\""
    rs = rs + ",\"branch\":\""           + git_branch + "\"}"
    rs = rs + ",\"model\":{\"arch\":\"vit\""
    rs = rs + ",\"name\":\"vit-tiny\""
    rs = rs + ",\"image_size\":"  + cfg.image_size.to_s
    rs = rs + ",\"patch_size\":"  + cfg.patch_size.to_s
    rs = rs + ",\"d_model\":"     + cfg.d_model.to_s
    rs = rs + ",\"n_layers\":"    + cfg.n_layers.to_s
    rs = rs + ",\"n_heads\":"     + cfg.n_heads.to_s
    rs = rs + ",\"d_ff\":"        + cfg.d_ff.to_s
    rs = rs + ",\"num_classes\":" + cfg.num_classes.to_s + "}"
    rs = rs + ",\"config\":{\"image_dir\":\"" + IMG_DIR + "\""
    rs = rs + ",\"n_images\":" + N_IMAGES.to_s
    rs = rs + ",\"steps\":"    + STEPS.to_s
    rs = rs + ",\"seed\":"     + SEED.to_s
    rs = rs + ",\"init\":\"scratch\"}"
    rs = rs + ",\"schedule\":{\"lr_max\":" + LR_MAX.to_s +
              ",\"lr_min\":" + LR_MIN.to_s +
              ",\"warmup\":" + WARMUP.to_s +
              ",\"n_steps\":" + STEPS.to_s + "}"
    rs = rs + "}"
    TinyNN.tnn_events_emit(rs)
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
    es  = "{\"kind\":\"step\",\"phase\":\"train\""
    es = es + ",\"t\":"       + TinyNN.tnn_events_now_seconds.to_s
    es = es + ",\"step\":"    + (step + 1).to_s
    es = es + ",\"loss\":"    + loss.to_s
    es = es + ",\"lr\":"      + lr.to_s
    es = es + ",\"wall_us\":" + step_wall_us.to_s
    es = es + "}"
    TinyNN.tnn_events_emit(es)
  end
  step = step + 1
end

# --- run_end (FILE only). NO checkpoint write: the ViT cache has no GGUF
# writer + ToyGGUFWriter/Fuser poly-collide with ViTTinyConfig (#169
# follow-up). Document the deferral in a note field (events, NOT stdout). ---
if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  re  = "{\"kind\":\"run_end\""
  re = re + ",\"t\":"           + TinyNN.tnn_events_now_seconds.to_s
  re = re + ",\"ended_at\":\""  + TinyNN.tnn_events_iso8601_now + "\""
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"final_step\":"  + STEPS.to_s
  re = re + ",\"final_loss\":"  + final_loss.to_s
  re = re + ",\"checkpoint\":\"deferred-169\""
  re = re + ",\"note\":\"ViT cache has no GGUF writer; checkpoint is the #169 follow-up\""
  re = re + ",\"exit_code\":0"
  re = re + "}"
  TinyNN.tnn_events_emit(re)
  TinyNN.tnn_events_close
end
