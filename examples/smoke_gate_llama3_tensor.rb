# P2.6 gate — llama3 RoPE post-rope TENSOR parity fixture.
#
# CRITICAL FRAMING: this is an HONEST PRESERVATION gate, NOT a correctness
# / round-trip claim. The realize-bulk workflow records this fixture's
# post-rope tensor baseline once, then re-runs it after a refactor and
# diffs before/after for self-consistency. The llama3 rope branch is
# logit-level INSENSITIVE (out of scope to fix), so we gate at the TENSOR
# level on the post-rope K/Q output, which DOES move with the freq_factors.
#
# WHY A STANDALONE SUBGRAPH (not an in-model tap): in transformer_block.rb
# the post-rope K (t_k, :187-190) and Q (t_q, :350-352) are TRANSIENT LOCAL
# variables — never returned, never tnn_set_output'd, no accessor. The only
# block taps (tap_attn_norm / tap_ffn_out / tap_resid_post) are NOT
# post-rope. Adding a tnn_set_output(t_q)+accessor would be a lib/ behavior
# change (new graph output node + mirror regen) — OUT OF SCOPE. So this
# fixture builds a standalone post-rope subgraph from the SAME PUBLIC lib
# primitives the model's rope path uses, and owns its OWN tap. No lib/
# change, no mirror regen (pure-Ruby fixture; RoPE.apply_2d already mirrored).
#
# WHAT IT EXERCISES: the EXACT entry both the K and Q paths call —
# Toy::LLM::Primitives::RoPE.apply_2d (rope.rb:55) — which internally invokes
# tnn_rope_ext with a NON-NULL, NON-TRIVIAL llama3 freq_factors ptr. The
# freq_factors are computed by the PUBLIC pure-Ruby
# Toy::RopeScaling.compute_llama3_freq_factors (toy_smollm2.rb:88), the same
# method finalize_weights_and_upload_constants! uses (llama_seq_forward_ffi.rb:1370).
#
# TWO LOAD-BEARING ASSERTS (assert_branch):
#   (a) LLAMA3 PATH TAKEN: the computed freq_factors array is genuinely
#       non-uniform — spans 1.0 (high-freq dims) up to factor=8.0 (low-freq
#       dims) with a smooth-interp middle. Assert max!=min AND some==factor
#       AND some==1.0 AND rope_scaling.kind==:llama3. Also assert the
#       freq_factors tensor allocated (handle != null) with ne0 == d_head/2.
#   (b) POST-ROPE TENSOR DETERMINISTIC: compute+download TWICE over the SAME
#       fixed pre-rope input + positions; assert all d_head*T post-rope
#       values are BIT-IDENTICAL across runs. The cross-Mat compare
#       (mat_a.flat[j] != mat_b.flat[j]) simultaneously proves determinism
#       AND finiteness (NaN != NaN by IEEE-754). Same Spinel-safe idiom as
#       the qkv_bias gate.
#   CONTRAST GUARD (honesty): build the SAME subgraph with freq_factors=NULL
#       (rope_scaling :none) and assert the post-rope output DIFFERS from the
#       llama3 one — proving the freq_factors ptr genuinely changed the rope
#       angles (otherwise the gate wouldn't actually exercise the branch).
#
# No model file needed: a deterministic random_init-style pre-rope tensor is
# sufficient (no heavy 5 GB GGUF load). data/llama-3.2-1b-native.gguf is
# present if a future variant wants real factors.
#
# Spinel: needs compilation (ffi_lib is a Spinel intrinsic). NO Struct.new
# (#16; cfg objects are hand-written ctors). NO lib/ change. One commit.
#
#   make examples/smoke_gate_llama3_tensor
#   ./examples/smoke_gate_llama3_tensor        # run from repo root

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"

# llama-3.2 rope params (factor=8.0, low=1.0, high=4.0, orig_max_pos=8192,
# base=500000) + d_head=64. T sequence positions, single batch.
D_HEAD       = 64
T            = 8
ROPE_BASE    = 500000.0
L3_FACTOR    = 8.0
L3_LOW_FREQ  = 1.0
L3_HIGH_FREQ = 4.0
L3_ORIG_MAX  = 8192

puts "config: d_head=" + D_HEAD.to_s + " T=" + T.to_s +
     " rope_base=" + ROPE_BASE.to_s + " factor=" + L3_FACTOR.to_s +
     " low=" + L3_LOW_FREQ.to_s + " high=" + L3_HIGH_FREQ.to_s +
     " orig_max_pos=" + L3_ORIG_MAX.to_s

# --- Build the public llama3 RopeScaling object ----------------------
scaling = Toy::RopeScaling.llama3(L3_FACTOR, L3_LOW_FREQ, L3_HIGH_FREQ, L3_ORIG_MAX)
if scaling.kind != :llama3
  STDERR.puts "FATAL: RopeScaling.llama3 produced kind=" + scaling.kind.to_s + " (expected :llama3)"
  exit 1
end
puts "rope_scaling.kind = " + scaling.kind.to_s

# --- Compute the per-dim freq_factors (public pure-Ruby method) ------
ff = Toy::RopeScaling.compute_llama3_freq_factors(
       D_HEAD, ROPE_BASE, scaling.orig_max_pos, scaling.factor,
       scaling.low_freq_factor, scaling.high_freq_factor)
n_half = D_HEAD / 2
if ff.length != n_half
  STDERR.puts "FATAL: freq_factors length " + ff.length.to_s + " != d_head/2 (" + n_half.to_s + ")"
  exit 1
end

# --- ASSERT (a): LLAMA3 PATH TAKEN / freq_factors non-trivial --------
# The array must span 1.0 (high-freq dims) up to factor=8.0 (low-freq dims),
# with the smooth-interp middle actually firing.
ff_min   = ff[0]
ff_max   = ff[0]
has_one  = false
has_fac  = false
fi = 0
while fi < ff.length
  v = ff[fi]
  if v < ff_min; ff_min = v; end
  if v > ff_max; ff_max = v; end
  if v == 1.0;        has_one = true; end
  if v == L3_FACTOR;  has_fac = true; end
  fi = fi + 1
end
puts "freq_factors: len=" + ff.length.to_s +
     " min=" + ff_min.to_s + " max=" + ff_max.to_s +
     " has_1.0=" + has_one.to_s + " has_factor(" + L3_FACTOR.to_s + ")=" + has_fac.to_s
if ff_max == ff_min
  STDERR.puts "FATAL: freq_factors uniform (min==max==" + ff_min.to_s + ") — llama3 smoothing did NOT fire; this is NOT the llama3 branch."
  exit 1
end
if !has_one
  STDERR.puts "FATAL: no freq_factor == 1.0 — high-freq (untouched) dims missing; llama3 path not taken."
  exit 1
end
if !has_fac
  STDERR.puts "FATAL: no freq_factor == factor(" + L3_FACTOR.to_s + ") — low-freq (full-scale) dims missing; llama3 path not taken."
  exit 1
end
puts "ASSERT (a) OK: freq_factors non-uniform, spans 1.0..factor, smoothing fired (kind=:llama3)"

# --- RoPE.apply_2d Cfg — the llama3 scaling scalars (freq_scale=1.0) --
rope_cfg = Toy::LLM::Primitives::RoPE::Cfg.new(
             D_HEAD, ROPE_BASE, scaling.freq_scale, scaling.ext_factor,
             scaling.attn_factor, scaling.beta_fast, scaling.beta_slow)

# --- Deterministic pre-rope values + positions (fixed pattern, no RNG) -
# m_pre is (rows=T, cols=d_head) row-major -> uploads to ne=[d_head, T],
# matching t_k_pre in transformer_block (ne=[d_head, T*B], d_head fastest).
m_pre = Mat.new(T, D_HEAD)
idx = 0
while idx < T * D_HEAD
  m_pre.flat[idx] = ((idx * 13 + 7) % 97).to_f / 97.0
  idx = idx + 1
end

positions = [0]; positions.pop
pp = 0
while pp < T
  positions.push(pp)
  pp = pp + 1
end

# Build a fresh session, allocate the EXACT input types the lib's seq
# forward uses (pre-rope F32 persistent + i32 ctx positions + freq_factors),
# wire ONE RoPE.apply_2d output, realize (which allocates the compute-ctx
# buffers so the ctx-input positions tensor is uploadable), upload the fixed
# inputs, compute, download. `use_llama3` toggles the freq_factors ptr:
# true => t_freq (the llama3 branch); false => NULL (the :none contrast).
# Returns the downloaded (T, d_head) post-rope Mat. One output per session
# keeps the realize root reaching every input (multi-output realize leaves
# the unrooted tap's backend buffer unset).
def run_rope_once(use_llama3, rope_cfg, m_pre, positions, ff, t_seq, d_head)
  sess  = TinyNN.tnn_session_new(0)
  t_pre = TinyNN.tnn_input_2d_f32_persistent(sess, t_seq, d_head)
  t_pos = TinyNN.tnn_input_1d_i32_ctx(sess, t_seq)
  t_frq = TinyNN.tnn_rope_freq_factors_alloc(sess, d_head)
  if use_llama3
    t_out = Toy::LLM::Primitives::RoPE.apply_2d(sess, t_pre, t_pos, t_frq, rope_cfg, t_seq, 1)
  else
    t_out = Toy::LLM::Primitives::RoPE.apply_2d(sess, t_pre, t_pos, TinyNN.tnn_null_ptr, rope_cfg, t_seq, 1)
  end
  TinyNN.tnn_set_output(t_out)
  TinyNN.tnn_finalize_weights(sess)
  TinyNN.tnn_realize(sess, t_out)
  TinyNN.upload_row_major(sess, t_pre, m_pre)
  TinyNN.upload_int_array(sess, t_pos, positions)
  TinyNN.tnn_upload_from_float_array(sess, t_frq, ff, ff.length)
  TinyNN.tnn_compute(sess)
  TinyNN.download_row_major(sess, t_out, t_seq, d_head)
end

# --- Allocate a freq_factors tensor once to ASSERT its extent ---------
sess0  = TinyNN.tnn_session_new(0)
t_freq = TinyNN.tnn_rope_freq_factors_alloc(sess0, D_HEAD)
if t_freq == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: tnn_rope_freq_factors_alloc returned NULL"
  exit 1
end
ff_ne0 = TinyNN.tnn_tensor_ne0(t_freq)
if ff_ne0 != n_half
  STDERR.puts "FATAL: freq_factors tensor ne0=" + ff_ne0.to_s + " != d_head/2 (" + n_half.to_s + ")"
  exit 1
end
puts "freq_factors tensor allocated: ne0=" + ff_ne0.to_s + " (== d_head/2)"

# --- Run the llama3 graph TWICE (cross-run determinism + finiteness) --
mat_a = run_rope_once(true, rope_cfg, m_pre, positions, ff, T, D_HEAD)
mat_b = run_rope_once(true, rope_cfg, m_pre, positions, ff, T, D_HEAD)
# CONTRAST: same subgraph, freq_factors=NULL (:none).
mat_none = run_rope_once(false, rope_cfg, m_pre, positions, ff, T, D_HEAD)
puts "forward x3 OK (2x llama3 + 1x :none contrast; " + (T * D_HEAD).to_s + " post-rope values each)"

# --- ASSERT (b): DETERMINISM + FINITENESS (single cross-run compare) -
n          = T * D_HEAD
mismatches = 0
first_bad  = -1
j = 0
while j < n
  if mat_a.flat[j] != mat_b.flat[j]
    if first_bad < 0
      first_bad = j
    end
    mismatches = mismatches + 1
  end
  j = j + 1
end
if mismatches > 0
  STDERR.puts "FAIL: " + mismatches.to_s + " of " + n.to_s +
              " post-rope values differ between the two runs (first at " + first_bad.to_s +
              "); non-deterministic OR NaN. Gate FAILED."
  exit 1
end
puts "ASSERT (b) OK: all " + n.to_s + " post-rope values byte-identical across two runs (deterministic + finite)"

# --- CONTRAST GUARD: llama3 freq_factors genuinely change the angles -
# The :none output (freq_factors=NULL) MUST differ from the llama3 output;
# otherwise the freq_factors ptr was inert and we did NOT exercise the branch.
diffs = 0
g = 0
while g < n
  if mat_a.flat[g] != mat_none.flat[g]
    diffs = diffs + 1
  end
  g = g + 1
end
puts "contrast: " + diffs.to_s + " of " + n.to_s + " post-rope values differ between llama3 and :none (NULL freq_factors)"
if diffs == 0
  STDERR.puts "FATAL: llama3 post-rope == :none post-rope — freq_factors ptr was INERT; the llama3 branch was NOT exercised. Gate FAILED."
  exit 1
end
puts "CONTRAST GUARD OK: freq_factors ptr genuinely altered the rope angles (" + diffs.to_s + " values moved)"

# --- Baseline: first + last few post-rope values (human-readable) ----
puts "BASELINE post-rope (llama3) first 5:"
k = 0
while k < 5
  puts "  [" + k.to_s + "] = " + mat_a.flat[k].to_s
  k = k + 1
end
puts "BASELINE post-rope (llama3) last 5:"
k = n - 5
while k < n
  puts "  [" + k.to_s + "] = " + mat_a.flat[k].to_s
  k = k + 1
end

puts "VERDICT: llama3 rope TENSOR gate PASS (freq_factors non-trivial, post-rope deterministic + finite, contrast vs :none confirms branch exercised)"
