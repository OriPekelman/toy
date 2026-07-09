# tinynn/toy_llama_smoke.rb — smoke for Toy::RMSNorm + Toy::SwiGLU.
#
# Verifies the new llama-family building blocks compile under Spinel
# and produce correct math on tiny inputs. No GGUF, no real weights.
#
# Spinel gotcha: don't name a local `rms` — TransformerLM#rms_norm_backward
# has a `rms` Array<Float> parameter. Two types under the same local
# name unify to sp_RbVal and break the existing backward code.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"

# ---- Toy::RMSNorm ----
puts "=== Toy::RMSNorm ==="
rnorm = Toy::RMSNorm.new(4)
xn    = Mat.new(2, 4)
# Row 0: [1, 1, 1, 1]   → mean(x^2)=1 → y = [1, 1, 1, 1] (gamma=1)
# Row 1: [1, 2, 3, 4]   → mean(x^2)=7.5 → y = x / sqrt(7.5)
xn.flat[0]=1.0; xn.flat[1]=1.0; xn.flat[2]=1.0; xn.flat[3]=1.0
xn.flat[4]=1.0; xn.flat[5]=2.0; xn.flat[6]=3.0; xn.flat[7]=4.0
yn = rnorm.forward(xn)
puts "  y[0] ≈ [1, 1, 1, 1]: [" + yn.flat[0].to_s + ", " +
     yn.flat[1].to_s + "]"
puts "  y[1,0] = " + yn.flat[4].to_s
puts "  expected y[1,0] = " + (1.0 / Math.sqrt(7.5)).to_s

# ---- Toy.silu! ----
puts "=== Toy.silu! ==="
ms = Mat.new(1, 3)
ms.flat[0] = 0.0
ms.flat[1] = 1.0
ms.flat[2] = -1.0
Toy.silu!(ms)
puts "  silu([0, 1, -1]) = [" + ms.flat[0].to_s + ", " +
     ms.flat[1].to_s + ", " + ms.flat[2].to_s + "]"
puts "  expected:          [0, 0.7311, -0.2689]"

# ---- Toy.hadamard! ----
puts "=== Toy.hadamard! ==="
mh_dst = Mat.new(1, 3); mh_dst.flat[0]=2.0; mh_dst.flat[1]=3.0; mh_dst.flat[2]=4.0
mh_src = Mat.new(1, 3); mh_src.flat[0]=0.5; mh_src.flat[1]=2.0; mh_src.flat[2]=0.0
Toy.hadamard!(mh_dst, mh_src)
puts "  [2,3,4] * [0.5,2,0] = [" + mh_dst.flat[0].to_s + ", " +
     mh_dst.flat[1].to_s + ", " + mh_dst.flat[2].to_s + "]"
puts "  expected:             [1, 6, 0]"

# ---- Toy::SwiGLU end-to-end ----
puts "=== Toy::SwiGLU ==="
swi = Toy::SwiGLU.new(8, 16)   # d_model=8, d_ff=16
swi.w_gate.fill_random(0.1)
swi.w_up.fill_random(0.1)
swi.w_down.fill_random(0.1)
swi_in = Mat.new(3, 8)
swi_in.fill_random(1.0)
swi_out = swi.forward(swi_in)
puts "  out shape: [" + swi_out.nrows.to_s + ", " + swi_out.ncols.to_s + "]"
puts "  (expected [3, 8])"
puts "  out[0,0] = " + swi_out.flat[0].to_s

# ---- Toy::RoPE ----
puts "=== Toy::RoPE ==="
# d_head=4 → half=2.  max_seq=4.  theta_base=100000 (SmolLM2 value).
rp = Toy::RoPE.new(4, 4, 100000.0)
# Identity check: position 0 should leave the tensor unchanged.
xr = Mat.new(1, 4)
xr.flat[0]=1.0; xr.flat[1]=2.0; xr.flat[2]=3.0; xr.flat[3]=4.0
rp.rotate!(xr, 0)
puts "  rotate at pos 0 (should be identity): [" +
     xr.flat[0].to_s + ", " + xr.flat[1].to_s + ", " +
     xr.flat[2].to_s + ", " + xr.flat[3].to_s + "]"
puts "  expected:                              [1, 2, 3, 4]"

# At pos=1 with d_head=4 and theta_base=100000:
#   k=0: theta_0 = 100000^(-0/4) = 1     → angle = 1 * 1 = 1 rad
#   k=1: theta_1 = 100000^(-2/4) ≈ 0.00316 → angle ≈ 0.00316 rad
xr2 = Mat.new(1, 4)
xr2.flat[0]=1.0; xr2.flat[1]=0.0; xr2.flat[2]=0.0; xr2.flat[3]=0.0
rp.rotate!(xr2, 1)
# x[0]=1, x[2]=0 → x_new[0] = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.5403
# x_new[2] = 0*cos(1) + 1*sin(1) = sin(1) ≈ 0.8415
puts "  rotate [1,0,0,0] at pos 1: [" +
     xr2.flat[0].to_s + ", " + xr2.flat[1].to_s + ", " +
     xr2.flat[2].to_s + ", " + xr2.flat[3].to_s + "]"
puts "  expected:                  [cos(1)=" + Math.cos(1.0).to_s +
     ", 0, sin(1)=" + Math.sin(1.0).to_s + ", 0]"

# ---- Toy::GQAttention ----
puts "=== Toy::GQAttention ==="
# Tiny: d_model=12, n_heads=6, n_kv=2 (3 queries per KV head). d_head=2.
rp2 = Toy::RoPE.new(2, 8, 10000.0)
gqa = Toy::GQAttention.new(12, 6, 2, rp2)
gqa.w_q.length.times { |h| gqa.w_q[h].fill_random(0.1) }
gqa.w_k.length.times { |h| gqa.w_k[h].fill_random(0.1) }
gqa.w_v.length.times { |h| gqa.w_v[h].fill_random(0.1) }
gqa.w_o.fill_random(0.1)
xa = Mat.new(3, 12)
xa.fill_random(1.0)
ya = gqa.forward(xa, 0)
puts "  out shape: [" + ya.nrows.to_s + ", " + ya.ncols.to_s +
     "] (expected [3, 12])"
puts "  n_q=" + gqa.n_heads.to_s + " n_kv=" + gqa.n_kv.to_s +
     " group_size=" + gqa.group_size.to_s

# ---- Toy::SmolLM2 (random weights, shape check) ----
puts "=== Toy::SmolLM2 (tiny) ==="
# Tiny llama-shape config: vocab=32, d=12, n_q=6, n_kv=2, d_ff=24,
# n_layers=2, ctx=16, rope_base=10000, rms_eps=1e-5.
tiny = Toy::SmolLM2Config.new(32, 12, 6, 2, 24, 2, 16, 10000.0, 1.0e-5)
sm   = Toy::SmolLM2.new(tiny)
# Fill weights with small randoms (no parity expected; just shape flow).
sm.token_embed.weight.fill_random(0.1)
sm.final_norm.gamma.length.times { |j| sm.final_norm.gamma[j] = 1.0 }
li = 0
while li < tiny.n_layers
  sblk2 = sm.stack[li]
  sblk2.rn1.gamma.length.times { |j| sblk2.rn1.gamma[j] = 1.0 }
  sblk2.rn2.gamma.length.times { |j| sblk2.rn2.gamma[j] = 1.0 }
  sblk2.attn.w_q.length.times  { |h| sblk2.attn.w_q[h].fill_random(0.1) }
  sblk2.attn.w_k.length.times  { |h| sblk2.attn.w_k[h].fill_random(0.1) }
  sblk2.attn.w_v.length.times  { |h| sblk2.attn.w_v[h].fill_random(0.1) }
  sblk2.attn.w_o.fill_random(0.1)
  sblk2.ffn.w_gate.fill_random(0.1)
  sblk2.ffn.w_up.fill_random(0.1)
  sblk2.ffn.w_down.fill_random(0.1)
  li += 1
end

prompt = [3, 7, 1, 9]
logits = sm.forward(prompt, 0)
puts "  forward([3,7,1,9], 0) → logits shape [" +
     logits.nrows.to_s + ", " + logits.ncols.to_s + "]"
puts "  expected: [4, " + tiny.vocab.to_s + "]"
puts "  logits[0,0] = " + logits.flat[0].to_s
puts "  (any finite number is a passing smoke; parity comes with real weights)"

puts "ok"
