# F1.0 — minimal LoRA training-step smoke.
#
# Toy task: regress to zero output. y = (W_base + B @ A) @ x; loss = sum(y²);
# W_base is frozen, A and B are trainable. Hand-coded backward (no ggml
# autograd needed at this scale):
#
#   ∂L/∂y = 2y
#   ∂L/∂A = Bᵀ · (∂L/∂y) · xᵀ     where (∂L/∂y) is (out × 1) and x is (k × 1)
#   ∂L/∂B = (∂L/∂y) · (A · x)ᵀ
#
# Pure Mat math (no FFI). This is the algorithm scaffold; F1.1 swaps
# it into FFI graph form for performance.

require_relative "../lib/transformer"

K = 4      # input dim
OUT = 3    # output dim
R = 2      # LoRA rank
STEPS = 30
LR = 0.05

$seed = 42
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

# Forward pieces. Spinel doesn't like Mat multi-return, so we expose two
# separate functions and recompute as needed.
def fwd_ax(a, x)
  a.matmul(x)               # (R × K) · (K × 1) → (R × 1)
end

def fwd_y(w_base, b, ax, x)
  base_y = w_base.matmul(x) # (OUT × K) · (K × 1) → (OUT × 1)
  bax    = b.matmul(ax)     # (OUT × R) · (R × 1) → (OUT × 1)
  out = Mat.new(base_y.nrows, base_y.ncols)
  i = 0
  while i < base_y.nrows * base_y.ncols
    out.flat[i] = base_y.flat[i] + bax.flat[i]
    i = i + 1
  end
  out
end

def loss_of(y)
  total = 0.0
  i = 0
  while i < y.nrows * y.ncols
    total = total + y.flat[i] * y.flat[i]
    i = i + 1
  end
  total
end

# W_base — frozen.
w_base = Mat.new(OUT, K)
i = 0
while i < OUT * K
  w_base.flat[i] = next_normal(0.5)
  i = i + 1
end

# A — (R × K), initialized small.
a = Mat.new(R, K)
i = 0
while i < R * K
  a.flat[i] = next_normal(0.1)
  i = i + 1
end

# B — (OUT × R), zero-init (standard LoRA).
b = Mat.new(OUT, R)
i = 0
while i < OUT * R
  b.flat[i] = 0.0
  i = i + 1
end

# x — fixed input.
x = Mat.new(K, 1)
i = 0
while i < K
  x.flat[i] = 0.5 + i * 0.3
  i = i + 1
end

ax_init = fwd_ax(a, x)
y_init = fwd_y(w_base, b, ax_init, x)
loss0 = loss_of(y_init)
puts "step 0 (initial): loss=" + loss0.to_s
$stdout.flush

s = 1
while s <= STEPS
  ax = fwd_ax(a, x)
  y  = fwd_y(w_base, b, ax, x)

  # dL/dy = 2y
  dy = Mat.new(y.nrows, y.ncols)
  i = 0
  while i < y.nrows * y.ncols
    dy.flat[i] = 2.0 * y.flat[i]
    i = i + 1
  end

  # dL/dB = dy · (A · x)ᵀ              (OUT × 1) · (1 × R) → (OUT × R)
  dB = dy.matmul_t(ax)

  # dL/dA = Bᵀ · dy · xᵀ                (R × OUT) · (OUT × 1) · (1 × K) → (R × K)
  bt_dy = b.t_matmul(dy)        # Bᵀ · dy → (R × 1)
  dA    = bt_dy.matmul_t(x)     # (R × 1) · (1 × K) → (R × K)

  # SGD update
  i = 0
  while i < a.nrows * a.ncols
    a.flat[i] = a.flat[i] - LR * dA.flat[i]
    i = i + 1
  end
  i = 0
  while i < b.nrows * b.ncols
    b.flat[i] = b.flat[i] - LR * dB.flat[i]
    i = i + 1
  end

  if s % 5 == 0 || s == 1
    ax_v = fwd_ax(a, x)
    y_v  = fwd_y(w_base, b, ax_v, x)
    puts "step " + s.to_s + ": loss=" + loss_of(y_v).to_s
    $stdout.flush
  end
  s = s + 1
end

ax_f = fwd_ax(a, x)
y_f  = fwd_y(w_base, b, ax_f, x)
final = loss_of(y_f)
puts ""
puts "final loss=" + final.to_s
puts "initial→final loss ratio: " + (final / loss0).to_s
puts "training " + (final < loss0 ? "DECREASED" : "INCREASED") + " the loss"
