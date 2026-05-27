# E1.1 / GH#13 — Conv2D FFI smoke + PyTorch parity dump.
#
# Builds a small Conv2D op in ggml via the new tnn_conv_2d primitive
# and writes inputs + output as JSON for the Python parity check
# (tinynn/conv2d_parity.py). Use the inline-script PEP 723 form so
# uv handles torch transparently.
#
#   make tinynn/ab_smoke_conv2d
#   ./tinynn/ab_smoke_conv2d                          # writes /tmp/conv2d_ref.json
#   uv run tinynn/conv2d_parity.py /tmp/conv2d_ref.json
#
# Shapes match a ViT-Tiny-shaped first slice (patch_embed):
#   input  ne=[W=4, H=4, IC=3, N=1]    — 48 floats
#   kernel ne=[KW=2, KH=2, IC=3, OC=2] — 24 floats
#   stride 1, no padding → output ne=[3, 3, 2, 1] = 18 floats

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

OUT = ENV["OUT"] || "/tmp/conv2d_ref.json"
W = 4; H = 4; IC = 3; N = 1
KW = 2; KH = 2; OC = 2
STRIDE = 1; PAD = 0; DIL = 1

sess = TinyNN.tnn_session_new(0)

# kernel ne=[KW, KH, IC, OC] - 4D persistent f32
t_k = TinyNN.tnn_input_4d_f32_persistent(sess, KW, KH, IC, OC)
# data ne=[W, H, IC, N] - 3D since N=1 (ne[3]=1 implicit)
t_x = TinyNN.tnn_input_3d_f32_persistent(sess, W, H, IC)

# Add conv_2d op.
t_y = TinyNN.tnn_conv_2d(sess, t_k, t_x, STRIDE, STRIDE, PAD, PAD, DIL, DIL)
TinyNN.tnn_set_output(t_y)
TinyNN.tnn_finalize_weights(sess)
TinyNN.tnn_add_to_graph(sess, t_y)
TinyNN.tnn_realize(sess, t_y)

# Fill kernel + input with deterministic values.
n_k = KW * KH * IC * OC
n_x = W * H * IC

kbuf = Mat.new(1, n_k)
i = 0
while i < n_k
  # interleaved sign + decimal to avoid all-positive
  kbuf.flat[i] = ((i % 7) - 3) * 0.1
  i = i + 1
end
xbuf = Mat.new(1, n_x)
i = 0
while i < n_x
  xbuf.flat[i] = ((i % 5) - 2) * 0.25
  i = i + 1
end

TinyNN.tnn_upload_from_float_array(sess, t_k, kbuf.flat, n_k)
TinyNN.tnn_upload_from_float_array(sess, t_x, xbuf.flat, n_x)

TinyNN.tnn_compute(sess)

# Read output dimensions from the tensor.
OW = TinyNN.tnn_tensor_ne0(t_y)
OH = TinyNN.tnn_tensor_ne1(t_y)
out_oc = TinyNN.tnn_tensor_ne2(t_y)
out_n  = TinyNN.tnn_tensor_ne3(t_y)
puts "conv_2d output ne=[" + OW.to_s + "," + OH.to_s + "," + out_oc.to_s + "," + out_n.to_s + "]"

n_y = OW * OH * out_oc * out_n
ybuf = Mat.new(1, n_y)
TinyNN.tnn_download_to_f64_array(sess, t_y, ybuf.flat, n_y)

# Pretty-print the output so we can eyeball.
puts "y[0..5] = " + ybuf.flat[0].to_s + " " + ybuf.flat[1].to_s + " " +
     ybuf.flat[2].to_s + " " + ybuf.flat[3].to_s + " " + ybuf.flat[4].to_s

# Write JSON for the parity script. Use the block form (memory:
# spinel-type-inference-landmines — File.open(w) without block is
# documented working for other parity dumps via the block form).
File.open(OUT, "w") do |f|
  f.write("{")
  f.write("\"kernel_ne\":[" + KW.to_s + "," + KH.to_s + "," + IC.to_s + "," + OC.to_s + "],")
  f.write("\"input_ne\":["  + W.to_s  + "," + H.to_s  + "," + IC.to_s + "," + N.to_s  + "],")
  f.write("\"output_ne\":[" + OW.to_s + "," + OH.to_s + "," + out_oc.to_s + "," + out_n.to_s + "],")
  f.write("\"stride\":[" + STRIDE.to_s + "," + STRIDE.to_s + "],")
  f.write("\"padding\":[" + PAD.to_s + "," + PAD.to_s + "],")
  f.write("\"dilation\":[" + DIL.to_s + "," + DIL.to_s + "],")
  f.write("\"kernel\":[")
  i = 0
  while i < n_k
    f.write((i == 0 ? "" : ",") + kbuf.flat[i].to_s)
    i = i + 1
  end
  f.write("],\"input\":[")
  i = 0
  while i < n_x
    f.write((i == 0 ? "" : ",") + xbuf.flat[i].to_s)
    i = i + 1
  end
  f.write("],\"output\":[")
  i = 0
  while i < n_y
    f.write((i == 0 ? "" : ",") + ybuf.flat[i].to_s)
    i = i + 1
  end
  f.write("]}")
end

puts "wrote " + OUT
