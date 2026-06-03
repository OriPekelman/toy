# E1.2 / GH#13 — ViT patch_embed smoke + PyTorch parity dump.
#
# Composes ggml_conv_2d + permute + cont_2d (via ToyVit.patch_embed)
# and writes inputs + output to JSON for the parity script.
#
#   make tinynn/ab_smoke_patch_embed
#   OUT=$HOME/tmp/patch_embed_ref.json ./tinynn/ab_smoke_patch_embed
#   uv run tinynn/patch_embed_parity.py $HOME/tmp/patch_embed_ref.json
#
# Shapes chosen small for hand-verifiability but ViT-Tiny-shaped at
# the axes: image 8×8×3, patch 4, d_model=16 → 4 patches × 16 dim.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/toy/models/toy_vit"

OUT = ENV["OUT"] || "/tmp/patch_embed_ref.json"
W = 8; H = 8; IC = 3
KW = 4; KH = 4; OC = 16   # d_model = 16
N = 1

sess = TinyNN.tnn_session_new(0)

# kernel ne=[KW, KH, IC, OC] - the patch_embed conv kernel
t_k = TinyNN.tnn_input_4d_f32_persistent(sess, KW, KH, IC, OC)
# image ne=[W, H, IC, N] - using 3D primitive with N=1 implicit
t_x = TinyNN.tnn_input_3d_f32_persistent(sess, W, H, IC)

# Composite: conv2d + permute + cont_2d → ne=[OC, OW*OH]
t_y = ToyVit.patch_embed(sess, t_k, t_x, KW)   # patch_size = KW = KH = 4
TinyNN.tnn_set_output(t_y)
TinyNN.tnn_finalize_weights(sess)
TinyNN.tnn_add_to_graph(sess, t_y)
TinyNN.tnn_realize(sess, t_y)

# Fill kernel + image with deterministic values.
n_k = KW * KH * IC * OC
n_x = W * H * IC

kbuf = Mat.new(1, n_k)
i = 0
while i < n_k
  kbuf.flat[i] = ((i % 11) - 5) * 0.05
  i = i + 1
end
xbuf = Mat.new(1, n_x)
i = 0
while i < n_x
  xbuf.flat[i] = ((i % 7) - 3) * 0.1
  i = i + 1
end

TinyNN.tnn_upload_from_float_array(sess, t_k, kbuf.flat, n_k)
TinyNN.tnn_upload_from_float_array(sess, t_x, xbuf.flat, n_x)
TinyNN.tnn_compute(sess)

# Output ne=[OC, N_patches]
out_oc = TinyNN.tnn_tensor_ne0(t_y)
out_np = TinyNN.tnn_tensor_ne1(t_y)
n_y    = out_oc * out_np
puts "patch_embed output ne=[" + out_oc.to_s + "," + out_np.to_s + "]  (d_model × N_patches)"

ybuf = Mat.new(1, n_y)
TinyNN.tnn_download_to_f64_array(sess, t_y, ybuf.flat, n_y)

puts "y[0..3] = " + ybuf.flat[0].to_s + " " + ybuf.flat[1].to_s + " " +
     ybuf.flat[2].to_s + " " + ybuf.flat[3].to_s

# Write JSON for parity.
File.open(OUT, "w") do |f|
  f.write("{")
  f.write("\"image_ne\":["  + W.to_s  + "," + H.to_s  + "," + IC.to_s + "," + N.to_s  + "],")
  f.write("\"kernel_ne\":[" + KW.to_s + "," + KH.to_s + "," + IC.to_s + "," + OC.to_s + "],")
  f.write("\"output_ne\":[" + out_oc.to_s + "," + out_np.to_s + "],")
  f.write("\"patch_size\":" + KW.to_s + ",")
  f.write("\"kernel\":[")
  i = 0
  while i < n_k
    f.write((i == 0 ? "" : ",") + kbuf.flat[i].to_s)
    i = i + 1
  end
  f.write("],\"image\":[")
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
