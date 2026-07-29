# lib/toy/llm/primitives/situ_glu.rb — L1 primitive: SiTU-GLU (Sigmoid
# Tanh Unit GLU, Kimi K3 §2.3.2 — toy#136 / K-series M6).
#
# SwiGLU with a smooth softcap on BOTH multiplicative factors:
#
#   SiTU-GLU(gate, up) = [β₁·tanh(gate/β₁) ⊙ σ(gate)] ⊙ [β₂·tanh(up/β₂)]
#
# β₁=4 (gate branch), β₂=25 (up branch) — K3's pinned values; near the
# origin tanh(x/β)·β ≈ x so the local SwiGLU response is preserved,
# while |f| ≤ β₁·β₂ bounds large coordinates (SwiGLU is unbounded —
# the activation-outlier / low-precision overflow motivation).
#
# Same L1 contract as swiglu.rb: pure module, block owns the gate/up
# projections, this composes only the activation. tanh/sigmoid
# BACKWARD ride vendor-patch 0013 (composed from mul/sub — no new
# kernels, every backend). No `require_relative "tinynn"` — the
# loading module provides the backend's TinyNN (mirror-generated
# variants rename TinyNN. → TinyNN<Backend>.).

module Toy
  module LLM
    module Primitives
      module SiTUGLU
        NAME = :situ_glu
        BETA1 = 4.0
        BETA2 = 25.0

        def self.gate(sess, t_gate, t_up)
          t_capg = TinyNN.tnn_scale(sess,
                     TinyNN.tnn_tanh(sess,
                       TinyNN.tnn_scale(sess, t_gate, 1.0 / BETA1)), BETA1)
          t_sig  = TinyNN.tnn_sigmoid(sess, t_gate)
          t_capu = TinyNN.tnn_scale(sess,
                     TinyNN.tnn_tanh(sess,
                       TinyNN.tnn_scale(sess, t_up, 1.0 / BETA2)), BETA2)
          TinyNN.tnn_mul(sess, TinyNN.tnn_mul(sess, t_capg, t_sig), t_capu)
        end
      end
    end
  end
end
