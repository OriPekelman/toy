# lib/toy/llm/primitives/rope.rb — L1 primitive: rotary positional
# embedding with extended (NTK/llama3) scaling.
#
# Pure module: `self.` methods only, no module ivars, no state. The
# block (L2) owns the position tensor + freq_factors ptr and passes
# them in. See lib/toy/llm/primitives/README.md for the L1 contract.
#
# Spinel hygiene: the Cfg ctor takes all 7 args positionally with NO
# defaults (default-arg poisoning, landmine #4). apply_2d does int
# math + FFI passthrough only — no STDERR, no Optional returns, no
# Array<Array<mixed>> destructure.
#
# This file does NOT `require_relative "tinynn"` — the loading module
# (lib/llama_seq_forward_ffi.rb) already loads the correct backend's
# TinyNN before requiring this primitive. Keeping the require out lets
# the mirror generator pick the backend via the monolith's require
# rewrite (the CUDA/Metal mirror of the monolith requires rope_<backend>,
# and that mirror's TinyNN. -> TinyNN<Backend>. rename handles the rest).

module Toy
  module LLM
    module Primitives
      module RoPE
        NAME = :rope

        # Cfg value object — concrete scalar fields, Spinel-safe. All
        # five scaling values are copied as plain Floats at realize
        # time (decoupling the primitive from the RopeScaling object).
        # freq_factors is intentionally NOT a field here: it is a
        # per-realize FFI ptr passed explicitly to apply_2d.
        class Cfg
          attr_accessor :d_head, :base,
                        :freq_scale, :ext_factor, :attn_factor,
                        :beta_fast, :beta_slow

          def initialize(d_head, base, freq_scale, ext_factor,
                         attn_factor, beta_fast, beta_slow)
            @d_head      = d_head
            @base        = base
            @freq_scale  = freq_scale
            @ext_factor  = ext_factor
            @attn_factor = attn_factor
            @beta_fast   = beta_fast
            @beta_slow   = beta_slow
          end
        end

        # Shape-lift + rope + un-lift, identical to both seq-forward
        # call sites (K path + Q path). t_pre is ne=[d_head, T*B]
        # (ne[2]==1); ggml_rope_ext requires a->ne[2] == positions->
        # ne[0], so lift to ne=[d_head, 1, T*B], rope, then reshape
        # back to 2D. Reshape is metadata-only on contiguous tensors;
        # at T=1,B=1 it is a no-op (1 == 1). Returns the rotated 2D
        # tensor handle.
        def self.apply_2d(sess, t_pre, positions, freq_factors, cfg, t_seq, t_batch)
          tb = t_seq * t_batch
          t_pre3 = TinyNN.tnn_reshape_3d(sess, t_pre, cfg.d_head, 1, tb)
          t3     = TinyNN.tnn_rope_ext(sess, t_pre3, positions,
                                       cfg.d_head, cfg.base,
                                       cfg.freq_scale, cfg.ext_factor,
                                       cfg.attn_factor, cfg.beta_fast,
                                       cfg.beta_slow, freq_factors)
          TinyNN.tnn_reshape_2d(sess, t3, cfg.d_head, tb)
        end
      end
    end
  end
end
