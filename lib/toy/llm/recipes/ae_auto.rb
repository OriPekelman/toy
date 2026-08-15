# lib/toy/llm/recipes/ae_auto.rb — L4 recipe for the toy#165 (capstone
# P1a) latent-autoencoder lane: realize the encoder + bottleneck + decode
# head on a Toy::LLM::Engine::AeEngine, then drive one window at a time.
#
# Same class shape as the other lane recipes (plain class, no-arg ctor,
# uniquely ae_-prefixed ivars, no backend require of its own — the runner
# requires the engine first). CPU-only, all BP.

module Toy; module LLM; module Recipes
  class AeAuto
    attr_accessor :ae_cache, :ae_t_loss, :ae_t_labels, :ae_t_hp

    def initialize
      @ae_cache    = Toy::LLM::Engine::AeEngine.new
      @ae_t_loss   = nil
      @ae_t_labels = nil
      @ae_t_hp     = nil
    end

    def realize!(d_model, heads, d_ff, n_blocks, context, latent, seed,
                 init_scale, kl_beta, kl_learned)
      @ae_cache.realize_for_random_init(d_model, heads, d_ff, n_blocks,
                                        context, latent, seed, init_scale,
                                        kl_beta, kl_learned)
      result = @ae_cache.build_training_step
      @ae_t_loss   = result[0]
      @ae_t_labels = result[1]
      @ae_t_hp     = result[2]
      nil
    end

    # THE PROBE, uploaded as data. `perm` is a position gather (identity
    # in training and at clean/noisy eval, a shuffle for the control),
    # `gain` an elementwise multiplier (ones, or zeros for the zeroed
    # control), `noise` an elementwise addend (zeros, or sigma * std * z).
    #
    # They are set once per pass rather than per step because none of
    # them varies within a pass — and because uploading them from the
    # step path would make it possible to forget one, which is the
    # failure the single-graph design exists to remove.
    def upload_probe!(perm, gain, noise)
      c = @ae_cache.ae_context
      d = @ae_cache.ae_latent
      TinyNN.tnn_upload_from_int_array(@ae_cache.sess, @ae_cache.t_perm, perm, c)
      TinyNN.tnn_upload_from_float_array(@ae_cache.sess, @ae_cache.t_gain, gain, c * d)
      TinyNN.tnn_upload_from_float_array(@ae_cache.sess, @ae_cache.t_noise, noise, c * d)
      nil
    end

    # ONE window. `tokens` are the byte ids at each position; the labels
    # are the SAME ids one-hot — this is reconstruction, not prediction,
    # so there is no shift anywhere in this lane.
    def step!(tokens, m_labels, m_hp, is_first)
      s = @ae_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.tnn_upload_from_int_array(s, @ae_cache.t_tokens, tokens,
                                       @ae_cache.ae_context)
      TinyNN.upload_row_major(s, @ae_t_labels, m_labels)
      TinyNN.upload_row_major(s, @ae_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @ae_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
