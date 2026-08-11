# lib/toy/llm/recipes/diff_denoiser.rb — L4 recipe for the toy#156
# (DFA-arch T2) latent diffusion lane: realize a time-conditioned
# eps-prediction denoiser + MSE + backward graph on a
# Toy::LLM::Engine::DiffEngine, then drive one step at a time.
#
# Same class shape as mlp_classifier.rb / gnn_node.rb / ssm_seq.rb
# (plain class, no-arg ctor, uniquely dn_-prefixed ivars, no backend
# require of its own — the runner requires the engine first). CPU-only.
#
# `denoise!` is the SAMPLING entry point and is deliberately separate
# from `step!`: ancestral sampling runs the same graph with lr = 0 many
# times, and it must never touch the epsilon target or the optimizer
# state. Keeping the two paths apart is what stops the sampler from
# quietly becoming a training loop (the toy#139/#146 class of bug, in
# its generative-lane form).

module Toy; module LLM; module Recipes
  class DiffDenoiser
    attr_accessor :dn_cache, :dn_t_loss, :dn_t_eps, :dn_t_hp

    def initialize
      @dn_cache  = Toy::LLM::Engine::DiffEngine.new
      @dn_t_loss = nil
      @dn_t_eps  = nil
      @dn_t_hp   = nil
    end

    def realize!(d_in, latent_dim, d_hidden, n_layers, batch, seed,
                 init_scale, policy, b_seed, b_dist, b_scale, b_sigma)
      @dn_cache.realize_for_random_init(d_in, latent_dim, d_hidden,
                                        n_layers, batch, seed, init_scale)
      result = @dn_cache.build_training_step(policy, b_seed, b_dist,
                                              b_scale, b_sigma)
      @dn_t_loss = result[0]
      @dn_t_eps  = result[1]
      @dn_t_hp   = result[2]
      nil
    end

    # ONE training step. `x_flat` is [x_t ; time features] laid out
    # sample-major (sample i at i * d_in), `m_eps` the [batch, latent]
    # epsilon target.
    def step!(x_flat, m_eps, m_hp, is_first)
      s = @dn_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      n = @dn_cache.df_batch * @dn_cache.df_d_in
      TinyNN.tnn_upload_from_float_array(s, @dn_cache.t_x, x_flat, n)
      TinyNN.upload_row_major(s, @dn_t_eps, m_eps)
      TinyNN.upload_row_major(s, @dn_t_hp,  m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @dn_t_loss, 1, 1)
      loss_mat.flat[0]
    end

    # ONE denoising evaluation: upload [x_t ; time features], run the
    # graph with the caller's ZEROED hp, and read the predicted epsilon
    # into `out`. The epsilon input keeps whatever was last uploaded —
    # it only feeds the loss, and the loss is not read here.
    def denoise!(x_flat, m_hp, out)
      s = @dn_cache.sess
      TinyNN.tnn_graph_reset_grads_only(s)
      n = @dn_cache.df_batch * @dn_cache.df_d_in
      TinyNN.tnn_upload_from_float_array(s, @dn_cache.t_x, x_flat, n)
      TinyNN.upload_row_major(s, @dn_t_hp, m_hp)
      TinyNN.tnn_compute_backward(s)
      TinyNN.tnn_download_to_f64_array(s, @dn_cache.t_pred, out,
                                       @dn_cache.df_batch * @dn_cache.df_latent)
    end
  end
end; end; end
