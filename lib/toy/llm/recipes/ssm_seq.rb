# lib/toy/llm/recipes/ssm_seq.rb — L4 recipe for the toy#155 (DFA-arch
# T2) selective-scan lane: realize an unrolled selective-recurrence
# forward + CE + backward graph on a Toy::LLM::Engine::SsmEngine, then
# drive one step at a time.
#
# Same class shape as mlp_classifier.rb / ctr_tower.rb / gnn_node.rb
# (plain class, no-arg ctor, uniquely sq_-prefixed ivars, no backend
# require of its own — the runner requires the engine first). CPU-only
# (tao#18; the F19 CUDA twin is deferred to the long-sequence
# memory measurement and is not this slice).

module Toy; module LLM; module Recipes
  class SsmSeq
    attr_accessor :sq_cache, :sq_t_loss, :sq_t_labels, :sq_t_hp

    def initialize
      @sq_cache    = Toy::LLM::Engine::SsmEngine.new
      @sq_t_loss   = nil
      @sq_t_labels = nil
      @sq_t_hp     = nil
    end

    def realize!(d_model, d_inner, t_len, batch, n_classes, n_layers,
                 conv_k, selection, seed, init_scale, dt_init,
                 policy, cut, b_seed, b_dist, b_scale, b_sigma)
      @sq_cache.realize_for_random_init(d_model, d_inner, t_len, batch,
                                        n_classes, n_layers, conv_k,
                                        selection, seed, init_scale,
                                        dt_init)
      result = @sq_cache.build_training_step(policy, cut, b_seed,
                                              b_dist, b_scale, b_sigma)
      @sq_t_loss   = result[0]
      @sq_t_labels = result[1]
      @sq_t_hp     = result[2]
      nil
    end

    # ONE step. `x_flat` is the whole batch of sequences laid out
    # step-major ((t * batch + b) * d_model + j) — the column order the
    # engine's per-step views slice. `m_labels` is the [batch, classes]
    # one-hot the CE reads.
    def step!(x_flat, m_labels, m_hp, is_first)
      s = @sq_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      n = @sq_cache.ssm_t * @sq_cache.ssm_batch * @sq_cache.ssm_d_model
      TinyNN.tnn_upload_from_float_array(s, @sq_cache.t_x, x_flat, n)
      TinyNN.upload_row_major(s, @sq_t_labels, m_labels)
      TinyNN.upload_row_major(s, @sq_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @sq_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
