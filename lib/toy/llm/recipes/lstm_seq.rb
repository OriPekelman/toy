# lib/toy/llm/recipes/lstm_seq.rb — L4 recipe for the toy#157 (DFA-arch
# T3) LSTM lane: realize an unrolled LSTM + CE + backward graph on a
# Toy::LLM::Engine::LstmEngine, then drive one step at a time.
#
# Same class shape as the other lane recipes (plain class, no-arg ctor,
# uniquely lr_-prefixed ivars, no backend require of its own — the
# runner requires the engine first). CPU-only.

module Toy; module LLM; module Recipes
  class LstmSeq
    attr_accessor :lr_cache, :lr_t_loss, :lr_t_labels, :lr_t_hp

    def initialize
      @lr_cache    = Toy::LLM::Engine::LstmEngine.new
      @lr_t_loss   = nil
      @lr_t_labels = nil
      @lr_t_hp     = nil
    end

    def realize!(d_in, hidden, t_len, batch, n_classes, n_layers, seed,
                 init_scale, policy, cut, b_seed, b_dist, b_scale, b_sigma,
                 clip)
      @lr_cache.realize_for_random_init(d_in, hidden, t_len, batch,
                                        n_classes, n_layers, seed,
                                        init_scale)
      result = @lr_cache.build_training_step(policy, cut, b_seed, b_dist,
                                              b_scale, b_sigma, clip)
      @lr_t_loss   = result[0]
      @lr_t_labels = result[1]
      @lr_t_hp     = result[2]
      nil
    end

    # ONE step over the whole batch of sequences. `x_flat` is laid out
    # step-major ((t * batch + b) * d_in + j) — the same order SsmTask
    # writes, which is why this lane reuses that generator unchanged.
    def step!(x_flat, m_labels, m_hp, is_first)
      s = @lr_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      n = @lr_cache.lstm_t * @lr_cache.lstm_batch * @lr_cache.lstm_d_in
      TinyNN.tnn_upload_from_float_array(s, @lr_cache.t_x, x_flat, n)
      TinyNN.upload_row_major(s, @lr_t_labels, m_labels)
      TinyNN.upload_row_major(s, @lr_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @lr_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
