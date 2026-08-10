# lib/toy/llm/recipes/ctr_tower.rb — L4 recipe for the toy#154
# (DFA-arch T1) CTR tower: realize an embeddings + MLP-tower + scalar
# logloss graph on a Toy::LLM::Engine::CtrEngine, then drive one step
# at a time.
#
# Same class shape as mlp_classifier.rb / vit_tiny.rb (plain class,
# no-arg ctor, uniquely ct_-prefixed ivars, no backend require of its
# own — the runner requires the engine first). CPU-only (tao#18).

module Toy; module LLM; module Recipes
  class CtrTower
    attr_accessor :ct_cache, :ct_t_loss, :ct_t_hp

    def initialize
      @ct_cache  = Toy::LLM::Engine::CtrEngine.new
      @ct_t_loss = nil
      @ct_t_hp   = nil
    end

    def realize!(n_fields, cardinality, n_numeric, d_emb, d_hidden,
                 n_layers, batch, seed, init_scale, policy,
                 b_seed, b_dist, b_scale, b_sigma, wide)
      @ct_cache.realize_for_random_init(n_fields, cardinality, n_numeric,
                                        d_emb, d_hidden, n_layers, batch,
                                        seed, init_scale, wide)
      result = @ct_cache.build_training_step(policy, b_seed, b_dist,
                                             b_scale, b_sigma)
      @ct_t_loss = result[0]
      @ct_t_hp   = result[1]
      nil
    end

    # ONE step. `idx` is the flat field-major Int array (n_fields*batch),
    # `m_num` the numeric Mat, `m_labels` the [batch, 2] one-hot the CE
    # reads, `m_y` the [batch, 1] scalar label the DFA error reads.
    # is_first selects graph_reset vs reset_grads_only.
    def step!(idx, m_num, m_labels, m_y, m_hp, is_first)
      s = @ct_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      b = @ct_cache.ctr_batch
      fi = 0
      while fi < @ct_cache.ctr_fields
        # One upload per field: the engine's index leaves are separate
        # persistent tensors (#1449), so the flat field-major buffer is
        # sliced here rather than uploaded whole.
        col = [0]; col.pop
        k = 0
        while k < b
          col.push(idx[fi * b + k])
          k = k + 1
        end
        TinyNN.upload_int_array(s, @ct_cache.t_idx[fi], col)
        fi = fi + 1
      end
      if @ct_cache.ctr_numeric > 0
        TinyNN.upload_row_major(s, @ct_cache.t_numeric, m_num)
      end
      TinyNN.upload_row_major(s, @ct_cache.t_labels, m_labels)
      # Only when a DFA layer put it in the graph — see ctr_uses_y.
      if @ct_cache.ctr_uses_y == 1
        TinyNN.upload_row_major(s, @ct_cache.t_y, m_y)
      end
      TinyNN.upload_row_major(s, @ct_cache.t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @ct_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
