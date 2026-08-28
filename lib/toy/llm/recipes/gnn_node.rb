# lib/toy/llm/recipes/gnn_node.rb — L4 recipe for the toy#153
# (DFA-arch T1) GNN node-classification lane: realize a message-passing
# forward + masked CE + backward graph on a Toy::LLM::Engine::GnnEngine,
# then drive one full-graph step at a time.
#
# Same class shape as mlp_classifier.rb / ctr_tower.rb (plain class,
# no-arg ctor, uniquely gn_-prefixed ivars, no backend require of its
# own — the runner requires the engine first). CPU-only (tao#18).
#
# NOTE what step! does NOT upload. In this lane the "batch" is the whole
# graph and it never changes, so the features, the adjacency, the label
# one-hots and the training mask are PERSISTENT tensors uploaded once by
# upload_graph!. The only per-step upload is the hp vector — which is
# also why the eval pass is exactly one more step! at lr = 0.

module Toy; module LLM; module Recipes
  class GnnNode
    attr_accessor :gn_cache, :gn_t_loss, :gn_t_hp

    def initialize
      @gn_cache  = Toy::LLM::Engine::GnnEngine.new
      @gn_t_loss = nil
      @gn_t_hp   = nil
    end

    def realize!(n_nodes, feat_dim, d_hidden, n_layers, n_classes,
                 n_train, seed, init_scale,
                 x_flat, s_flat, train_idx, y_flat, mask_flat, lab_flat,
                 policy, feedback, hops, b_seed, b_dist, b_scale, b_sigma)
      @gn_cache.realize_for_random_init(n_nodes, feat_dim, d_hidden,
                                        n_layers, n_classes, n_train,
                                        seed, init_scale)
      @gn_cache.upload_graph!(x_flat, s_flat, train_idx, y_flat,
                              mask_flat, lab_flat)
      result = @gn_cache.build_training_step(policy, feedback, hops,
                                              b_seed, b_dist, b_scale,
                                              b_sigma)
      @gn_t_loss = result[0]
      @gn_t_hp   = result[1]
      nil
    end

    # ONE full-graph step. is_first selects graph_reset vs
    # reset_grads_only, so the caller stays in control of the step == 0
    # branch (the gate's branch).
    # D3b: the mask upload must land BEFORE the graph reset+compute,
    # exactly like the graph's other persistent inputs.
    def step_with_dropout!(m_hp, is_first, step, training)
      @gn_cache.refresh_dropout!(step, training)
      step!(m_hp, is_first)
    end

    def step!(m_hp, is_first)
      s = @gn_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.upload_row_major(s, @gn_t_hp, m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @gn_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
