# lib/toy/llm/recipes/mlp_classifier.rb — L4 recipe for the toy#152
# (DFA-arch T0) MLP classifier anchor: realize a random-init
# forward + CE + backward + per-layer-policy update graph on a
# Toy::LLM::Engine::MlpEngine, then drive one step at a time.
#
# Mirrors the CLASS SHAPE of vit_tiny.rb / franken_from_scratch.rb
# (plain class, no-arg ctor, uniquely mc_-prefixed ivars for type
# isolation, NO backend require of its own — the runner requires the
# engine first). CPU-only by construction: tao#18 settled that T0–T3
# get no CUDA twins (the anchors are small, and twin-drift has bitten
# twice recently — toy#150, toy#151).
#
# The realize! signature takes PLAIN INTS rather than a cfg object +
# RecipeOptions: this lane's knob set is disjoint from the llama one
# (no t_seq, no vocab, no untied/qkv_bias), and threading it through
# RecipeOptions would put MLP-only fields on an object shared with
# every other recipe.

module Toy; module LLM; module Recipes
  class MlpClassifier
    attr_accessor :mc_cache, :mc_t_loss, :mc_t_labels, :mc_t_hp

    def initialize
      @mc_cache    = Toy::LLM::Engine::MlpEngine.new
      @mc_t_loss   = nil
      @mc_t_labels = nil
      @mc_t_hp     = nil
    end

    def realize!(d_in, d_hidden, n_layers, n_classes, batch, seed,
                 init_scale, policy, b_seed, b_dist, b_scale, b_sigma)
      @mc_cache.realize_for_random_init(d_in, d_hidden, n_layers,
                                        n_classes, batch, seed, init_scale)
      result = @mc_cache.build_training_step(policy, b_seed, b_dist,
                                             b_scale, b_sigma)
      @mc_t_loss   = result[0]
      @mc_t_labels = result[1]
      @mc_t_hp     = result[2]
      nil
    end

    # ONE step: uploads (x, labels, hp), compute_backward, read the
    # loss. is_first selects graph_reset vs reset_grads_only, so the
    # caller stays in control of the step==0 branch (the gate's branch).
    def step!(m_x, m_labels, m_hp, is_first)
      s = @mc_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.upload_row_major(s, @mc_cache.t_x,   m_x)
      TinyNN.upload_row_major(s, @mc_t_labels,    m_labels)
      TinyNN.upload_row_major(s, @mc_t_hp,        m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @mc_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
