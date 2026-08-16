# lib/toy/llm/recipes/gtx_graph.rb — L4 recipe for the toy#160 (DFA-arch
# T4) graph-transformer lane: realize an adjacency-masked transformer +
# a small relation head + the backward graph on a
# Toy::LLM::Engine::GtxEngine, then drive one step at a time.
#
# Same class shape as the other lane recipes (plain class, no-arg ctor,
# uniquely gr_-prefixed ivars, no backend require of its own — the
# runner requires the engine first). CPU-only.

module Toy; module LLM; module Recipes
  class GtxGraph
    attr_accessor :gr_cache, :gr_t_loss, :gr_t_labels, :gr_t_hp

    def initialize
      @gr_cache    = Toy::LLM::Engine::GtxEngine.new
      @gr_t_loss   = nil
      @gr_t_labels = nil
      @gr_t_hp     = nil
    end

    def realize!(d_in, d_model, heads, d_ff, n_blocks, n_nodes, n_pairs,
                 n_classes, seed, init_scale, policy, cut, b_seed,
                 b_dist, b_scale, b_sigma, retro_classes, adapters,
                 adapter_rank, bytelm, gnorm)
      @gr_cache.realize_for_random_init(d_in, d_model, heads, d_ff,
                                        n_blocks, n_nodes, n_pairs,
                                        n_classes, seed, init_scale,
                                        retro_classes, adapters,
                                        adapter_rank, bytelm, gnorm)
      result = @gr_cache.build_training_step(policy, cut, b_seed, b_dist,
                                             b_scale, b_sigma)
      @gr_t_loss   = result[0]
      @gr_t_labels = result[1]
      @gr_t_hp     = result[2]
      nil
    end

    # toy#161 — REBUILD the graph for the retrofit phase. The weights are
    # persistent tensors, so the pretrained backbone survives the rebuild
    # untouched; only the graph over them changes (detached backbone,
    # adapter stack, new head). Doing both phases in ONE process is what
    # guarantees every retrofit arm starts from a BIT-IDENTICAL backbone
    # — the comparison's whole premise.
    def rebuild_retrofit!(policy, cut, b_seed, b_dist, b_scale, b_sigma,
                          adapter_policy, freeze_backbone)
      result = @gr_cache.build_retrofit_step(policy, cut, b_seed, b_dist,
                                             b_scale, b_sigma,
                                             adapter_policy, freeze_backbone)
      @gr_t_loss   = result[0]
      @gr_t_labels = result[1]
      @gr_t_hp     = result[2]
      nil
    end

    # The TOPOLOGY is constant for the life of the run, so the adjacency
    # mask is uploaded ONCE. The CONTENT is not: features are redrawn
    # and re-uploaded every step, because a fixed instance is
    # memorisable and a memorised instance needs no retrieval — which is
    # the one thing this lane exists to measure.
    def upload_mask!(mask_flat)
      TinyNN.tnn_upload_from_float_array(@gr_cache.sess, @gr_cache.t_mask,
        mask_flat, @gr_cache.gx_nodes * @gr_cache.gx_nodes)
      nil
    end

    def upload_features!(x_flat)
      TinyNN.tnn_upload_from_float_array(@gr_cache.sess, @gr_cache.t_x,
        x_flat, @gr_cache.gx_nodes * @gr_cache.gx_d_in)
      nil
    end

    # toy#170 — ONE bytelm step. Uploads ONLY the tensors the bytelm
    # graph actually consumes: tokens, labels, hp. NOT t_x / t_idx_a /
    # t_idx_b / t_inc — under bytelm those are allocated but unreferenced,
    # and an input tensor nothing in the graph consumes has NO BACKEND
    # BUFFER, so the upload aborts inside ggml_backend_tensor_set
    # (toy#154's landmine, same shape as the empty-tap-family guard).
    def step_bytelm!(tokens, m_labels, m_hp, is_first)
      s = @gr_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.tnn_upload_from_int_array(s, @gr_cache.t_tokens, tokens,
                                       @gr_cache.gx_nodes)
      TinyNN.upload_row_major(s, @gr_t_labels, m_labels)
      TinyNN.upload_row_major(s, @gr_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @gr_t_loss, 1, 1)
      loss_mat.flat[0]
    end

    # ONE step over a batch of labelled pairs. `inc_flat` is the
    # pair->node incidence the DFA route needs; it is uploaded on every
    # arm so the two arms build the identical graph.
    def step!(idx_a, idx_b, inc_flat, m_labels, m_hp, is_first)
      s = @gr_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.tnn_upload_from_int_array(s, @gr_cache.t_idx_a, idx_a, @gr_cache.gx_pairs)
      TinyNN.tnn_upload_from_int_array(s, @gr_cache.t_idx_b, idx_b, @gr_cache.gx_pairs)
      TinyNN.tnn_upload_from_float_array(s, @gr_cache.t_inc, inc_flat,
        @gr_cache.gx_pairs * @gr_cache.gx_nodes)
      TinyNN.upload_row_major(s, @gr_t_labels, m_labels)
      TinyNN.upload_row_major(s, @gr_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @gr_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
