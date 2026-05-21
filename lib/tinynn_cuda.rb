# CUDA-flavored FFI bridge (sibling of lib/tinynn.rb).
#
# Same API surface as lib/tinynn.rb — same `TinyNNCuda.matmul(a, b)`
# semantics — but links against the CUDA-built ggml backend and asks
# tnn_session_new for the CUDA device. Drivers that need CPU use
# `require_relative "lib/tinynn"`; drivers that need GPU use
# `require_relative "lib/tinynn_cuda"`. We deliberately do *not* try to
# pick at runtime: spinel's FFI library linkage is static so a single
# binary either has the CUDA archives linked in or it doesn't.
#
# Build: `make ab-smoke-cuda` (requires `make setup-ggml-cuda` to have
# produced vendor/ggml/build-cuda first).

# Same AdamStepResult definition as lib/tinynn.rb — drivers require
# exactly one of {tinynn, tinynn_cuda} so we duplicate to keep each
# file self-sufficient.
class AdamStepResult
  attr_accessor :param, :mom_m, :mom_v
  def initialize(param, mom_m, mom_v)
    @param = param
    @mom_m = mom_m
    @mom_v = mom_v
  end
end

# Same as FFNFFICache in lib/tinynn.rb but uses TinyNNCuda. The class
# name differs so that drivers requiring BOTH modules (e.g. the CUDA
# parity smoke loads tinynn_cuda directly, and lib/transformer.rb
# transitively pulls in tinynn) don't trip Spinel's same-class-defined-
# twice path. For CUDA training, sed-swap `FFNFFICache` to
# `FFNFFICacheCuda` in lib/transformer.rb along with `TinyNN.` to
# `TinyNNCuda.` in feed_forward_ffi.
class FFNFFICacheCuda
  attr_accessor :sess, :t_h, :t_w1_t, :t_w2_t,
                :t_pre, :t_hidden, :t_out,
                :t_seq, :d_model, :d_ff, :realized

  def initialize
    @realized = false
    @t_seq    = 0
    @d_model  = 0
    @d_ff     = 0
    @sess     = TinyNNCuda.tnn_null_ptr
    @t_h      = TinyNNCuda.tnn_null_ptr
    @t_w1_t   = TinyNNCuda.tnn_null_ptr
    @t_w2_t   = TinyNNCuda.tnn_null_ptr
    @t_pre    = TinyNNCuda.tnn_null_ptr
    @t_hidden = TinyNNCuda.tnn_null_ptr
    @t_out    = TinyNNCuda.tnn_null_ptr
  end

  def realize_for(t_seq, d_model, d_ff)
    @t_seq   = t_seq
    @d_model = d_model
    @d_ff    = d_ff

    @sess   = TinyNNCuda.tnn_session_new(1)
    @t_h    = TinyNNCuda.tnn_input_2d_f32(@sess, t_seq,  d_model)
    @t_w1_t = TinyNNCuda.tnn_input_2d_f32(@sess, d_ff,   d_model)
    @t_w2_t = TinyNNCuda.tnn_input_2d_f32(@sess, d_model, d_ff)

    @t_pre    = TinyNNCuda.tnn_matmul(@sess, @t_w1_t, @t_h)
    @t_hidden = TinyNNCuda.tnn_gelu(@sess, @t_pre)
    @t_out    = TinyNNCuda.tnn_matmul(@sess, @t_w2_t, @t_hidden)
    TinyNNCuda.tnn_set_output(@t_pre)
    TinyNNCuda.tnn_set_output(@t_hidden)
    TinyNNCuda.tnn_set_output(@t_out)
    TinyNNCuda.tnn_realize(@sess, @t_out)

    @realized = true
  end
end

module TinyNNCuda
  ffi_lib "tinynn_ggml_cuda"
  ffi_lib "ggml"
  ffi_lib "ggml-cpu"
  ffi_lib "ggml-cuda"
  ffi_lib "ggml-base"
  ffi_lib "stdc++"
  ffi_lib "pthread"
  ffi_lib "gomp"
  ffi_lib "m"
  # CUDA: static runtime + cublas, plus the dynamic driver lib.
  ffi_lib "cublas_static"
  ffi_lib "cublasLt_static"
  ffi_lib "culibos"
  ffi_lib "cudart_static"
  ffi_lib "cuda"
  ffi_lib "rt"
  ffi_lib "dl"

  ffi_cflags "-L. -Ltinynn -Lvendor/ggml/build-cuda/src -Lvendor/ggml/build-cuda/src/ggml-cpu -Lvendor/ggml/build-cuda/src/ggml-cuda -L/usr/local/cuda/lib64 -Wno-int-conversion"

  ffi_func :tnn_session_new,      [:int],                   :ptr
  ffi_func :tnn_session_free,     [:ptr],                   :void
  ffi_func :tnn_backend_name,     [:ptr],                   :str
  ffi_func :tnn_input_2d_f32,     [:ptr, :int, :int],       :ptr
  ffi_func :tnn_input_1d_i32,     [:ptr, :int],             :ptr
  ffi_func :tnn_matmul,           [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_add,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_gelu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_rms_norm,         [:ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_softmax,          [:ptr, :ptr],             :ptr
  ffi_func :tnn_diag_mask_inf,    [:ptr, :ptr, :int],       :ptr
  # Llama-family ops
  ffi_func :tnn_silu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_mul,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_rope_ext,         [:ptr, :ptr, :ptr, :int, :double], :ptr
  ffi_func :tnn_input_1d_i32_ctx, [:ptr, :int],             :ptr
  ffi_func :tnn_concat,           [:ptr, :ptr, :ptr, :int], :ptr
  ffi_func :tnn_null_ptr,         [],                       :ptr
  ffi_func :tnn_layer_norm,       [:ptr, :ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_view_1d,          [:ptr, :ptr, :int, :long],         :ptr
  ffi_func :tnn_view_2d,          [:ptr, :ptr, :int, :int, :long, :long], :ptr
  ffi_func :tnn_reshape_3d,       [:ptr, :ptr, :int, :int, :int], :ptr
  ffi_func :tnn_reshape_2d,       [:ptr, :ptr, :int, :int],       :ptr
  ffi_func :tnn_cpy,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_set_rows,         [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_soft_max_ext,     [:ptr, :ptr, :ptr, :double, :double], :ptr
  ffi_func :tnn_set_2d,           [:ptr, :ptr, :ptr, :long, :long], :ptr
  ffi_func :tnn_reset_for_rebuild, [:ptr],                  :int
  ffi_func :tnn_scale,            [:ptr, :ptr, :double],    :ptr
  ffi_func :tnn_transpose,        [:ptr, :ptr],             :ptr
  ffi_func :tnn_softmax_back,     [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_silu_back,        [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_rope_ext_back,    [:ptr, :ptr, :ptr, :int, :double], :ptr
  ffi_func :tnn_get_rows,         [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_get_rows_back,    [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_realize,          [:ptr, :ptr],             :int
  ffi_func :tnn_build_forward_only, [:ptr, :ptr],           :int
  ffi_func :tnn_add_to_graph,     [:ptr, :ptr],             :int
  ffi_func :tnn_compute,          [:ptr],                   :int
  # Training primitives — CUDA mirror of the CPU binding set. See
  # tinynn/tinynn_ggml.c for the contracts. Same C library is linked;
  # we only need the FFI declarations on the CUDA module surface.
  ffi_func :tnn_build_backward,       [:ptr],                       :int
  ffi_func :tnn_extend_backward_graph,[:ptr, :ptr],                 :int
  ffi_func :tnn_realize_backward,     [:ptr],                       :int
  ffi_func :tnn_graph_reset,          [:ptr],                       :int
  ffi_func :tnn_compute_backward,     [:ptr],                       :int
  ffi_func :tnn_tensor_grad,          [:ptr, :ptr],                 :ptr
  ffi_func :tnn_set_loss,             [:ptr],                       :void
  ffi_func :tnn_sum,                  [:ptr, :ptr],                 :ptr
  ffi_func :tnn_cross_entropy_loss,   [:ptr, :ptr, :ptr],           :ptr
  ffi_func :tnn_opt_step_sgd,         [:ptr, :ptr, :ptr, :ptr],     :ptr
  ffi_func :tnn_pin_all_graph_b_nodes,[:ptr],                       :int
  ffi_func :tnn_graph_reset_grads_only,[:ptr],                      :int
  ffi_func :tnn_scratch_set,      [:ptr, :int, :double],    :void
  ffi_func :tnn_scratch_get,      [:ptr, :int],             :double
  ffi_func :tnn_scratch_set_i32,  [:ptr, :int, :int],       :void
  ffi_func :tnn_scratch_get_i32,  [:ptr, :int],             :int
  ffi_func :tnn_upload,           [:ptr, :ptr],             :int
  ffi_func :tnn_download,         [:ptr, :ptr],             :int
  ffi_func :tnn_upload_from_float_array, [:ptr, :ptr, :float_array, :size_t], :int
  ffi_func :tnn_upload_transposed_f64,   [:ptr, :ptr, :float_array, :int, :int], :int
  ffi_func :tnn_upload_from_int_array,   [:ptr, :ptr, :int_array,   :size_t], :int
  # CPU-scratch-backed custom kernels (gelu_back, adam_step). They
  # operate on host memory regardless of the session's backend; what
  # makes them "CUDA-mirrored" is just exposing them under TinyNNCuda
  # for callers using a CUDA session.
  ffi_func :tnn_gelu_back_scratch,[:ptr, :int],             :void
  ffi_func :tnn_set_output,       [:ptr],                   :void
  ffi_func :tnn_set_param,        [:ptr],                   :void
  ffi_func :tnn_tensor_nelements, [:ptr],                   :int
  ffi_func :tnn_input_1d_f32,     [:ptr, :int],             :ptr
  ffi_func :tnn_input_2d_f32_persistent, [:ptr, :int, :int],   :ptr
  ffi_func :tnn_input_2d_persistent_typed, [:ptr, :int, :int, :int], :ptr
  ffi_func :tnn_input_1d_f32_persistent, [:ptr, :int],         :ptr
  # Phase 2 BYO-pointer mmap (CUDA path: ggml-cuda patched to expose
  # ggml_backend_cuda_buffer_from_ptr; weight tensors reference
  # cudaHostRegister'd pages and run via UVA on unified-memory SKUs).
  ffi_func :tnn_session_attach_weight_mmap, [:ptr, :ptr, :size_t], :int
  ffi_func :tnn_input_2d_persistent_mmap, [:ptr, :int, :int, :int, :size_t], :ptr
  ffi_func :tnn_input_1d_persistent_mmap, [:ptr, :int, :int, :size_t], :ptr
  # GGUF accessors — same C symbols as TinyNN; exposed here so CUDA-
  # only demos can avoid pulling in lib/tinynn.rb (which drags in extra
  # CPU-side classes that can trip Spinel's type unifier).
  ffi_func :tnn_gguf_load,                  [:str],           :ptr
  ffi_func :tnn_gguf_free,                  [:ptr],           :void
  ffi_func :tnn_gguf_find_index,            [:ptr, :str],     :int
  ffi_func :tnn_gguf_tensor_type,           [:ptr, :int],     :int
  ffi_func :tnn_gguf_get_bool,              [:ptr, :str],     :int
  ffi_func :tnn_gguf_get_str,               [:ptr, :str],     :str
  ffi_func :tnn_gguf_arr_n,                 [:ptr, :str],     :int
  ffi_func :tnn_gguf_arr_str,               [:ptr, :str, :int], :str
  ffi_func :tnn_gguf_mmap_base,             [:ptr],           :ptr
  ffi_func :tnn_gguf_mmap_size,             [:ptr],           :size_t
  ffi_func :tnn_gguf_tensor_file_offset,    [:ptr, :int],     :size_t
  ffi_func :tnn_gguf_copy_to_persistent,            [:ptr, :int, :ptr, :ptr], :int
  ffi_func :tnn_gguf_copy_1d_to_persistent,         [:ptr, :int, :ptr, :ptr], :int
  ffi_func :tnn_gguf_copy_head_slice_to_persistent_native,
           [:ptr, :int, :ptr, :ptr, :int, :int, :int, :int], :int
  ffi_func :tnn_gguf_copy_head_bias_slice_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int], :int
  ffi_func :tnn_finalize_weights, [:ptr],                   :int
  ffi_func :tnn_realize_b,        [:ptr, :ptr],             :int
  ffi_func :tnn_switch_a,         [:ptr],                   :int
  ffi_func :tnn_switch_b,         [:ptr],                   :int
  ffi_func :tnn_compute_b,        [:ptr],                   :int
  ffi_func :tnn_opt_step_adamw,   [:ptr, :ptr, :ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_adam_step_scratch,[:ptr, :int, :double, :double, :double, :double, :double, :double], :void
  ffi_func :tnn_tensor_ne0,       [:ptr],                   :int
  ffi_func :tnn_tensor_ne1,       [:ptr],                   :int

  def self.matmul(a, b)
    sess = TinyNNCuda.tnn_session_new(1)   # 1 = prefer CUDA

    ta   = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb_t = TinyNNCuda.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc   = TinyNNCuda.tnn_matmul(sess, ta, tb_t)
    TinyNNCuda.tnn_realize(sess, tc)

    i = 0
    na = a.nrows * a.ncols
    while i < na
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)

    bc = b.ncols
    br = b.nrows
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNNCuda.tnn_scratch_set(sess, j * br + i, b.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tb_t)

    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)

    out = Mat.new(a.nrows, b.ncols)
    m = a.nrows
    n = b.ncols
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNNCuda.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end

    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.add(a, b)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNNCuda.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNNCuda.tnn_add(sess, ta, tb)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, b.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tb)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.gelu(a)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNNCuda.tnn_gelu(sess, ta)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.rms_norm(x, gamma, eps)
    sess = TinyNNCuda.tnn_session_new(1)
    tx = TinyNNCuda.tnn_input_2d_f32(sess, x.nrows, x.ncols)
    tg = TinyNNCuda.tnn_input_2d_f32(sess, 1, x.ncols)
    tc = TinyNNCuda.tnn_rms_norm(sess, tx, tg, eps)
    TinyNNCuda.tnn_realize(sess, tc)
    nx = x.nrows * x.ncols
    i = 0
    while i < nx
      TinyNNCuda.tnn_scratch_set(sess, i, x.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tx)
    i = 0
    while i < x.ncols
      TinyNNCuda.tnn_scratch_set(sess, i, gamma[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tg)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < nx
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.softmax(a)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNNCuda.tnn_softmax(sess, ta)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.scale(a, s)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNNCuda.tnn_scale(sess, ta, s)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # gelu(h * w1) * w2 chained via the persistent CUDA engine.
  def self.ffn_pipeline(h, w1, w2)
    pre    = TinyNNCuda.matmul(h, w1)
    hidden = TinyNNCuda.gelu(pre)
    TinyNNCuda.matmul(hidden, w2)
  end

  # a * b^T (matches Mat#matmul_t).
  def self.matmul_t(a, b)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNNCuda.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNNCuda.tnn_matmul(sess, ta, tb)
    TinyNNCuda.tnn_realize(sess, tc)
    na = a.nrows * a.ncols
    i = 0
    while i < na
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    nb = b.nrows * b.ncols
    i = 0
    while i < nb
      TinyNNCuda.tnn_scratch_set(sess, i, b.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tb)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, b.nrows)
    m = a.nrows
    n = b.nrows
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNNCuda.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # a^T * b (matches Mat#t_matmul). Both uploaded transposed.
  def self.t_matmul(a, b)
    sess = TinyNNCuda.tnn_session_new(1)
    ta_t = TinyNNCuda.tnn_input_2d_f32(sess, a.ncols, a.nrows)
    tb_t = TinyNNCuda.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc = TinyNNCuda.tnn_matmul(sess, ta_t, tb_t)
    TinyNNCuda.tnn_realize(sess, tc)
    ar = a.nrows
    ac = a.ncols
    i = 0
    while i < ar
      j = 0
      while j < ac
        TinyNNCuda.tnn_scratch_set(sess, j * ar + i, a.flat[i * ac + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta_t)
    br = b.nrows
    bc = b.ncols
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNNCuda.tnn_scratch_set(sess, j * br + i, b.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tb_t)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.ncols, b.ncols)
    m = a.ncols
    n = b.ncols
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNNCuda.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # Per-row softmax backward.
  def self.softmax_back(a_softmax, dy)
    sess = TinyNNCuda.tnn_session_new(1)
    tdy = TinyNNCuda.tnn_input_2d_f32(sess, dy.nrows, dy.ncols)
    ta  = TinyNNCuda.tnn_input_2d_f32(sess, a_softmax.nrows, a_softmax.ncols)
    tc  = TinyNNCuda.tnn_softmax_back(sess, tdy, ta)
    TinyNNCuda.tnn_realize(sess, tc)
    n = dy.nrows * dy.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, dy.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tdy)
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a_softmax.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a_softmax.nrows, a_softmax.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # Embedding lookup: gather rows.
  def self.embed_lookup(table, indices)
    n_idx = indices.length
    sess  = TinyNNCuda.tnn_session_new(1)
    ttab  = TinyNNCuda.tnn_input_2d_f32(sess, table.nrows, table.ncols)
    tidx  = TinyNNCuda.tnn_input_1d_i32(sess, n_idx)
    tout  = TinyNNCuda.tnn_get_rows(sess, ttab, tidx)
    TinyNNCuda.tnn_realize(sess, tout)
    nt = table.nrows * table.ncols
    i = 0
    while i < nt
      TinyNNCuda.tnn_scratch_set(sess, i, table.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ttab)
    i = 0
    while i < n_idx
      TinyNNCuda.tnn_scratch_set_i32(sess, i, indices[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tidx)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tout)
    out = Mat.new(n_idx, table.ncols)
    n = n_idx * table.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # Embedding scatter-add (backward).
  def self.embed_back(d_out, indices, vocab_size)
    n_idx = indices.length
    sess  = TinyNNCuda.tnn_session_new(1)
    td    = TinyNNCuda.tnn_input_2d_f32(sess, d_out.nrows, d_out.ncols)
    tidx  = TinyNNCuda.tnn_input_1d_i32(sess, n_idx)
    tshape = TinyNNCuda.tnn_input_2d_f32(sess, vocab_size, d_out.ncols)
    tout  = TinyNNCuda.tnn_get_rows_back(sess, td, tidx, tshape)
    TinyNNCuda.tnn_realize(sess, tout)
    nd = d_out.nrows * d_out.ncols
    i = 0
    while i < nd
      TinyNNCuda.tnn_scratch_set(sess, i, d_out.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, td)
    i = 0
    while i < n_idx
      TinyNNCuda.tnn_scratch_set_i32(sess, i, indices[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tidx)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tout)
    out = Mat.new(vocab_size, d_out.ncols)
    n = vocab_size * d_out.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # SGD: param_new = param - lr * grad. Composed.
  def self.sgd_step(param, grad, lr)
    TinyNNCuda.add(param, TinyNNCuda.scale(grad, -lr))
  end

  # GeLU backward (tanh approx) via custom CPU kernel. Mirrors
  # TinyNN.gelu_back; same scratch-layout protocol.
  def self.gelu_back(x, dh)
    sess = TinyNNCuda.tnn_session_new(1)
    n = x.nrows * x.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, x.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, n + i, dh.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_gelu_back_scratch(sess, n)
    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, 2 * n + i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # cross_entropy_grad = (softmax(logits) - one_hot(targets)) / n_pred.
  # Composable from TinyNNCuda.softmax + scale + add.
  def self.cross_entropy_grad(logits, targets, n_pred)
    oh = Mat.new(logits.nrows, logits.ncols)
    i = 0
    while i < n_pred
      oh.flat[i * logits.ncols + targets[i]] = 1.0
      i = i + 1
    end
    sm = TinyNNCuda.softmax(logits)
    inv_n = 1.0 / n_pred.to_f
    sm_s = TinyNNCuda.scale(sm, inv_n)
    oh_s = TinyNNCuda.scale(oh, -inv_n)
    TinyNNCuda.add(sm_s, oh_s)
  end

  # Adam step via custom CPU kernel. Returns AdamStepResult (param,
  # mom_m, mom_v) — same shape as TinyNN.adam_step.
  def self.adam_step(param, grad, m, v, lr, b1, b2, eps, omc1, omc2)
    sess = TinyNNCuda.tnn_session_new(1)
    n = param.nrows * param.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, param.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, n + i, grad.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, 2 * n + i, m.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, 3 * n + i, v.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_adam_step_scratch(sess, n, lr, b1, b2, eps, omc1, omc2)
    new_param = Mat.new(param.nrows, param.ncols)
    new_mom_m = Mat.new(param.nrows, param.ncols)
    new_mom_v = Mat.new(param.nrows, param.ncols)
    i = 0
    while i < n
      new_param.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      new_mom_m.flat[i] = TinyNNCuda.tnn_scratch_get(sess, 2 * n + i)
      new_mom_v.flat[i] = TinyNNCuda.tnn_scratch_get(sess, 3 * n + i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    AdamStepResult.new(new_param, new_mom_m, new_mom_v)
  end

  # ----- Persistent-session API (mirrors TinyNN's; see lib/tinynn.rb) -----
  def self.persistent_new(prefer_cuda);  TinyNNCuda.tnn_session_new(prefer_cuda); end
  def self.persistent_free(sess);        TinyNNCuda.tnn_session_free(sess); end
  def self.alloc_2d(sess, r, c);         TinyNNCuda.tnn_input_2d_f32(sess, r, c); end
  def self.alloc_1d_i32(sess, n);        TinyNNCuda.tnn_input_1d_i32(sess, n); end
  def self.build_matmul(sess, ta, tb);   TinyNNCuda.tnn_matmul(sess, ta, tb); end
  def self.build_add(sess, ta, tb);      TinyNNCuda.tnn_add(sess, ta, tb); end
  def self.build_gelu(sess, ta);         TinyNNCuda.tnn_gelu(sess, ta); end
  def self.build_softmax(sess, ta);      TinyNNCuda.tnn_softmax(sess, ta); end
  def self.build_scale(sess, ta, s);     TinyNNCuda.tnn_scale(sess, ta, s); end
  def self.build_rms_norm(sess, x, g, e); TinyNNCuda.tnn_rms_norm(sess, x, g, e); end
  def self.realize(sess, r);             TinyNNCuda.tnn_realize(sess, r); end
  def self.compute(sess);                TinyNNCuda.tnn_compute(sess); end

  def self.upload_row_major(sess, tensor, mat)
    TinyNNCuda.tnn_upload_from_float_array(sess, tensor, mat.flat, mat.nrows * mat.ncols)
  end

  # Upload an Array<Int> to a 1D int32 tensor in one FFI call.
  def self.upload_int_array(sess, tensor, indices)
    TinyNNCuda.tnn_upload_from_int_array(sess, tensor, indices, indices.length)
  end

  def self.upload_transposed(sess, tensor, mat)
    # Chunked in C — works for tensors larger than the 16 MiB scratch
    # buffer (Qwen2.5-0.5B's ffn_* are 17 MB; the per-element path
    # silently truncated at the 4M float boundary).
    TinyNNCuda.tnn_upload_transposed_f64(sess, tensor, mat.flat,
                                          mat.nrows, mat.ncols)
  end

  # Alias to match the CPU module's name; used by feed_forward_ffi.
  def self.stage_transposed_and_upload(sess, target, b)
    TinyNNCuda.upload_transposed(sess, target, b)
  end

  def self.download_row_major(sess, tensor, rows, cols)
    TinyNNCuda.tnn_download(sess, tensor)
    out = Mat.new(rows, cols)
    n = rows * cols
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    out
  end

  def self.download_matmul(sess, tensor, m, n)
    TinyNNCuda.tnn_download(sess, tensor)
    out = Mat.new(m, n)
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNNCuda.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end
    out
  end
end

# === FullForwardFFICacheCuda mirror (M1 family) ===
# Per-block tensor handles for FullForwardFFICache. One instance per
# transformer block. All ivars are :ptr handles (or arrays thereof);
# the actual ggml tensors live in the FullForwardFFICache's session.
class BlockFFICacheCuda
  attr_accessor :t_norm1_gamma, :t_norm2_gamma,
                :t_w_q, :t_w_k, :t_w_v,   # Array<:ptr>, one per head
                :t_w_o, :t_w_ff1, :t_w_ff2

  def initialize
    @t_norm1_gamma = TinyNNCuda.tnn_null_ptr
    @t_norm2_gamma = TinyNNCuda.tnn_null_ptr
    # Note: Spinel currently types these arrays as `IntArray` rather
    # than `PtrArray` because the FFI's `:ptr` value model is integer-
    # backed (a long-sized address). The runtime is correct because
    # pointer values fit in `mrb_int` on 64-bit platforms — every
    # FFI call that takes the element gets a `void *` produced by a
    # silent long-to-pointer conversion (cc emits a warning but the
    # bit pattern round-trips). On a 32-bit platform this would break.
    # Seeded with `tnn_null_ptr` rather than `[nil]` so the intent is
    # clear in source even though Spinel infers the same way either way.
    @t_w_q   = [TinyNNCuda.tnn_null_ptr]
    @t_w_k   = [TinyNNCuda.tnn_null_ptr]
    @t_w_v   = [TinyNNCuda.tnn_null_ptr]
    @t_w_o   = TinyNNCuda.tnn_null_ptr
    @t_w_ff1 = TinyNNCuda.tnn_null_ptr
    @t_w_ff2 = TinyNNCuda.tnn_null_ptr
  end
end

# Full forward of a TransformerLM as one persistent ggml graph. Built
# incrementally; M1.1 covered embed + positional embedding + tied
# unembed (the bookends). M1.2 adds one full transformer block:
# pre-RMSNorm, multi-head causal attention, residual, pre-RMSNorm, FFN,
# residual. M1.3+ will scale to n_layers blocks.
#
# Layout conventions (see project_chained_ffn_2026_05_14):
#   - Mat (rows, cols) row-major upload  -> ggml ne=[cols, rows]
#   - Per-block intermediates carry ne=[d_model, T]: elem(d, t) is the
#     logical value at (row=t, col=d).
#
# Persistent (ctx_w):
#   - t_token_embed (vocab, d_model)
#   - t_pos_slice   (T, d_model)
#   - t_final_norm_gamma (d_model)
#   - per block (in @blocks_ffi):
#     - t_norm1_gamma, t_norm2_gamma (d_model)
#     - t_w_q[h], t_w_k[h], t_w_v[h] (d_model, d_head) per head
#     - t_w_o   (d_model, d_model)
#     - t_w_ff1 (d_model, d_ff), t_w_ff2 (d_ff, d_model)
#
# Compute (ctx):       t_token_ids (T int32), intermediates, t_logits
class FullForwardFFICacheCuda
  attr_accessor :sess, :t_token_embed, :t_pos_slice, :t_token_ids,
                :t_final_norm_gamma,
                :blocks_ffi,
                :t_x_embed, :t_x_final, :t_logits,
                :t_seq, :d_model, :d_ff, :n_heads, :d_head, :n_layers,
                :vocab_size, :realized

  def initialize
    @realized   = false
    @t_seq      = 0
    @d_model    = 0
    @d_ff       = 0
    @n_heads    = 0
    @d_head     = 0
    @n_layers   = 0
    @vocab_size = 0
    @sess               = TinyNNCuda.tnn_null_ptr
    @t_token_embed      = TinyNNCuda.tnn_null_ptr
    @t_pos_slice        = TinyNNCuda.tnn_null_ptr
    @t_token_ids        = TinyNNCuda.tnn_null_ptr
    @t_final_norm_gamma = TinyNNCuda.tnn_null_ptr
    @t_x_embed          = TinyNNCuda.tnn_null_ptr
    @t_x_final          = TinyNNCuda.tnn_null_ptr
    @t_logits           = TinyNNCuda.tnn_null_ptr
    @blocks_ffi         = [BlockFFICacheCuda.new]
  end

  def realize_for(t_seq, d_model, d_ff, n_heads, n_layers, vocab_size)
    @t_seq      = t_seq
    @d_model    = d_model
    @d_ff       = d_ff
    @n_heads    = n_heads
    @d_head     = d_model / n_heads
    @n_layers   = n_layers
    @vocab_size = vocab_size

    @sess = TinyNNCuda.tnn_session_new(1)

    # === Persistent weights (ctx_w) ===
    @t_token_embed      = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, vocab_size, d_model)
    @t_pos_slice        = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, t_seq,      d_model)
    @t_final_norm_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)

    # Build per-block tensor handles (seed-then-push for Spinel's
    # Array<BlockFFICache> inference).
    @blocks_ffi = [BlockFFICacheCuda.new]
    li = 1
    while li < n_layers
      @blocks_ffi.push(BlockFFICacheCuda.new)
      li = li + 1
    end

    li = 0
    while li < n_layers
      blk = @blocks_ffi[li]
      blk.t_norm1_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_norm2_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      # Per-head Q/K/V: shape (d_model, d_head). Uploaded TRANSPOSED so
      # ggml ne=[d_model, d_head] holds w.elem(r, c) = w[r][c].
      blk.t_w_q = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_k = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_v = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      h = 1
      while h < n_heads
        blk.t_w_q.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_k.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_v.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        h = h + 1
      end
      blk.t_w_o   = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_model, d_model)
      blk.t_w_ff1 = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_ff,    d_model)
      blk.t_w_ff2 = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_model, d_ff)
      li = li + 1
    end

    TinyNNCuda.tnn_finalize_weights(@sess)

    # === Compute input ===
    @t_token_ids = TinyNNCuda.tnn_input_1d_i32(@sess, t_seq)

    # === Forward graph ===
    # x_embed = token_embed[ids] + pos_slice  (ne=[d_model, T])
    t_embedded = TinyNNCuda.tnn_get_rows(@sess, @t_token_embed, @t_token_ids)
    @t_x_embed = TinyNNCuda.tnn_add(@sess, t_embedded, @t_pos_slice)
    TinyNNCuda.tnn_set_output(@t_x_embed)

    # Through each block.
    t_cur = @t_x_embed
    eps   = 1.0e-5
    scale = 1.0 / Math.sqrt(d_head.to_f)
    li = 0
    while li < n_layers
      t_cur = build_block(t_cur, @blocks_ffi[li], eps, scale)
      li = li + 1
    end

    # Final RMSNorm on the post-blocks x.
    @t_x_final = TinyNNCuda.tnn_rms_norm(@sess, t_cur, @t_final_norm_gamma, eps)
    TinyNNCuda.tnn_set_output(@t_x_final)

    # Tied unembed: logits = mul_mat(token_embed, x_final)  ne=[vocab, T]
    @t_logits = TinyNNCuda.tnn_matmul(@sess, @t_token_embed, @t_x_final)
    TinyNNCuda.tnn_set_output(@t_logits)

    TinyNNCuda.tnn_realize(@sess, @t_logits)
    @realized = true
  end

  # Build one transformer block's graph nodes. Returns the block's
  # output tensor (post-FFN residual). Mathematics:
  #   h1 = rms_norm(x, norm1_gamma)
  #   per head h:
  #     q_h = w_q[h]^T @ h1     (mul_mat(w_q_t_h, h1)  ne=[d_head, T])
  #     k_h = w_k[h]^T @ h1
  #     v_h = h1 @ w_v[h]       (mul_mat(h1, w_v_t_h)  ne=[T, d_head])
  #     scores_h = mul_mat(k_h, q_h)   ne=[T_key, T_query]
  #     scaled_h = scale(scores_h, 1/sqrt(d_head))
  #     masked_h = diag_mask_inf(scaled_h, 0)         -- causal
  #     attn_h   = soft_max(masked_h)  -- per-query softmax over keys
  #     head_out_h = mul_mat(v_h, attn_h)  ne=[d_head, T_query]
  #   concat = concat_along_d(head_out_h for h in heads)  ne=[d_model, T]
  #   out_proj = mul_mat(w_o_t, concat)  ne=[d_model, T]
  #   x_attn = x + out_proj
  #   h2 = rms_norm(x_attn, norm2_gamma)
  #   ffn:
  #     pre    = mul_mat(w_ff1_t, h2)   ne=[d_ff,    T]
  #     hidden = gelu(pre)
  #     ffn_out= mul_mat(w_ff2_t, hidden) ne=[d_model, T]
  #   x_out = x_attn + ffn_out
  def build_block(t_x, blk, eps, scale)
    # Pre-norm before attention.
    t_h1 = TinyNNCuda.tnn_rms_norm(@sess, t_x, blk.t_norm1_gamma, eps)

    # Per-head attention. Build each head's output, then concat.
    t_head_outs = [build_attention_head(t_h1, blk.t_w_q[0], blk.t_w_k[0], blk.t_w_v[0], scale)]
    h = 1
    while h < @n_heads
      t_head_outs.push(build_attention_head(t_h1, blk.t_w_q[h], blk.t_w_k[h], blk.t_w_v[h], scale))
      h = h + 1
    end

    # Concat along ne0 (d_head -> d_model).
    t_concat = t_head_outs[0]
    h = 1
    while h < @n_heads
      t_concat = TinyNNCuda.tnn_concat(@sess, t_concat, t_head_outs[h], 0)
      h = h + 1
    end

    # Output projection + residual.
    t_out_proj = TinyNNCuda.tnn_matmul(@sess, blk.t_w_o, t_concat)
    t_x_attn   = TinyNNCuda.tnn_add(@sess, t_x, t_out_proj)

    # Pre-norm before FFN.
    t_h2 = TinyNNCuda.tnn_rms_norm(@sess, t_x_attn, blk.t_norm2_gamma, eps)

    # FFN (matches FFNFFICache's chained design).
    t_pre    = TinyNNCuda.tnn_matmul(@sess, blk.t_w_ff1, t_h2)
    t_hidden = TinyNNCuda.tnn_gelu(@sess, t_pre)
    t_ffn    = TinyNNCuda.tnn_matmul(@sess, blk.t_w_ff2, t_hidden)

    # Second residual.
    TinyNNCuda.tnn_add(@sess, t_x_attn, t_ffn)
  end

  # Single attention head, given pre-normed x and the head's persistent
  # Q/K/V weights. See build_block's docstring for the math.
  def build_attention_head(t_x, t_w_q, t_w_k, t_w_v, scale)
    t_q = TinyNNCuda.tnn_matmul(@sess, t_w_q, t_x)   # ne=[d_head, T]
    t_k = TinyNNCuda.tnn_matmul(@sess, t_w_k, t_x)   # ne=[d_head, T]
    # v in Pattern A (ne=[T, d_head]) so head_out's k_dim matches.
    # mul_mat(x, w_v_t) where x.ne=[d_model, T] and w_v_t.ne=[d_model, d_head]
    # yields ne=[T, d_head]. ✓
    t_v = TinyNNCuda.tnn_matmul(@sess, t_x, t_w_v)

    t_scores = TinyNNCuda.tnn_matmul(@sess, t_k, t_q)            # ne=[T_key, T_query]
    t_scaled = TinyNNCuda.tnn_scale(@sess, t_scores, scale)
    t_masked = TinyNNCuda.tnn_diag_mask_inf(@sess, t_scaled, 0)
    t_attn   = TinyNNCuda.tnn_softmax(@sess, t_masked)           # softmax along ne0 = key dim

    TinyNNCuda.tnn_matmul(@sess, t_v, t_attn)                    # ne=[d_head, T_query]
  end
end

# Spinel anchor block: mirror of the one at the end of lib/tinynn.rb,
# this time pinning TinyNNCuda.upload_int_array's `indices` param to
# Array<Int>. Without this, library-style methods with no reachable
# caller default `indices` to mrb_int, the `:int_array` FFI spec's
# `indices->data` access fails C compile.
if false
  _aic_sess   = TinyNNCuda.tnn_null_ptr
  _aic_tensor = TinyNNCuda.tnn_null_ptr
  _aic_ids    = [0]
  TinyNNCuda.upload_int_array(_aic_sess, _aic_tensor, _aic_ids)
  # Also pin the literal `indices` name as Array<Int> (same trick as
  # the CPU anchor) so embed_lookup / embed_back / scratch_set_i32
  # see int elements when `indices[i]` is evaluated. Without this the
  # local-name collapse can rewire `indices` to Array<aggregate>
  # depending on what else is loaded.
  indices = [0]
  _aic_table = Mat.new(1, 1)
  TinyNNCuda.embed_lookup(_aic_table, indices)
  TinyNNCuda.embed_back(_aic_table, indices, 1)
  # Pin upload_row_major / upload_transposed as returning int (the
  # underlying FFI int return). Without a concrete callsite Spinel
  # may infer aggregate, then the `return` of the FFI call fails to
  # compile in CUDA-only demos that don't transitively load
  # toy_smollm2_ffi_kv_cuda.
  _aic_mat = Mat.new(1, 1)
  TinyNNCuda.upload_row_major(_aic_sess, _aic_tensor, _aic_mat)
  TinyNNCuda.upload_transposed(_aic_sess, _aic_tensor, _aic_mat)
end
