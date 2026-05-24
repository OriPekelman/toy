# lib/tinynn_metal.rb — Metal-flavored FFI bridge.
#
# Sibling of lib/tinynn.rb and lib/tinynn_cuda.rb. Same API surface,
# different backend underneath. Drivers needing Metal `require_relative
# "lib/tinynn_metal"` and pass `2` to `tnn_session_new`.
#
# Build chain: `make setup-ggml-metal` then any `_metal` target. The
# ggml-metal kernels JIT-compile on first device load (~15s, then
# cached) because GGML_METAL_EMBED_LIBRARY=ON; this is the price we
# pay for the toolchain working with Command Line Tools rather than
# requiring full Xcode.
#
# Hardware: any Apple Silicon Mac (M1 / M2 / M3 / M4 / M5). Intel Macs
# with a discrete AMD GPU also work in principle; not tested in this
# project.

# Same AdamStepResult definition as lib/tinynn.rb and lib/tinynn_cuda.rb
# — drivers require exactly one of {tinynn, tinynn_cuda, tinynn_metal}
# so we duplicate to keep each file self-sufficient.
class AdamStepResult
  attr_accessor :param, :mom_m, :mom_v
  def initialize(param, mom_m, mom_v)
    @param = param
    @mom_m = mom_m
    @mom_v = mom_v
  end
end

# Same as FFNFFICache / FFNFFICacheCuda — Metal-flavored. Class name
# differs so drivers requiring multiple backends don't trip Spinel's
# same-class-defined-twice path.
class FFNFFICacheMetal
  attr_accessor :sess, :t_h, :t_w1_t, :t_w2_t,
                :t_pre, :t_hidden, :t_out,
                :t_seq, :d_model, :d_ff, :realized

  def initialize
    @realized = false
    @t_seq    = 0
    @d_model  = 0
    @d_ff     = 0
    @sess     = TinyNNMetal.tnn_null_ptr
    @t_h      = TinyNNMetal.tnn_null_ptr
    @t_w1_t   = TinyNNMetal.tnn_null_ptr
    @t_w2_t   = TinyNNMetal.tnn_null_ptr
    @t_pre    = TinyNNMetal.tnn_null_ptr
    @t_hidden = TinyNNMetal.tnn_null_ptr
    @t_out    = TinyNNMetal.tnn_null_ptr
  end

  def realize_for(t_seq, d_model, d_ff)
    @t_seq   = t_seq
    @d_model = d_model
    @d_ff    = d_ff

    @sess   = TinyNNMetal.tnn_session_new(2)
    @t_h    = TinyNNMetal.tnn_input_2d_f32(@sess, t_seq,  d_model)
    @t_w1_t = TinyNNMetal.tnn_input_2d_f32(@sess, d_ff,   d_model)
    @t_w2_t = TinyNNMetal.tnn_input_2d_f32(@sess, d_model, d_ff)
    @t_pre    = TinyNNMetal.tnn_matmul(@sess, @t_w1_t, @t_h)
    @t_hidden = TinyNNMetal.tnn_gelu(@sess, @t_pre)
    @t_out    = TinyNNMetal.tnn_matmul(@sess, @t_w2_t, @t_hidden)
    TinyNNMetal.tnn_set_output(@t_pre)
    TinyNNMetal.tnn_set_output(@t_hidden)
    TinyNNMetal.tnn_set_output(@t_out)
    TinyNNMetal.tnn_realize(@sess, @t_out)
    @realized = true
  end
end

module TinyNNMetal
  # Linking order: the Metal-specific archive first (force-link symbol
  # lives there), then the common tinynn_ggml C wrappers, then ggml-
  # metal, then the rest of ggml. Foundation/Metal/MetalKit frameworks
  # are linked via ffi_cflags (-framework …).
  ffi_lib "tinynn_ggml_metal"
  ffi_lib "tinynn_ggml"
  ffi_lib "ggml"
  ffi_lib "ggml-cpu"
  ffi_lib "ggml-metal"
  ffi_lib "ggml-base"
  ffi_lib "c++"
  ffi_lib "pthread"
  ffi_lib "m"

  ffi_cflags "-L. -Ltinynn -Lvendor/ggml/build-metal/src -Lvendor/ggml/build-metal/src/ggml-cpu -Lvendor/ggml/build-metal/src/ggml-metal -Wno-int-conversion -framework Foundation -framework Metal -framework MetalKit"

  ffi_func :tnn_session_new,      [:int],                   :ptr
  ffi_func :tnn_session_free,     [:ptr],                   :void
  ffi_func :tnn_shutdown_engines, [],                       :void
  # _exit() trampoline — skips C++ static destructors. Use at end of
  # main on Metal programs. See tinynn_backend_metal.m for the why.
  ffi_func :tnn_force_exit,       [:int],                   :void
  ffi_func :tnn_backend_name,     [:ptr],                   :str
  ffi_func :tnn_link_check,       [],                       :int
  ffi_func :tnn_input_2d_f32,     [:ptr, :int, :int],       :ptr
  ffi_func :tnn_input_1d_i32,     [:ptr, :int],             :ptr
  ffi_func :tnn_matmul,           [:ptr, :ptr, :ptr],       :ptr
  # M2: MoE primitives.
  ffi_func :tnn_mul_mat_id,       [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_add_id,           [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_argsort,          [:ptr, :ptr, :int],       :ptr
  ffi_func :tnn_top_k,            [:ptr, :ptr, :int],       :ptr
  ffi_func :tnn_input_3d_f32_persistent, [:ptr, :int, :int, :int], :ptr
  ffi_func :tnn_input_3d_persistent_typed, [:ptr, :int, :int, :int, :int], :ptr
  ffi_func :tnn_add,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_gelu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_rms_norm,         [:ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_softmax,          [:ptr, :ptr],             :ptr
  ffi_func :tnn_diag_mask_inf,    [:ptr, :ptr, :int],       :ptr
  ffi_func :tnn_flash_attn_ext,   [:ptr, :ptr, :ptr, :ptr, :ptr,
                                   :double, :double, :double], :ptr
  ffi_func :tnn_silu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_mul,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_rope_ext,         [:ptr, :ptr, :ptr, :int,
                                   :double, :double, :double, :double,
                                   :double, :double, :ptr], :ptr
  ffi_func :tnn_rope_freq_factors_alloc, [:ptr, :int], :ptr
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
  ffi_func :tnn_rope_ext_back,    [:ptr, :ptr, :ptr, :int,
                                   :double, :double, :double, :double,
                                   :double, :double, :ptr], :ptr
  ffi_func :tnn_get_rows,         [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_get_rows_back,    [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_realize,          [:ptr, :ptr],             :int
  ffi_func :tnn_build_forward_only, [:ptr, :ptr],           :int
  ffi_func :tnn_add_to_graph,     [:ptr, :ptr],             :int
  ffi_func :tnn_compute,          [:ptr],                   :int
  ffi_func :tnn_build_backward,       [:ptr],                       :int
  ffi_func :tnn_extend_backward_graph,[:ptr, :ptr],                 :int
  ffi_func :tnn_realize_backward,     [:ptr],                       :int
  ffi_func :tnn_graph_reset,          [:ptr],                       :int
  ffi_func :tnn_compute_backward,     [:ptr],                       :int
  ffi_func :tnn_tensor_grad,          [:ptr, :ptr],                 :ptr
  ffi_func :tnn_trace_open,           [:str],                       :int
  ffi_func :tnn_trace_close,          [],                           :void
  ffi_func :tnn_trace_begin,          [:str],                       :long
  ffi_func :tnn_trace_end,            [:str, :long],                :void
  ffi_func :tnn_trace_mark,           [:str],                       :void
  ffi_func :tnn_trace_active,         [],                           :int
  ffi_func :tnn_trace_set_op_capture, [:int],                       :void
  ffi_func :tnn_trace_op_capture_active, [],                        :int
  ffi_func :tnn_set_loss,             [:ptr],                       :void
  ffi_func :tnn_sum,                  [:ptr, :ptr],                 :ptr
  ffi_func :tnn_sum_rows,             [:ptr, :ptr],                 :ptr
  ffi_func :tnn_cross_entropy_loss,   [:ptr, :ptr, :ptr],           :ptr
  ffi_func :tnn_opt_step_sgd,         [:ptr, :ptr, :ptr, :ptr],     :ptr
  ffi_func :tnn_pin_all_graph_b_nodes,[:ptr],                       :int
  ffi_func :tnn_graph_reset_grads_only,[:ptr],                      :int
  ffi_func :tnn_scratch_set,      [:ptr, :int, :double],    :void
  ffi_func :tnn_scratch_get,      [:ptr, :int],             :double
  ffi_func :tnn_scratch_set_i32,  [:ptr, :int, :int],       :void
  ffi_func :tnn_scratch_get_i32,  [:ptr, :int],             :int
  ffi_func :tnn_scratch_min_f32,        [:ptr, :int], :double
  ffi_func :tnn_scratch_max_f32,        [:ptr, :int], :double
  ffi_func :tnn_scratch_sum_abs_f32,    [:ptr, :int], :double
  ffi_func :tnn_scratch_sum_sq_f32,     [:ptr, :int], :double
  ffi_func :tnn_scratch_sum_f32,        [:ptr, :int], :double
  ffi_func :tnn_scratch_nan_count_f32,  [:ptr, :int], :int
  ffi_func :tnn_upload,           [:ptr, :ptr],             :int
  ffi_func :tnn_download,         [:ptr, :ptr],             :int
  ffi_func :tnn_upload_from_float_array, [:ptr, :ptr, :float_array, :size_t], :int
  ffi_func :tnn_upload_transposed_f64,   [:ptr, :ptr, :float_array, :int, :int], :int
  ffi_func :tnn_upload_from_int_array,   [:ptr, :ptr, :int_array,   :size_t], :int
  ffi_func :tnn_gelu_back_scratch,[:ptr, :int],             :void
  ffi_func :tnn_set_output,       [:ptr],                   :void
  ffi_func :tnn_set_param,        [:ptr],                   :void
  ffi_func :tnn_tensor_nelements, [:ptr],                   :int
  ffi_func :tnn_input_1d_f32,     [:ptr, :int],             :ptr
  ffi_func :tnn_input_2d_f32_persistent, [:ptr, :int, :int],   :ptr
  ffi_func :tnn_input_2d_persistent_typed, [:ptr, :int, :int, :int], :ptr
  ffi_func :tnn_row_size,                  [:int, :int],              :long
  ffi_func :tnn_input_1d_f32_persistent, [:ptr, :int],         :ptr
  # Phase 2 BYO-pointer mmap. On Metal the buffer-from-ptr path falls
  # through to ggml_backend_cpu_buffer_from_ptr (no public Metal
  # buffer_from_ptr API); the scheduler then copies host pages to
  # Metal buffers on first compute. Apple Silicon's unified memory
  # keeps this cheap but it's not literally zero-copy.
  ffi_func :tnn_session_attach_weight_mmap, [:ptr, :ptr, :size_t], :int
  ffi_func :tnn_input_2d_persistent_mmap, [:ptr, :int, :int, :int, :size_t], :ptr
  ffi_func :tnn_input_3d_persistent_mmap, [:ptr, :int, :int, :int, :int, :size_t], :ptr
  ffi_func :tnn_input_1d_persistent_mmap, [:ptr, :int, :int, :size_t], :ptr
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
  ffi_func :tnn_gguf_copy_verbatim_to_persistent,    [:ptr, :int, :ptr, :ptr], :int
  ffi_func :tnn_gguf_copy_verbatim_head_slice_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int], :int
  ffi_func :tnn_finalize_weights, [:ptr],                   :int
  ffi_func :tnn_zero_tensor,      [:ptr, :ptr],             :int
  ffi_func :tnn_realize_b,        [:ptr, :ptr],             :int
  ffi_func :tnn_switch_a,         [:ptr],                   :int
  ffi_func :tnn_switch_b,         [:ptr],                   :int
  ffi_func :tnn_compute_b,        [:ptr],                   :int
  ffi_func :tnn_opt_step_adamw,   [:ptr, :ptr, :ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_adam_step_scratch,[:ptr, :int, :double, :double, :double, :double, :double, :double], :void
  ffi_func :tnn_tensor_ne0,       [:ptr],                   :int
  ffi_func :tnn_tensor_ne1,       [:ptr],                   :int

  # CPU-parity bindings (file-utility helpers + extra GGUF accessors).
  ffi_func :tnn_rms_norm_back,    [:ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_gguf_load_empty,            [],                              :ptr
  ffi_func :tnn_gguf_n_tensors,             [:ptr],                          :int
  ffi_func :tnn_gguf_tensor_name,           [:ptr, :int],                    :str
  ffi_func :tnn_gguf_tensor_ne,             [:ptr, :int, :int],              :int
  ffi_func :tnn_gguf_tensor_nbytes,         [:ptr, :int],                    :size_t
  ffi_func :tnn_gguf_tensor_is_quantized,   [:ptr, :int],                    :int
  ffi_func :tnn_gguf_get_u32,               [:ptr, :str],                    :int
  ffi_func :tnn_gguf_get_f32,               [:ptr, :str],                    :double
  ffi_func :tnn_gguf_read_f32_to_doubles,   [:ptr, :int, :float_array, :size_t], :int
  ffi_func :tnn_gguf_write_demo_file,       [:str],                          :int
  ffi_func :tnn_gguf_copy_transposed_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int], :int
  ffi_func :tnn_gguf_copy_head_slice_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int, :int, :int], :int
  ffi_func :tnn_list_ggufs,                 [:str],                          :str
  ffi_func :tnn_file_size,                  [:str],                          :size_t
  ffi_func :tnn_download_to_f64_array,      [:ptr, :ptr, :float_array, :size_t], :int

  # Ruby-level convenience helpers — these are pure-Ruby wrappers over
  # the FFI surface. The mirror generator only rewrites the per-method
  # `TinyNN.` qualifiers, not the helper-method bodies themselves;
  # callers in the SmolLM2 mirror reference `TinyNNMetal.upload_int_array`
  # etc., so we define them here. Bodies are identical to lib/tinynn.rb.
  def self.upload_row_major(sess, tensor, mat)
    TinyNNMetal.tnn_upload_from_float_array(sess, tensor, mat.flat, mat.nrows * mat.ncols)
  end

  def self.upload_int_array(sess, tensor, indices)
    TinyNNMetal.tnn_upload_from_int_array(sess, tensor, indices, indices.length)
  end

  def self.upload_transposed(sess, tensor, mat)
    br = mat.nrows
    bc = mat.ncols
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNNMetal.tnn_scratch_set(sess, j * br + i, mat.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNNMetal.tnn_upload(sess, tensor)
  end

  def self.download_row_major(sess, dl_handle, rows, cols)
    TinyNNMetal.tnn_download(sess, dl_handle)
    out = Mat.new(rows, cols)
    n = rows * cols
    i = 0
    while i < n
      out.flat[i] = TinyNNMetal.tnn_scratch_get(sess, i)
      i = i + 1
    end
    out
  end

  def self.download_to_mat(sess, dl_handle, rows, cols)
    out = Mat.new(rows, cols)
    TinyNNMetal.tnn_download_to_f64_array(sess, dl_handle, out.flat, rows * cols)
    out
  end

  def self.download_matmul(sess, tensor, m, n)
    TinyNNMetal.tnn_download(sess, tensor)
    out = Mat.new(m, n)
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNNMetal.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end
    out
  end

  def self.stage_transposed_and_upload(sess, target, b)
    TinyNNMetal.tnn_upload_transposed_f64(sess, target, b.flat, b.nrows, b.ncols)
  end

  def self.stage_row_major_and_upload(sess, target, m)
    TinyNNMetal.tnn_upload_from_float_array(sess, target, m.flat, m.nrows * m.ncols)
  end
end
