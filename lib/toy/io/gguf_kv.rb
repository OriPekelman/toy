# GGUF metadata FFI — slim, independent of lib/tinynn.rb / Mat.
#
# Exists to work around a Spinel cross-class type-inference issue
# where loading lib/tokenizer.rb alongside lib/transformer.rb (which
# pulls in Mat-using FFNFFICache via lib/tinynn.rb) triggers
# polymorphic dispatch widening Mat#nrows to sp_RbVal, producing
# generated C that fails to compile.
#
# This module binds ONLY the GGUF metadata kv-read FFI (handle open
# + scalar/array reads + close). No tensor types, no Mat, no FFNFFICache.
# Callers requiring just metadata access load this module standalone.

module GgufKV
  ffi_lib "tinynn_ggml"
  ffi_lib "ggml"
  ffi_lib "ggml-cpu"
  ffi_lib "ggml-base"
  ffi_lib "stdc++"
  ffi_lib "pthread"
  ffi_lib "m"

  # Keep BYTE-IDENTICAL to lib/tinynn.rb's line — it doubles as the
  # spinel-ext.json "tinynn" placeholder (toy#45) and the vendor step
  # substitutes every vendored .rb carrying it.
  ffi_cflags "-L. -Ltinynn -Lvendor/ggml/build/src -Wno-int-conversion"

  ffi_func :tnn_gguf_load,     [:str],            :ptr
  ffi_func :tnn_gguf_free,     [:ptr],            :void
  ffi_func :tnn_gguf_get_u32,  [:ptr, :str],      :int
  ffi_func :tnn_gguf_get_f32,  [:ptr, :str],      :double
  ffi_func :tnn_gguf_get_bool, [:ptr, :str],      :int
  ffi_func :tnn_gguf_get_str,  [:ptr, :str],      :str
  ffi_func :tnn_gguf_arr_n,    [:ptr, :str],      :int
  ffi_func :tnn_gguf_arr_str,  [:ptr, :str, :int], :str
end
