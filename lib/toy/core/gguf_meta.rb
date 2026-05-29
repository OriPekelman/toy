# lib/toy/core/gguf_meta.rb — pure-Ruby GGUF header + kv + tensor-info reader.
#
# CRuby ONLY. No FFI, no ggml, no Spinel. This is the CLI shell's
# (lib/toy/core/) means of reading model metadata WITHOUT realizing a
# ggml session — that path (Arch.from_gguf via tnn_gguf_*) is Spinel-
# compiled and unreachable under MRI.
#
# Reads:
#   magic "GGUF" + u32 version (==3) + u64 n_tensors + u64 n_kv,
#   the type-tagged KV section (scalars + arrays; arrays of strings are
#   skipped past, not materialized), and the tensor-info section
#   (name, n_dims, dims[], type, offset) for tensor-NAME presence.
#
# Fail-loud (memory: feedback_never_mask_fail_loud): an unknown KV type
# tag or a bad magic raises Toy::Core::GGUFMeta::ParseError with context,
# rather than silently returning garbage. Callers that want best-effort
# degrade-to-unknown rescue ParseError explicitly.

module Toy
  module Core
    # Parsed view of a GGUF file's metadata (no tensor data read).
    #
    #   meta = GGUFMeta.read("model.gguf")
    #   meta.kv["llama.embedding_length"]   # => 576
    #   meta.tensor?("output.weight")       # => false (tied)
    class GGUFMeta
      class ParseError < StandardError; end

      # GGUF metadata value type tags (gguf.h: enum gguf_metadata_value_type).
      T_UINT8   = 0
      T_INT8    = 1
      T_UINT16  = 2
      T_INT16   = 3
      T_UINT32  = 4
      T_INT32   = 5
      T_FLOAT32 = 6
      T_BOOL    = 7
      T_STRING  = 8
      T_ARRAY   = 9
      T_UINT64  = 10
      T_INT64   = 11
      T_FLOAT64 = 12

      attr_reader :version, :n_tensors, :kv, :tensor_names

      def self.read(path)
        File.open(path, "rb") { |io| new(io, path).parse }
      end

      def initialize(io, path)
        @io = io
        @path = path
        @kv = {}
        @tensor_names = []
      end

      def parse
        magic = read_bytes(4)
        unless magic == "GGUF"
          raise ParseError, "#{@path}: not a GGUF file (magic=#{magic.inspect})"
        end
        @version = read_u32
        unless @version == 3
          raise ParseError, "#{@path}: unsupported GGUF version #{@version} (only 3)"
        end
        @n_tensors = read_u64
        n_kv = read_u64

        n_kv.times do
          key = read_gguf_string
          @kv[key] = read_value
        end

        @n_tensors.times do
          name = read_gguf_string
          @tensor_names << name
          n_dims = read_u32
          n_dims.times { read_u64 }  # dims
          read_u32                   # ggml type
          read_u64                   # offset
        end

        self
      end

      def tensor?(name)
        @tensor_names.include?(name)
      end

      # --- value readers -------------------------------------------------

      private

      def read_value
        type = read_u32
        read_typed(type)
      end

      def read_typed(type)
        case type
        when T_UINT8   then read_bytes(1).unpack1("C")
        when T_INT8    then read_bytes(1).unpack1("c")
        when T_UINT16  then read_bytes(2).unpack1("S<")
        when T_INT16   then read_bytes(2).unpack1("s<")
        when T_UINT32  then read_u32
        when T_INT32   then read_bytes(4).unpack1("l<")
        when T_FLOAT32 then read_bytes(4).unpack1("e")
        when T_BOOL    then read_bytes(1).unpack1("C") != 0
        when T_UINT64  then read_u64
        when T_INT64   then read_bytes(8).unpack1("q<")
        when T_FLOAT64 then read_bytes(8).unpack1("E")
        when T_STRING  then read_gguf_string
        when T_ARRAY   then read_array
        else
          raise ParseError, "#{@path}: unknown GGUF value type tag #{type}"
        end
      end

      # Arrays: <elem_type:u32><count:u64><elems...>. For string arrays
      # (e.g. tokenizer.ggml.tokens, ~50k entries) we only need the
      # length, so we skip past the bytes rather than building 50k Strings.
      def read_array
        elem_type = read_u32
        count = read_u64
        if elem_type == T_STRING
          count.times { skip_gguf_string }
          { array: :string, length: count }
        else
          vals = Array.new(count) { read_typed(elem_type) }
          { array: elem_type, length: count, values: vals }
        end
      end

      def read_gguf_string
        len = read_u64
        read_bytes(len).force_encoding("UTF-8")
      end

      def skip_gguf_string
        len = read_u64
        @io.seek(len, IO::SEEK_CUR)
      end

      def read_u32
        read_bytes(4).unpack1("L<")
      end

      def read_u64
        read_bytes(8).unpack1("Q<")
      end

      def read_bytes(n)
        buf = @io.read(n)
        if buf.nil? || buf.bytesize < n
          raise ParseError, "#{@path}: unexpected EOF (wanted #{n} bytes)"
        end
        buf
      end
    end
  end
end
