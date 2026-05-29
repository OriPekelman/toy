# lib/toy/core/model_scan.rb — find + classify GGUF models on disk.
#
# CRuby reimplementation of lib/model_index.rb's scan logic. The Spinel
# library (model_index.rb) delegates dir-walking to a C shim
# (tnn_list_ggufs) and reads arch metadata via the ggml FFI — neither is
# reachable under MRI. This port keeps the SAME source-path order,
# classify_path conventions, and estimate_params arithmetic, but uses
# Dir.glob (which follows HF-cache symlinks natively) and the pure-Ruby
# Toy::Core::GGUFMeta header reader.
#
# Behavior delta vs the old binary (flagged in the P3 UX contract):
# this lists ALL *.gguf, degrading non-llama / unparseable files to
# family=:unknown, params=0 — rather than silently dropping them.

require_relative "gguf_meta"

module Toy
  module Core
    # One discovered model. Idiomatic struct (CRuby shell; no Spinel
    # type-pin restrictions apply here).
    ModelEntry = Struct.new(
      :name, :path, :family, :n_params, :size_b, :source,
      keyword_init: true
    ) do
      def params_summary
        if n_params >= 1_000_000_000
          format("%.1fB", n_params.to_f / 1_000_000_000.0)
        elsif n_params >= 1_000_000
          "#{n_params / 1_000_000}M"
        else
          n_params.to_s
        end
      end

      def size_summary
        if size_b >= 1_000_000_000
          format("%.1f GB", size_b.to_f / 1_000_000_000.0)
        else
          "#{size_b / 1_000_000} MB"
        end
      end
    end

    module ModelScan
      module_function

      # Search-path order matters for first-found dedup. Project-local
      # paths first, then standard caches. Ported verbatim from
      # ModelIndex.default_sources (model_index.rb:65-77).
      def default_sources
        home = ENV["HOME"] || "/"
        paths = []
        env = ENV["TOY_MODEL_DIR"]
        paths << env if env && !env.empty?
        paths << "./data"
        paths << "./models"
        paths << File.join(home, ".cache/huggingface/hub")
        paths << File.join(home, ".ollama/models")
        paths << File.join(home, ".lmstudio/models")
        paths << File.join(home, "models")
        paths
      end

      # Walk a directory tree for *.gguf files. Replaces the C shim
      # tnn_list_ggufs with Dir.glob, which follows symlinks (HF cache
      # stores blobs + symlinked snapshots). Returns absolute paths.
      def find_ggufs(root)
        return [] if root.nil? || root.empty?
        return [] unless File.directory?(root)
        Dir.glob(File.join(root, "**", "*.gguf"))
           .select { |f| File.file?(f) }
           .map { |f| File.expand_path(f) }
      end

      # Friendly name + source-kind from an absolute path. Ported from
      # ModelIndex.classify_path (model_index.rb:108-125).
      # Returns [source_kind, friendly_name].
      def classify_path(path)
        home = ENV["HOME"] || "/"
        bn = File.basename(path)
        bn_no_gguf = bn.end_with?(".gguf") ? bn[0...-5] : bn

        if path.start_with?(File.join(home, ".cache/huggingface/hub") + "/")
          ["hf", bn_no_gguf]
        elsif path.start_with?(File.join(home, ".ollama/models") + "/")
          ["ollama", bn]
        elsif path.start_with?(File.join(home, ".lmstudio/models") + "/")
          ["lmstudio", bn_no_gguf]
        else
          ["local", bn_no_gguf]
        end
      end

      # Read arch dims + family from GGUF metadata via the pure-Ruby
      # header reader. Mirrors the key list + family logic of
      # Arch.from_gguf (arch.rb:159-218). Returns a Hash, or nil if the
      # file can't be parsed as a llama-family GGUF.
      def read_arch(meta)
        prefix = arch_prefix(meta)
        return nil if prefix.nil?

        kv = meta.kv
        vocab = kv["#{prefix}.vocab_size"]
        vocab = array_length(kv["tokenizer.ggml.tokens"]) if vocab.nil?
        d_model  = kv["#{prefix}.embedding_length"]
        d_ff     = kv["#{prefix}.feed_forward_length"]
        n_q      = kv["#{prefix}.attention.head_count"]
        n_kv     = kv["#{prefix}.attention.head_count_kv"] || n_q
        n_layers = kv["#{prefix}.block_count"]

        return nil if vocab.nil? || d_model.nil? || n_layers.nil? || n_q.nil?
        return nil if vocab <= 0 || d_model <= 0 || n_layers <= 0 || n_q <= 0

        # Family detection (arch.rb:214-218): general.architecture is
        # unreliable (converter writes "llama" for all), so use tensor
        # presence. QKV bias → qwen2, else llama. Coarse by design.
        has_qkv_bias = meta.tensor?("blk.0.attn_q.bias") ||
                       meta.tensor?("blk.0.attn_q.head_0.bias")
        family = has_qkv_bias ? :qwen2 : :llama

        {
          family:   family,
          vocab:    vocab,
          d_model:  d_model,
          d_ff:     d_ff,
          n_q:      n_q,
          n_kv:     n_kv,
          n_layers: n_layers,
          d_head:   d_model / n_q,
          untied:   meta.tensor?("output.weight"),
          ctx:      kv["#{prefix}.context_length"] || 8192,
          rope_base: kv["#{prefix}.rope.freq_base"],
          rms_eps:   kv["#{prefix}.attention.layer_norm_rms_epsilon"],
          moe:       meta.tensor?("blk.0.ffn_gate_inp.weight"),
          n_experts:      kv["#{prefix}.expert_count"] || 0,
          n_experts_used: kv["#{prefix}.expert_used_count"] || 0,
          arch_prefix: prefix
        }
      end

      # Probe the arch-prefix the converter used (arch.rb:159-166).
      def arch_prefix(meta)
        %w[llama olmoe gemma2 qwen2 qwen3].each do |p|
          return p if meta.kv.key?("#{p}.embedding_length")
        end
        # fall back to general.architecture if it names a prefix we see
        ga = meta.kv["general.architecture"]
        return ga if ga.is_a?(String) && meta.kv.key?("#{ga}.embedding_length")
        nil
      end

      def array_length(v)
        v.is_a?(Hash) ? v[:length] : nil
      end

      # Estimate parameter count from arch dims. Ported verbatim from
      # ModelIndex.estimate_params (model_index.rb:131-144).
      def estimate_params(a)
        v = a[:vocab]; d = a[:d_model]; l = a[:n_layers]
        ff = a[:d_ff] || 0
        nq = a[:n_q]; nkv = a[:n_kv]; dh = a[:d_head]
        embed = v * d
        attn_per_layer = (d * nq * dh) + (d * nkv * dh) * 2 + (nq * dh * d)
        ffn_per_layer = 3 * d * ff
        untied = a[:untied] ? (v * d) : 0
        embed + l * (attn_per_layer + ffn_per_layer) + untied
      end

      # Scan source dirs → [ModelEntry]. De-dup by absolute path,
      # first-found wins. Unparseable/non-llama files degrade to
      # family=:unknown, params=0 (NOT dropped — the UX delta).
      def scan(sources = default_sources)
        seen = {}
        out = []
        sources.each do |src|
          find_ggufs(src).each do |path|
            next if seen[path]
            seen[path] = true
            src_kind, name = classify_path(path)
            size = File.size(path)
            entry = build_entry(path, name, src_kind, size)
            out << entry
          end
        end
        out
      end

      def build_entry(path, name, src_kind, size)
        begin
          meta = GGUFMeta.read(path)
          arch = read_arch(meta)
        rescue GGUFMeta::ParseError, SystemCallError
          arch = nil
        end
        if arch
          ModelEntry.new(name: name, path: path, family: arch[:family],
                         n_params: estimate_params(arch), size_b: size,
                         source: src_kind)
        else
          ModelEntry.new(name: name, path: path, family: :unknown,
                         n_params: 0, size_b: size, source: src_kind)
        end
      end
    end
  end
end
