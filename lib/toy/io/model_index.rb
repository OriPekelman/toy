# lib/model_index.rb — find GGUF models on disk.
#
# Toy doesn't ship its own model registry. Instead, scan the same
# places users already have models stashed (HuggingFace cache, Ollama,
# LM Studio, the project's own data/, an env-var override) and present
# what's there. Drop the binary on a machine, scan caches, serve.
#
# Use case: an inference daemon's startup banner can call
# ModelIndex.scan and list what it found. A user app picks a model by
# `name` or `path`. Everything else (loading, decoding) is unchanged.
#
# Library-first design — not a CLI. For the user-facing model listing,
# use the `toy list` command (lib/toy/core/cli/list.rb, a pure-MRI port);
# this Spinel library is the smallest useful in-binary integration.
#
# What we recognise today:
#   *.gguf files (any directory).
#
# Filesystem walking goes through tnn_list_ggufs (a small C shim
# in tinynn/tinynn_gguf.c) because Spinel's stdlib doesn't have
# Dir.entries / opendir. That keeps the library Spinel-compilable into
# a single binary (the link edge survives via examples/01_inference_metal.rb).

require_relative "../../transformer"   # Mat is referenced via tinynn FFI bindings
require_relative "../../arch"

class ModelEntry
  attr_reader :name      # friendly name shown to the user
  attr_reader :path      # absolute path to the GGUF
  attr_reader :family    # :qwen2 / :llama / etc. (from Arch.from_gguf)
  attr_reader :n_params  # approx parameter count
  attr_reader :size_b    # file size in bytes
  attr_reader :source    # "hf", "ollama", "lmstudio", "local"

  def initialize(name, path, family, n_params, size_b, source)
    @name = name
    @path = path
    @family = family
    @n_params = n_params
    @size_b = size_b
    @source = source
  end

  def params_summary
    if @n_params >= 1_000_000_000
      ((@n_params.to_f / 1_000_000_000.0 * 10.0).round.to_f / 10.0).to_s + "B"
    elsif @n_params >= 1_000_000
      (@n_params / 1_000_000).to_s + "M"
    else
      @n_params.to_s
    end
  end

  def size_summary
    if @size_b >= 1_000_000_000
      ((@size_b.to_f / 1_000_000_000.0 * 10.0).round.to_f / 10.0).to_s + " GB"
    else
      (@size_b / 1_000_000).to_s + " MB"
    end
  end
end

class ModelIndex
  # The search-path order matters for first-found dedup. Project-local
  # paths first, then standard caches.
  def self.default_sources
    home = ENV["HOME"] || "/"
    paths = [""]; paths.pop   # seed-then-pop to type-pin as String[]
    env = ENV["TOY_MODEL_DIR"]
    if env != nil && env.length > 0; paths.push(env); end
    paths.push("./data")
    paths.push("./models")
    paths.push(home + "/.cache/huggingface/hub")
    paths.push(home + "/.ollama/models")
    paths.push(home + "/.lmstudio/models")
    paths.push(home + "/models")
    paths
  end

  # Walk a directory tree and collect .gguf paths. Delegates to the
  # C-side tnn_list_ggufs (tinynn/tinynn_gguf.c) because Spinel's
  # stdlib doesn't ship Dir.entries / opendir wrappers. Returns
  # absolute paths.
  def self.find_ggufs(root)
    out = [""]; out.pop   # String[] type-pin
    return out if root == nil || root.length == 0
    blob = TinyNN.tnn_list_ggufs(root)
    return out if blob == nil
    return out if blob.length == 0
    lines = blob.split("\n")
    li = 0
    while li < lines.length
      ln = lines[li]
      out.push(ln) if ln.length > 0
      li = li + 1
    end
    out
  end

  # Friendly name + source-kind from an absolute path. Each cache
  # has its own convention; normalize so the output is readable.
  # Returns [source_kind_string, friendly_name_string].
  #
  # `p = "" + path` is a type-pin: this self-method's `path` parameter
  # has no callsite Spinel can use to deduce String, so it defaults
  # to int. The concat against a String literal pins it. (Spinel
  # commits b876243 and fe91c01 made the in-method String ops below
  # work idiomatically; the param-inference workaround is what's left.)
  def self.classify_path(path)
    home = ENV["HOME"] || "/"
    p = "" + path
    slash = p.rindex("/")
    bn = slash == nil ? p : p[(slash + 1)..(p.length - 1)]
    bn_no_gguf = bn.end_with?(".gguf") ? bn[0..(bn.length - 6)] : bn

    if p.start_with?(home + "/.cache/huggingface/hub/")
      ["hf", bn_no_gguf]
    elsif p.start_with?(home + "/.ollama/models/")
      # blobs/sha256-<hash> — no friendly name without manifest crawl.
      ["ollama", bn]
    elsif p.start_with?(home + "/.lmstudio/models/")
      ["lmstudio", bn_no_gguf]
    else
      ["local", bn_no_gguf]
    end
  end

  # Estimate parameter count from an Arch. Counts the dominant tensors:
  # embeddings + N × (attention + FFN) + final norm + unembed.
  # Good to ~5% for standard transformer shapes — close enough for a
  # banner.
  def self.estimate_params(arch)
    v = arch.vocab_size
    d = arch.d_model
    l = arch.n_layers
    ff = arch.d_ff
    nq = arch.n_heads_q
    nkv = arch.n_heads_kv
    dh = arch.d_head
    embed = v * d
    attn_per_layer = (d * nq * dh) + (d * nkv * dh) * 2 + (nq * dh * d)
    ffn_per_layer = 3 * d * ff   # gate + up + down for SwiGLU
    untied = arch.untied_lm_head ? (v * d) : 0
    embed + l * (attn_per_layer + ffn_per_layer) + untied
  end

  # Scan a list of source dirs and return ModelEntry[]. Skips paths
  # that don't open as GGUF. De-dup by absolute path; first-found wins.
  # Paths returned by tnn_list_ggufs are already absolute (they're
  # built from absolute roots inside the C shim), so no expand_path.
  def self.scan_sources(sources)
    seen = {}
    out = [ModelEntry.new("", "", :unknown, 0, 0, "")]; out.pop
    si = 0
    while si < sources.length
      src = sources[si]
      ggufs = find_ggufs(src)
      gi = 0
      while gi < ggufs.length
        path = ggufs[gi]
        if !seen.has_key?(path)
          arch = Arch.from_gguf(path)
          if arch != nil
            # Arch.from_gguf reads `llama.*` keys (our converter's
            # convention). GGUFs from other tooling (gpt2, bert, ...)
            # leave those at -1 and the param math wraps to nonsense.
            # Skip + warn rather than emit garbage; "bail loud".
            if arch.vocab_size > 0 && arch.d_model > 0 && arch.n_layers > 0
              src_kind, name = classify_path(path)
              params = estimate_params(arch)
              size = TinyNN.tnn_file_size(path)
              out.push(ModelEntry.new(name, path, arch.family, params, size, src_kind))
            else
              puts "model_index: skipping " + path +
                " (missing llama.* metadata — non-llama-family GGUF?)"
            end
            seen[path] = true
          end
        end
        gi = gi + 1
      end
      si = si + 1
    end
    out
  end


  # Cheap human-readable index dump. Drop-in for a daemon's startup
  # banner.
  def self.print_summary(entries)
    if entries.length == 0
      puts "No GGUF models found. Set TOY_MODEL_DIR to a directory containing them, or"
      puts "download via huggingface-cli / ollama pull / similar."
      return
    end
    puts "Found " + entries.length.to_s + " model(s):"
    i = 0
    while i < entries.length
      e = entries[i]
      puts "  [" + e.source + "] " + e.name + "  " +
           e.family.to_s + " · " + e.params_summary + " · " + e.size_summary +
           "\n    " + e.path
      i = i + 1
    end
  end
end
