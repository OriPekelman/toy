# T1.2 diagnostic: prove or rebut the theory that SmolLM2 BPE
# produces merged pieces NOT present in the vocab. Llama-3 and Qwen
# pass the same prompts with the same Ruby encoder, so the bug is
# data-driven (SmolLM2-specific). Two candidate causes:
#
#   A. Merge dict has entries whose output isn't a vocab token.
#   B. Pre-tokenizer regex feeds the wrong chunks to BPE for SmolLM2.
#
# This smoke isolates the question by reading the raw vocab + merges
# arrays from each GGUF and asking, for the *characters that appear
# in the failing prompts*, whether the corresponding merges produce
# in-vocab strings.

require_relative "../lib/toy/io/gguf_kv"

# Pull vocab + merges directly. Don't go through Tokenizer; we want
# the raw arrays so we can search them.
def load_vocab_and_merges(path)
  handle = GgufKV.tnn_gguf_load(path)
  if handle == nil
    puts "load: failed to open " + path
    empty = [""]; empty.pop
    return [empty, empty]
  end
  n_tok    = GgufKV.tnn_gguf_arr_n(handle, "tokenizer.ggml.tokens")
  n_merges = GgufKV.tnn_gguf_arr_n(handle, "tokenizer.ggml.merges")
  vocab  = [""]; vocab.pop
  merges = [""]; merges.pop
  i = 0
  while i < n_tok
    s = GgufKV.tnn_gguf_arr_str(handle, "tokenizer.ggml.tokens", i)
    if s == nil; vocab.push(""); else; vocab.push(s); end
    i = i + 1
  end
  i = 0
  while i < n_merges
    s = GgufKV.tnn_gguf_arr_str(handle, "tokenizer.ggml.merges", i)
    if s == nil; merges.push(""); else; merges.push(s); end
    i = i + 1
  end
  GgufKV.tnn_gguf_free(handle)
  [vocab, merges]
end

# linear lookup (slow but our test inputs are tiny)
def vocab_has?(vocab, s)
  i = 0
  while i < vocab.length
    if vocab[i] == s; return true; end
    i = i + 1
  end
  false
end

def merges_has?(merges, key)
  i = 0
  while i < merges.length
    if merges[i] == key; return true; end
    i = i + 1
  end
  false
end

# The failing prompt-fragments and the byte-mapped pieces they
# decompose to. "Ċ" is the GPT-2 byte-map char for \n (0x0A → 256).
# These are the literal piece-strings BPE would see internally.
TARGETS = [
  ["?\n",   ["?", "Ċ"],   "? Ċ",   "?Ċ"],
  [".txt",  [".", "t", "x", "t"], ". t",  ".t"],
  [".txt-2", [".", "t", "x", "t"], "t x",  "tx"],
  [".txt-3", [".", "t", "x", "t"], "tx t", "txt"],
  ["I/O",   ["I", "/", "O"], "/ O",  "/O"],
]

MODELS = ["data/smollm2-135m-tok-new.gguf",
          "data/llama-3.2-1b-tok.gguf",
          "data/qwen25-0.5b-tok.gguf"]

mi = 0
while mi < MODELS.length
  path = MODELS[mi]
  res = load_vocab_and_merges(path)
  vocab  = res[0]
  merges = res[1]
  puts "===================================================================="
  puts "model: " + path
  puts "  vocab.length=" + vocab.length.to_s + "  merges.length=" + merges.length.to_s
  ti = 0
  while ti < TARGETS.length
    tag       = TARGETS[ti][0]
    merge_key = TARGETS[ti][2]
    out_piece = TARGETS[ti][3]
    has_merge = merges_has?(merges, merge_key)
    has_piece = vocab_has?(vocab, out_piece)
    ok = "OK "
    if has_merge && !has_piece
      ok = "BUG"   # merge fires but produces a non-vocab token
    end
    if !has_merge
      ok = "no-merge"
    end
    puts "  " + ok + "  prompt=" + tag.inspect +
         "  merge=" + merge_key.inspect + " (" + (has_merge ? "yes" : "no") + ")" +
         "  result=" + out_piece.inspect + " in vocab? " + (has_piece ? "yes" : "no")
    ti = ti + 1
  end
  mi = mi + 1
end
