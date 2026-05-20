# Smoke test for lib/arch.rb. Builds Arch instances from three GGUFs
# and prints summaries; verifies the factory does what the docs say.

require_relative "../lib/transformer"
require_relative "../lib/arch"

paths = ["data/qwen25-1.5b-native.gguf",
         "data/llama-3.2-1b-f32.gguf",
         "data/smollm2-135m-f32.gguf"]

i = 0
while i < paths.length
  p = paths[i]
  puts "=== " + p + " ==="
  a = Arch.from_gguf(p)
  if a == nil
    puts "  (failed)"
  else
    puts "  " + a.summary
    puts "  family=" + a.family.to_s +
         " untied_lm=" + a.untied_lm_head.to_s +
         " tokenizer=" + a.tokenizer_kind.to_s +
         " gqa?=" + a.gqa?.to_s
  end
  puts ""
  i = i + 1
end
