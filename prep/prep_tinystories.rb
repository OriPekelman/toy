# Prep step (CRuby): download a HuggingFace dataset, tokenize it with the
# French/English tokenizer, and write three flat-text files that the
# Spinel-compiled training program can read without any HTTP / Unicode /
# regex machinery of its own:
#
#   data/ts_vocab.txt    — one word per line; line index = token id
#   data/ts_seqs.txt     — one sequence per line, space-separated token IDs
#   data/ts_prompt.txt   — the seed prompt's token IDs, space-separated
#
# Usage:
#   ruby prep_tinystories.rb \
#     --repo roneneldan/TinyStories \
#     --file TinyStoriesV2-GPT4-valid.txt \
#     --max_lines 500 \
#     --context_length 64 \
#     --prompt "Once upon a time"
#
# FROZEN-VOCAB LARGE PACK (toy#123, the F6 full-horizon corpus):
#   ruby prep/prep_tinystories.rb --frozen_vocab data/ts_vocab.txt \
#       --out_bin data/ts_seqs_large.bin --max_lines 0
# Reuses the PINNED vocab (line index = token id — the trainers' 627
# shape contract); streams EVERY line of the source file; OOV words are
# DROPPED (not UNK'd: the vocab has no spare slot, and dropping whole
# lines retains almost nothing at 627 words). Emits a flat packed-i32
# token stream the --corpus flag reads at arbitrary offsets, plus
# retention stats. Nothing about the classic 3-file mode changes.
require "optparse"
require "fileutils"
require_relative "dataset_loader"
require_relative "tokenizer"

opts = {
  repo:           "roneneldan/TinyStories",
  file:           "TinyStoriesV2-GPT4-valid.txt",
  max_lines:      500,
  context_length: 64,
  prompt:         "Once upon a time",
  out_dir:        "data",
}
OptionParser.new do |o|
  o.on("--repo R")               { |v| opts[:repo] = v }
  o.on("--file F")               { |v| opts[:file] = v }
  o.on("--max_lines N", Integer) { |v| opts[:max_lines] = v }
  o.on("--context_length N", Integer) { |v| opts[:context_length] = v }
  o.on("--prompt P")             { |v| opts[:prompt] = v }
  o.on("--out_dir D")            { |v| opts[:out_dir] = v }
  o.on("--frozen_vocab P")       { |v| opts[:frozen_vocab] = v }
  o.on("--out_bin P")            { |v| opts[:out_bin] = v }
end.parse!

if opts[:frozen_vocab]
  vocab_words = File.readlines(opts[:frozen_vocab]).map(&:chomp)
  word_to_index = vocab_words.each_with_index.to_h
  abort "frozen vocab is empty: #{opts[:frozen_vocab]}" if vocab_words.empty?
  out_bin = opts[:out_bin] || File.join(opts[:out_dir], "ts_seqs_large.bin")
  n = opts[:max_lines]
  puts "Frozen-vocab pack: #{vocab_words.size} words from #{opts[:frozen_vocab]}"
  puts "Loading #{opts[:repo]} / #{opts[:file]} (#{n == 0 ? 'ALL' : "first #{n}"} lines)…"
  lines = n == 0 ? DatasetLoader.lines(opts[:repo], opts[:file]) :
                   DatasetLoader.head(opts[:repo], opts[:file], n)
  ids = []
  total_words = 0
  lines.each do |l|
    toks = Tokenizer.tokenize_french(l)
    total_words += toks.size
    toks.each do |w|
      id = word_to_index[w]
      ids << id if id
    end
  end
  File.open(out_bin, "wb") { |f| f.write(ids.pack("l<*")) }
  max_id = ids.max
  puts "Wrote #{out_bin}: #{ids.size} tokens (#{(ids.size * 100.0 / total_words).round(1)}% of #{total_words} words retained; max id #{max_id}, vocab #{vocab_words.size})"
  abort "PACK INVALID: max id #{max_id} >= vocab #{vocab_words.size}" if max_id >= vocab_words.size
  exit 0
end

FileUtils.mkdir_p(opts[:out_dir])

puts "Loading #{opts[:repo]} / #{opts[:file]} (first #{opts[:max_lines]} lines)…"
lines = DatasetLoader.head(opts[:repo], opts[:file], opts[:max_lines])
puts "  #{lines.size} non-empty lines"

# Build a single shared vocabulary from the corpus + the prompt so any
# prompt token has an ID in the trained model.
tokenized      = lines.map { |l| Tokenizer.tokenize_french(l) }
prompt_tokens  = Tokenizer.tokenize_french(opts[:prompt])
vocab_words    = (tokenized.flatten + prompt_tokens).uniq
word_to_index  = vocab_words.each_with_index.to_h
puts "  vocab size: #{vocab_words.size}"

# Tokenize each line into IDs, then chunk over-long lines into windows
# of at most context_length tokens (same trick we use in CRuby).
sequences = tokenized.flat_map do |toks|
  ids = toks.map { |w| word_to_index[w] }.compact
  ids.each_slice(opts[:context_length]).to_a
end.reject { |s| s.size < 2 }
puts "  sequences (after chunking): #{sequences.size}"

# Write the three files.
File.write(
  File.join(opts[:out_dir], "ts_vocab.txt"),
  vocab_words.join("\n") + "\n"
)
File.open(File.join(opts[:out_dir], "ts_seqs.txt"), "w") do |f|
  sequences.each { |seq| f.puts seq.join(" ") }
end
prompt_ids = prompt_tokens.map { |w| word_to_index[w] }.compact
File.write(
  File.join(opts[:out_dir], "ts_prompt.txt"),
  prompt_ids.join(" ") + "\n"
)

puts "Wrote #{opts[:out_dir]}/ts_vocab.txt, ts_seqs.txt, ts_prompt.txt"
puts "Prompt token IDs: #{prompt_ids.inspect}"
