# prep/smokes/smoke_tokenize.rb — tokenizer-only probe (toy#208).
#
# Loads ONLY the tokenizer metadata of a GGUF (no weights, no graph) and
# prints the ids for a string, so a gate can compare them against
# HuggingFace reference ids on a multi-GB model in well under a second.
#
#   GGUF=model.gguf TEXT='<|im_start|>user' prep/smokes/smoke_tokenize
#   PLAIN=1 …    # encode_plain: no special-token parsing
#
#   TEXT_FILE=f  # one text per line (\\n = newline), one ids line each
#
# Output:
#   specials: <count of CONTROL/USER_DEFINED tokens registered>
#   pre: <pre-tokenizer kind: 0 llama-bpe, 1 qwen2, 2 smollm, 3 gpt-2>
#   ids: <space-separated ids>   (one line per text)

require_relative "../../lib/toy/io/tokenizer"

GGUF  = ENV["GGUF"] || ""
TEXT  = ENV["TEXT"] || ""
PLAIN = ENV["PLAIN"] || ""
TEXT_FILE = ENV["TEXT_FILE"] || ""

tok = Tokenizer.from_gguf(GGUF)
if !tok.present
  puts "smoke_tokenize: no embedded tokenizer in " + GGUF
  exit 1
end
def show(tok, text)
  ids = [0]
  ids.pop
  if PLAIN == "1"
    ids = tok.encode_plain(text)
  else
    ids = tok.encode(text)
  end
  line = "ids:"
  i = 0
  while i < ids.length
    line = line + " " + ids[i].to_s
    i = i + 1
  end
  puts line
end

puts "specials: " + tok.n_special.to_s
puts "pre: " + tok.pre_kind.to_s
if TEXT_FILE != ""
  # One text per line; the two characters \n inside a line are a newline.
  lines = File.read(TEXT_FILE).split("\n")
  li = 0
  while li < lines.length
    show(tok, lines[li].gsub("\\n", "\n"))
    li = li + 1
  end
else
  show(tok, TEXT)
end
