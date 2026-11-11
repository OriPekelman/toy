# prep/smokes/smoke_tokenize.rb — tokenizer-only probe (toy#208).
#
# Loads ONLY the tokenizer metadata of a GGUF (no weights, no graph) and
# prints the ids for a string, so a gate can compare them against
# HuggingFace reference ids on a multi-GB model in well under a second.
#
#   GGUF=model.gguf TEXT='<|im_start|>user' prep/smokes/smoke_tokenize
#   PLAIN=1 …    # encode_plain: no special-token parsing
#
# Output (two lines, both always printed):
#   specials: <count of CONTROL/USER_DEFINED tokens registered>
#   ids: <space-separated ids>

require_relative "../../lib/toy/io/tokenizer"

GGUF  = ENV["GGUF"] || ""
TEXT  = ENV["TEXT"] || ""
PLAIN = ENV["PLAIN"] || ""

tok = Tokenizer.from_gguf(GGUF)
if !tok.present
  puts "smoke_tokenize: no embedded tokenizer in " + GGUF
  exit 1
end
ids = [0]
ids.pop
if PLAIN == "1"
  ids = tok.encode_plain(TEXT)
else
  ids = tok.encode(TEXT)
end
line = "ids:"
i = 0
while i < ids.length
  line = line + " " + ids[i].to_s
  i = i + 1
end
puts "specials: " + tok.n_special.to_s
puts line
