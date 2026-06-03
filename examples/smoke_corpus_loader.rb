# E2.4 / GH#14 — smoke for the streaming corpus loader + cosine LR.
#
#   uv run prep/pretokenize_corpus.py            # one-time: ts_seqs.bin
#   make examples/smoke_corpus_loader
#   ./examples/smoke_corpus_loader
#
# Reads 5 successive 32-token sequences from data/ts_seqs.bin and
# prints the first few tokens of each — should not repeat. Then
# prints cosine LR at a few representative steps to verify the
# schedule shape (warmup → cosine → tail).

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/toy/io/toy_corpus_loader"
require_relative "../lib/toy/train/toy_lr_schedule"

PATH    = ENV["CORPUS"] || "data/ts_seqs.bin"
SEQ_LEN = (ENV["SEQ_LEN"] || "32").to_i
N_SEQS  = (ENV["N_SEQS"]  || "5").to_i

puts "=== streaming reads from " + PATH + " (SEQ_LEN=" + SEQ_LEN.to_s + ") ==="
offset = 0
i = 0
while i < N_SEQS
  buf = ToyCorpusLoader.read_seq(PATH, offset, SEQ_LEN)
  preview = ""
  k = 0
  while k < 8 && k < buf.length
    preview = preview + (k == 0 ? "" : " ") + buf[k].to_s
    k = k + 1
  end
  puts "seq[" + i.to_s + "] offset=" + offset.to_s + " first 8: " + preview
  offset = offset + SEQ_LEN * 4   # i32 = 4 bytes
  i = i + 1
end

puts ""
puts "=== cosine LR schedule (n_steps=100, warmup=5, lr_max=1e-3, lr_min=1e-5) ==="
[0, 1, 4, 5, 10, 50, 90, 99].each do |s|
  lr = ToyLR.cosine(s, 100, 1.0e-3, 1.0e-5, 5)
  puts "  step " + s.to_s + "  lr=" + lr.to_s
end
