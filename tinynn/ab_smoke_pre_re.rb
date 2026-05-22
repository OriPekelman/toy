# Inspect what the BPE pre-tokenizer regex actually does with the
# failing prompts. We don't go through Tokenizer.encode here — we
# just scan with the same regex used in lib/tokenizer.rb:179 and
# print each chunk.

PRE_RE = /'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D|[^\r\na-zA-Z0-9]?[a-zA-Z]+|[0-9]{1,3}| ?[^\sa-zA-Z0-9]+[\r\n]*|\s+/

SAMPLES = ["?\n", ".txt", "I/O", "?", "\n", "/", "."]

i = 0
while i < SAMPLES.length
  text = SAMPLES[i]
  chunks = text.scan(PRE_RE)
  puts text.inspect + "  → " + chunks.length.to_s + " chunks:"
  j = 0
  while j < chunks.length
    c = chunks[j]
    puts "    [" + j.to_s + "] " + c.inspect + " (" + c.bytesize.to_s + " bytes, " + c.length.to_s + " chars)"
    j = j + 1
  end
  i = i + 1
end
