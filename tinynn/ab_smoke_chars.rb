# Pin down whether Spinel's String#chars returns codepoints or bytes
# on multi-byte UTF-8 strings. "Ċ" is U+010A = 0xC4 0x8A in UTF-8.
# MRI Ruby returns ["Ċ"] (codepoint-aware); if Spinel returns
# ["\xC4", "\x8A"] (bytewise), that's the SmolLM2 tokenizer bug.

s = "?" + [196, 138].pack("C*")   # "?Ċ" — 3 bytes (0x3F 0xC4 0x8A)
puts "bytesize=" + s.bytesize.to_s
puts "length="   + s.length.to_s

cs = s.chars
puts "chars.length=" + cs.length.to_s
i = 0
while i < cs.length
  c = cs[i]
  puts "  chars[" + i.to_s + "] bytesize=" + c.bytesize.to_s +
       " bytes=" + c.bytes.inspect
  i = i + 1
end

# Bonus: print bytes() directly, which IS byte-level.
puts "bytes=" + s.bytes.inspect
