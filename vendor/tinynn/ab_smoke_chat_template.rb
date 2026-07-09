# Smoke: render the 4 chat-template families against a fixed 3-message
# conversation. Verifies the wire, not Tokenizer integration.

require_relative "../lib/toy/train/toy_chat_template"

# msgs: Array<Array<String>> — [role, content]. Sample chat:
msgs = [["", ""]]; msgs.pop
msgs.push(["system",    "Be concise."])
msgs.push(["user",      "What is 2+2?"])
msgs.push(["assistant", "4"])
msgs.push(["user",      "And 3+3?"])

puts "=== chatml ==="
puts ToyChatTemplate.apply("chatml", msgs, true)
puts ""
puts "=== llama3 ==="
puts ToyChatTemplate.apply("llama3", msgs, true)
puts ""
puts "=== mistral ==="
puts ToyChatTemplate.apply("mistral", msgs, true)
puts ""
puts "=== gemma2 ==="
puts ToyChatTemplate.apply("gemma2", msgs, true)
puts ""
puts "=== detect_family probes ==="
puts "gemma2 arch       → " + ToyChatTemplate.detect_family("gemma2", false, false)
puts "qwen2 arch        → " + ToyChatTemplate.detect_family("qwen2",  false, false)
puts "llama + im_start  → " + ToyChatTemplate.detect_family("llama",  true,  false)
puts "llama + BoT only  → " + ToyChatTemplate.detect_family("llama",  false, true)
puts "llama bare        → " + ToyChatTemplate.detect_family("llama",  false, false)
puts "unknown arch      → " + ToyChatTemplate.detect_family("xyz",    false, false)
