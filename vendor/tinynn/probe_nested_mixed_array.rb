# Probe landmine #13 — Array<Array<int_or_float>> literal seeds
# faulted at startup-class-init under Spinel (toy_logprobs.rb pattern).
#
# Original symptom: `result = [[ids[0], vals[0]]]` where ids was
# Array<Int> and vals was Array<Float> segfaulted in main() BEFORE
# any user puts ran. Even `STDOUT.sync = true; puts "smoke start"`
# produced no output — the fault was in module initialization.

STDOUT.sync = true
puts "probe-nested-mixed: about to declare nested array"

ids  = [1, 2, 3]
vals = [0.5, 1.0, 1.5]
result = [[ids[0], vals[0]]]
i = 1
while i < ids.length
  result.push([ids[i], vals[i]])
  i = i + 1
end

puts "constructed " + result.length.to_s + " rows"
j = 0
while j < result.length
  row = result[j]
  puts "  row " + j.to_s + ": id=" + row[0].to_s + " val=" + row[1].to_s
  j = j + 1
end
puts "PROBE-PASS"
