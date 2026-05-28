# Probe landmine #11 — File.open(r) block form + tnn_session_new in
# the same program. On Spinel d59926a/568cf0d this segfaulted during
# FFI init. Verify current Spinel HEAD a03bb49 fixes it.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

# Pin Mat by touching it once so Spinel resolves sp_Mat.
m = Mat.new(2, 2)
m.flat[0] = 1.0

path = "/tmp/probe_file_block_input.txt"
File.open(path, "w") do |f|
  f.write("0 1 2 3 4\n")
  f.write("5 6 7 8 9\n")
end

lines = ["?"]; lines.pop
File.open(path, "r") do |f|
  f.each_line { |l| lines.push(l.chomp) }
end

puts "read " + lines.length.to_s + " lines via block form"
puts "line 0: " + lines[0]
puts "Mat.flat[0] = " + m.flat[0].to_s

# Now FFI init AFTER the block-form file reads
sess = TinyNN.tnn_session_new(0)
t_a  = TinyNN.tnn_input_2d_f32_persistent(sess, 4, 8)
TinyNN.tnn_finalize_weights(sess)

puts "tnn session OK; tensor allocated and finalized"
puts "PROBE-PASS"
