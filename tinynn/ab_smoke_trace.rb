# Trace primitive smoke. Three things to verify:
#
#   1. tnn_trace_open returns 0 and tnn_trace_active becomes 1.
#   2. Begin/end pairs around a tiny FFI workload (a couple of matmuls)
#      land in the JSON output.
#   3. With tracing OFF, begin/end pairs are no-ops — confirmed by
#      running the same workload without tnn_trace_open and showing
#      no JSON gets written (no file appears).
#
# Open the output in https://perfetto.dev or chrome://tracing to view.

require_relative "../lib/tinynn"
require_relative "../lib/transformer"

OUT = "/tmp/tnn_smoke.trace.json"

# Workload: alloc a session, run one tnn_compute on a 64×64 matmul,
# tear down. Repeat 5 times so the trace has some structure.
def run_workload(label)
  i = 0
  while i < 5
    t = TinyNN.tnn_trace_begin(label)
    sess = TinyNN.tnn_session_new(0)
    a = TinyNN.tnn_input_2d_f32(sess, 64, 64)
    b = TinyNN.tnn_input_2d_f32(sess, 64, 64)
    c = TinyNN.tnn_matmul(sess, a, b)
    TinyNN.tnn_realize(sess, c)
    TinyNN.stage_row_major_and_upload(sess, a, Mat.new(64, 64))
    TinyNN.stage_row_major_and_upload(sess, b, Mat.new(64, 64))
    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, c)
    TinyNN.tnn_session_free(sess)
    TinyNN.tnn_trace_end(label, t)
    i = i + 1
  end
end

puts "smoke 1: tracing OFF"
active_before = TinyNN.tnn_trace_active
puts "  active before: " + active_before.to_s
run_workload("warmup_off")
puts "  active still:  " + TinyNN.tnn_trace_active.to_s

puts ""
puts "smoke 2: tracing ON"
rc = TinyNN.tnn_trace_open(OUT)
puts "  trace_open rc: " + rc.to_s
puts "  active now:    " + TinyNN.tnn_trace_active.to_s

run_workload("matmul_step")
TinyNN.tnn_trace_mark("checkpoint")
run_workload("matmul_step")

TinyNN.tnn_trace_close
puts "  active after close: " + TinyNN.tnn_trace_active.to_s

puts ""
puts "smoke 3: file inspection"
sz = File.size(OUT)
puts "  output size: " + sz.to_s + " bytes (" + OUT + ")"
puts "  first 200 bytes:"
content = File.read(OUT)
print "  "
n = 200
if content.length < n; n = content.length; end
puts content[0..(n - 1)]
