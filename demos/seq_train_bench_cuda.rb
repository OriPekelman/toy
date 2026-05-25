require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/llama_seq_forward_ffi_cuda"

# CUDA sequence-mode training bench (LoRA-Q or full-FT).
#
# Env knobs:
#   GGUF       = model path (default smollm2-135m-native.gguf)
#   MODE       = "lora" (default) | "ft"
#   STEPS      = total iterations (default 10; first is dropped as warmup)
#   SEQ_LEN    = sequence length T (default 4). Heavy bench uses 256.
#   BENCH_TAG  = non-empty → emit `BENCH <tag>_<metric> <value>` lines that
#                bench/check_heavy.rb can parse.
#   TRACE      = path to Chrome Trace JSON (opened AFTER step 1 to skip the
#                graph-compile / allocator-warmup transient, then closed
#                after a small captured window so the file stays bite-sized).
#   TRACE_OPS  = "1" to also capture per-ggml-op durations. CUDA per-op
#                numbers reflect enqueue latency, not kernel wallclock —
#                relative attribution is still useful.
#   TRACE_FROM = first step (1-indexed) to include in the trace (default 3,
#                so first 2 are warmup; opening after step 1's compile is
#                what matters most).
#   TRACE_FOR  = number of steps to trace (default 2). Keeps the JSON small.
#   SKIP_LABELS_UPLOAD = "1" → upload t_labels ONCE before the timed
#                loop (rather than per step). For attribution only: real
#                training varies labels per step. Reveals the upper
#                bound of "what if labels upload cost was zero?"
#
# Output:
#   Always: human-readable mean step time excluding warmup.
#   When tagged: `BENCH <tag>_step_ms_{mean,p95,stddev}` lines.

GGUF    = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
MODE    = ENV["MODE"]    || "lora"  # lora | ft
STEPS   = (ENV["STEPS"]  || "10").to_i
SEQ_LEN = (ENV["SEQ_LEN"] || "4").to_i
BENCH_TAG = ENV["BENCH_TAG"] || ""
TRACE     = ENV["TRACE"]      || ""
TRACE_OPS = ENV["TRACE_OPS"]  || ""
TRACE_FROM = (ENV["TRACE_FROM"] || "3").to_i
TRACE_FOR  = (ENV["TRACE_FOR"]  || "2").to_i
SKIP_LABELS_UPLOAD = ENV["SKIP_LABELS_UPLOAD"] == "1"

cfg = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)

# Synthesise deterministic but spread-out token IDs and positions of
# length SEQ_LEN. Token-content has no effect on per-step wallclock at
# fixed shapes; we just need IDs inside the vocab.
TOKENS = [0]; TOKENS.pop
positions = [0]; positions.pop
i = 0
while i < SEQ_LEN
  tok = (i * 1103515245 + 12345) & 0x3FFF        # stays inside vocab
  TOKENS.push(tok)
  positions.push(i)
  i = i + 1
end

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = LlamaSeqForwardFFICacheCuda.new
if MODE == "ft"
  seq.enable_full_finetune!
  seq.realize_for_full_finetune(gguf, cfg, SEQ_LEN, flags.untied, flags.qkv_bias)
else
  seq.enable_lora_q!(8)
  seq.enable_lora_q_adamw!
  seq.realize_for_mmap(gguf, cfg, SEQ_LEN, flags.untied, flags.qkv_bias)
  seq.upload_lora_q_init!(42, 0.01)
end

result = seq.build_training_step
t_loss, t_labels, t_hp = result[0], result[1], result[2]

m_labels = Mat.new(SEQ_LEN, cfg.vocab)
i = 0; while i < SEQ_LEN * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
i = 0; while i < SEQ_LEN; m_labels.flat[i * cfg.vocab + 99] = 1.0; i = i + 1; end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = 0.001; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0

if SKIP_LABELS_UPLOAD
  TinyNNCuda.upload_row_major(seq.sess, t_labels, m_labels)
  puts "t_labels uploaded once before timed loop (SKIP_LABELS_UPLOAD=1)"
end

times = [0.0]; times.pop
trace_opened = false
trace_first = TRACE_FROM
trace_last  = TRACE_FROM + TRACE_FOR - 1
step = 1
while step <= STEPS
  # Open the trace just before the first step we want to capture, after
  # graph-compile/allocator transients have settled into steady state.
  if TRACE.length > 0 && !trace_opened && step == trace_first
    rc = TinyNNCuda.tnn_trace_open(TRACE)
    if rc != 0
      puts "trace_open failed: rc=" + rc.to_s + " (TRACE=" + TRACE + ")"
    else
      puts "tracing to " + TRACE + " (steps " + trace_first.to_s + ".." + trace_last.to_s + ")"
      if TRACE_OPS == "1"
        TinyNNCuda.tnn_trace_set_op_capture(1)
        puts "per-op trace enabled (TRACE_OPS=1)"
      end
      trace_opened = true
    end
  end

  m_hp.flat[5] = 1.0 / (1.0 - (0.9 ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNNCuda.tnn_graph_reset(seq.sess)
  else
    TinyNNCuda.tnn_graph_reset_grads_only(seq.sess)
  end
  _t_step = trace_opened ? TinyNNCuda.tnn_trace_begin("step") : 0
  t0 = Time.now
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_token_ids, TOKENS)
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_positions, positions)
  if !SKIP_LABELS_UPLOAD
    TinyNNCuda.upload_row_major(seq.sess, t_labels, m_labels)
  end
  TinyNNCuda.upload_row_major(seq.sess, t_hp, m_hp)
  TinyNNCuda.tnn_compute_backward(seq.sess)
  ms = (Time.now - t0) * 1000.0
  if trace_opened
    TinyNNCuda.tnn_trace_end("step", _t_step)
  end
  times.push(ms)

  # Close the trace after we've captured the requested window.
  if trace_opened && step == trace_last
    TinyNNCuda.tnn_trace_close
    puts "trace closed: " + TRACE
    trace_opened = false
  end
  step = step + 1
end

# Drop first step (compile warmup), then stats over the rest.
samples = [0.0]; samples.pop
i = 1
while i < times.length; samples.push(times[i]); i = i + 1; end
n = samples.length

# Mean
sum = 0.0
i = 0; while i < n; sum = sum + samples[i]; i = i + 1; end
mean_ms = sum / n.to_f

# Stddev (population)
var = 0.0
i = 0
while i < n
  d = samples[i] - mean_ms
  var = var + d * d
  i = i + 1
end
stddev_ms = (var / n.to_f) ** 0.5

# P95: copy + insertion sort (Spinel-friendly), pick ceil(0.95 * (n-1))
sorted = [0.0]; sorted.pop
i = 0; while i < n; sorted.push(samples[i]); i = i + 1; end
i = 1
while i < n
  v = sorted[i]; j = i - 1
  while j >= 0 && sorted[j] > v
    sorted[j + 1] = sorted[j]
    j = j - 1
  end
  sorted[j + 1] = v
  i = i + 1
end
p95_idx = ((n - 1).to_f * 0.95).to_i
p95_ms = sorted[p95_idx]

puts MODE.upcase + " step time (excl warmup): mean=" + mean_ms.to_s +
     " ms  p95=" + p95_ms.to_s + " ms  stddev=" + stddev_ms.to_s +
     " ms  over " + n.to_s + " steps  (SEQ_LEN=" + SEQ_LEN.to_s + ")"

if BENCH_TAG.length > 0
  puts "BENCH " + BENCH_TAG + "_step_ms_mean "   + mean_ms.to_s
  puts "BENCH " + BENCH_TAG + "_step_ms_p95 "    + p95_ms.to_s
  puts "BENCH " + BENCH_TAG + "_step_ms_stddev " + stddev_ms.to_s
end
