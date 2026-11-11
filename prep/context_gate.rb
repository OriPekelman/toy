#!/usr/bin/env ruby
# prep/context_gate.rb — toy#207 regression gate: the KV-cache context.
#
# The context used to be a literal 256 with no way to change it, and going
# past it was a ggml view assert in the middle of prefill:
#
#   ggml.c:1755: GGML_ASSERT(view_src == NULL || data_size == 0 || ...)
#
# Now it comes from CTX= (`toy infer --ctx N`) or, unset, from the model's
# own context_length clamped to [256, 4096]; and a request that does not
# fit is refused by name before any decode.
#
# Legs (all on libexec/toy-infer, CPU, greedy):
#   unchanged  a short prompt decodes the SAME ids at the default context
#              and at CTX=256 — the default moved, the output must not
#   long       a 300-token prompt decodes at the default context (the
#              case that asserted), and matches a run at an explicit CTX
#   refused    the same prompt at CTX=256 exits 1 with the plain message,
#              and never reaches ggml
#   boundary   prompt + n == CTX is accepted; CTX one smaller is refused
#   malformed  CTX=abc is fatal, not a silent fall back to the default
#
#   ruby prep/context_gate.rb     # exit 0 pass, 1 fail, 2 unusable

require "open3"

ROOT   = File.expand_path("..", __dir__)
RUNNER = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, "libexec", "toy-infer")
SRC    = ENV["TOY_GATE_SRC"] || File.join(ROOT, "data", "smollm2-135m-native.gguf")
SHORT  = "6403 1980 253 655 28"
LONG_N = 300
LONG   = (0...LONG_N).map { |i| 100 + (i * 37) % 4000 }.join(" ")

unless File.file?(SRC)
  warn "context_gate: missing fixture #{SRC} (never skipped silently)"
  exit 2
end
unless ENV["TOY_GATE_RUNNER"]
  _o, st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-infer")
  unless st.success?
    warn "context_gate: could not build libexec/toy-infer"
    exit 2
  end
end

def infer(ids, n, ctx = nil)
  env = { "GGUF" => SRC, "PROMPT_IDS" => ids, "N_NEW" => n.to_s, "CTX" => ctx && ctx.to_s }
  Open3.capture2e(env, RUNNER)
end

def ids_line(out)
  out.lines.map(&:strip).reverse.find { |l| l.start_with?("ids:") }.to_s
end

failures = []

# ---- unchanged ----
d_out, d_st = infer(SHORT, 8)
c_out, c_st = infer(SHORT, 8, 256)
if !d_st.success? || !c_st.success? || ids_line(d_out).empty?
  failures << "unchanged: a short prompt did not decode (default #{d_st.exitstatus}, CTX=256 #{c_st.exitstatus})"
elsif ids_line(d_out) != ids_line(c_out)
  failures << "unchanged: ids depend on the context size\n      default: #{ids_line(d_out)}\n      CTX=256: #{ids_line(c_out)}"
else
  puts "  ok: unchanged — #{ids_line(d_out)} at the default context and at 256"
end

# ---- long ----
l_out, l_st = infer(LONG, 4)
e_out, e_st = infer(LONG, 4, 512)
want = 1 + LONG_N + 4
if !l_st.success?
  failures << "long: a #{LONG_N}-token prompt failed at the default context:\n      #{l_out.lines.last.to_s.strip}"
elsif ids_line(l_out).split.length != want
  failures << "long: expected #{want - 1} ids, got #{ids_line(l_out).split.length - 1}"
elsif !e_st.success? || ids_line(e_out) != ids_line(l_out)
  failures << "long: default-context ids differ from CTX=512 ids"
else
  puts "  ok: long — #{LONG_N}-token prompt decodes at the default context (…#{ids_line(l_out).split.last(4).join(' ')})"
end

# ---- refused ----
r_out, r_st = infer(LONG, 4, 256)
msg = "prompt #{LONG_N} tokens + n 4 exceeds context 256"
if r_st.success?
  failures << "refused: an over-long prompt was ACCEPTED at CTX=256"
elsif !r_out.include?(msg)
  failures << "refused: failed without the plain message (#{r_out.lines.last.to_s.strip})"
elsif r_out.include?("GGML_ASSERT")
  failures << "refused: reached a ggml assert before the check"
elsif r_st.exitstatus != 1
  failures << "refused: exit #{r_st.exitstatus.inspect}, want 1"
else
  puts "  ok: refused — \"#{msg}\", exit 1, ggml never reached"
end

# ---- boundary ----
fit_out, fit_st   = infer(LONG, 4, LONG_N + 4)
_s_out, short_st  = infer(LONG, 4, LONG_N + 3)
if !fit_st.success? || ids_line(fit_out) != ids_line(l_out)
  failures << "boundary: prompt + n == CTX should decode identically (exit #{fit_st.exitstatus.inspect})"
elsif short_st.success?
  failures << "boundary: CTX one short of prompt + n was accepted"
else
  puts "  ok: boundary — exact fit accepted, one short refused"
end

# ---- malformed ----
m_out, m_st = infer(SHORT, 2, "abc")
if m_st.success? || !m_out.include?("context must be a positive integer")
  failures << "malformed: CTX=abc was not fatal (#{m_out.lines.last.to_s.strip})"
else
  puts "  ok: malformed — CTX=abc is fatal"
end

if failures.empty?
  puts "GATE PASS [context]: context comes from --ctx or the GGUF, output is context-independent, and an overflow is a plain error before decode"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [context]: #{failures.length} leg(s)"
exit 1
