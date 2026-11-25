#!/usr/bin/env ruby
# prep/serve_adapter_gate.rb — toy#218 gate: a low-rank weight delta applied
# at inference (`toy serve --adapter`, request field "lora_scale") and the
# residual stream read at a layer (POST /v1/hidden).
#
# Boots the serve runner (CPU by default, `--device cuda`) on
# data/smollm2-135m-tok.gguf (F32, embedded tokenizer). The gate writes its
# own fixtures in pure Ruby: a llama.cpp-format LoRA GGUF (all seven
# matrices of block 12 at rank 2 or 3, attn_output and ffn_down of block 20
# at rank 1, alpha 4) and a MERGED model, a copy of the fixture whose
# matrices are W + (alpha/r) B A. The merged model served with no adapter
# is the reference for the edited model.
#
#   base      with the adapter loaded, "lora_scale": 0 gives the ids,
#             log-probabilities and hidden states of a server without it
#   merged    at the default scale the ids equal the merged model's, the
#             log-probabilities and the hidden states of every layer agree
#             with it to rounding, and they are NOT the base model's
#   layers    hidden states up to block 11 are untouched by the edit, the
#             ones from block 12 on change (the edit sits where the file
#             says, and layer i is the stream after block i)
#   scale     "lora_scale": 2 equals a server whose adapter file carries
#             alpha 8
#   embed     layer 0 is the token-embedding row of the last prompt token,
#             read from the GGUF
#   head      layer L through the final norm and the output matrix (done
#             here in Ruby) gives the first generated id and its
#             log-probability
#   position  "position": p (or its negative form) reads another token than
#             the last; a subset of layers comes back in ascending order;
#             a layer outside 0..L is HTTP 400
#   mixed     on 2 slots, base and edited requests sent at once share
#             decode steps and each gives what it gives alone
#   reuse     on one slot, a base request after an edited one with the same
#             prompt reuses none of its cached rows, and a repeat at the
#             same scale reuses them
#   prefix    (toy#219) with a boot prefix (--prefix-file), the first
#             request at an adapter scale evaluates the system turn and
#             leaves it as that scale's prefix; later requests at the scale
#             read it (cached_tokens = the system turn), at two scales and
#             at scale 0 side by side, and every reply equals a server
#             without reuse (KV_REUSE=0)
#   label     (toy#221) every reply names its weights: the "toy" object of
#             completions, chat and hidden replies carries the sha256 of
#             the model file, of the adapter file ("" without one) and
#             the scale; the request event carries the scale
#   refuse    an adapter whose matrix has the wrong width, one carrying a
#             tensor that cannot be applied, and an adapter on the per-head
#             graph (KV_FUSED=0) each exit with a message
#
#   ruby prep/serve_adapter_gate.rb [--device cuda]   # exit 0 pass, 1 fail, 2 unusable

require "digest"
require "net/http"
require "json"
require "socket"
require "timeout"
require "tmpdir"
require "open3"
require "fileutils"

ROOT    = File.expand_path("..", __dir__)
DEVICE  = (i = ARGV.index("--device")) ? ARGV[i + 1].to_s : "cpu"
TARGET  = DEVICE == "cuda" ? "libexec/toy-serve-cuda" : "libexec/toy-serve"
RUNNER  = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, TARGET)
GGUF    = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
LABEL   = DEVICE == "cuda" ? "serve-adapter-cuda" : "serve-adapter"
MAX_TOKENS = 12

unless %w[cpu cuda].include?(DEVICE)
  warn "serve_adapter_gate: --device must be cpu or cuda"
  exit 2
end
unless File.file?(GGUF)
  warn "serve_adapter_gate: missing fixture #{GGUF} (never skipped silently)"
  exit 2
end
unless ENV["TOY_GATE_RUNNER"]
  out, st = Open3.capture2e("make", "-C", ROOT, TARGET)
  unless st.success?
    warn "serve_adapter_gate: `make #{TARGET}` failed:\n#{out.lines.last(15).join}"
    exit 2
  end
end

STDBUF_BIN = %w[/usr/bin/stdbuf /bin/stdbuf /opt/homebrew/bin/gstdbuf].find { |b| File.executable?(b) }
STDBUF = STDBUF_BIN ? [STDBUF_BIN, "-oL"].freeze : [].freeze
WORK = Dir.mktmpdir("serve_adapter_gate_")

# ---- server lifecycle (one at a time; always torn down) --------------------

@pid = nil
@logf = nil
@run_dir = nil

def stop_server
  pid = @pid
  @pid = nil
  return if pid.nil?
  begin
    Process.kill("TERM", -Process.getpgid(pid))
  rescue Errno::ESRCH, Errno::ECHILD
  end
  begin
    Timeout.timeout(5) { Process.wait(pid) }
  rescue Timeout::Error
    begin
      Process.kill("KILL", -Process.getpgid(pid))
      Process.wait(pid)
    rescue Errno::ESRCH, Errno::ECHILD
    end
  rescue Errno::ECHILD
  end
end

at_exit do
  stop_server
  FileUtils.rm_rf(WORK)
end
%w[INT TERM].each { |s| trap(s) { exit 130 } }

# Boots the runner; returns the port, or raises with the log tail. Each
# server gets its own run directory, so its request events can be read.
def start_server(extra_env = {}, model = GGUF)
  stop_server
  sock = TCPServer.new("127.0.0.1", 0)
  port = sock.addr[1]
  sock.close
  @run_dir = File.join(WORK, "run_#{port}")
  FileUtils.mkdir_p(@run_dir)
  @logf = File.join(WORK, "serve_#{port}.log")
  env = { "MODEL_PATH" => model, "MODEL_NAME" => "gate", "PORT" => port.to_s,
          "TOY_RUN_DIR" => @run_dir, "TOY_RUN_ID" => "adapter-gate",
          "MODEL_SHA256" => Digest::SHA256.file(model).hexdigest }.merge(extra_env)
  env["ADAPTER_SHA256"] = Digest::SHA256.file(env["ADAPTER"]).hexdigest if env["ADAPTER"] && File.file?(env["ADAPTER"])
  File.open(@logf, "w") do |lf|
    @pid = Process.spawn(env, *STDBUF, RUNNER, chdir: ROOT, pgroup: true, out: lf, err: lf, in: File::NULL)
  end
  deadline = Time.now + 180
  while Time.now < deadline
    if Process.waitpid(@pid, Process::WNOHANG)
      @pid = nil
      raise "server exited before ready:\n#{File.read(@logf).lines.last(8).join}"
    end
    begin
      return port if Net::HTTP.get_response(URI("http://127.0.0.1:#{port}/health")).code == "200"
    rescue StandardError
      # not listening yet
    end
    sleep 0.2
  end
  raise "server not ready after 180 s:\n#{File.read(@logf).lines.last(8).join}"
end

def post(port, path, body)
  res = Net::HTTP.start("127.0.0.1", port, read_timeout: 300) do |h|
    h.post(path, JSON.generate(body), "Content-Type" => "application/json")
  end
  [res.code.to_i, res.body]
end

def chatml(user, system = nil)
  s = +""
  s << "<|im_start|>system\n#{system}<|im_end|>\n" if system
  s << "<|im_start|>user\n#{user}<|im_end|>\n<|im_start|>assistant\n"
  s
end

# /v1/completions with a text prompt: the generated ids, or the HTTP status.
def ids_of(port, prompt, extra = {})
  code, body = post(port, "/v1/completions", { model: "gate", prompt: prompt, max_tokens: MAX_TOKENS }.merge(extra))
  return code unless code == 200
  JSON.parse(body).dig("choices", 0, "ids")
end

def all_at_once(items)
  items.map { |it| Thread.new { yield it } }.map(&:value)
end

# The request events of the CURRENT server, in order.
def request_events
  f = File.join(@run_dir, "events.jsonl")
  return [] unless File.file?(f)
  File.readlines(f).map { |l| JSON.parse(l) }.select { |e| e["name"] == "request" }.map { |e| e["extra"] }
end


# ---- GGUF, in Ruby ---------------------------------------------------------

# Reads a GGUF header: { kv: {key => scalar or array}, tensors: {name =>
# {dims:, type:, off:}}, data: byte offset of the tensor data }.
def gguf_read(path)
  File.open(path, "rb") do |f|
    raise "#{path}: not a GGUF" unless f.read(4) == "GGUF"
    f.read(4)
    n_t, n_kv = f.read(16).unpack("Q<Q<")
    str = -> { f.read(f.read(8).unpack1("Q<")) }
    val = nil
    val = lambda do |t|
      case t
      when 0 then f.read(1).unpack1("C")
      when 1 then f.read(1).unpack1("c")
      when 2 then f.read(2).unpack1("S<")
      when 3 then f.read(2).unpack1("s<")
      when 4 then f.read(4).unpack1("L<")
      when 5 then f.read(4).unpack1("l<")
      when 6 then f.read(4).unpack1("e")
      when 7 then f.read(1).unpack1("C") != 0
      when 8 then str.call
      when 9
        et, n = f.read(12).unpack("L<Q<")
        Array.new(n) { val.call(et) }
      when 10 then f.read(8).unpack1("Q<")
      when 11 then f.read(8).unpack1("q<")
      when 12 then f.read(8).unpack1("E")
      else raise "#{path}: GGUF value type #{t}"
      end
    end
    kv = {}
    n_kv.times do
      k = str.call
      kv[k] = val.call(f.read(4).unpack1("L<"))
    end
    tensors = {}
    n_t.times do
      name = str.call
      nd = f.read(4).unpack1("L<")
      dims = f.read(8 * nd).unpack("Q<*")
      type, off = f.read(12).unpack("L<Q<")
      tensors[name] = { dims: dims, type: type, off: off }
    end
    al = kv["general.alignment"] || 32
    { kv: kv, tensors: tensors, data: (f.pos + al - 1) / al * al }
  end
end

# Writes a GGUF of F32 tensors: `tensors` is [[name, dims (ggml ne), floats]].
def gguf_write(path, strs, f32s, tensors)
  s = ->(x) { [x.bytesize].pack("Q<") + x.b }
  out = "GGUF".b + [3, tensors.length, strs.length + f32s.length].pack("L<Q<Q<")
  strs.each { |k, v| out << s.call(k) << [8].pack("L<") << s.call(v) }
  f32s.each { |k, v| out << s.call(k) << [6].pack("L<") << [v].pack("e") }
  off = 0
  tensors.each do |name, dims, vals|
    out << s.call(name) << [dims.length].pack("L<") << dims.pack("Q<*") << [0, off].pack("L<Q<")
    off += (vals.length * 4 + 31) / 32 * 32
  end
  out << "\0".b * ((32 - out.bytesize % 32) % 32)
  tensors.each do |_n, _d, vals|
    blob = vals.pack("e*")
    out << blob << "\0".b * ((32 - blob.bytesize % 32) % 32)
  end
  File.binwrite(path, out)
end

def f32_tensor(path, hdr, name)
  t = hdr[:tensors].fetch(name)
  raise "#{name}: type #{t[:type]}, the gate needs F32" unless t[:type] == 0
  IO.binread(path, t[:dims].inject(:*) * 4, hdr[:data] + t[:off]).unpack("e*")
end

HDR      = gguf_read(GGUF)
D_MODEL  = HDR[:kv].fetch("llama.embedding_length")
D_FF     = HDR[:kv].fetch("llama.feed_forward_length")
N_LAYERS = HDR[:kv].fetch("llama.block_count")
N_HEADS  = HDR[:kv].fetch("llama.attention.head_count")
N_KV     = HDR[:kv].fetch("llama.attention.head_count_kv")
RMS_EPS  = HDR[:kv].fetch("llama.attention.layer_norm_rms_epsilon")
D_HEAD   = D_MODEL / N_HEADS
EDIT_LO  = 12   # first edited block
ALPHA    = 4.0

# [tensor base name, d_in, d_out, rank]
EDITS = [["blk.12.attn_q",      D_MODEL, D_MODEL,         3],
         ["blk.12.attn_k",      D_MODEL, N_KV * D_HEAD,   2],
         ["blk.12.attn_v",      D_MODEL, N_KV * D_HEAD,   2],
         ["blk.12.attn_output", D_MODEL, D_MODEL,         2],
         ["blk.12.ffn_gate",    D_MODEL, D_FF,            2],
         ["blk.12.ffn_up",      D_MODEL, D_FF,            2],
         ["blk.12.ffn_down",    D_FF,    D_MODEL,         2],
         ["blk.20.attn_output", D_MODEL, D_MODEL,         1],
         ["blk.20.ffn_down",    D_FF,    D_MODEL,         1]].freeze

rng = Random.new(218)
# A is [r][d_in] (ne=[d_in, r]); B is [d_out][r] (ne=[r, d_out]).
AB = EDITS.map do |_n, d_in, d_out, r|
  [Array.new(r * d_in) { (rng.rand - 0.5) * 0.6 }, Array.new(d_out * r) { (rng.rand - 0.5) * 0.6 }]
end

def write_adapter(path, alpha, edits = EDITS, ab = AB, extra = [])
  tensors = []
  edits.each_with_index do |(n, d_in, d_out, r), i|
    tensors << ["#{n}.weight.lora_a", [d_in, r], ab[i][0]]
    tensors << ["#{n}.weight.lora_b", [r, d_out], ab[i][1]]
  end
  gguf_write(path, [["general.type", "adapter"], ["adapter.type", "lora"], ["general.architecture", "llama"]],
             [["adapter.lora.alpha", alpha]], tensors + extra)
end

ADAPTER  = File.join(WORK, "adapter.gguf")
ADAPTER8 = File.join(WORK, "adapter-alpha8.gguf")
MERGED   = File.join(WORK, "merged.gguf")
write_adapter(ADAPTER, ALPHA)
write_adapter(ADAPTER8, 2 * ALPHA)

# The merged model: W[o][i] += (alpha / r) * sum_k B[o][k] A[k][i].
FileUtils.cp(GGUF, MERGED)
File.open(MERGED, "r+b") do |f|
  EDITS.each_with_index do |(n, d_in, d_out, r), i|
    name = "#{n}.weight"
    t = HDR[:tensors].fetch(name)
    raise "#{name}: dims #{t[:dims]}, expected #{[d_in, d_out]}" unless t[:dims] == [d_in, d_out]
    w = f32_tensor(GGUF, HDR, name)
    a, b = AB[i]
    mul = ALPHA / r
    d_out.times do |o|
      r.times do |k|
        bk = b[o * r + k] * mul
        base = o * d_in
        ak = k * d_in
        d_in.times { |x| w[base + x] += bk * a[ak + x] }
      end
    end
    f.seek(HDR[:data] + t[:off])
    f.write(w.pack("e*"))
  end
end

# ---- requests --------------------------------------------------------------

USERS = ["What is 2+2?", "Name three colours.", "Tell me about the sea.", "Why is the sky blue?"].freeze
PROMPTS = USERS.map { |u| chatml(u) }.freeze

# [ids, [log-probability of each id]]
def gen(port, prompt, extra = {})
  code, body = post(port, "/v1/completions",
                    { model: "gate", prompt: prompt, max_tokens: MAX_TOKENS, logprobs: 0 }.merge(extra))
  raise "completions HTTP #{code}: #{body[0, 200]}" unless code == 200
  c = JSON.parse(body).dig("choices", 0)
  [c["ids"], c.dig("logprobs", "content").map { |e| e["logprob"] }]
end

# The "toy" object of a completions, a chat and a hidden reply (toy#221).
def labels(port, extra = {})
  reqs = [["/v1/completions", { model: "gate", prompt: PROMPTS[0], max_tokens: 2 }],
          ["/v1/chat/completions", { model: "gate", max_tokens: 2, messages: [{ role: "user", content: USERS[0] }] }],
          ["/v1/hidden", { model: "gate", prompt: PROMPTS[0], layers: [1] }]]
  reqs.map do |path, body|
    code, out = post(port, path, body.merge(extra))
    raise "#{path} HTTP #{code}: #{out[0, 200]}" unless code == 200
    JSON.parse(out)["toy"]
  end
end

# { layer => [d_model floats] } plus the reply's other fields under :meta.
def hidden(port, prompt, extra = {})
  code, body = post(port, "/v1/hidden", { model: "gate", prompt: prompt }.merge(extra))
  raise "hidden HTTP #{code}: #{body[0, 200]}" unless code == 200
  j = JSON.parse(body)
  h = { meta: j.reject { |k, _| k == "data" } }
  j["layers"].each_with_index { |l, i| h[l] = j["data"][i] }
  h
end

# Largest difference between two vectors, relative to the largest value.
def rel_diff(a, b)
  return Float::INFINITY unless a.length == b.length
  top = [a.map(&:abs).max, b.map(&:abs).max, 1e-30].max
  a.each_index.map { |i| (a[i] - b[i]).abs }.max / top
end

def lp_diff(a, b)
  return Float::INFINITY unless a.length == b.length && !a.empty?
  a.each_index.map { |i| (a[i] - b[i]).abs }.max
end

# Rounding. On CPU the two ways of computing the edited model agree to
# 5e-7 (hidden, relative) and 5e-6 (log-probability). ggml-cuda picks its
# kernels by batch size, so there the same request evaluated in another
# batch already differs by about 1e-3 / 5e-3.
TOL_H  = DEVICE == "cuda" ? 4e-3 : 1e-4
TOL_LP = DEVICE == "cuda" ? 3e-2 : 1e-3
AWAY   = 1e-2   # what counts as a different hidden state
ALL    = (0..N_LAYERS).to_a

failures = []
ok = ->(leg, msg) { puts "  ok: #{leg} — #{msg}" }
bad = ->(leg, msg) { failures << leg; puts "  FAIL: #{leg} — #{msg}" }

begin
  # References: the plain model and the merged model, no adapter.
  port = start_server
  ref_gen = PROMPTS.map { |p| gen(port, p) }
  ref_hid = PROMPTS.map { |p| hidden(port, p, layers: ALL) }
  plain_labels = labels(port)
  port = start_server({}, MERGED)
  mer_gen = PROMPTS.map { |p| gen(port, p) }
  mer_hid = PROMPTS.map { |p| hidden(port, p, layers: ALL) }

  port = start_server("ADAPTER" => ADAPTER)

  # base
  got_gen = PROMPTS.map { |p| gen(port, p, lora_scale: 0) }
  got_hid = PROMPTS.map { |p| hidden(port, p, layers: ALL, lora_scale: 0) }
  same_h = PROMPTS.each_index.all? { |i| ALL.all? { |l| got_hid[i][l] == ref_hid[i][l] } }
  if got_gen == ref_gen && same_h
    ok.call("base", "lora_scale 0: ids, log-probabilities and #{ALL.length} layers of hidden state equal the server without an adapter, #{PROMPTS.length} prompts")
  else
    bad.call("base", "lora_scale 0 differs from the server without an adapter (ids/logprobs equal: #{got_gen == ref_gen}, hidden equal: #{same_h})")
  end

  # merged
  ed_gen = PROMPTS.map { |p| gen(port, p) }
  ed_hid = PROMPTS.map { |p| hidden(port, p, layers: ALL) }
  ids_eq = PROMPTS.each_index.all? { |i| ed_gen[i][0] == mer_gen[i][0] }
  lp_d   = PROMPTS.each_index.map { |i| lp_diff(ed_gen[i][1], mer_gen[i][1]) }.max
  h_d    = PROMPTS.each_index.map { |i| ALL.map { |l| rel_diff(ed_hid[i][l], mer_hid[i][l]) }.max }.max
  away_h = PROMPTS.each_index.map { |i| rel_diff(ed_hid[i][N_LAYERS], ref_hid[i][N_LAYERS]) }.min
  away_lp = PROMPTS.each_index.map { |i| (ed_gen[i][1][0] - ref_gen[i][1][0]).abs }.max
  if ids_eq && lp_d < TOL_LP && h_d < TOL_H && away_h > AWAY && away_lp > 2 * TOL_LP && ed_hid[0][:meta]["lora_scale"] == 1.0
    ok.call("merged", "adapter == merged weights: ids equal, log-probabilities within #{'%.1e' % lp_d}, hidden within #{'%.1e' % h_d}; the base model is #{'%.2f' % away_h} away")
  else
    bad.call("merged", "ids equal #{ids_eq}, logprob diff #{lp_d}, hidden diff #{h_d}, distance to base #{away_h} (hidden) #{away_lp} (logprob)")
  end

  # layers
  below = PROMPTS.each_index.map { |i| (0..EDIT_LO).map { |l| rel_diff(ed_hid[i][l], ref_hid[i][l]) }.max }.max
  above = PROMPTS.each_index.map { |i| ((EDIT_LO + 1)..N_LAYERS).map { |l| rel_diff(ed_hid[i][l], ref_hid[i][l]) }.min }.min
  if below < 1e-6 && above > AWAY
    ok.call("layers", "layers 0..#{EDIT_LO} unchanged by the edit (#{'%.1e' % below}), #{EDIT_LO + 1}..#{N_LAYERS} changed (at least #{'%.2f' % above})")
  else
    bad.call("layers", "below the edit #{below} (want 0), above it #{above} (want a change)")
  end

  # position / subset / bad layer (still on the adapter server, base scale)
  long = PROMPTS[2]
  n_tok = ref_hid[2][:meta]["prompt_tokens"]
  cut = n_tok - 4
  h_pos = hidden(port, long, layers: [N_LAYERS, 3], position: cut, lora_scale: 0)
  h_neg = hidden(port, long, layers: [3, N_LAYERS], position: -4, lora_scale: 0)
  code_bad, = post(port, "/v1/hidden", { model: "gate", prompt: long, layers: [N_LAYERS + 1] })
  code_pos, = post(port, "/v1/hidden", { model: "gate", prompt: long, position: n_tok })
  full3 = hidden(port, long, layers: [3], lora_scale: 0)
  pos_ok = h_pos[:meta]["layers"] == [3, N_LAYERS] && h_pos[:meta]["position"] == cut &&
           h_pos[:meta]["prompt_tokens"] == cut + 1 &&
           h_neg[:meta]["position"] == n_tok - 4 &&
           rel_diff(h_neg[3], h_pos[3]) < TOL_H && rel_diff(h_neg[N_LAYERS], h_pos[N_LAYERS]) < TOL_H &&
           rel_diff(h_pos[3], full3[3]) > AWAY
  if pos_ok && code_bad == 400 && code_pos == 400
    ok.call("position", "position #{cut} of #{n_tok} tokens equals position -4 and differs from the last position; layers come back ascending; layer #{N_LAYERS + 1} and position #{n_tok} are HTTP 400")
  else
    bad.call("position", "meta #{h_pos[:meta]} / #{h_neg[:meta]}, bad layer HTTP #{code_bad}, bad position HTTP #{code_pos}")
  end

  # embed + head, against the GGUF's own tensors
  emb  = f32_tensor(GGUF, HDR, "token_embd.weight")
  gam  = f32_tensor(GGUF, HDR, "output_norm.weight")
  vocab = emb.length / D_MODEL
  code_i, body_i = post(port, "/v1/completions", { model: "gate", prompt: PROMPTS[0], max_tokens: 1, logprobs: 0, lora_scale: 0 })
  raise "completions HTTP #{code_i}" unless code_i == 200
  first_id = JSON.parse(body_i).dig("choices", 0, "ids", 0)
  first_lp = JSON.parse(body_i).dig("choices", 0, "logprobs", "content", 0, "logprob")
  # The prompt ends with "assistant\n": its last token id is the one whose
  # embedding row matches layer 0.
  h0 = ref_hid[0][0]
  row = (0...vocab).find { |v| (0...D_MODEL).all? { |x| emb[v * D_MODEL + x] == h0[x].to_f } || rel_diff(emb[v * D_MODEL, D_MODEL], h0) < 1e-6 }
  nl_h = hidden(port, "\n", layers: [0], lora_scale: 0)
  row_nl = (0...vocab).find { |v| rel_diff(emb[v * D_MODEL, D_MODEL], nl_h[0]) < 1e-6 }
  if row && row_nl && row == row_nl
    ok.call("embed", "layer 0 is token_embd row #{row}, the prompt's last token (\"\\n\")")
  else
    bad.call("embed", "layer 0 matches embedding row #{row.inspect}; a lone \"\\n\" matches #{row_nl.inspect}")
  end
  hl = ref_hid[0][N_LAYERS]
  rms = Math.sqrt(hl.sum { |x| x * x } / D_MODEL + RMS_EPS)
  xn = hl.each_index.map { |x| hl[x] / rms * gam[x] }
  out_w = HDR[:tensors].key?("output.weight") ? f32_tensor(GGUF, HDR, "output.weight") : emb
  logits = Array.new(vocab) { |v| s = 0.0; o = v * D_MODEL; D_MODEL.times { |x| s += out_w[o + x] * xn[x] }; s }
  mx = logits.max
  lse = mx + Math.log(logits.sum { |l| Math.exp(l - mx) })
  best = logits.index(mx)
  if best == first_id && ((mx - lse) - first_lp).abs < 1e-3
    ok.call("head", "layer #{N_LAYERS} through the final norm and output matrix gives id #{best} at log-probability #{'%.4f' % (mx - lse)} (the server: #{first_id}, #{'%.4f' % first_lp})")
  else
    bad.call("head", "from the hidden state: id #{best}, log-probability #{mx - lse}; the server: #{first_id}, #{first_lp}")
  end

  # reuse (one slot): cached_tokens per request event
  n0 = request_events.length
  gen(port, PROMPTS[1])                    # edited
  gen(port, PROMPTS[1])                    # edited again: reuses
  g_base = gen(port, PROMPTS[1], lora_scale: 0)   # base: must not reuse edited rows
  g_ed   = gen(port, PROMPTS[1])           # edited: must not reuse base rows
  ev = request_events[n0..]
  cached = ev.map { |e| e["cached_tokens"] }
  if g_base == ref_gen[1] && g_ed == ed_gen[1] && cached[1] > 0 && cached[2] == 0 && cached[3] == 0
    ok.call("reuse", "cached tokens #{cached.inspect}: a repeat at the same scale reuses its rows, a change of scale reuses none, and the replies are the base / edited ones")
  else
    bad.call("reuse", "cached tokens #{cached.inspect}; base reply right #{g_base == ref_gen[1]}, edited reply right #{g_ed == ed_gen[1]}")
  end

  # scale
  s2_gen = PROMPTS.map { |p| gen(port, p, lora_scale: 2) }
  s2_hid = PROMPTS.map { |p| hidden(port, p, layers: [N_LAYERS], lora_scale: 2) }
  port = start_server("ADAPTER" => ADAPTER8)
  a8_gen = PROMPTS.map { |p| gen(port, p) }
  a8_hid = PROMPTS.map { |p| hidden(port, p, layers: [N_LAYERS]) }
  h_d2 = PROMPTS.each_index.map { |i| rel_diff(s2_hid[i][N_LAYERS], a8_hid[i][N_LAYERS]) }.max
  far  = PROMPTS.each_index.map { |i| rel_diff(s2_hid[i][N_LAYERS], ed_hid[i][N_LAYERS]) }.min
  if s2_gen.map(&:first) == a8_gen.map(&:first) && h_d2 < TOL_H && far > AWAY
    ok.call("scale", "lora_scale 2 equals an adapter file with alpha #{2 * ALPHA} (hidden within #{'%.1e' % h_d2}), and is #{'%.2f' % far} from scale 1")
  else
    bad.call("scale", "ids equal #{s2_gen.map(&:first) == a8_gen.map(&:first)}, hidden diff #{h_d2}, distance to scale 1 #{far}")
  end

  # mixed, on 2 slots
  port = start_server("ADAPTER" => ADAPTER, "SLOTS" => "2")
  jobs = PROMPTS.each_index.flat_map { |i| [[i, 0], [i, 1]] }
  n0 = request_events.length
  res = all_at_once(jobs) { |i, s| gen(port, PROMPTS[i], lora_scale: s) }
  hres = all_at_once(jobs) { |i, s| hidden(port, PROMPTS[i], layers: [N_LAYERS], lora_scale: s) }
  ev = request_events[n0, jobs.length]
  right = jobs.each_index.all? do |j|
    i, s = jobs[j]
    want = s == 0 ? ref_gen[i] : ed_gen[i]
    want_h = s == 0 ? ref_hid[i][N_LAYERS] : ed_hid[i][N_LAYERS]
    res[j][0] == want[0] && lp_diff(res[j][1], want[1]) < TOL_LP && rel_diff(hres[j][N_LAYERS], want_h) < TOL_H
  end
  shared = ev.count { |e| e["shared_steps"] > 0 }
  if right && shared >= jobs.length / 2
    ok.call("mixed", "#{jobs.length} base and edited requests at once on 2 slots each give what they give alone (generation and hidden state); #{shared} of them shared decode steps")
  else
    bad.call("mixed", "replies right #{right}; #{shared}/#{jobs.length} requests shared steps")
  end

  # prefix (toy#219): one slot, a boot prefix, three scales
  system = "You are a careful assistant. Answer in one short sentence, and say so when you do not know."
  sysfile = File.join(WORK, "system.txt")
  File.write(sysfile, system + "\n")
  chat = lambda do |pt, user, scale|
    code, body = post(pt, "/v1/chat/completions",
                      { model: "gate", max_tokens: MAX_TOKENS, logprobs: true, lora_scale: scale,
                        messages: [{ role: "system", content: system }, { role: "user", content: user }] })
    raise "chat HTTP #{code}: #{body[0, 200]}" unless code == 200
    c = JSON.parse(body).dig("choices", 0)
    [c.dig("message", "content"), c.dig("logprobs", "content").map { |e| e["logprob"] }]
  end
  seq = [[0, 1], [1, 1], [0, 2], [1, 2], [1, 0], [2, 1], [2, 2]]   # [user, scale]
  port = start_server("ADAPTER" => ADAPTER, "KV_REUSE" => "0")
  want = seq.map { |u, sc| chat.call(port, USERS[u], sc) }
  port = start_server("ADAPTER" => ADAPTER, "PREFIX_FILE" => sysfile)
  n0 = request_events.length
  got = seq.map { |u, sc| chat.call(port, USERS[u], sc) }
  cached = request_events[n0..].map { |e| e["cached_tokens"] }
  sys_n = cached[4]   # the base request reads the boot prefix
  same = seq.each_index.all? { |i| got[i][0] == want[i][0] && lp_diff(got[i][1], want[i][1]) < TOL_LP }
  kept = File.read(@logf).scan(/KV prefix: (\d+) tokens kept for lora_scale/).flatten.map(&:to_i)
  differ = want[1][0] != want[4][0] || lp_diff(want[1][1], want[4][1]) > TOL_LP
  # Requests 1 and 3 follow one at their own scale in the slot and reuse
  # its rows (the system turn and more); 5 and 6 follow another scale, so
  # what they have cached can only be their scale's prefix.
  shape = cached[0] == 0 && cached[2] == 0 && cached[1] >= sys_n && cached[3] >= sys_n && cached[4..] == [sys_n] * 3
  # The same on 3 slots, all requests at once, twice: the second round finds
  # the sets the first one left.
  port = start_server("ADAPTER" => ADAPTER, "PREFIX_FILE" => sysfile, "SLOTS" => "3")
  rounds = 2.times.map { all_at_once(seq) { |u, sc| chat.call(port, USERS[u], sc) } }
  n_ev = request_events.length
  cached2 = request_events[n_ev - seq.length, seq.length].map { |e| e["cached_tokens"] }
  same = same && rounds.all? { |r| seq.each_index.all? { |i| r[i][0] == want[i][0] && lp_diff(r[i][1], want[i][1]) < TOL_LP } }
  shape = shape && cached2.all? { |c| c >= sys_n }
  if same && differ && sys_n > 10 && shape && kept == [sys_n, sys_n]
    ok.call("prefix", "cached tokens #{cached.inspect}: after the first request at a scale, requests at scales 1, 2 and 0 each read the #{sys_n}-token system turn, and all #{seq.length} replies equal a server without reuse, also sent at once on 3 slots (cached #{cached2.inspect})")
  else
    bad.call("prefix", "cached tokens #{cached.inspect}, on 3 slots second round #{cached2.inspect} (system turn #{sys_n}), sets kept #{kept.inspect}, replies equal the no-reuse server #{same}, edited differs from base #{differ}")
  end

  # label
  port = start_server("ADAPTER" => ADAPTER, "SLOTS" => "2")
  sha_m = Digest::SHA256.file(GGUF).hexdigest
  sha_a = Digest::SHA256.file(ADAPTER).hexdigest
  want = ->(a, s) { { "model_sha256" => sha_m, "adapter_sha256" => a, "lora_scale" => s } }
  n0 = request_events.length
  got = { "no adapter" => [plain_labels, want.call("", 0.0)],
          "adapter, no scale given" => [labels(port), want.call(sha_a, 1.0)],
          "adapter, lora_scale 0" => [labels(port, lora_scale: 0), want.call(sha_a, 0.0)],
          "adapter, lora_scale 0.5" => [labels(port, lora_scale: 0.5), want.call(sha_a, 0.5)] }
  ev_scales = request_events[n0..].map { |e| e["lora_scale"] }
  wrong_l = got.reject { |_, (g, w)| g.length == 3 && g.all? { |x| x == w } }
  if wrong_l.empty? && ev_scales == [1.0] * 3 + [0.0] * 3 + [0.5] * 3 && sha_m != sha_a
    ok.call("label", "completions, chat and hidden replies name the model file, the adapter file and the scale in 4 cases; 9 request events carry the scale")
  else
    bad.call("label", "wrong: #{wrong_l.map { |k, (g, w)| "#{k}: got #{g.inspect}, want #{w.inspect}" }.join('; ')}; event scales #{ev_scales.inspect}")
  end
  stop_server

  # refuse
  wrong = File.join(WORK, "wrong.gguf")
  write_adapter(wrong, ALPHA, [["blk.3.ffn_down", D_MODEL, D_MODEL, 1]], [[Array.new(D_MODEL, 0.1), Array.new(D_MODEL, 0.1)]])
  stray = File.join(WORK, "stray.gguf")
  write_adapter(stray, ALPHA, EDITS, AB, [["token_embd.weight.lora_a", [D_MODEL, 1], Array.new(D_MODEL, 0.1)]])
  cases = [["a matrix of the wrong width", { "ADAPTER" => wrong }, /blk\.3\.ffn_down/],
           ["a tensor that cannot be applied", { "ADAPTER" => stray }, /cannot be applied/],
           ["the per-head graph", { "ADAPTER" => ADAPTER, "KV_FUSED" => "0" }, /fused decode graph/]]
  refused = cases.map do |what, env, re|
    sock = TCPServer.new("127.0.0.1", 0); rport = sock.addr[1]; sock.close
    out = ""
    st = nil
    begin
      Timeout.timeout(120) do
        out, st = Open3.capture2e({ "MODEL_PATH" => GGUF, "MODEL_NAME" => "gate", "PORT" => rport.to_s }.merge(env),
                                  RUNNER, chdir: ROOT, stdin_data: "")
      end
    rescue Timeout::Error
      out = "(still running after 120 s: it served instead of refusing)"
    end
    [what, !st.nil? && !st.success? && out =~ re ? true : false, out.lines.last(2).join.strip]
  end
  if refused.all? { |_w, r, _o| r }
    ok.call("refuse", "an adapter with #{refused.map(&:first).join(', one with ')} each exit with a message")
  else
    bad.call("refuse", refused.reject { |_w, r, _o| r }.map { |w, _r, o| "#{w}: #{o}" }.join(" | "))
  end
rescue StandardError => e
  failures << "error"
  puts "  FAIL: #{e.class}: #{e.message}"
  puts e.backtrace.first(4).map { |l| "        #{l}" }.join("\n")
ensure
  stop_server
end

if failures.empty?
  puts "GATE PASS [#{LABEL}]: low-rank delta at inference and hidden-state read (toy#218), #{DEVICE}"
  exit 0
else
  puts "GATE FAIL [#{LABEL}]: #{failures.join(', ')}"
  exit 1
end
