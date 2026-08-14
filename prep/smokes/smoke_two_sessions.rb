# prep/smokes/smoke_two_sessions.rb — can ONE process hold two model
# sessions in sequence?
#
# toy#166 (capstone P1b) needs three models in one run — the frozen
# autoencoder, the diffusion denoiser, and an AR judge — and the P1b
# ticket assumes they compose in a single process. But
# tnn_session_new_on() calls ggml_backend_sched_reset() on the ENGINE's
# scheduler, which every session on that backend shares:
#
#     /* Reset the (shared) scheduler so any prior allocation state is
#      * wiped before this session builds its graph. */
#
# so creating session B may invalidate session A's realized graph. If it
# does, P1b cannot hold two models live at once and the lane has to
# sequence them — train A, extract what is needed from A into plain Ruby
# arrays, and only then create B. That is a different runner shape, and
# it is much cheaper to learn it here than 800 lines into the runner.
#
# Prints PASS/FAIL for two questions:
#   1. does a SECOND session train correctly after a first one has?
#   2. can the FIRST session still be computed on afterwards?
#
# (2) is the one P1b actually needs the answer to: if it is false, every
# cross-model quantity has to leave the session as a Ruby array before
# the next model is built.

require_relative "../../lib/toy/llm/engine/ae_engine"
require_relative "../../lib/toy/llm/adamw"

CTX = 16
VOC = 256

def train3(eng, tag)
  eng.realize_for_random_init(32, 2, 64, 1, CTX, 4, 0, 1.0)
  r = eng.build_training_step
  t_loss = r[0]; t_labels = r[1]; t_hp = r[2]
  adamw = Toy::AdamW.for_from_scratch
  adamw.lr = 0.001
  toks = Array.new(CTX, 0)
  i = 0
  while i < CTX
    toks[i] = (i * 7 + 3) % VOC
    i = i + 1
  end
  perm = Array.new(CTX, 0)
  p2 = 0
  while p2 < CTX
    perm[p2] = p2
    p2 = p2 + 1
  end
  gain = Array.new(CTX * 4, 1.0)
  zero = Array.new(CTX * 4, 0.0)
  TinyNN.tnn_upload_from_int_array(eng.sess, eng.t_perm, perm, CTX)
  TinyNN.tnn_upload_from_float_array(eng.sess, eng.t_gain, gain, CTX * 4)
  TinyNN.tnn_upload_from_float_array(eng.sess, eng.t_noise, zero, CTX * 4)
  m_lab = Mat.new(CTX, VOC)
  last = 0.0
  s = 0
  while s < 3
    k = 0
    while k < CTX * VOC
      m_lab.flat[k] = 0.0
      k = k + 1
    end
    b = 0
    while b < CTX
      m_lab.flat[b * VOC + toks[b]] = 1.0
      b = b + 1
    end
    if s == 0
      TinyNN.tnn_graph_reset(eng.sess)
    else
      TinyNN.tnn_graph_reset_grads_only(eng.sess)
    end
    TinyNN.tnn_upload_from_int_array(eng.sess, eng.t_tokens, toks, CTX)
    TinyNN.upload_row_major(eng.sess, t_labels, m_lab)
    TinyNN.upload_row_major(eng.sess, t_hp, adamw.hp(s))
    TinyNN.tnn_compute_backward(eng.sess)
    lm = TinyNN.download_row_major(eng.sess, t_loss, 1, 1)
    last = lm.flat[0]
    s = s + 1
  end
  puts tag + ": loss=" + last.to_s
  last
end

a = Toy::LLM::Engine::AeEngine.new
la = train3(a, "session A")

b = Toy::LLM::Engine::AeEngine.new
lb = train3(b, "session B (created after A)")

if lb == la
  puts "PASS q1: a second session trains, and identically (same seed)"
else
  puts "FAIL q1: second session gives " + lb.to_s + " vs " + la.to_s
end

# q2 — the one that decides the runner's shape. Recompute A's graph now
# that B exists. If the shared scheduler reset invalidated A, this is
# where it shows up (wrong number, or a crash).
TinyNN.tnn_graph_reset_grads_only(a.sess)
TinyNN.tnn_compute_backward(a.sess)
lm2 = TinyNN.download_row_major(a.sess, a.t_loss, 1, 1)
puts "session A recomputed after B exists: loss=" + lm2.flat[0].to_s
d = lm2.flat[0] - la
if d < 0.0
  d = -d
end
if d < 1.0
  puts "PASS q2: session A is still usable after B was created"
else
  puts "FAIL q2: session A returns garbage after B was created — models must be SEQUENCED"
end
