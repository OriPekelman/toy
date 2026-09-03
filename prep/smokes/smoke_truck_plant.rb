#!/usr/bin/env ruby
# prep/smokes/smoke_truck_plant.rb — the toy#188 parity gate for the
# truck-and-trailer plant (lib/toy/io/toy_truck_task.rb).
#
# WHY THIS GATE IS BUILT THE WAY IT IS. A plant is the worst possible
# place for a silent error: every arm downstream reports a NUMBER, and a
# plant off by a sign produces plausible numbers rather than a crash.
# dfa-vs-bp paid for that shape three times ("a broken feature looks
# like the hoped-for answer, not like an error"), so every leg below
# asserts TWO-SIDED — it also proves its own comparator can FAIL, on a
# deliberately corrupted input, because an assertion that cannot fail is
# not coverage.
#
# The legs:
#
#   1  GOLDEN      20 steps against a trajectory emitted by a verbatim
#                  transcription of the frontend's `drive()` under node,
#                  at 1e-12 per entry. Both sides are IEEE doubles in
#                  the same operation order.
#   1b LIVE        the same comparator REJECTS the same run at r = 3, so
#                  leg 1 is comparing something.
#   2  CLAMP       the jack-knife clamp, which toy#188 states fires from
#                  step 8 of the golden. IT DOES NOT — see the note in
#                  the leg. Coverage comes from a constructed rollout
#                  instead, where the flag fires and Oc is pinned.
#   3  JACOBIAN    every one of the 20 analytic entries against a central
#                  difference of the plant itself.
#   3b LIVE        one entry sign-flipped by hand; the same comparator
#                  must catch it.
#   4  OBJECTIVES  the frontend's error scalar and the derivative it
#                  actually backpropagates ARE NOT THE SAME OBJECTIVE,
#                  asserted against a finite difference of the scalar.
#   5  SR-D2       the one-full-turn tolerance in the paper's distance,
#                  and the signed branch a broadcast arm reads.
#   6  BEST        nearest-approach tracking is a running minimum, on a
#                  trajectory that passes the dock and recedes.
#   7  WRAP        divergence 2 MEASURED: WRAP_NONE leaves (-pi, pi],
#                  WRAP_PI does not.
#   8  DOCK        divergence 1 MEASURED: the two reference points are
#                  16 m apart on a straight rig.
#   9  OBS         the three input sets, their dims, and the slot order.
#  10  ASIN        the saturation landmine fires at large r and not at
#                  the paper's r, and the state stays finite.
#  11  STARTS      the paper's 15 fixed starts, its single point, the
#                  frontend's curriculum endpoints, and yard bounds.
#
# Spinel hygiene: while loops, popped-empty literals, no interpolation.

require_relative "../../lib/toy/io/toy_truck_task"

TOL_GOLD = 1.0e-12
TOL_FD   = 1.0e-6

fails = 0

def fabs(v)
  v < 0.0 ? -v : v
end

# ---------------------------------------------------------------- leg 1
# The golden protocol: WRAP_NONE, r = 1, Ls = 14, Lc = 6, u_max = 70 deg,
# no validity or dock termination — drive the full sequence regardless.

SIG = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
       -1.0, -1.0, 0.5, -0.25, 0.0, 1.0, -1.0, 0.75, -0.75, 0.0]

# signal x y Oc Os, one row per step, flattened
GOLD = [
  44.295533694724, 11.702156423300, -0.300000000000, 0.446031803704,
  44.068931706192, 11.593802091636, -0.252998064901, 0.462613942411,
  43.837943125566, 11.478606103730, -0.205996129803, 0.478642655612,
  43.602772354760, 11.356579088358, -0.158994194704, 0.494092643560,
  43.363623780571, 11.227760109040, -0.111992259606, 0.508937862240,
  43.120701473660, 11.092217944298, -0.064990324507, 0.523151523273,
  42.874209002002, 10.950052403937, -0.017988389409, 0.536706100084,
  42.624349379646, 10.801395678227, 0.029013545690,  0.549573341423,
  42.371325173697, 10.646413714633, -0.017988389409, 0.561724293514,
  42.129153262070, 10.494001807068, -0.064990324507, 0.575107051551,
  41.898967862437, 10.344796203180, -0.111992259606, 0.589698968155,
  41.681867108823, 10.199529801704, -0.158994194704, 0.605469444790,
  41.195740378381, 9.863034615499,  -0.130311440163, 0.645978813925,
  40.652358600974, 9.453390705700,  -0.145347296675, 0.693726214942,
  40.138637039737, 9.026169634235,  -0.145347296675, 0.746895863549,
  39.981108349265, 8.880327204833,  -0.098345361577, 0.765915325217,
  39.821073157641, 8.726409481100,  -0.145347296675, 0.784498239356,
  39.563445109379, 8.469244707399,  -0.105669219285, 0.819358090660,
  39.313334376607, 8.201542315370,  -0.145347296675, 0.854092183188,
  38.958098254986, 7.793812392775,  -0.145347296675, 0.914211821159
]

# Drives the golden sequence at step length `r` and returns the worst
# absolute deviation from the golden over all 80 entries. Also records
# the clamp count and the widest |Os - Oc| reached, which leg 2 reads.
$gold_clamps = 0
$gold_widest = 0.0

def run_golden(r)
  t = TruckTask.new
  t.demo_defaults!
  t.tt_r = r
  t.reset_state!(45.0, 12.0, -0.3, 0.4)
  worst = 0.0
  $gold_clamps = 0
  $gold_widest = 0.0
  i = 0
  while i < 20
    t.step!(SIG[i])
    b = i * 4
    e0 = fabs(t.tt_x - GOLD[b])
    e1 = fabs(t.tt_y - GOLD[b + 1])
    e2 = fabs(t.tt_oc - GOLD[b + 2])
    e3 = fabs(t.tt_os - GOLD[b + 3])
    if e0 > worst; worst = e0; end
    if e1 > worst; worst = e1; end
    if e2 > worst; worst = e2; end
    if e3 > worst; worst = e3; end
    w = fabs(t.tt_os - t.tt_oc)
    if w > $gold_widest; $gold_widest = w; end
    i = i + 1
  end
  $gold_clamps = t.tt_clamp_count
  worst
end

worst = run_golden(1.0)
if worst <= TOL_GOLD
  puts "truck GOLDEN ok — 80 entries, worst dev " + worst.to_s
else
  puts "truck GOLDEN FAIL — worst dev " + worst.to_s + " > " + TOL_GOLD.to_s
  fails = fails + 1
end

# 1b — the comparator must be live. r = 3 is the paper's step length and
# a different problem; if leg 1 passed with the comparator disabled this
# would pass too.
worst_r3 = run_golden(3.0)
if worst_r3 > TOL_GOLD
  puts "truck GOLDEN-LIVE ok — r=3 rejected, dev " + worst_r3.to_s
else
  puts "truck GOLDEN-LIVE FAIL — r=3 matched the r=1 golden; comparator is dead"
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 2
# THE CLAMP. toy#188 states of the golden sequence: "it fires from step 8
# on — note Oc pinned at Os - pi/2". IT DOES NOT FIRE ANYWHERE in that
# sequence: |Os - Oc| peaks at 1.0596 rad = 60.7 deg, well inside the
# 90 deg limit. What makes it LOOK pinned is that rows 13/14/16/18/19
# repeat Oc = -0.145347296675 exactly — the exact reversal of symmetric
# +/-s signals about one base, not a bound. That is this lane's recurring
# defect shape (a plausible number, not an error) appearing in the
# GOLDEN's own annotation, so the fact is asserted rather than trusted,
# and clamp coverage is taken from a rollout that really does clamp.

run_golden(1.0)
if $gold_clamps == 0 && $gold_widest < 0.5 * Math::PI
  puts "truck CLAMP-GOLDEN ok — 0 clamps, widest |Os-Oc| " + $gold_widest.to_s +
       " rad (issue says it fires at step 8; it does not)"
else
  puts "truck CLAMP-GOLDEN FAIL — clamps " + $gold_clamps.to_s +
       " widest " + $gold_widest.to_s
  fails = fails + 1
end

# Constructed: hold full lock and the cab runs into the jack-knife limit.
tk = TruckTask.new
tk.paper_defaults!
tk.tt_wrap = TruckTask::WRAP_NONE
tk.reset_state!(60.0, 0.0, 0.0, 0.0)
pin_worst = 0.0
n = 0
while n < 60
  tk.step!(1.0)
  if tk.tt_clamped == 1
    dev = fabs(fabs(tk.tt_os - tk.tt_oc) - 0.5 * Math::PI)
    if dev > pin_worst; pin_worst = dev; end
  end
  n = n + 1
end
if tk.tt_clamp_count > 0 && pin_worst < 1.0e-12
  puts "truck CLAMP-PINNED ok — fired " + tk.tt_clamp_count.to_s +
       "/60 steps, |Os-Oc| pinned to pi/2 within " + pin_worst.to_s
else
  puts "truck CLAMP-PINNED FAIL — fired " + tk.tt_clamp_count.to_s +
       " worst pin dev " + pin_worst.to_s
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 3
# THE JACOBIAN, against a central difference of the plant. Every entry:
# a sign error hides here, and it hides in the direction of a plausible
# result. States are chosen away from the clamp and away from asin
# saturation, and WRAP_NONE is forced because wrapping is a
# discontinuity the Jacobian does not model.

ST_X  = [60.0, 40.0, 75.0]
ST_Y  = [10.0, -20.0, 5.0]
ST_OC = [0.3, -0.5, 0.9]
ST_OS = [0.1, -0.2, 0.4]
ST_U  = [0.4, -0.9, 0.15]

# Returns the four post-step state components for a perturbed input.
# `slot` 0..3 perturbs a state component, 4 perturbs u.
def probe(t, x, y, oc, os, u, slot, h, out)
  px = x; py = y; poc = oc; pos = os; pu = u
  if slot == 0; px  = px  + h; end
  if slot == 1; py  = py  + h; end
  if slot == 2; poc = poc + h; end
  if slot == 3; pos = pos + h; end
  if slot == 4; pu  = pu  + h; end
  t.reset_state!(px, py, poc, pos)
  t.step_u!(pu)
  out[0] = t.tt_x
  out[1] = t.tt_y
  out[2] = t.tt_oc
  out[3] = t.tt_os
  nil
end

tj = TruckTask.new
tj.paper_defaults!
tj.tt_wrap = TruckTask::WRAP_NONE

fd   = Array.new(20, 0.0)
hi   = Array.new(4, 0.0)
lo   = Array.new(4, 0.0)
h    = 1.0e-6
jac_worst = 0.0
jac_worst_i = 0

si = 0
while si < 3
  x  = ST_X[si]
  y  = ST_Y[si]
  oc = ST_OC[si]
  os = ST_OS[si]
  u  = ST_U[si]

  col = 0
  while col < 5
    probe(tj, x, y, oc, os, u, col, h, hi)
    probe(tj, x, y, oc, os, u, col, -h, lo)
    row = 0
    while row < 4
      fd[row * 5 + col] = (hi[row] - lo[row]) / (2.0 * h)
      row = row + 1
    end
    col = col + 1
  end

  tj.reset_state!(x, y, oc, os)
  tj.jacobian!(u)
  k = 0
  while k < 20
    d = fabs(tj.tt_jac[k] - fd[k])
    if d > jac_worst
      jac_worst = d
      jac_worst_i = k
    end
    k = k + 1
  end
  si = si + 1
end

if jac_worst <= TOL_FD
  puts "truck JACOBIAN ok — 60 entries over 3 states, worst dev " + jac_worst.to_s
else
  puts "truck JACOBIAN FAIL — worst dev " + jac_worst.to_s +
       " at entry " + jac_worst_i.to_s + " (row " + (jac_worst_i / 5).to_s +
       " col " + (jac_worst_i % 5).to_s + ")"
  fails = fails + 1
end

# 3b — corrupt one entry by hand and require the SAME comparator to
# catch it. dOs/dOc (entry 17) is the one a sign slip would land on.
tj.reset_state!(ST_X[0], ST_Y[0], ST_OC[0], ST_OS[0])
tj.jacobian!(ST_U[0])
col = 0
while col < 5
  probe(tj, ST_X[0], ST_Y[0], ST_OC[0], ST_OS[0], ST_U[0], col, h, hi)
  probe(tj, ST_X[0], ST_Y[0], ST_OC[0], ST_OS[0], ST_U[0], col, -h, lo)
  row = 0
  while row < 4
    fd[row * 5 + col] = (hi[row] - lo[row]) / (2.0 * h)
    row = row + 1
  end
  col = col + 1
end
tj.reset_state!(ST_X[0], ST_Y[0], ST_OC[0], ST_OS[0])
tj.jacobian!(ST_U[0])
tj.tt_jac[17] = -tj.tt_jac[17]
caught = fabs(tj.tt_jac[17] - fd[17])
if caught > TOL_FD
  puts "truck JACOBIAN-LIVE ok — sign flip on dOs/dOc caught, dev " + caught.to_s
else
  puts "truck JACOBIAN-LIVE FAIL — a sign flip on entry 17 passed the FD check"
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 4
# THE TWO OBJECTIVES ARE NOT ONE OBJECTIVE. The frontend's scalar
# weights x, y and the trailer angle equally; the derivative it
# backpropagates weights y by 10 and zeroes x. Asserted against a
# finite difference of the scalar, so "ported with the asymmetry
# intact" is measured and not claimed.

to = TruckTask.new
to.demo_defaults!
to.reset_state!(30.0, 8.0, -0.2, 0.35)
to.demo_error_grad!
g_y = to.tt_grad4[1]
g_x = to.tt_grad4[0]
g_c = to.tt_grad4[2]
g_s = to.tt_grad4[3]

# d(demo_error)/d(y_normalised): perturb y physically, undo the 1/50.
hy = 1.0e-4
to.reset_state!(30.0, 8.0 + hy, -0.2, 0.35)
e_hi = to.demo_error
to.reset_state!(30.0, 8.0 - hy, -0.2, 0.35)
e_lo = to.demo_error
fd_y = (e_hi - e_lo) / (2.0 * hy / 50.0)

# d(demo_error)/d(x_normalised) is NOT zero here (x is inside the max()),
# while the derivative reports exactly zero. That gap is the point.
hx = 1.0e-4
to.reset_state!(30.0 + hx, 8.0, -0.2, 0.35)
ex_hi = to.demo_error
to.reset_state!(30.0 - hx, 8.0, -0.2, 0.35)
ex_lo = to.demo_error
fd_x = (ex_hi - ex_lo) / (2.0 * hx / 50.0)

ratio_ok = fabs(g_y / fd_y - 10.0) < 1.0e-5
zero_ok  = g_x == 0.0 && g_c == 0.0 && fabs(fd_x) > 1.0e-6
if ratio_ok && zero_ok && fabs(g_s) > 0.0
  puts "truck OBJECTIVES ok — grad/scalar ratio on y is " + (g_y / fd_y).to_s +
       ", grad x is 0 while the scalar's is " + fd_x.to_s
else
  puts "truck OBJECTIVES FAIL — y ratio " + (g_y / fd_y).to_s +
       " grad_x " + g_x.to_s + " grad_c " + g_c.to_s + " fd_x " + fd_x.to_s
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 5
# THE PAPER'S DISTANCE. The min term tolerates one full turn either way,
# which is why the paper needs no wrapping; and the SIGNED branch is
# what a broadcast arm reads as its third error component.

td = TruckTask.new
td.paper_defaults!
td.tt_wrap = TruckTask::WRAP_NONE

td.reset_state!(0.0, 0.0, 0.0, 2.0 * Math::PI)
d_turn = td.sr_d2
a_turn = td.sr_angle_err
td.reset_state!(0.0, 0.0, 0.0, Math::PI)
d_half = td.sr_d2
td.reset_state!(3.0, 4.0, 0.0, 0.0)
d_xy = td.sr_d2

turn_ok = d_turn < 1.0e-20 && fabs(a_turn) < 1.0e-12
half_ok = fabs(d_half - Math::PI * Math::PI) < 1.0e-12
xy_ok   = fabs(d_xy - 25.0) < 1.0e-12
if turn_ok && half_ok && xy_ok
  puts "truck SR-D2 ok — full turn scores " + d_turn.to_s +
       ", half turn " + d_half.to_s + ", (3,4) " + d_xy.to_s
else
  puts "truck SR-D2 FAIL — turn " + d_turn.to_s + " angle_err " + a_turn.to_s +
       " half " + d_half.to_s + " xy " + d_xy.to_s
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 6
# NEAREST APPROACH. The GA scored the best point on the trajectory, not
# the terminal state, so the plant tracks a running minimum AND the
# state at it. Driven straight through the dock and out the far side: a
# terminal-only read would report 25 where the episode achieved 1.

tb = TruckTask.new
tb.paper_defaults!
tb.reset_state!(10.0, 0.0, 0.0, 0.0)
n = 0
while n < 5
  tb.step!(0.0)
  n = n + 1
end
term = tb.sr_d2
best = tb.tt_best_d2
best_ok = fabs(best - 1.0) < 1.0e-9 && fabs(term - 25.0) < 1.0e-9 &&
          tb.tt_best_step == 3 && fabs(tb.tt_best_x - 1.0) < 1.0e-9
if best_ok
  puts "truck BEST ok — best d2 " + best.to_s + " at step " +
       tb.tt_best_step.to_s + ", terminal " + term.to_s
else
  puts "truck BEST FAIL — best " + best.to_s + " step " + tb.tt_best_step.to_s +
       " x " + tb.tt_best_x.to_s + " terminal " + term.to_s
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 7
# DIVERGENCE 2, MEASURED. Under WRAP_NONE a rollout leaves the range its
# own [-1,1] normalisation assumes; under WRAP_PI it cannot. Both sides
# are asserted, so "the frontend does not wrap" is a fact here and not a
# comment.

def spin_extreme(wrap)
  t = TruckTask.new
  t.paper_defaults!
  t.tt_wrap = wrap
  t.reset_state!(60.0, 0.0, 0.0, 0.0)
  worst = 0.0
  n = 0
  while n < 200
    t.step!(1.0)
    a = fabs(t.tt_os)
    if a > worst; worst = a; end
    n = n + 1
  end
  worst
end

w_none = spin_extreme(TruckTask::WRAP_NONE)
w_pi   = spin_extreme(TruckTask::WRAP_PI)
if w_none > Math::PI && w_pi <= Math::PI + 1.0e-12
  puts "truck WRAP ok — WRAP_NONE reached |Os| " + w_none.to_s +
       ", WRAP_PI held at " + w_pi.to_s
else
  puts "truck WRAP FAIL — none " + w_none.to_s + " pi " + w_pi.to_s
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 8
# DIVERGENCE 1, MEASURED. On a straight rig the frontend's termination
# point sits Ls + 2 = 16 m ahead of the trailer rear the paper scores.
# A lane that mixed them would dock 16 m from the dock and report
# success.

tm = TruckTask.new
tm.demo_defaults!
tm.reset_state!(10.0, 0.0, 0.0, 0.0)
tm.tt_dock = TruckTask::DOCK_TRAILER
rx_t = tm.ref_x
tm.tt_dock = TruckTask::DOCK_MIDRIG
rx_m = tm.ref_x
if fabs(rx_m - rx_t - 16.0) < 1.0e-12
  puts "truck DOCK ok — trailer ref " + rx_t.to_s + ", midrig ref " + rx_m.to_s
else
  puts "truck DOCK FAIL — trailer " + rx_t.to_s + " midrig " + rx_m.to_s
  fails = fails + 1
end

# ---------------------------------------------------------------- leg 9
# THE THREE INPUT SETS, and the slot order. Distinct values in every
# slot, so swapping the two angles (which toy#188's prose invites) is
# caught rather than absorbed.

tv = TruckTask.new
tv.paper_defaults!
tv.reset_state!(45.0, 12.0, -0.3, 0.4)

tv.tt_obs = TruckTask::OBS4
tv.obs!
o4_ok = tv.obs_dim == 4 &&
        fabs(tv.tt_obs_buf[0] - (-0.1)) < 1.0e-12 &&
        fabs(tv.tt_obs_buf[1] - 0.24) < 1.0e-12 &&
        fabs(tv.tt_obs_buf[2] - (-0.3 / Math::PI)) < 1.0e-12 &&
        fabs(tv.tt_obs_buf[3] - (0.4 / Math::PI)) < 1.0e-12

tv.tt_obs = TruckTask::OBS3
tv.obs!
o3_ok = tv.obs_dim == 3 &&
        fabs(tv.tt_obs_buf[2] - (0.4 / Math::PI)) < 1.0e-12

tv.tt_obs = TruckTask::OBS8
tv.obs!
o8_ok = tv.obs_dim == 8
k = 0
while k < 4
  if fabs(tv.tt_obs_buf[4 + k] - 10.0 * tv.tt_obs_buf[k]) > 1.0e-12
    o8_ok = false
  end
  k = k + 1
end

if o4_ok && o3_ok && o8_ok
  puts "truck OBS ok — dims 4/3/8, slot order (x,y,Oc,Os), 10x duplicate exact"
else
  puts "truck OBS FAIL — obs4 " + o4_ok.to_s + " obs3 " + o3_ok.to_s +
       " obs8 " + o8_ok.to_s
  fails = fails + 1
end

# --------------------------------------------------------------- leg 10
# THE ASIN LANDMINE. The reformulation does not remove the domain: at
# large r the ratio leaves [-1,1]. This port saturates and counts it
# (the frontend NaNs and finds out downstream). Asserted BOTH ways —
# fires at r = 30, silent at the paper's r = 3 — because a counter that
# always fires would be as useless as one that never does.

ta = TruckTask.new
ta.paper_defaults!
ta.reset_state!(60.0, 0.0, 0.0, 0.0)
ta.step!(1.0)
quiet = ta.tt_asin_sat

ta.tt_r = 30.0
ta.reset_state!(60.0, 0.0, 0.0, 0.0)
ta.step!(1.0)
loud = ta.tt_asin_sat
finite = ta.tt_oc == ta.tt_oc && ta.tt_os == ta.tt_os && ta.tt_x == ta.tt_x

if quiet == 0 && loud == 1 && finite
  puts "truck ASIN ok — silent at r=3, fired once at r=30, state finite"
else
  puts "truck ASIN FAIL — quiet " + quiet.to_s + " loud " + loud.to_s +
       " finite " + finite.to_s
  fails = fails + 1
end

# --------------------------------------------------------------- leg 11
# THE START SCHEMES, against the published numbers: the paper's 15 fixed
# starts, its single point, the frontend's curriculum endpoints, and the
# generalisation region.

ts = TruckTask.new
ts.paper_defaults!

ens_ok = ts.ensemble_size == 15
ts.ensemble_start!(0)
ens_ok = ens_ok && fabs(ts.tt_x - 100.0) < 1.0e-12 && fabs(ts.tt_y) < 1.0e-12 &&
         fabs(ts.tt_os + 0.5 * Math::PI) < 1.0e-12 &&
         fabs(ts.tt_oc - ts.tt_os) < 1.0e-12
ts.ensemble_start!(14)
ens_ok = ens_ok && fabs(ts.tt_x - 80.0) < 1.0e-12 &&
         fabs(ts.tt_y + 50.0) < 1.0e-12 &&
         fabs(ts.tt_os - 0.5 * Math::PI) < 1.0e-12

ts.point_start!
pt_ok = fabs(ts.tt_x - 20.0) < 1.0e-12 && fabs(ts.tt_y - 10.0) < 1.0e-12 &&
        fabs(ts.tt_os + 2.0) < 1.0e-12

# Curriculum endpoints, with L = 20 and the interpolation over i/19.
# Lesson 0's TRAILER angle is exactly 0 but its CAB angle is +/-10 deg —
# toy#188's "straight rig" holds for the trailer only.
ts.lesson_bounds!(0)
l0_ok = ts.lesson_count == 20 &&
        fabs(ts.tt_bounds[0] - 4.0) < 1.0e-12 &&
        fabs(ts.tt_bounds[1] - 10.0) < 1.0e-12 &&
        fabs(ts.tt_bounds[2] + 2.0) < 1.0e-12 &&
        fabs(ts.tt_bounds[3] - 2.0) < 1.0e-12 &&
        fabs(ts.tt_bounds[4]) < 1.0e-12 &&
        fabs(ts.tt_bounds[5]) < 1.0e-12 &&
        fabs(ts.tt_bounds[7] - 10.0 * Math::PI / 180.0) < 1.0e-12
ts.lesson_bounds!(19)
l19_ok = fabs(ts.tt_bounds[0] - 40.0) < 1.0e-12 &&
         fabs(ts.tt_bounds[1] - 80.0) < 1.0e-12 &&
         fabs(ts.tt_bounds[3] - 20.0) < 1.0e-12 &&
         fabs(ts.tt_bounds[5] - 0.5 * Math::PI) < 1.0e-12 &&
         fabs(ts.tt_bounds[7] - 0.5 * Math::PI) < 1.0e-12

# The generalisation region, sampled: x in [50,100], y in [-50,50].
yard_ok = true
ts.seed!(11)
n = 0
while n < 200
  if !ts.sample_yard!(64); yard_ok = false; end
  if ts.tt_x < 50.0 || ts.tt_x > 100.0; yard_ok = false; end
  if ts.tt_y < -50.0 || ts.tt_y > 50.0; yard_ok = false; end
  n = n + 1
end

if ens_ok && pt_ok && l0_ok && l19_ok && yard_ok
  puts "truck STARTS ok — 15 fixed, point (20,10,-2), curriculum 0/19, yard bounds"
else
  puts "truck STARTS FAIL — ens " + ens_ok.to_s + " point " + pt_ok.to_s +
       " lesson0 " + l0_ok.to_s + " lesson19 " + l19_ok.to_s +
       " yard " + yard_ok.to_s
  fails = fails + 1
end

# ----------------------------------------------------------------------

puts ""
puts "provenance: " + TruckTask.new.provenance
if fails == 0
  puts "truck-plant: PASS"
else
  puts "truck-plant: FAIL (" + fails.to_s + ")"
  exit 1
end
