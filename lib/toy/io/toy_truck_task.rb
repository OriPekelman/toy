# lib/toy/io/toy_truck_task.rb — the truck-and-trailer PLANT behind the
# `dfa-for-dynamic-control` arc (toy#188).
#
# THIS FILE IS A PLANT AND NOTHING ELSE. No net, no credit rule, no
# runner, no policy surface. Which learning signal the arms carry is
# still an OPEN QUESTION in the programme (C-SIGNAL) and deciding it
# here by accident — by shipping, say, a convenient "expert action" —
# would pre-empt it. The plant integrates state and hands out
# derivatives; who consumes them is somebody else's issue.
#
# ── WHY THIS FIXTURE ──
#
# Nguyen & Widrow's truck backer-upper: back a truck-and-trailer to a
# loading dock from anywhere in the yard. One control variable, four
# state variables, non-holonomic. It is the FIRST fixture in this
# programme where the policy determines the states it subsequently
# sees — every dfa-vs-bp fixture was a static dataset, so nothing
# measured there says whether DFA's noisier updates drift and compound
# in a loop or act as exploration that makes a policy robust to its own
# mistakes.
#
# Two PUBLISHED poles already exist on this exact plant, which is most
# of why it was chosen:
#
#   Nguyen & Widrow 1990        BPTT through a LEARNED emulator.
#   Schoenauer & Ronald ICEC'94 a GA over the weights of a 4-9-1 net.
#                               No gradient at all.
#
# DFA sits between them: a per-step signal, but a randomly projected
# one. Inserting a third credit rule between two published poles on one
# plant is a better-shaped experiment than another arm on a fresh
# fixture. Schoenauer & Ronald also state the sharper version of why
# the task is here at all — BP "is not a priori suitable as a learning
# paradigm, because no set of training vectors is available" — which is
# dfa-vs-bp's "where BP structurally cannot run at all", said by
# somebody else, thirty years earlier, about this plant.
#
# ── THE KINEMATICS ARE SCHOENAUER & RONALD'S, NOT NGUYEN & WIDROW'S ──
#
# S&R adopt an arcsin reformulation to "resolve more easily the
# uncertainty on the angles Os and Oc occasioned by the use of the
# arctangent in the paper cited above", noting both agree to first
# order. The TypeScript frontend (OriPekelman/truck_backer_upper,
# upstream TiFu/truck_backer_upper, `Truck.drive()`) implements that
# same reformulation, so paper and frontend AGREE and this port has one
# target rather than two:
#
#   A       = r * cos(u)
#   B       = A * cos(Oc[t] - Os[t])
#   x[t+1]  = x[t] - B * cos(Os[t])
#   y[t+1]  = y[t] - B * sin(Os[t])
#   Os[t+1] = Os[t] - asin(A * sin(Oc[t] - Os[t]) / Ls)
#   Oc[t+1] = Oc[t] + asin(r * sin(u) / (Ls + Lc))
#   then Oc[t+1] is clamped to respect |Os[t+1] - Oc[t+1]| <= 90 deg
#
# (x, y) is the CENTRE REAR OF THE TRAILER. The clamp reads the NEW
# trailer angle, not the old one — verified against the frontend.
#
# ── TWO CONVENTIONS, BOTH FIRST-CLASS ──
#
# The paper and the frontend disagree in three places, and the arc
# needs both: the published result must stay reproducible AND the
# frontend must stay replayable. So they are SWITCHES, never fixes, and
# there are two no-arg presets:
#
#   paper_defaults!  r = 3, 300-step cap, DOCK_TRAILER, WRAP_PI
#                    -> what ARMS run under. The arc's goal is to
#                       reproduce S&R's result with a DFA-trained net
#                       in place of the GA, so this is the default.
#   demo_defaults!   r = 1, 1000-step cap, DOCK_MIDRIG, WRAP_NONE
#                    -> the parity-gate protocol and frontend replay.
#
# The three divergences, exposed and NOT resolved:
#
#   1. DOCK REFERENCE POINT. The paper docks the centre rear of the
#      trailer at the origin. The frontend's termination test
#      (`Truck.continue()`) uses `getEndOfTruck()` — a point 2 units
#      PAST the coupling toward the cab — while its error function
#      scores the TRAILER state. It stops on one point and grades
#      another. DOCK_TRAILER (paper) / DOCK_MIDRIG (frontend).
#
#   2. ANGLE WRAPPING. The frontend wraps into (-pi, pi] in its
#      constructor and setters but NOT after a drive step, so a rollout
#      can leave the range its own [-1,1] normalisation assumes. The
#      paper instead tolerates a full turn either way inside its
#      distance term (see sr_d2). WRAP_PI / WRAP_NONE (frontend).
#
#   3. STEP LENGTH r. The paper uses r = 3 m and states convergence was
#      "only achieved by setting a fairly large value r = 3". The
#      frontend uses velocity 1 with up to 1000 steps an episode.
#      Neither is wrong; they are DIFFERENT PROBLEMS, and the
#      difference is 3x in how far one credit assignment reaches.
#
# ── THE ANALYTIC JACOBIAN, AND WHY IT BELONGS IN THE PLANT ──
#
# `jacobian!` gives d(x', y', Oc', Os')/d(x, y, Oc, Os, u) at the
# current state. It buys the fixture a TRUE-GRADIENT arm — a ceiling
# owing nothing to a learned model. The frontend routes credit through
# a trained 5-100-4 emulator, so "BPTT through a learned model" vs
# "BPTT through the real Jacobian" becomes a MEASURED gap rather than
# an assumption. That matters more here than it usually would: D1b
# found the TRUER signal is not reliably the better one — an oracle B
# routing 99.999% of the error trained the body no better than never
# training it, while a random one carrying 1.6% beat both. This plant
# lets that be asked again with a real physical gradient as the oracle
# instead of a constructed one.
#
# Two places the Jacobian does NOT describe the plant:
#
#   - the JACK-KNIFE CLAMP is a projection, not a smooth map. On steps
#     where it fires, d(Oc')/d(anything) is not what the Jacobian
#     reports. `tt_clamped` / `tt_clamp_count` expose it so a caller
#     can count how often an episode's credit path crosses it.
#   - the [-1,1] SIGNAL SATURATION. d(u)/d(signal) is u_max inside and
#     0 outside; `dsteer_dsignal` is that factor, and it is the
#     caller's job to apply it.
#
# ── LANDMINES ──
#
#  1. VECTOR SLOT ORDER IS (x, y, Oc, Os) EVERYWHERE — Jacobian rows,
#     Jacobian columns, obs4, obs8 and demo_error_grad. That is the
#     FRONTEND's state order (`Truck.getStateVector()`), chosen so a
#     shipped weight file and a toy arm index the same slots. toy#188's
#     prose writes obs4 as "(x, y, Os, Oc)"; taking that literally
#     would swap the two angles against every other vector here and
#     against the frontend's weights, and a swapped angle pair is a
#     plausible-looking wrong number, not an error. Raised on the issue.
#  2. demo_error AND demo_error_grad ARE NOT THE SAME OBJECTIVE. The
#     scalar weights x, y and the trailer angle EQUALLY; the derivative
#     the frontend backpropagates weights y by 10 and ZEROES x. Both are
#     ported with that asymmetry intact, because anyone comparing the
#     reported number across arms would otherwise be comparing
#     something other than what was descended. Do not reconcile them.
#  3. asin's DOMAIN survives the reformulation: at large r the ratio can
#     leave [-1,1]. The frontend does not guard it and yields NaN its
#     trainer only notices downstream. HERE THE ARGUMENT SATURATES —
#     the physically sensible reading, and it keeps a rollout scoreable
#     — counted in tt_asin_sat and named in `provenance`.
#  4. sr_fitness IS A RECONSTRUCTION, not a citation. See its comment.
#  5. THE SAMPLING BOX IS NOT THE WORLD BOX. tt_x_min/_max/tt_y_min/_max
#     are where starts are DRAWN (the paper's generalisation region);
#     tt_world_* is where the rig may BE (the frontend's rendering
#     world). Conflating them makes the paper's own y = +/-50 ensemble
#     starts instantly invalid and every episode zero steps long — see
#     valid?. The frontend's validity test also uses the four CORNERS of
#     a 3 m-wide rectangle where valid? tests rig POINTS, so a marginal
#     state can differ.
#
# Spinel hygiene: plain class, no-arg ctor, no default args, no Struct,
# while loops, typed-empty array seeds, no #{} interpolation.

class TruckTask
  # ---- divergence 1: which point docks ----
  DOCK_TRAILER = 0
  DOCK_MIDRIG  = 1

  # ---- divergence 2: angle wrapping after a step ----
  WRAP_PI   = 0
  WRAP_NONE = 1

  # ---- observation sets, all three from the paper ----
  OBS4 = 0   # (x, y, Oc, Os) — the 4-9-1 net's input, and the demo's 4-45-1
  OBS3 = 1   # drops the cab angle: the paper's 3-7-1, "fly by instruments"
  OBS8 = 2   # the four inputs, plus the same four again at 10x

  # ---- start-state schemes ----
  START_LESSON   = 0   # the frontend's 20-lesson curriculum
  START_ENSEMBLE = 1   # the paper's 15 fixed starts
  START_YARD     = 2   # the paper's generalisation region
  START_POINT    = 3   # the paper's single-point run, (20, 10, -2)

  attr_accessor :tt_ls, :tt_lc, :tt_r, :tt_umax,
                :tt_dock, :tt_wrap, :tt_obs, :tt_start,
                :tt_max_steps,
                :tt_x_min, :tt_x_max, :tt_y_min, :tt_y_max,
                :tt_yard_x_min, :tt_yard_x_max,
                :tt_world_x_max, :tt_world_y_min, :tt_world_y_max,
                :tt_dock_x, :tt_dock_y,
                :tt_x, :tt_y, :tt_oc, :tt_os,
                :tt_steps, :tt_clamped, :tt_clamp_count,
                :tt_asin_sat, :tt_sat, :tt_best_d2,
                :tt_best_x, :tt_best_y, :tt_best_oc, :tt_best_os,
                :tt_best_step,
                :tt_jac, :tt_obs_buf, :tt_grad4,
                :tt_ens_x, :tt_ens_y, :tt_ens_os,
                :tt_bounds, :tt_s

  def initialize
    # Plant geometry. Ls = trailer, rear axle to pivot; Lc = cab, pivot
    # to front axle; both in metres, both from the paper's DETAILS OF
    # PLANT table and both matching the frontend.
    @tt_ls   = 14.0
    @tt_lc   = 6.0
    @tt_r    = 3.0
    @tt_umax = 70.0 * Math::PI / 180.0

    # Paper conventions are the DEFAULT: arms reproduce S&R.
    @tt_dock      = DOCK_TRAILER
    @tt_wrap      = WRAP_PI
    @tt_obs       = OBS4
    @tt_start     = START_ENSEMBLE
    @tt_max_steps = 300

    # The start-sampling yard, and the paper's generalisation region as
    # its default: x in [50,100], y in [-50,50]. The frontend samples
    # x from 0 instead (`randomizeNoLimits`), which is a DIFFERENT and
    # easier population, so it is a field and not a constant.
    @tt_x_min = 50.0
    @tt_x_max = 100.0
    @tt_y_min = -50.0
    @tt_y_max = 50.0

    # The dock wall.
    @tt_yard_x_min = 0.0
    @tt_yard_x_max = 100.0

    # The WORLD box the rig may occupy — the frontend's rendering
    # world, deliberately WIDER than the sampling box above. See valid?.
    @tt_world_x_max = 200.0
    @tt_world_y_min = -100.0
    @tt_world_y_max = 100.0

    @tt_dock_x = 0.0
    @tt_dock_y = 0.0

    @tt_x  = 0.0
    @tt_y  = 0.0
    @tt_oc = 0.0
    @tt_os = 0.0

    @tt_steps       = 0
    @tt_clamped     = 0
    @tt_clamp_count = 0
    @tt_asin_sat    = 0
    @tt_sat         = 0
    @tt_best_d2     = 0.0

    # The state AT the best approach, not just its score. toy#189's
    # headline arm broadcasts the state error at the best-approach step,
    # so a caller that kept only the scalar would have to re-run the
    # episode to find out where it happened.
    @tt_best_x    = 0.0
    @tt_best_y    = 0.0
    @tt_best_oc   = 0.0
    @tt_best_os   = 0.0
    @tt_best_step = 0

    @tt_jac     = Array.new(20, 0.0)   # 4 rows x 5 cols, row-major
    @tt_obs_buf = Array.new(8, 0.0)
    @tt_grad4   = Array.new(4, 0.0)
    @tt_bounds  = Array.new(8, 0.0)

    @tt_ens_x  = [0.0]; @tt_ens_x.pop
    @tt_ens_y  = [0.0]; @tt_ens_y.pop
    @tt_ens_os = [0.0]; @tt_ens_os.pop
    build_ensemble!

    @tt_s = [0]
    seed!(7)
  end

  # ---- the two conventions, as no-arg presets ----

  # What ARMS run under. r = 3 is the paper's, and it states convergence
  # was "only achieved" with it.
  def paper_defaults!
    @tt_r         = 3.0
    @tt_max_steps = 300
    @tt_dock      = DOCK_TRAILER
    @tt_wrap      = WRAP_PI
    nil
  end

  # The parity-gate protocol and frontend replay. NOT for arms.
  def demo_defaults!
    @tt_r         = 1.0
    @tt_max_steps = 1000
    @tt_dock      = DOCK_MIDRIG
    @tt_wrap      = WRAP_NONE
    nil
  end

  # ---- state ----

  # Resets the episode: counters, and the nearest-approach tracker,
  # which INCLUDES the start state (an episode that begins at the dock
  # has already achieved its best approach).
  def reset_state!(x, y, oc, os)
    @tt_x  = x
    @tt_y  = y
    @tt_oc = oc
    @tt_os = os
    if @tt_wrap == WRAP_PI
      @tt_oc = wrap_pi(@tt_oc)
      @tt_os = wrap_pi(@tt_os)
    end
    @tt_steps       = 0
    @tt_clamped     = 0
    @tt_clamp_count = 0
    @tt_asin_sat    = 0
    @tt_sat         = 0
    @tt_best_d2     = sr_d2
    mark_best!
    nil
  end

  # Snapshots the current state as the best approach so far.
  def mark_best!
    @tt_best_x    = @tt_x
    @tt_best_y    = @tt_y
    @tt_best_oc   = @tt_oc
    @tt_best_os   = @tt_os
    @tt_best_step = @tt_steps
    nil
  end

  # signal in [-1,1] -> physical steering angle, SATURATING. That
  # saturation is a real non-linearity of the closed loop, not a
  # numerical guard. Sets tt_sat so a caller can see it bind.
  def steer_angle(signal)
    s = signal
    @tt_sat = 0
    if s > 1.0
      s = 1.0
      @tt_sat = 1
    end
    if s < -1.0
      s = -1.0
      @tt_sat = 1
    end
    s * @tt_umax
  end

  # d(u)/d(signal): u_max inside the saturation, 0 outside. The caller
  # applies it — the Jacobian is in u, not in signal.
  def dsteer_dsignal(signal)
    if signal > 1.0;  return 0.0; end
    if signal < -1.0; return 0.0; end
    @tt_umax
  end

  # One plant step from a STEERING SIGNAL in [-1,1].
  #
  # No validity or dock test here: termination is `continue?`, kept
  # separate so a caller (the parity gate, for one) can drive a fixed
  # sequence regardless.
  def step!(signal)
    u = steer_angle(signal)
    step_u!(u)
  end

  # One plant step from a physical steering angle in RADIANS.
  def step_u!(u)
    r  = @tt_r
    ls = @tt_ls
    lc = @tt_lc
    os = @tt_os
    oc = @tt_oc

    a = r * Math.cos(u)
    d = oc - os
    b = a * Math.cos(d)

    nx = @tt_x - b * Math.cos(os)
    ny = @tt_y - b * Math.sin(os)

    # Landmine 3: the reformulation does not remove asin's domain. At
    # large r either ratio can leave [-1,1]; saturate and count it.
    g1 = a * Math.sin(d) / ls
    g2 = r * Math.sin(u) / (ls + lc)
    if g1 > 1.0 || g1 < -1.0 || g2 > 1.0 || g2 < -1.0
      @tt_asin_sat = @tt_asin_sat + 1
    end
    if g1 > 1.0;  g1 = 1.0;  end
    if g1 < -1.0; g1 = -1.0; end
    if g2 > 1.0;  g2 = 1.0;  end
    if g2 < -1.0; g2 = -1.0; end

    nos = os - Math.asin(g1)
    noc = oc + Math.asin(g2)

    # The jack-knife clamp, against the NEW trailer angle. A
    # projection, not a smooth map — see landmine 1 on the Jacobian.
    lim = 0.5 * Math::PI
    @tt_clamped = 0
    if nos - noc > lim
      noc = nos - lim
      @tt_clamped = 1
    end
    if noc - nos > lim
      noc = nos + lim
      @tt_clamped = 1
    end
    if @tt_clamped == 1
      @tt_clamp_count = @tt_clamp_count + 1
    end

    @tt_x  = nx
    @tt_y  = ny
    @tt_os = nos
    @tt_oc = noc

    # Divergence 2. The frontend does NOT wrap here, which is how a
    # rollout leaves the range its own normalisation assumes.
    if @tt_wrap == WRAP_PI
      @tt_os = wrap_pi(@tt_os)
      @tt_oc = wrap_pi(@tt_oc)
    end

    @tt_steps = @tt_steps + 1

    # The paper's score is the NEAREST APPROACH over the episode — the
    # GA scored the best point on the trajectory, not the terminal
    # state — so the running minimum is maintained here rather than
    # left to a caller to remember to take.
    d2 = sr_d2
    if d2 < @tt_best_d2
      @tt_best_d2 = d2
      mark_best!
    end
    nil
  end

  def wrap_pi(angle)
    a = angle % (2.0 * Math::PI)
    if a > Math::PI
      a = a - 2.0 * Math::PI
    end
    if a < -Math::PI
      a = a + 2.0 * Math::PI
    end
    a
  end

  # ---- rig geometry ----

  # The coupling device: Ls along the trailer from the centre rear.
  def coupling_x; @tt_x + @tt_ls * Math.cos(@tt_os); end
  def coupling_y; @tt_y + @tt_ls * Math.sin(@tt_os); end

  # The cab's front axle: Lc further along the cab heading.
  def cab_front_x; coupling_x + @tt_lc * Math.cos(@tt_oc); end
  def cab_front_y; coupling_y + @tt_lc * Math.sin(@tt_oc); end

  # DOCK_MIDRIG's reference: 2 m past the coupling toward the cab.
  # This is the frontend's `getEndOfTruck()`, and the point its
  # TERMINATION test uses while its ERROR scores the trailer.
  def midrig_x; coupling_x + 2.0 * Math.cos(@tt_oc); end
  def midrig_y; coupling_y + 2.0 * Math.sin(@tt_oc); end

  def ref_x
    @tt_dock == DOCK_MIDRIG ? midrig_x : @tt_x
  end

  def ref_y
    @tt_dock == DOCK_MIDRIG ? midrig_y : @tt_y
  end

  # ---- termination ----

  # The frontend's test: under 10 cm on both axes, in PHYSICAL units.
  def at_dock?
    dx = ref_x - @tt_dock_x
    dy = ref_y - @tt_dock_y
    if dx < 0.0; dx = -dx; end
    if dy < 0.0; dy = -dy; end
    dx < 0.1 && dy < 0.1
  end

  # TWO DIFFERENT BOXES, and conflating them is a bug toy#189 found
  # within an hour of consuming this file.
  #
  #  - THE DOCK WALL (x > 0) is a hard physical constraint and applies
  #    to EVERY rig point: no part of the truck may pass through the
  #    dock.
  #  - THE WORLD BOX is where the rig may be, and it is NOT the box
  #    starts are sampled from. The paper's own 15-start ensemble sits
  #    AT y = +/-50, so a rig pointing outward has its coupling at
  #    y = 64 the instant it starts. Testing every rig point against
  #    the sampling box made all ten of those starts instantly invalid,
  #    every episode zero steps long, and every arm report a flat loss
  #    curve at d^2 ~ 9157 — a plausible number, not an error.
  #    So the world box defaults to the FRONTEND's rendering world,
  #    [0,200] x [-100,100], and is tested on the trailer reference
  #    point, while tt_x_min/tt_x_max/tt_y_min/tt_y_max stay the
  #    SAMPLING box (the paper's generalisation region).
  def valid?
    if !above_dock?(@tt_x, @tt_y);             return false; end
    if !above_dock?(coupling_x, coupling_y);   return false; end
    if !above_dock?(cab_front_x, cab_front_y); return false; end
    if @tt_x > @tt_world_x_max;  return false; end
    if @tt_y < @tt_world_y_min;  return false; end
    if @tt_y > @tt_world_y_max;  return false; end
    true
  end

  # The dock wall. `x <= 0` is through the wall, not merely at it.
  def above_dock?(x, y)
    if x <= @tt_yard_x_min; return false; end
    true
  end

  def continue?
    if @tt_steps >= @tt_max_steps; return false; end
    if !valid?;                    return false; end
    if at_dock?;                   return false; end
    true
  end

  # ---- objective (a): the frontend's terminal error ----
  #
  # In NORMALISED coordinates, and note the dock sits at x = -1 there,
  # not 0 — `(0 - 50)/50`. The frontend's own comment flags this as the
  # reason it "tries to drive a circle" with the max() below.
  #
  # LANDMINE 2: this scalar and demo_error_grad! are NOT the same
  # objective, and both are ported as-is.
  def demo_error
    xd = obs_x(@tt_x)
    if xd < -1.0; xd = -1.0; end
    xdiff = xd - obs_x(@tt_dock_x)
    ydiff = obs_y(@tt_y) - obs_y(@tt_dock_y)
    tdiff = obs_a(@tt_os)
    xdiff * xdiff + ydiff * ydiff + tdiff * tdiff
  end

  # The derivative the frontend actually backpropagates, in the slot
  # order (x, y, Oc, Os): x ZEROED, y weighted TEN, cab angle zeroed.
  # It is not the gradient of demo_error and is not meant to be.
  def demo_error_grad!
    ydiff = obs_y(@tt_y) - obs_y(@tt_dock_y)
    tdiff = obs_a(@tt_os)
    @tt_grad4[0] = 0.0
    @tt_grad4[1] = 10.0 * 2.0 * ydiff
    @tt_grad4[2] = 0.0
    @tt_grad4[3] = 2.0 * tdiff
    nil
  end

  # ---- objective (b): Schoenauer & Ronald's distance ----
  #
  #   d^2 = x^2 + y^2 + min(Os^2, (Os - 2pi)^2, (Os + 2pi)^2)
  #
  # In PHYSICAL units. The min term is the one-full-turn tolerance —
  # which is also why the paper needs no angle wrapping. The CAB ANGLE
  # DOES NOT ENTER: the paper scores where the trailer ended up and how
  # straight it is.
  def sr_d2
    dx = @tt_x - @tt_dock_x
    dy = @tt_y - @tt_dock_y
    a0 = @tt_os
    a1 = @tt_os - 2.0 * Math::PI
    a2 = @tt_os + 2.0 * Math::PI
    t = a0 * a0
    t1 = a1 * a1
    t2 = a2 * a2
    if t1 < t; t = t1; end
    if t2 < t; t = t2; end
    dx * dx + dy * dy + t
  end

  # The SIGNED angle whose square sr_d2's min term selected — the third
  # component of the state error a broadcast arm needs. Returning d2's
  # scalar alone would leave a caller to re-derive which of the three
  # branches won, and picking the wrong branch near a half-turn costs a
  # full 2pi of error with no sign that anything went wrong.
  def sr_angle_err
    a0 = @tt_os
    a1 = @tt_os - 2.0 * Math::PI
    a2 = @tt_os + 2.0 * Math::PI
    best = a0
    t = a0 * a0
    if a1 * a1 < t
      t = a1 * a1
      best = a1
    end
    if a2 * a2 < t
      t = a2 * a2
      best = a2
    end
    best
  end

  # RECONSTRUCTED — NOT A CITATION. The paper's AMENDED fitness (the one
  # adding the trajectory-length penalty) is a bitmap figure in the scan
  # and did not survive OCR. What the TEXT states: GAs maximise, so the
  # distance is inverted; the amendment adds a term in l, the number of
  # steps; and eps = 0.1 with gamma = 0.001 "ensure a good compromise
  # between trajectory accuracy and length". The form below satisfies
  # all three and reduces to the un-amended 1/(eps + d^2) at gamma = 0.
  #
  # THE FORM IS toy#189's, not toy#188's. #188 asked only for "a form"
  # satisfying the three constraints; #189's `ga` arm names
  # 1/(eps + d^2) * 1/(1 + gamma*l) specifically. Two reconstructions
  # that both satisfy the text would have scored the GA differently
  # while both looking right, so the consumer's form is the one here.
  #
  # IT MUST BE CONFIRMED WITH EDMUND RONALD BEFORE ANY ARM IS SCORED ON
  # IT. A plausible formula must not acquire the authority of a cited
  # one; sr_d2 above is the cited quantity and is safe to score on.
  def sr_fitness(d2, l, eps, gamma)
    (1.0 / (eps + d2)) * (1.0 / (1.0 + gamma * l.to_f))
  end

  def sr_fitness_default(d2, l)
    sr_fitness(d2, l, 0.1, 0.001)
  end

  # ---- observation ----
  #
  # The frontend's normalisation, kept because its shipped weight files
  # were trained in it: x -> (x-50)/50, y -> y/50, angle -> angle/pi.
  # Under WRAP_NONE the angle terms CAN leave [-1,1] — that is
  # divergence 2 surfacing, a fact about the frontend, not a bug here.
  #
  # The paper's net is a 4-9-1 with LOGISTIC SIGMOID units (55 weights
  # including biases), so a consumer must not assume a symmetric
  # activation downstream of these.
  def obs_x(x); (x - 50.0) / 50.0; end
  def obs_y(y); y / 50.0; end
  def obs_a(a); a / Math::PI; end

  def obs_dim
    if @tt_obs == OBS3; return 3; end
    if @tt_obs == OBS8; return 8; end
    4
  end

  # Slot order (x, y, Oc, Os) — landmine 1.
  #
  # OBS8 is the paper's rescaled-duplicate trick: the same four inputs
  # again at 10x. It cut its GA from ~1500 generations to ~800 AND
  # improved precision — an input-conditioning result on the very
  # fixture we are about to run conditioning-sensitive credit rules on.
  # Cheap to carry, expensive to rediscover.
  def obs!
    xn = obs_x(@tt_x)
    yn = obs_y(@tt_y)
    cn = obs_a(@tt_oc)
    sn = obs_a(@tt_os)
    if @tt_obs == OBS3
      @tt_obs_buf[0] = xn
      @tt_obs_buf[1] = yn
      @tt_obs_buf[2] = sn
      return nil
    end
    @tt_obs_buf[0] = xn
    @tt_obs_buf[1] = yn
    @tt_obs_buf[2] = cn
    @tt_obs_buf[3] = sn
    if @tt_obs == OBS8
      @tt_obs_buf[4] = 10.0 * xn
      @tt_obs_buf[5] = 10.0 * yn
      @tt_obs_buf[6] = 10.0 * cn
      @tt_obs_buf[7] = 10.0 * sn
    end
    nil
  end

  # ---- the analytic Jacobian ----
  #
  # d(x', y', Oc', Os') / d(x, y, Oc, Os, u) at the current state, u in
  # RADIANS. Row-major 4x5, rows (x, y, Oc, Os), columns
  # (x, y, Oc, Os, u) — the frontend's ordering, so a weight file and a
  # toy arm index the same slots.
  #
  # Derived by hand and cross-checked entry-by-entry against the
  # frontend's `TruckEmulator.backward()`; the two agree everywhere.
  # It is also finite-difference gated (prep/smokes/smoke_truck_plant.rb):
  # a sign error here would hide in the direction of a plausible result.
  def jacobian!(u)
    ls = @tt_ls
    lc = @tt_lc
    r  = @tt_r
    os = @tt_os
    d  = @tt_oc - os

    a  = r * Math.cos(u)
    b  = a * Math.cos(d)
    sd = Math.sin(d)
    cd = Math.cos(d)
    so = Math.sin(os)
    co = Math.cos(os)
    su = Math.sin(u)

    g = a * sd / ls
    if g > 1.0;  g = 1.0;  end
    if g < -1.0; g = -1.0; end
    q = 1.0 - g * g
    if q < 1.0e-12; q = 1.0e-12; end
    ig = 1.0 / Math.sqrt(q)

    h = r * su / (ls + lc)
    if h > 1.0;  h = 1.0;  end
    if h < -1.0; h = -1.0; end
    qh = 1.0 - h * h
    if qh < 1.0e-12; qh = 1.0e-12; end
    ih = 1.0 / Math.sqrt(qh)

    # row 0 — x'
    @tt_jac[0] = 1.0
    @tt_jac[1] = 0.0
    @tt_jac[2] = a * sd * co
    @tt_jac[3] = -a * sd * co + b * so
    @tt_jac[4] = r * su * cd * co

    # row 1 — y'
    @tt_jac[5] = 0.0
    @tt_jac[6] = 1.0
    @tt_jac[7] = a * sd * so
    @tt_jac[8] = -a * sd * so - b * co
    @tt_jac[9] = r * su * cd * so

    # row 2 — Oc'
    @tt_jac[10] = 0.0
    @tt_jac[11] = 0.0
    @tt_jac[12] = 1.0
    @tt_jac[13] = 0.0
    @tt_jac[14] = r * Math.cos(u) * ih / (ls + lc)

    # row 3 — Os'
    @tt_jac[15] = 0.0
    @tt_jac[16] = 0.0
    @tt_jac[17] = -ig * a * cd / ls
    @tt_jac[18] = 1.0 + ig * a * cd / ls
    @tt_jac[19] = ig * r * su * sd / ls

    nil
  end

  # ---- start states ----

  # The paper's multi-point training set: 5 trailer orientations at each
  # of 3 locations. Fifteen starts, FIXED, no sampling.
  #
  # This is worth more than it looks. The strongest inherited positive
  # is F17/D3b — DFA beats a fully regularised BP where labels are
  # scarce, +0.097 — and a simulator hands out episodes for free, so
  # scarcity has to be IMPOSED ON PURPOSE or that result has no home
  # here. The paper already imposed it, at n = 15, and reported a net
  # trained on those fifteen generalising across the whole yard.
  def build_ensemble!
    lx = [100.0, 80.0, 80.0]
    ly = [0.0, 50.0, -50.0]
    la = [-90.0, -30.0, 0.0, 30.0, 90.0]
    i = 0
    while i < 3
      j = 0
      while j < 5
        @tt_ens_x.push(lx[i])
        @tt_ens_y.push(ly[i])
        @tt_ens_os.push(la[j] * Math::PI / 180.0)
        j = j + 1
      end
      i = i + 1
    end
    nil
  end

  def ensemble_size; 15; end

  # Straight rig: the paper's ensemble states a trailer orientation per
  # start and no cab angle, so the cab is aligned with the trailer.
  def ensemble_start!(k)
    reset_state!(@tt_ens_x[k], @tt_ens_y[k], @tt_ens_os[k], @tt_ens_os[k])
    nil
  end

  # The paper's single-point run: (x, y, Os) = (20, 10, -2), which it
  # reports reaching d^2 < 5.
  #
  # The cab angle is NOT stated for it, so a straight rig is assumed —
  # flagged on toy#188 rather than buried, because obs3 drops the cab
  # angle entirely and a wrong assumption here would look like a
  # slightly worse arm, not like an error.
  def point_start!
    reset_state!(20.0, 10.0, -2.0, -2.0)
    nil
  end

  # The frontend's 20-lesson curriculum, widening linearly over i/19
  # with L = Ls + Lc = 20. Fills tt_bounds as
  # [x_min, x_max, y_min, y_max, os_min, os_max, cab_min, cab_max],
  # the cab bounds RELATIVE TO THE TRAILER.
  #
  # Transcribed from `createTruckControllerLessons`. Note lesson 0 is
  # NOT a straight rig: its trailer angle is exactly 0 but its cab
  # angle is +/-10 deg. toy#188's prose says "straight rig", which
  # holds for the trailer only.
  def lesson_bounds!(i)
    l = @tt_ls + @tt_lc
    f = i.to_f / 19.0
    d = 180.0 / Math::PI

    @tt_bounds[0] = (0.2 + f * (2.0 - 0.2)) * l
    @tt_bounds[1] = (0.5 + f * (4.0 - 0.5)) * l
    @tt_bounds[2] = (-0.1 + f * (-1.0 + 0.1)) * l
    @tt_bounds[3] = (0.1 + f * (1.0 - 0.1)) * l
    @tt_bounds[4] = (0.0 + f * (-90.0 - 0.0)) / d
    @tt_bounds[5] = (0.0 + f * (90.0 - 0.0)) / d
    @tt_bounds[6] = (-10.0 + f * (-90.0 + 10.0)) / d
    @tt_bounds[7] = (10.0 + f * (90.0 - 10.0)) / d
    nil
  end

  def lesson_count; 20; end

  # Reject-and-redraw on an invalid start. Bounded: `tries` draws, then
  # the last draw stands and the caller can see it with valid?. An
  # unbounded redraw loop would hang on a lesson whose box is entirely
  # invalid instead of reporting it.
  def sample_lesson!(i, tries)
    lesson_bounds!(i)
    n = 0
    while n < tries
      x  = @tt_bounds[0] + next_u * (@tt_bounds[1] - @tt_bounds[0])
      y  = @tt_bounds[2] + next_u * (@tt_bounds[3] - @tt_bounds[2])
      os = @tt_bounds[4] + next_u * (@tt_bounds[5] - @tt_bounds[4])
      oc = os + @tt_bounds[6] + next_u * (@tt_bounds[7] - @tt_bounds[6])
      reset_state!(x, y, oc, os)
      if valid?; return true; end
      n = n + 1
    end
    false
  end

  # The paper's generalisation test: uniform over the region, with an
  # arbitrary jack-knife. Bounds are fields — see the ctor on why the
  # default is [50,100] and not the frontend's [0,100].
  def sample_yard!(tries)
    n = 0
    while n < tries
      x  = @tt_x_min + next_u * (@tt_x_max - @tt_x_min)
      y  = @tt_y_min + next_u * (@tt_y_max - @tt_y_min)
      os = -Math::PI + next_u * 2.0 * Math::PI
      oc = os - 0.5 * Math::PI + next_u * Math::PI
      reset_state!(x, y, oc, os)
      if valid?; return true; end
      n = n + 1
    end
    false
  end

  # ---- provenance ----
  #
  # Every convention that changes a number, on one line, so a stored
  # cell can be re-run. `asin=saturate` is here because landmine 3 is a
  # CHOICE this port made and the frontend made differently.
  def provenance
    s = "plant=truck"
    s = s + " ls=" + @tt_ls.to_s
    s = s + " lc=" + @tt_lc.to_s
    s = s + " r=" + @tt_r.to_s
    s = s + " umax_deg=" + (@tt_umax * 180.0 / Math::PI).to_s
    s = s + " dock=" + (@tt_dock == DOCK_MIDRIG ? "midrig" : "trailer")
    s = s + " wrap=" + (@tt_wrap == WRAP_NONE ? "none" : "pi")
    s = s + " obs=" + obs_dim.to_s
    s = s + " max_steps=" + @tt_max_steps.to_s
    s = s + " asin=saturate"
    s
  end

  # ---- deterministic stream (the tree-wide 31-bit LCG; toy#114) ----

  def seed!(task_seed)
    @tt_s[0] = lcg_seed_state(task_seed)
    nil
  end

  def lcg_seed_state(seed)
    s = ((seed + 104729) * 2654435761) % 2147483647
    if s <= 0
      s = seed + 104729
    end
    w = 0
    while w < 8
      s = (s * 1103515245 + 12345) & 0x7FFFFFFF
      w = w + 1
    end
    s
  end

  def next_u
    s = @tt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @tt_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end
end
