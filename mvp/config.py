# ---------------- Shuttle ----------------
CRUISE_SPEED = 17  # m/s
CHECK_PERIOD_S = 10.0  # how often for speed check
SETTLE_S = 1.0  # after an avoidance maneuver, run a check this long afterwards
RETRY_S = 2.0  # if the corrected path is blocked, retry the check after this long
ARRIVE_RADIUS_M = 5.0  # stop issuing corrections inside this distance
VEL_TOL_ON = 0.3  # m/s velocity error that triggers a correction burn
VEL_TOL_OFF = 0.05  # m/s velocity error at which the correction burn stops
VEL_GAIN = 1.5  # 1/s, proportional gain of the velocity-hold burn
ACC_MAX = 0.5  # m/s^2 PLACEHOLDER: set from the thruster limits on the vehicle specs page

# ---------------- Attitude (view only: turns the nose toward the target) ----------------
REORIENT_SHUTTLE = (
    True  # master switch. False = no rotation commands at all, the shuttle rotates however it likes
)
REORIENT_START_DEG = 5.0  # start turning when the nose is this far off the target
REORIENT_STOP_DEG = (
    2.0  # stop turning when within this (the gap avoids constant fuel-burning corrections)
)
KP_ATT = 0.05  # rad/s^2 per rad of pointing error (untuned)
KD_ATT = 0.40  # rad/s^2 per rad/s of angular velocity (untuned)
ANG_ACC_MAX = 0.10  # rad/s^2 per axis

# ---------------- Radar / estimator ----------------
RADAR_N = 500
SIGMA_POS = 2.0  # radar position noise (m), from the spec
PRIOR_SIGMA_VEL = 2.0  # m/s, prior uncertainty of a new debris track's velocity (a guess)
ASSOC_GATE_M = 12.0  # base distance gate for matching a radar contact to a track
ASSOC_GATE_SIGMAS = 3.0
MAX_COAST_S = 10.0  # drop a track not seen for this long
MIN_UPDATES_FOR_THREAT = 2
MAX_EXTRAP_S = 2.0  # never extrapolate tracks further than this past the last scan
SHUTTLE_HISTORY_S = 15.0

# ---------------- Alpha-beta tracker ----------------
ALPHA = 0.2  # position gain
BETA = 0.02  # velocity gain (applied as BETA/dt)
GROWING_GAIN = (
    True  # use alpha = max(ALPHA, 1/(n+1)) so new tracks converge fast, then settle at ALPHA
)
# True: seed velocity from the radar range rate and keep correcting it.
# False: pure position alpha-beta.
USE_RANGE_RATE = True
RATE_GAIN = 0.5  # gain on the range-rate correction (only if USE_RANGE_RATE)
SIG_P = 1.0  # assumed position error of a settled track (m)
SIG_V = 0.4  # assumed velocity error of a settled track (m/s)
SETTLE_UPDATES = 5  # tracks with fewer updates than this are treated as unsettled (PRIOR_SIGMA_VEL)

# ---------------- Collision prediction ----------------
SHUTTLE_RADIUS_M = 3.36  # half the box diagonal: 0.5 * sqrt(2^2 + 4^2 + 5^2) = 3.354 m, rounded up
SAFETY_M = 0.5
K_SIGMA = 2.0  # safety margin = K_SIGMA * predicted position uncertainty ...
MARGIN_CAP_M = 6.0  # ... capped at this
HORIZON_S = 25.0
MIN_AVOID_SPEED = 1.0

# ---------------- Avoidance planner ----------------
PLAN_MAX_TRACKS = 40
PLAN_RELEVANT_M = 80.0  # only tracks passing within this of the nominal path are considered
PLAN_DT_S = 0.25
PLAN_N_DIRS = 12
PLAN_ACC_FRACTIONS = (0.5, 1.0)  # fractions of ACC_MAX to try

# ---------------- Threat trigger and re-planning ----------------
THREAT_LEAD_S = 15.0  # only react to a collision predicted within this many seconds
THREAT_MAX_SIGV = 1.0  # m/s: a track must be this well known to trigger avoidance...
IMMINENT_S = 4.0  # ...unless the collision is closer than this many seconds away
REPLAN_PERIOD_S = 0.5  # while avoiding, re-evaluate and possibly re-plan this often
REPLAN_MIN_GAIN_M = (
    1.0  # a new plan must beat the current one's clearance by this much to replace it
)
PLAN_BURN_TIMES_S = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)  # first-burn durations tried
PLAN_EXTRA_CLEAR_M = 1.0  # extra clearance a plan must have beyond the safety radius

# ---------------- Display ----------------
HUD_EVERY_N = 4  # redraw every N loop iterations
HUD_HALF_FOV_DEG = 40.0
HUD_FLIP_H = True
HUD_FLIP_V = True
SHOW_THREAT_IDS = True
DISPLAY_MAX_W, DISPLAY_MAX_H = 1000, 640
