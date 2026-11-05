"""
avoidance.py - fuel-aware collision avoidance using tracked debris motion.

Two pieces:

  Avoider      picks the velocity V to fly (Phobos frame), given the velocity you
               WANT (v_des), your CURRENT velocity (vel) and the debris tracks.
  BurnGovernor turns "velocity I want" into a thruster acceleration command, but
               stays silent (coasts) unless the velocity error is worth a burn.

Why this saves fuel
-------------------
Fuel is roughly proportional to total delta-v (sum of |acceleration| x time), so
the cheapest manoeuvre is the smallest velocity change that keeps you safe, and
the cheapest steady state is to not touch the thrusters at all.

  * v_des itself is always a candidate, so when nothing is in the way the avoider
    returns it EXACTLY (the old version snapped to a coarse grid of directions,
    which made the command jump around as the goal direction slowly rotated).
  * Candidates are small SIDEWAYS offsets from v_des (plus slowing down), tried
    from the smallest dodge upward, so the minimum-effort safe dodge wins.
  * Cost is in physical units (m/s of delta-v):
        cost(V) = |V - vel|            burn needed now to reach V
                + w_aim * |V - v_des|  burn needed later to get back on course
  * Hysteresis: the previous command is kept while it is still safe and nearly as
    cheap as the best option, so noise can't make the command flip-flop.
  * BurnGovernor: a deadband with hysteresis on the velocity error, so tiny
    errors (gravity drift, rounding, sensor noise) never cause thruster pulses.

Safety check (unchanged idea): if we fly at constant V, debris i is at
    p(t) = p0 + t * (v_debris - V)
Closest approach has a closed form, and the keep-out radius is
    diameter/2 + margin + k*sigma_p + k*sigma_v*t   (uncertainty term capped).
`lag_s` models that we can't change velocity instantly: the shuttle keeps its
current velocity for `lag_s` seconds before the new one takes effect.
"""

import numpy as np


class Avoider:
    def __init__(
        self,
        margin=6.0,
        horizon=15.0,
        k_sigma=1.5,
        max_inflate=10.0,
        lag_s=0.5,
        w_aim=1.5,
        switch_margin=0.3,
        speed_scales=(1.0, 0.7, 0.4),
        lateral_fracs=(0.05, 0.10, 0.18, 0.28, 0.40, 0.60, 0.90),
        n_phi=16,
    ):
        self.margin = margin  # clearance beyond debris radius (m); covers shuttle size
        self.horizon = horizon  # look-ahead time (s)
        self.k_sigma = k_sigma  # sigmas of track uncertainty to keep clear
        self.max_inflate = max_inflate  # cap on the uncertainty inflation (m)
        self.lag_s = lag_s  # seconds before a new velocity takes effect
        self.w_aim = w_aim  # weight on "burn needed to get back to v_des" (>1 favours v_des)
        self.switch_margin = (
            switch_margin  # m/s: keep previous command unless best is this much cheaper
        )
        self.scales = np.asarray(speed_scales, float)
        self.lat = np.asarray(lateral_fracs, float)  # sideways offsets as a fraction of speed
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.cphi, self.sphi = np.cos(phis), np.sin(phis)
        self.v_prev = None
        self.last_info = {}

    # ------------------------------------------------------------------ candidates
    def _candidates(self, v_des, vel):
        speed = np.linalg.norm(v_des)
        u = v_des / speed
        a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(u, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        ring = self.cphi[:, None] * e1 + self.sphi[:, None] * e2  # (n_phi,3)
        offsets = np.vstack(
            [np.zeros((1, 3)), (self.lat[:, None, None] * speed * ring[None]).reshape(-1, 3)]
        )
        V = (self.scales[:, None, None] * v_des[None, None, :] + offsets[None]).reshape(-1, 3)
        v_prev = self.v_prev if self.v_prev is not None else v_des
        # fixed slots first: [0] v_des, [1] coast at current velocity,
        # [2] previous command, [3] stop
        return np.vstack([v_des, vel, v_prev, np.zeros(3), V])

    # ------------------------------------------------------------------ main entry
    def pick(self, v_des, snap, vel=None):
        """
        v_des : velocity you would like to fly (Phobos frame)
        snap  : DebrisTracker.snapshot(...)
        vel   : your current velocity (Phobos frame) - needed for the fuel-aware cost
        """
        v_des = np.asarray(v_des, float)
        speed = np.linalg.norm(v_des)
        vel = v_des.copy() if vel is None else np.asarray(vel, float)
        if speed < 1e-3 or len(snap["id"]) == 0:
            self.v_prev = v_des
            self.last_info = {"blocked": False, "n_tracks": len(snap["id"]), "dv": 0.0}
            return v_des

        V = self._candidates(v_des, vel)

        # --- safety: closest approach of every (debris, candidate) pair ---
        P = snap["p_rel"] + self.lag_s * (
            snap["v_abs"] - vel
        )  # where debris is when V takes effect
        W = snap["v_abs"][:, None, :] - V[None, :, :]  # (m,n,3) relative velocity
        WW = np.einsum("mnk,mnk->mn", W, W)
        PW = np.einsum("mnk,mk->mn", W, P)
        t = np.clip(-PW / (WW + 1e-9), 0.0, self.horizon)
        d = np.linalg.norm(P[:, None, :] + t[..., None] * W, axis=2)
        infl = self.k_sigma * (
            snap["sigma_p"][:, None] + snap["sigma_v"][:, None] * (t + self.lag_s)
        )
        R = (snap["diameter"] / 2 + self.margin)[:, None] + np.minimum(infl, self.max_inflate)
        clearance = (d - R).min(axis=0)
        safe = clearance > 0

        # --- choose: cheapest safe candidate in delta-v, with hysteresis ---
        cost = np.linalg.norm(V - vel, axis=1) + self.w_aim * np.linalg.norm(V - v_des, axis=1)
        if safe.any():
            k = int(np.argmin(np.where(safe, cost, np.inf)))
            if safe[2] and cost[2] <= cost[k] + self.switch_margin:
                k = 2  # stick with the previous command
        else:
            k = int(np.argmax(clearance))  # nothing safe: least-bad option
        self.v_prev = V[k].copy()
        self.last_info = {
            "blocked": not bool(safe[0]),  # was flying straight on unsafe?
            "any_safe": bool(safe.any()),
            "clearance": float(clearance[k]),
            "dv": float(np.linalg.norm(V[k] - vel)),
            "n_tracks": len(snap["id"]),
        }
        return V[k]


class BurnGovernor:
    """Velocity command -> acceleration command, with a deadband so small errors coast."""

    def __init__(self, gain=5.0, start_err=0.3, stop_err=0.1):
        self.gain = gain  # same meaning as your accel_gain
        self.start_err = start_err  # m/s: begin a burn when velocity error exceeds this
        self.stop_err = stop_err  # m/s: end the burn once error falls below this
        self.firing = False

    def accel(self, v_cmd, vel):
        err = np.asarray(v_cmd, float) - np.asarray(vel, float)
        e = np.linalg.norm(err)
        if self.firing:
            if e < self.stop_err:
                self.firing = False
        elif e > self.start_err:
            self.firing = True
        return self.gain * err if self.firing else np.zeros(3)
