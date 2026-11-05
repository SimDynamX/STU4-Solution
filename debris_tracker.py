"""
debris_tracker.py - estimate debris velocities from the shuttle's noisy radar.

Idea
----
The radar gives each contact's position RELATIVE to the shuttle (Phobos axes)
plus its range rate. We:
  1. convert to ABSOLUTE Phobos-frame positions (shuttle position + radar offset),
     so the debris velocity doesn't depend on how the shuttle is moving;
  2. match each new contact to an existing track (nearest-neighbour with a
     statistical gate, plus a diameter check - the radar has no object IDs);
  3. run a small Kalman filter per track, state = [position(3), velocity(3)],
     constant-velocity model, with two measurement updates:
        - position (noise ~2 m)
        - range rate. It is the debris velocity projected on the line of sight:
              v_debris . u = range_rate + v_shuttle . u        (noise ~0.02 m/s)
     which pins down the radial velocity almost immediately; the sideways
     velocity is then learned from how the position changes over time.

Limits: debris gravity/rotation effects are lumped into process noise; tracks
can be lost or swapped in dense fields; tangential velocity needs a few
seconds of observation to converge.
"""

import numpy as np


class Track:
    _next_id = 0

    def __init__(self, x, P, diameter, t):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = x  # [px,py,pz,vx,vy,vz], absolute Phobos-frame
        self.P = P  # 6x6 covariance
        self.diameter = diameter
        self.t = t  # time the state refers to
        self.last_seen = t
        self.hits = 1


class DebrisTracker:
    def __init__(
        self,
        pos_sigma=2.0,
        rr_sigma=0.02,
        accel_sigma=0.02,
        init_tangential_vel_sigma=10.0,
        gate_chi2=16.0,
        diameter_tol=0.3,
        drop_after_s=5.0,
        min_hits=3,
    ):
        self.pos_sigma = pos_sigma  # radar position noise (wiki: 2 m)
        self.rr_sigma = rr_sigma  # radar range-rate noise (wiki: 0.02 m/s)
        self.accel_sigma = accel_sigma  # unmodelled debris accel, m/s^2 (tune)
        self.v0_sigma = init_tangential_vel_sigma  # prior on unknown sideways speed (tune)
        self.gate_chi2 = gate_chi2  # Mahalanobis^2 gate, 3 dof (16.3 ~ 99.9%)
        self.diameter_tol = diameter_tol  # max fractional diameter mismatch
        self.drop_after_s = drop_after_s  # delete tracks unseen this long
        self.min_hits = min_hits  # hits before a track is "confirmed"
        self.tracks: list[Track] = []
        self.t_last = -np.inf

    # ------------------------------------------------------------------ public
    def update(self, shuttle_pos, shuttle_vel, debris_info):
        """
        shuttle_pos, shuttle_vel : shuttle state in the Phobos frame (3-vectors)
        debris_info              : output of GetNearestDebrisLocations(N)
        Returns the list of confirmed tracks.
        """
        sp = np.asarray(shuttle_pos, float)
        sv = np.asarray(shuttle_vel, float)
        if len(debris_info) == 0:
            return self.confirmed()

        t = max(float(d[2, 0]) for d in debris_info)  # radar measurement time
        if t <= self.t_last + 1e-9:  # no fresh radar data yet
            return self.confirmed()
        self.t_last = t

        # --- measurements (absolute positions, diameters, projected rates) ---
        meas = []
        for d in debris_info:
            rel = np.array(d[0, :], float)
            meas.append({"p": sp + rel, "rel": rel, "vr": float(d[1, 1]), "diam": float(d[1, 2])})

        for tr in self.tracks:
            self._predict(tr, t)

        # --- association: greedy on Mahalanobis distance, with gates ---
        R = self.pos_sigma**2 * np.eye(3)
        pairs = []
        for i, tr in enumerate(self.tracks):
            Sinv = np.linalg.inv(tr.P[:3, :3] + R)
            for j, m in enumerate(meas):
                if abs(m["diam"] - tr.diameter) > self.diameter_tol * max(tr.diameter, 0.5):
                    continue
                r = m["p"] - tr.x[:3]
                d2 = (r @ Sinv @ r).item()
                if d2 < self.gate_chi2:
                    pairs.append((d2, i, j))
        pairs.sort()
        used_t, used_m = set(), set()
        for _d2, i, j in pairs:
            if i in used_t or j in used_m:
                continue
            used_t.add(i)
            used_m.add(j)
            self._correct(self.tracks[i], meas[j], sp, sv, t)

        # --- new tracks for unmatched contacts ---
        for j, m in enumerate(meas):
            if j not in used_m:
                self.tracks.append(self._new_track(m, sp, sv, t))

        # --- drop stale tracks ---
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.drop_after_s]
        return self.confirmed()

    def confirmed(self):
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]

    def snapshot(self, t_now, shuttle_pos, shuttle_vel=None, max_extrap_s=1.0):
        """
        Every live track (confirmed or not) as arrays, for the avoidance layer.
        Positions are relative to the shuttle (Phobos axes) and nudged forward to
        t_now (sim seconds since start, same clock as the radar timestamps).
        Velocities are ABSOLUTE (Phobos frame). sigma_p / sigma_v are the worst-axis
        1-sigma uncertainties.
        """
        sp = np.asarray(shuttle_pos, float)
        n = len(self.tracks)
        snap = {
            "id": np.zeros(n, int),
            "p_rel": np.zeros((n, 3)),
            "v_abs": np.zeros((n, 3)),
            "diameter": np.zeros(n),
            "sigma_p": np.zeros(n),
            "sigma_v": np.zeros(n),
        }
        for i, tr in enumerate(self.tracks):
            dt = float(np.clip(t_now - tr.t, 0.0, max_extrap_s))
            snap["id"][i] = tr.id
            snap["p_rel"][i] = tr.x[:3] + tr.x[3:] * dt - sp
            snap["v_abs"][i] = tr.x[3:]
            snap["diameter"][i] = tr.diameter
            snap["sigma_p"][i] = np.sqrt(np.max(np.diag(tr.P)[:3]))
            snap["sigma_v"][i] = np.sqrt(np.max(np.diag(tr.P)[3:]))
        return snap

    @staticmethod
    def relative_state(track, shuttle_pos, shuttle_vel):
        """Debris position and velocity relative to the shuttle (Phobos axes)."""
        return track.x[:3] - np.asarray(shuttle_pos, float), track.x[3:] - np.asarray(
            shuttle_vel, float
        )

    # ----------------------------------------------------------------- private
    def _new_track(self, m, sp, sv, t):
        rng = max(np.linalg.norm(m["rel"]), 1e-6)
        u = m["rel"] / rng
        v_rad = m["vr"] + sv @ u  # debris velocity along line of sight
        x = np.concatenate([m["p"], v_rad * u])  # sideways velocity unknown -> 0
        P = np.zeros((6, 6))
        P[:3, :3] = self.pos_sigma**2 * np.eye(3)
        P[3:, 3:] = self.v0_sigma**2 * (np.eye(3) - np.outer(u, u)) + self.rr_sigma**2 * np.outer(
            u, u
        )
        return Track(x, P, m["diam"], t)

    def _predict(self, tr, t):
        dt = t - tr.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        q = self.accel_sigma**2
        Q = np.zeros((6, 6))
        Q[:3, :3] = q * dt**3 / 3 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = q * dt**2 / 2 * np.eye(3)
        Q[3:, 3:] = q * dt * np.eye(3)
        tr.x = F @ tr.x
        tr.P = F @ tr.P @ F.T + Q
        tr.t = t

    def _correct(self, tr, m, sp, sv, t):
        I6 = np.eye(6)
        # 1) position update
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        S = H @ tr.P @ H.T + self.pos_sigma**2 * np.eye(3)
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (m["p"] - H @ tr.x)
        A = I6 - K @ H
        tr.P = A @ tr.P @ A.T + K @ (self.pos_sigma**2 * np.eye(3)) @ K.T

        # 2) range-rate update: v_debris . u = range_rate + v_shuttle . u
        rel = tr.x[:3] - sp
        rng = max(np.linalg.norm(rel), 1e-6)
        u = rel / rng
        z = m["vr"] + sv @ u
        # line-of-sight error (~pos_sigma/range rad) leaks sideways speed into this
        # measurement, so inflate its noise with the current relative-speed estimate
        v_rel = np.linalg.norm(tr.x[3:] - sv)
        r2 = self.rr_sigma**2 + (self.pos_sigma / rng * v_rel) ** 2
        Hr = np.concatenate([np.zeros(3), u]).reshape(1, 6)
        s = (Hr @ tr.P @ Hr.T).item() + r2
        k = (tr.P @ Hr.T) / s
        tr.x = tr.x + (k * (z - (Hr @ tr.x).item())).ravel()
        A = I6 - k @ Hr
        tr.P = A @ tr.P @ A.T + (k * r2) @ k.T

        # bookkeeping
        tr.hits += 1
        tr.last_seen = t
        alpha = max(1.0 / tr.hits, 0.1)
        tr.diameter = (1 - alpha) * tr.diameter + alpha * m["diam"]


# ------------------------------------------------------------------------------
# HOW TO USE IN Controller.py
#
#   from debris_tracker import DebrisTracker
#   tracker = DebrisTracker()                       # before the loop
#
#   # inside the loop:
#   debris = GetNearestDebrisLocations(50)
#   tracks = tracker.update(loc, vel, debris)       # loc, vel = shuttle state, Phobos frame
#   for tr in tracks:
#       p_rel, v_rel = tracker.relative_state(tr, loc, vel)
#       # tr.id, tr.x[3:] (absolute debris velocity), np.sqrt(np.diag(tr.P)[3:])
# ------------------------------------------------------------------------------
