from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config as cfg


class ShuttleState:
    """The shuttle's own state in the Phobos frame, with a short history for timing radar scans."""

    def __init__(self):
        self.hist = deque()
        self.t = 0.0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def update(self, t, pos, vel):
        self.t = float(t)
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.hist.append((self.t, self.pos.copy(), self.vel.copy()))
        while self.hist and self.t - self.hist[0][0] > cfg.SHUTTLE_HISTORY_S:
            self.hist.popleft()

    def at(self, t):
        """Return (pos, vel) at t time. If t is not at recorded time, interpolated from history"""
        h = list(self.hist)
        if not h:
            return self.pos, self.vel
        if t <= h[0][0]:
            return h[0][1], h[0][2]
        if t >= h[-1][0]:
            return h[-1][1], h[-1][2]
        for k in range(len(h) - 1, 0, -1):
            t0, p0, v0 = h[k - 1]
            t1, p1, v1 = h[k]
            if t0 <= t <= t1:
                w = (t - t0) / max(t1 - t0, 1e-9)
                return p0 + w * (p1 - p0), v0 + w * (v1 - v0)
        return self.pos, self.vel

    def trail(self, seconds: float) -> np.ndarray:
        pts = [p for (t, p, v) in self.hist if self.t - t <= seconds]
        return np.array(pts) if pts else np.zeros((0, 3))


@dataclass
class Snapshot:
    """Debris tracks propagated to 'now'."""

    pos: np.ndarray  # (N,3) Phobos frame
    vel: np.ndarray  # (N,3)
    sig_p: np.ndarray  # (N,) position std (m)
    sig_v: np.ndarray  # (N,) velocity std (m/s)
    diam: np.ndarray  # (N,)
    n: np.ndarray  # (N,) number of radar updates
    ids: np.ndarray  # (N,)
    age: np.ndarray  # (N,) seconds since last radar update


def _sigmas(n):
    """Assumed position/velocity error for a track with n updates.

    Starts at the prior, settles to steady state.
    """
    k = np.exp(-(np.asarray(n, dtype=float) - 1.0) / cfg.CONV_UPDATES)
    return (
        cfg.SIGP_STEADY + (cfg.SIGMA_POS - cfg.SIGP_STEADY) * k,
        cfg.SIGV_STEADY + (cfg.PRIOR_SIGMA_VEL - cfg.SIGV_STEADY) * k,
    )


class DebrisTracks:
    """Alpha-beta tracker: one position and velocity per debris, fixed gains, no covariance.

    Per scan, for each matched track:
        p_pred   = p + v * dt
        residual = z - p_pred
        p        = p_pred + alpha * residual
        v        = v + (beta / dt) * residual
    Uncertainty is not estimated. It is a fixed assumption used only for the matching gate
    and the safety margin.
    """

    def __init__(self):
        self.t = None
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.n = np.zeros(0, dtype=int)
        self.last_seen = np.zeros(0)
        self.diam = np.zeros(0)
        self.ids = np.zeros(0, dtype=int)
        self.next_id = 0
        self.scans = 0

    # ---------- public ----------
    def ingest(self, raw, shuttle: ShuttleState) -> bool:
        """Process a radar read if it holds new data. Returns True if the tracks were updated."""
        if not raw:
            return False
        A = np.asarray([np.asarray(r, dtype=float) for r in raw])  # (M,3,3)
        rel = A[:, 0, :]
        valid = np.all(np.isfinite(rel), axis=1) & (np.linalg.norm(rel, axis=1) > 1e-6)
        if not valid.any():
            return False
        A = A[valid]
        rel, rate, diam = A[:, 0, :], A[:, 1, 1], A[:, 1, 2]
        scan_t = float(np.max(A[:, 2, 0]))
        if self.t is not None and scan_t <= self.t + 1e-9:
            return False  # same scan as before
        p_s, v_s = shuttle.at(scan_t)
        self._step(scan_t, p_s + rel, rel, rate, diam, v_s)
        self.scans += 1
        return True

    def _sigmas(self, n, age):
        sig_v = np.where(n < cfg.SETTLE_UPDATES, cfg.PRIOR_SIGMA_VEL, cfg.SIG_V)
        sig_p = np.sqrt(cfg.SIG_P**2 + (sig_v * np.maximum(age, 0.0)) ** 2)
        return sig_p, sig_v

    def snapshot(self, t_now: float) -> Snapshot:
        dt = 0.0 if self.t is None else float(np.clip(t_now - self.t, 0.0, cfg.MAX_EXTRAP_S))
        age = t_now - self.last_seen
        sig_p, sig_v = self._sigmas(self.n, age)
        return Snapshot(
            pos=self.p + self.v * dt,
            vel=self.v.copy(),
            sig_p=sig_p,
            sig_v=sig_v,
            diam=self.diam.copy(),
            n=self.n.copy(),
            ids=self.ids.copy(),
            age=age,
        )

    # ---------- internals ----------
    def _step(self, t, meas, rel, rate, diam, v_s):
        M = len(meas)
        if self.t is None or len(self.p) == 0:
            self.t = t
            self._spawn(np.arange(M), meas, rel, rate, v_s, diam, t)
            return
        dt = t - self.t
        self.p = self.p + self.v * dt  # predict
        self.t = t
        ti, mi = self._associate(meas, t)
        if len(ti):
            self._update(ti, meas[mi], rel[mi], rate[mi], v_s, dt)
            self.n[ti] += 1
            self.last_seen[ti] = t
            self.diam[ti] = 0.7 * self.diam[ti] + 0.3 * diam[mi]
        unmatched = np.setdiff1d(np.arange(M), mi)
        if len(unmatched):
            self._spawn(unmatched, meas, rel, rate, v_s, diam, t)
        keep = (t - self.last_seen) <= cfg.MAX_COAST_S
        if not keep.all():
            self.p, self.v, self.n = self.p[keep], self.v[keep], self.n[keep]
            self.last_seen, self.diam, self.ids = (
                self.last_seen[keep],
                self.diam[keep],
                self.ids[keep],
            )

    def _associate(self, meas, t):
        N, M = len(self.p), len(meas)
        d = np.linalg.norm(self.p[:, None, :] - meas[None, :, :], axis=2)  # (N,M)
        sig_p, _ = self._sigmas(self.n, t - self.last_seen)
        gate = cfg.ASSOC_GATE_M + cfg.ASSOC_GATE_SIGMAS * sig_p
        ii, jj = np.nonzero(d < gate[:, None])
        if len(ii) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        used_t, used_m = np.zeros(N, bool), np.zeros(M, bool)
        ti, mi = [], []
        for k in np.argsort(d[ii, jj]):  # greedy, closest pairs first
            a, b = ii[k], jj[k]
            if used_t[a] or used_m[b]:
                continue
            used_t[a] = used_m[b] = True
            ti.append(a)
            mi.append(b)
        return np.array(ti, dtype=int), np.array(mi, dtype=int)

    def _update(self, ti, z, rel, rate, v_s, dt):
        r = z - self.p[ti]  # position residual
        alpha = np.full(len(ti), cfg.ALPHA)
        if cfg.GROWING_GAIN:
            alpha = np.maximum(alpha, 1.0 / (self.n[ti] + 1.0))  # n=1 -> 0.5, n=4 -> 0.2
        self.p[ti] += alpha[:, None] * r
        self.v[ti] += (cfg.BETA / max(dt, 0.05)) * r
        if cfg.USE_RANGE_RATE:  # optional line-of-sight velocity correction
            u = rel / np.linalg.norm(rel, axis=1)[:, None]
            innov = (rate + u @ v_s) - np.einsum("ij,ij->i", u, self.v[ti])
            self.v[ti] += (cfg.RATE_GAIN * innov)[:, None] * u

    def _spawn(self, idx, meas, rel, rate, v_s, diam, t):
        K = len(idx)
        v0 = np.zeros((K, 3))
        if cfg.USE_RANGE_RATE:  # line-of-sight velocity from the radar, sideways unknown
            u = rel[idx] / np.linalg.norm(rel[idx], axis=1)[:, None]
            v0 = (rate[idx] + u @ v_s)[:, None] * u
        self.p = np.vstack([self.p, meas[idx]])
        self.v = np.vstack([self.v, v0])
        self.n = np.concatenate([self.n, np.ones(K, dtype=int)])
        self.last_seen = np.concatenate([self.last_seen, np.full(K, t)])
        self.diam = np.concatenate([self.diam, diam[idx]])
        self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + K)])
        self.next_id += K
