from dataclasses import dataclass

import numpy as np

from . import config as cfg


def lateral_basis(f):
    """Two unit vectors (right, up) perpendicular to the unit direction f."""
    z = np.array([0.0, 0.0, 1.0])
    if abs(float(f @ z)) > 0.95:
        z = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, z)
    right /= np.linalg.norm(right)
    return right, np.cross(right, f)


def safe_radius(diam, sig_p, sig_v, t):
    """Collision centre-to-centre distance, including tracking uncertainty at time t."""
    margin = np.minimum(cfg.K_SIGMA * np.sqrt(sig_p**2 + (sig_v * t) ** 2), cfg.MARGIN_CAP_M)
    return cfg.SHUTTLE_RADIUS_M + cfg.SAFETY_M + 0.55 * diam + margin


@dataclass
class StraightLine:
    speed: float
    eta: float  # seconds to closest approach to the target (inf if not heading toward it)
    miss: float  # distance from the target at that closest approach
    heading_err_deg: float  # angle between the velocity and the direction to the target


def straight_line(p, v, target) -> StraightLine:
    to_t = target - p
    dist = float(np.linalg.norm(to_t))
    speed = float(np.linalg.norm(v))
    if speed < 0.1:
        return StraightLine(speed, float("inf"), dist, 0.0)
    t_c = float(to_t @ v) / (speed * speed)
    miss = float(np.linalg.norm(p + v * max(t_c, 0.0) - target))
    cosang = float(np.clip((v @ to_t) / (speed * max(dist, 1e-9)), -1.0, 1.0))
    return StraightLine(
        speed, t_c if t_c > 0 else float("inf"), miss, float(np.degrees(np.arccos(cosang)))
    )


def propagate(p0, v0, ts, v_des=None, along=None, burns=(), t_now=0.0):
    """Shuttle path on ts (ts[0] = now), under velocity hold plus any timed burns.

    v_des: velocity target of the hold (None = coast).
    along: if given, the hold only acts along this unit vector.
    """
    K = len(ts)
    p = np.empty((K, 3))
    v = np.empty((K, 3))
    p[0], v[0] = p0, v0
    for k in range(K - 1):
        dt = ts[k + 1] - ts[k]
        a = np.zeros(3)
        if v_des is not None:
            err = v_des - v[k]
            if along is not None:
                err = float(err @ along) * along
            if np.linalg.norm(err) >= cfg.VEL_TOL_OFF:
                a = cfg.VEL_GAIN * err
                m = float(np.linalg.norm(a))
                if m > cfg.ACC_MAX:
                    a = a * (cfg.ACC_MAX / m)
        tk = t_now + ts[k]
        for b in burns:
            if b.t0 <= tk < b.t1:
                a = a + b.accel
        p[k + 1] = p[k] + v[k] * dt + 0.5 * a * dt * dt
        v[k + 1] = v[k] + a * dt
    return p, v


@dataclass
class Conflicts:
    t_star: np.ndarray  # time of worst approach for each track
    d_star: np.ndarray  # centre-to-centre distance at that time
    safe: np.ndarray  # distance below which that counts as a collision
    threat: np.ndarray  # bool
    gap: np.ndarray  # worst (distance - safe radius) over the horizon; negative = collision
    dmin: np.ndarray  # closest centre distance to the path over the horizon


def predict_conflicts(path, snap, ts) -> Conflicts:
    """Compare the shuttle path with every debris track (straight line) at each time step."""
    N = len(snap.pos)
    if N == 0:
        z = np.zeros(0)
        return Conflicts(z, z, z, np.zeros(0, bool), z, z)
    pd = snap.pos[:, None, :] + snap.vel[:, None, :] * ts[None, :, None]  # (N,K,3)
    d = np.linalg.norm(pd - path[None, :, :], axis=2)  # (N,K)
    safe = safe_radius(snap.diam[:, None], snap.sig_p[:, None], snap.sig_v[:, None], ts[None, :])
    gap = d - safe
    r = np.arange(N)
    k = np.argmin(gap, axis=1)
    t_hit = ts[k]
    settled = (snap.n >= cfg.MIN_UPDATES_FOR_THREAT) & (
        (snap.sig_v <= cfg.THREAT_MAX_SIGV) | (t_hit < cfg.IMMINENT_S)
    )
    threat = (gap[r, k] < 0.0) & settled & (t_hit <= cfg.THREAT_LEAD_S)
    return Conflicts(t_hit, d[r, k], safe[r, k], threat, gap[r, k], d.min(axis=1))
