from dataclasses import dataclass

import numpy as np

from . import config as cfg
from .accel_controller import Burn, maneuver_offset
from .trajectory import lateral_basis, safe_radius


@dataclass
class AvoidPlan:
    burns: list
    dv: float  # total velocity change spent (m/s)
    duration: float
    accel: float
    clearance: float  # smallest (distance - safe radius) over the relevant tracks and the horizon
    feasible: bool
    t_end: float
    path: np.ndarray  # planned shuttle path on the time grid (Phobos frame)
    threat_ids: list


def plan_avoidance(t, p, v, vhat, nominal, snap, conf, ts):
    """Try many two-burn sidesteps from the shuttle's CURRENT state.

    Keep the cheapest one that clears every nearby track.

    nominal: the shuttle path with no maneuver.
    vhat: the flight-path direction the sidestep is perpendicular to.
    """
    if len(snap.pos) == 0:
        return None
    v_lat = v - float(v @ vhat) * vhat  # any sideways velocity the shuttle already has
    right, up = lateral_basis(vhat)

    cand = np.flatnonzero(conf.dmin < cfg.PLAN_RELEVANT_M)
    if len(cand) == 0:
        return None
    idx = cand[np.argsort(conf.gap[cand])][: cfg.PLAN_MAX_TRACKS]

    angles = np.linspace(0.0, 2.0 * np.pi, cfg.PLAN_N_DIRS, endpoint=False)
    dirs = np.cos(angles)[:, None] * right + np.sin(angles)[:, None] * up
    F = np.asarray(cfg.PLAN_ACC_FRACTIONS, dtype=float)
    T = np.asarray(cfg.PLAN_BURN_TIMES_S, dtype=float)
    iD, iF, iT = [
        g.ravel()
        for g in np.meshgrid(
            np.arange(len(dirs)), np.arange(len(F)), np.arange(len(T)), indexing="ij"
        )
    ]
    a_dir, a_mag, tA = dirs[iD], cfg.ACC_MAX * F[iF], T[iT]

    off, tB, dB = maneuver_offset(ts, a_dir, a_mag, tA, v_lat)  # (C,K,3)
    ps = nominal[None] + off
    pd = snap.pos[idx][:, None, :] + snap.vel[idx][:, None, :] * ts[None, :, None]  # (R,K,3)
    safe_t = safe_radius(
        snap.diam[idx][:, None], snap.sig_p[idx][:, None], snap.sig_v[idx][:, None], ts[None, :]
    )
    gap = np.linalg.norm(pd[None] - ps[:, None], axis=3) - safe_t[None]  # (C,R,K)
    clear = gap.min(axis=(1, 2))
    dv = a_mag * (tA + tB)
    feasible = clear > cfg.PLAN_EXTRA_CLEAR_M

    k = int(np.argmin(np.where(feasible, dv, np.inf))) if feasible.any() else int(np.argmax(clear))
    burns = [Burn(t, t + float(tA[k]), a_mag[k] * a_dir[k])]
    if tB[k] > 0.05:
        burns.append(Burn(t + float(tA[k]), t + float(tA[k] + tB[k]), a_mag[k] * dB[k]))
    return AvoidPlan(
        burns=burns,
        dv=float(dv[k]),
        duration=float(tA[k] + tB[k]),
        accel=float(a_mag[k]),
        clearance=float(clear[k]),
        feasible=bool(feasible[k]),
        t_end=t + float(tA[k] + tB[k]),
        path=nominal + off[k],
        threat_ids=[int(i) for i in snap.ids[idx[conf.threat[idx]]]],
    )
