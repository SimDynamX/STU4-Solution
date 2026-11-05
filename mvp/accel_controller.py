from dataclasses import dataclass

import numpy as np

from . import config as cfg

# ---------- acceleration <-> position ----------
# A "sidestep" is a symmetric burst: accelerate by +a for tau seconds, then by -a for tau seconds.
# It moves the shuttle sideways by d = a * tau^2 and leaves its sideways velocity at zero.


def displacement_for_accel(a, tau):
    return a * tau * tau


def accel_for_displacement(d, tau):
    return d / (tau * tau)


def duration_for_displacement(d, a):
    return np.sqrt(d / a)


def displacement_profile(a, tau, t):
    """Sideways displacement at time t (from the start of the burst) for a symmetric burst."""
    u = np.clip(t, 0.0, 2.0 * tau)
    return np.where(u <= tau, 0.5 * a * u * u, a * tau * tau - 0.5 * a * (2.0 * tau - u) ** 2)


@dataclass
class Burn:
    t0: float
    t1: float
    accel: np.ndarray  # Phobos-frame axes, m/s^2


def plan_sidestep(direction, d, a, t_start):
    """Position -> acceleration: the two burns that move the shuttle d metres along `direction`."""
    tau = float(duration_for_displacement(d, a))
    return [
        Burn(t_start, t_start + tau, a * direction),
        Burn(t_start + tau, t_start + 2 * tau, -a * direction),
    ]


class BurnSequencer:
    """Plays timed burns against the sim clock."""

    def __init__(self):
        self.burns = []

    def set(self, burns):
        self.burns = list(burns)

    def clear(self):
        self.burns = []

    def accel_at(self, t):
        a = np.zeros(3)
        for b in self.burns:
            if b.t0 <= t < b.t1:
                a = a + b.accel
        return a


class VelocityHold:
    """Closed-loop burn toward a velocity setpoint.

    Active until the error is within tolerance, then it stops.
    """

    def __init__(self):
        self.v_des = np.zeros(3)
        self.active = False

    def command(self, v_des):
        self.v_des = np.asarray(v_des, dtype=float)
        self.active = True

    def cancel(self):
        self.active = False

    def step(self, v):
        if not self.active:
            return np.zeros(3)
        err = self.v_des - v
        if np.linalg.norm(err) < cfg.VEL_TOL_OFF:
            self.active = False
            return np.zeros(3)
        a = cfg.VEL_GAIN * err
        mag = np.linalg.norm(a)
        return a * (cfg.ACC_MAX / mag) if mag > cfg.ACC_MAX else a


def maneuver_offset(ts, a_dir, a_mag, tA, v_lat):
    """Offset from the no-maneuver path for a two-burn sidestep, vectorised over C candidates.

    Burn A: a_mag along a_dir for tA seconds. Burn B: a_mag along whatever direction
    cancels the lateral velocity (v_lat + the velocity added by burn A), for as long as that takes.
    Returns (offset (C,K,3), tB (C,), dB (C,3)).
    """
    dvA = (a_mag * tA)[:, None] * a_dir  # (C,3) velocity added by burn A
    dvB = -(v_lat[None, :] + dvA)  # (C,3) velocity burn B must add
    mB = np.linalg.norm(dvB, axis=1)
    tB = mB / a_mag
    dB = dvB / np.maximum(mB, 1e-9)[:, None]
    t = ts[None, :, None]  # (1,K,1)
    tA_, tB_ = tA[:, None, None], tB[:, None, None]
    a1 = a_mag[:, None, None] * a_dir[:, None, :]  # (C,1,3)
    a2 = a_mag[:, None, None] * dB[:, None, :]
    u1 = np.clip(t, 0.0, tA_)  # time spent in burn A
    u2 = np.clip(t - tA_, 0.0, tB_)  # time spent in burn B
    u3 = np.maximum(t - tA_ - tB_, 0.0)  # coasting afterwards
    vA = dvA[:, None, :]
    vB = vA + dvB[:, None, :]  # = -v_lat: the lateral velocity ends at zero
    off = 0.5 * a1 * u1**2 + vA * u2 + 0.5 * a2 * u2**2 + vB * u3
    return off, tB, dB
