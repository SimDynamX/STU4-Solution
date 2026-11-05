import numpy as np

from . import config as cfg
from .accel_controller import BurnSequencer, VelocityHold
from .avoidance import plan_avoidance
from .trajectory import predict_conflicts, propagate, straight_line


def _unit(x):
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-9 else np.zeros(3)


def _grid(dist, speed):
    horizon = float(np.clip(dist / max(speed, 1.0) + 2.0, 4.0, cfg.HORIZON_S))
    return np.arange(0.0, horizon + 1e-9, cfg.PLAN_DT_S)


class Guidance:
    """CRUISE: fly the straight line, recheck every CHECK_PERIOD_S.
    AVOID: run the sidestep burns, re-check the predicted path every REPLAN_PERIOD_S
    and re-plan if it stops being clear. The shuttle's path is always predicted from
    its actual state, including the velocity-hold burn and any sidestep burns.
    """

    def __init__(self):
        self.mode = "CRUISE"
        self.vhold = VelocityHold()
        self.seq = BurnSequencer()
        self.v_des = np.zeros(3)
        self.next_check = None
        self.next_replan = 0.0
        self.plan = None
        self.vhat0 = np.array([1.0, 0.0, 0.0])
        self.avoid_end = 0.0
        self.n_replans = 0
        self.note = "starting"

    def _set_velocity_target(self, p, target):
        self.v_des = cfg.CRUISE_SPEED * _unit(target - p)
        self.vhold.command(self.v_des)

    def _recorrect(self, t, p, v, snap, target, dist):
        self.next_check = t + cfg.CHECK_PERIOD_S
        if dist < cfg.ARRIVE_RADIUS_M:
            self.note = "at target"
            return
        v_des = cfg.CRUISE_SPEED * (target - p) / dist
        if np.linalg.norm(v_des - v) < cfg.VEL_TOL_ON and not self.vhold.active:
            self.v_des = v_des
            self.note = "check: on line, no burn"
            return
        # Check the corrected line (flown from the actual state) before burning onto it.
        ts = _grid(dist, cfg.CRUISE_SPEED)
        path = propagate(p, v, ts, v_des=v_des, t_now=t)[0]
        if predict_conflicts(path, snap, ts).threat.any():
            self.next_check = t + cfg.RETRY_S
            self.note = "check: corrected line blocked, holding"
            return
        self._set_velocity_target(p, target)
        self.note = "check: correction burn"

    def _plan(self, t, p, v, snap, ts, conf, target, force):
        """Start an avoidance, or replace the running one. Returns True if a plan was adopted."""
        if self.mode == "AVOID":
            vhat = self.vhat0
        elif self.vhold.active:
            vhat = _unit(self.v_des)
        elif np.linalg.norm(v) >= cfg.MIN_AVOID_SPEED:
            vhat = _unit(v)
        else:
            vhat = _unit(target - p)
        v_des = self.v_des if self.vhold.active else None
        nominal = propagate(p, v, ts, v_des=v_des, along=vhat, t_now=t)[
            0
        ]  # no sidestep, along-track hold only
        plan = plan_avoidance(t, p, v, vhat, nominal, snap, conf, ts)
        if plan is None:
            return False
        if self.mode == "AVOID":
            if not force and plan.clearance <= float(conf.gap.min()) + cfg.REPLAN_MIN_GAIN_M:
                return False  # the new plan isn't clearly better: keep flying the current one
            self.n_replans += 1
        else:
            self.n_replans = 0
        self.mode, self.vhat0, self.plan = "AVOID", vhat, plan
        self.avoid_end = plan.t_end
        self.seq.set(plan.burns)
        self.note = (
            f"avoiding {plan.threat_ids[:3]}"
            + ("" if plan.feasible else " (best effort)")
            + (f" [replan {self.n_replans}]" if self.n_replans else "")
        )
        return True

    def update(self, t, shuttle, snap, target):
        p, v = shuttle.pos, shuttle.vel
        dist = float(np.linalg.norm(target - p))

        if self.next_check is None:  # first call: the initial burn
            self._set_velocity_target(p, target)
            self.next_check = t + cfg.CHECK_PERIOD_S
            self.note = "initial burn"

        # Predict from its ACTUAL state (velocity-hold burn + any sidestep burns), then test it.
        avoiding = self.mode == "AVOID"
        speed_hint = max(
            float(np.linalg.norm(v)),
            float(np.linalg.norm(self.v_des)) if self.vhold.active else 0.0,
        )
        ts = _grid(dist, speed_hint)
        v_des = self.v_des if self.vhold.active else None
        path = propagate(
            p,
            v,
            ts,
            v_des=v_des,
            along=self.vhat0 if avoiding else None,
            burns=self.seq.burns if avoiding else (),
            t_now=t,
        )[0]
        conf = predict_conflicts(path, snap, ts)
        line = straight_line(p, self.v_des if self.vhold.active else v, target)

        if avoiding:
            done = t >= self.avoid_end
            if conf.threat.any() and (done or t >= self.next_replan):
                self._plan(t, p, v, snap, ts, conf, target, force=done)
                self.next_replan = t + cfg.REPLAN_PERIOD_S
            elif done and not conf.threat.any():  # burns finished and the path is clear
                self.mode, self.plan = "CRUISE", None
                self.seq.clear()
                self.next_check = t + cfg.SETTLE_S
                self.note = "avoidance done, recorrecting"
        else:
            started = conf.threat.any() and self._plan(t, p, v, snap, ts, conf, target, force=True)
            if started:
                self.next_replan = t + cfg.REPLAN_PERIOD_S
            elif t >= self.next_check:
                self._recorrect(t, p, v, snap, target, dist)

        accel = np.zeros(3)
        if self.mode == "AVOID":
            accel = accel + self.seq.accel_at(t)
            if self.vhold.active:  # keep any initial speed-up going, along the path only
                ah = self.vhold.step(v)
                accel = accel + float(ah @ self.vhat0) * self.vhat0
        else:
            accel = accel + self.vhold.step(v)

        speed = float(np.linalg.norm(v))
        to_t = target - p
        cosang = (
            float(np.clip((v @ to_t) / max(speed * max(dist, 1e-9), 1e-9), -1, 1))
            if speed > 0.1
            else 1.0
        )
        tel = dict(
            mode=self.mode,
            note=self.note,
            conf=conf,
            line=line,
            plan=self.plan,
            dist=dist,
            v_des=self.v_des,
            vhold=self.vhold.active,
            replans=self.n_replans,
            heading_err=float(np.degrees(np.arccos(cosang))),
            next_check_in=max(0.0, self.next_check - t),
        )
        return accel, tel
