import numpy as np

from . import config as cfg


class Reorienter:
    """Turns the shuttle's nose (body +X) toward the target. Fully disabled by cfg.REORIENT_SHUTTLE.

    A PD controller with a deadband: it starts turning when the pointing error exceeds
    REORIENT_START_DEG, and stops commanding once it is below REORIENT_STOP_DEG,
    so it doesn't keep spending fuel on tiny corrections.
    """

    def __init__(self):
        self.active = False
        self.error_deg = 0.0

    def command(self, target_dir_body, angvel_body) -> np.ndarray:
        """Angular acceleration command in the body frame (rad/s^2)."""
        d = np.asarray(target_dir_body, dtype=float)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        angle = float(
            np.arctan2(np.hypot(d[1], d[2]), d[0])
        )  # angle between body +X and the target
        self.error_deg = float(np.degrees(angle))

        if not cfg.REORIENT_SHUTTLE:
            self.active = False
            return np.zeros(3)

        if self.active:
            self.active = self.error_deg > cfg.REORIENT_STOP_DEG
        else:
            self.active = self.error_deg > cfg.REORIENT_START_DEG
        if not self.active:
            return np.zeros(3)

        axis = np.cross([1.0, 0.0, 0.0], d)  # rotation axis, body frame = [0, -dz, dy]
        s = float(np.linalg.norm(axis))
        axis = axis / s if s > 1e-9 else np.zeros(3)
        cmd = cfg.KP_ATT * angle * axis - cfg.KD_ATT * np.asarray(angvel_body, dtype=float)
        return np.clip(cmd, -cfg.ANG_ACC_MAX, cfg.ANG_ACC_MAX)
