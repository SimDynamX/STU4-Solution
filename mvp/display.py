import cv2
import numpy as np

from . import config as cfg
from .trajectory import lateral_basis

HUD = 500
MAP_W, MAP_H = 500, 250
MAP_S = 2.0  # pixels per metre on the maps
MAP_BACK_M = 40.0  # metres shown behind the shuttle


def _col(threat, closing, confident):
    if threat:
        return (0, 0, 255)
    if not confident:
        return (70, 70, 70)
    return (0, 165, 255) if closing else (110, 110, 110)


class Hud:
    """One window with forward view, top-down and side maps, and status text."""

    def draw(self, shuttle, snap, tel, target, scans):
        conf, _line, plan = tel["conf"], tel["line"], tel["plan"]
        p, v = shuttle.pos, shuttle.vel
        to_t = target - p
        f = (
            v / shuttle.speed
            if shuttle.speed > 1.0
            else to_t / max(float(np.linalg.norm(to_t)), 1e-9)
        )
        right, up = lateral_basis(f)

        rel, relv = snap.pos - p, snap.vel - v
        al, la, ve = rel @ f, rel @ right, rel @ up
        val, vla, vve = relv @ f, relv @ right, relv @ up
        rng = np.linalg.norm(rel, axis=1)
        closing = np.einsum("ij,ij->i", rel, relv) < 0
        confident = snap.n >= cfg.MIN_UPDATES_FOR_THREAT

        hud = self._hud(
            al, la, ve, rng, conf, closing, confident, snap.ids, target - p, f, right, up
        )

        trail = shuttle.trail(10.0) - p if len(shuttle.trail(10.0)) > 1 else np.zeros((0, 3))
        path = np.zeros((0, 3))
        if plan is not None and tel["mode"] == "AVOID":
            path = plan.path - p
        tgt = target - p
        top = self._map(
            "TOP-DOWN (ahead ->, lateral)",
            al,
            la,
            val,
            vla,
            conf,
            closing,
            confident,
            snap.ids,
            tgt @ f,
            tgt @ right,
            trail @ f,
            trail @ right,
            path @ f,
            path @ right,
        )
        side = self._map(
            "SIDE (ahead ->, up)",
            al,
            ve,
            val,
            vve,
            conf,
            closing,
            confident,
            snap.ids,
            tgt @ f,
            tgt @ up,
            trail @ f,
            trail @ up,
            path @ f,
            path @ up,
        )
        row = np.hstack((hud, np.vstack((top, side))))
        status = self._status(shuttle, snap, tel, scans, confident)
        full = np.vstack((row, status))

        scale = min(1.0, cfg.DISPLAY_MAX_W / full.shape[1], cfg.DISPLAY_MAX_H / full.shape[0])
        if scale < 1.0:
            full = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Tracking", full)
        cv2.waitKey(1)

    # ---------- panels ----------
    def _hud(self, al, la, ve, rng, conf, closing, confident, ids, tgt, f, right, up):
        panel = np.zeros((HUD, HUD, 3), np.uint8)
        c, half = HUD // 2, cfg.HUD_HALF_FOV_DEG
        ppd = c / half
        sh = -1 if cfg.HUD_FLIP_H else 1
        sv = -1 if cfg.HUD_FLIP_V else 1
        for deg in range(10, int(half) + 1, 10):
            cv2.circle(panel, (c, c), int(deg * ppd), (45, 45, 45), 1)
        cv2.line(panel, (0, c), (HUD, c), (45, 45, 45), 1)
        cv2.line(panel, (c, 0), (c, HUD), (45, 45, 45), 1)

        for i in np.flatnonzero(al > 1.0):
            az = np.degrees(np.arctan2(la[i], al[i]))
            el = np.degrees(np.arctan2(ve[i], np.hypot(al[i], la[i])))
            if abs(az) > 1.4 * half or abs(el) > 1.4 * half:
                continue
            px, py = int(c + sh * az * ppd), int(c - sv * el * ppd)
            ang = np.degrees(np.arctan(conf.safe[i] / max(rng[i], 1.0)))
            col = _col(conf.threat[i], closing[i] and rng[i] < 150, confident[i])
            cv2.circle(
                panel, (px, py), int(np.clip(ang * ppd, 3, 400)), col, 2 if conf.threat[i] else 1
            )
            if cfg.SHOW_THREAT_IDS and conf.threat[i]:
                cv2.putText(
                    panel, str(ids[i]), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
                )

        cv2.drawMarker(panel, (c, c), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)  # flight path
        ta, tl, tv = tgt @ f, tgt @ right, tgt @ up
        if ta > 1.0:
            az = np.degrees(np.arctan2(tl, ta))
            el = np.degrees(np.arctan2(tv, np.hypot(ta, tl)))
            cv2.drawMarker(
                panel,
                (int(c + sh * az * ppd), int(c - sv * el * ppd)),
                (0, 255, 255),
                cv2.MARKER_DIAMOND,
                16,
                2,
            )  # target
        cv2.putText(
            panel,
            "FORWARD VIEW (along flight path)",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        return panel

    def _map(self, label, x, y, vx, vy, conf, closing, confident, ids, tx, ty, trx, try_, pax, pay):
        panel = np.zeros((MAP_H, MAP_W, 3), np.uint8)

        def px(a):
            return int((a + MAP_BACK_M) * MAP_S)

        def py(o):
            return int(MAP_H / 2 - o * MAP_S)

        for a in range(0, int(MAP_W / MAP_S - MAP_BACK_M), 50):
            cv2.line(panel, (px(a), 0), (px(a), MAP_H), (40, 40, 40), 1)
        cv2.line(panel, (0, py(0)), (MAP_W, py(0)), (60, 60, 60), 1)

        vis = np.flatnonzero(
            (x > -MAP_BACK_M)
            & (x < MAP_W / MAP_S - MAP_BACK_M)
            & (np.abs(y) < MAP_H / MAP_S / 2 + 10)
        )
        for i in vis:
            col = _col(conf.threat[i], closing[i], confident[i])
            cv2.circle(
                panel,
                (px(x[i]), py(y[i])),
                int(max(2, conf.safe[i] * MAP_S)),
                col,
                2 if conf.threat[i] else 1,
            )
            cv2.line(
                panel, (px(x[i]), py(y[i])), (px(x[i] + vx[i] * 3), py(y[i] + vy[i] * 3)), col, 1
            )  # 3 s of relative motion
            if cfg.SHOW_THREAT_IDS and conf.threat[i]:
                cv2.putText(
                    panel,
                    str(ids[i]),
                    (px(x[i]) + 4, py(y[i]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    col,
                    1,
                )

        def poly(a, o, col):
            if len(a) > 1:
                pts = np.stack([[px(u) for u in a], [py(u) for u in o]], axis=1).astype(np.int32)
                cv2.polylines(panel, [pts.reshape(-1, 1, 2)], False, col, 1)

        poly(trx, try_, (255, 160, 0))  # where the shuttle has been
        poly(pax, pay, (255, 255, 0))  # planned sidestep path
        cv2.line(panel, (px(0), py(0)), (MAP_W, py(0)), (200, 200, 200), 1)
        cv2.drawMarker(panel, (px(0), py(0)), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)
        if -MAP_BACK_M < tx < MAP_W / MAP_S - MAP_BACK_M and abs(ty) < MAP_H / MAP_S / 2:
            cv2.drawMarker(panel, (px(tx), py(ty)), (0, 255, 255), cv2.MARKER_DIAMOND, 14, 2)
        cv2.putText(
            panel,
            label + "  (10 m grid = 50 m)",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )
        return panel

    def _status(self, shuttle, snap, tel, scans, confident):
        conf, line, plan = tel["conf"], tel["line"], tel["plan"]
        eta = f"{line.eta:.0f}s" if np.isfinite(line.eta) else "--"
        lines = [
            f"MODE {tel['mode']} | {tel['note']}",
            f"speed {shuttle.speed:.2f} / {cfg.CRUISE_SPEED:.1f} m/s | "
            f"heading err {tel['heading_err']:.1f} deg | dist {tel['dist']:.0f} m | "
            f"ETA {eta} | miss at target {line.miss:.1f} m",
            f"next check in {tel['next_check_in']:.1f}s | "
            f"velocity-hold {'ON' if tel['vhold'] else 'off'} | "
            f"tracks {len(snap.ids)} ({int(confident.sum())} confident) | "
            f"scans {scans} | threats {int(conf.threat.sum())}",
        ]
        thr = np.flatnonzero(conf.threat)
        for i in thr[np.argsort(conf.t_star[thr])][:3]:
            lines.append(
                f"  threat #{snap.ids[i]}: closest approach in {conf.t_star[i]:.1f}s, "
                f"miss {conf.d_star[i]:.1f} m (needs > {conf.safe[i]:.1f} m)"
            )
        if plan is not None:
            lines.append(
                f"  avoidance plan: dv {plan.dv:.2f} m/s over {plan.duration:.1f}s "
                f"at {plan.accel:.2f} m/s^2 | clearance {plan.clearance:.1f} m | "
                f"{'ok' if plan.feasible else 'BEST EFFORT'} | replans {tel['replans']}"
            )
        strip = np.zeros((170, 2 * HUD if False else 1000, 3), np.uint8)
        for k, txt in enumerate(lines[:7]):
            cv2.putText(
                strip,
                txt[:150],
                (8, 20 + 22 * k),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return strip
