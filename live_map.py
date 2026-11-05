"""
live_map.py - the simplest live top-down map of tracked rocks.

    from live_map import LiveMap
    live = LiveMap(axes=(0, 1), every=5)               # once, before the loop
    live.update(tracker.rocks, shuttle_pos, target_pos)  # every loop

`rocks` is any list of objects with .position (absolute Phobos-frame 3-vector)
and .diameter. The map redraws every `every` calls, because a matplotlib redraw
is slow (tens of ms) compared with your control loop. Marker size shows relative
diameter on screen, not true scale.

axes=(0, 1) is a top-down X-Y view; (0, 2) is a side view; (1, 2) a front view.
Needs matplotlib:  pip install matplotlib
"""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np

NAMES = ["X", "Y", "Z"]


class LiveMap:
    def __init__(self, axes=(0, 1), every=5, title="Live rock map"):
        self.i, self.j = axes
        self.every = every
        self.count = 0
        self.enabled = True
        plt.ion()
        self.fig = plt.figure(title)  # same name every time -> always the same window
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(f"{NAMES[self.i]} (m)")
        self.ax.set_ylabel(f"{NAMES[self.j]} (m)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(alpha=0.3)
        self.rocks = self.ax.scatter([], [], s=[], color="tab:red", alpha=0.5)
        (self.trail,) = self.ax.plot([], [], "k-", lw=0.8, alpha=0.5)
        (self.ship,) = self.ax.plot([], [], "k^", markersize=10)
        (self.goal,) = self.ax.plot([], [], "g*", markersize=14)
        self.path = deque(maxlen=2000)  # capped, so the trail can never grow without bound

    def update(self, rocks, shuttle_pos, target_pos=None):
        if not self.enabled:
            return
        self.count += 1
        ship = np.asarray(shuttle_pos, float)[[self.i, self.j]]
        self.path.append(ship)
        if self.count % self.every:
            return
        try:
            pts = [ship]
            if len(rocks):
                xy = np.array([np.asarray(r.position, float)[[self.i, self.j]] for r in rocks])
                self.rocks.set_offsets(xy)
                self.rocks.set_sizes(0.5 * np.array([r.diameter for r in rocks]) ** 2)
                pts.append(xy)
            else:
                self.rocks.set_offsets(np.empty((0, 2)))
            if target_pos is not None:
                tgt = np.asarray(target_pos, float)[[self.i, self.j]]
                self.goal.set_data([tgt[0]], [tgt[1]])
                pts.append(tgt[None])
            trail = np.array(self.path)
            self.trail.set_data(trail[:, 0], trail[:, 1])
            self.ship.set_data([ship[0]], [ship[1]])

            allp = np.vstack([np.atleast_2d(p) for p in pts])
            lo, hi = allp.min(axis=0), allp.max(axis=0)
            centre = (lo + hi) / 2
            half = max(0.6 * float((hi - lo).max()), 10.0)  # square view, 20% margin
            self.ax.set_xlim(centre[0] - half, centre[0] + half)
            self.ax.set_ylim(centre[1] - half, centre[1] + half)
            self.ax.set_title(f"{len(rocks)} tracked rocks")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:  # never let plotting crash the controller
            print(f"LiveMap disabled: {e}")
            self.enabled = False
