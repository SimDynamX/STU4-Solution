"""
radar_display.py - simple live view of the shuttle's radar debris data.

Three shuttle-centred 2D views of the debris:
  * Top-down : forward (+X) is UP, +Y (left of the shuttle) is on the LEFT
  * Side     : forward (+X) is RIGHT, +Z (up) is UP
  * Rear     : looking forward along +X, +Z up, +Y on the LEFT

Radar positions are already measured relative to the shuttle, so they are
plotted as-is. Each debris object is a circle at its radar diameter.
Red = closing in, blue = moving away. Green star = target.

If you pass a `to_body` function to update(), everything is rotated into the
shuttle's body frame, so "up" in the top-down view really is the direction the
shuttle's nose points. Without it, the axes are the Phobos-fixed axes.

Needs matplotlib:  pip install matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# (title, horizontal axis index, vertical axis index, flip horizontal?, shuttle marker)
#  axis index: 0 = X (forward), 1 = Y (left), 2 = Z (up)
VIEWS = [
    ("Top-down (forward = up)", 1, 0, True, "^"),
    ("Side (forward = right)", 0, 2, False, ">"),
    ("Rear view (looking forward)", 1, 2, True, "o"),
]
AXIS_NAMES = ["X", "Y", "Z"]


class RadarDisplay:
    def __init__(self, view_range_m: float = 150.0, update_every: int = 5):
        self.view_range = view_range_m
        self.update_every = update_every  # redraw every N controller loops
        self.enabled = True
        plt.ion()
        # A fixed figure name means matplotlib re-uses the same window even if
        # RadarDisplay() gets constructed again (e.g. accidentally inside the loop).
        self.fig = plt.figure("Radar debris view", figsize=(13, 5.4))
        if len(self.fig.axes) == 3:
            self.axs = self.fig.axes
        else:
            self.fig.clf()
            self.axs = self.fig.subplots(1, 3)

    def update(self, shuttle_loc, debris_info, target_loc=None, iteration: int = 0, to_body=None):
        """
        shuttle_loc : shuttle position in the Phobos frame (3-vector); only used to
                      place the target relative to the shuttle
        debris_info : output of GetNearestDebrisLocations(N)
        target_loc  : optional target position in the Phobos frame (3-vector)
        to_body     : optional function converting a Phobos-frame 3-vector into the
                      shuttle body frame (see the usage notes at the bottom)
        """
        if not self.enabled or iteration % self.update_every != 0:
            return
        try:
            self._draw(
                np.asarray(shuttle_loc, dtype=float),
                debris_info,
                None if target_loc is None else np.asarray(target_loc, dtype=float),
                to_body,
            )
        except Exception as e:  # never let plotting crash the controller
            print(f"RadarDisplay disabled: {e}")
            self.enabled = False

    def _draw(self, shuttle_loc, debris_info, target_loc, to_body):
        convert = to_body if to_body is not None else (lambda v: np.asarray(v, dtype=float))
        frame = "body" if to_body is not None else "Phobos"

        centers, rates, diameters, ranges = [], [], [], []
        for item in debris_info:
            centers.append(
                convert(np.array(item[0, :], dtype=float))
            )  # already relative to shuttle
            ranges.append(float(item[1, 0]))
            rates.append(float(item[1, 1]))
            diameters.append(float(item[1, 2]))
        t_rel = None if target_loc is None else convert(target_loc - shuttle_loc)

        R = self.view_range
        for ax, (title, h, v, flip_h, marker) in zip(self.axs, VIEWS, strict=False):
            ax.clear()
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            if flip_h:
                ax.invert_xaxis()  # +Y (left of the shuttle) is drawn on the left
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(f"{frame} {AXIS_NAMES[h]} (m)")
            ax.set_ylabel(f"{frame} {AXIS_NAMES[v]} (m)")
            ax.grid(alpha=0.3)

            for c, rate, d in zip(centers, rates, diameters, strict=False):
                colour = "tab:red" if rate < 0 else "tab:blue"
                ax.add_patch(Circle((c[h], c[v]), d / 2.0, color=colour, alpha=0.5))

            ax.plot(0, 0, "k", marker=marker, markersize=10)  # shuttle
            if t_rel is not None:
                ax.plot(t_rel[h], t_rel[v], "g*", markersize=14)  # target

        msg = "no debris detected"
        if ranges:
            k = int(np.argmin(ranges))
            msg = (
                f"{len(ranges)} contacts | closest surface {ranges[k]:.1f} m, "
                f"range rate {rates[k]:+.1f} m/s"
            )
        self.fig.suptitle(
            f"Shuttle-centred, {frame}-frame axes  (black = shuttle, green * = target)\n" + msg
        )
        self.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.80, wspace=0.25)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ------------------------------------------------------------------------------
# HOW TO USE IN Controller.py
#
# 1) Put radar_display.py next to Controller.py. Before the while loop add:
#
#       from radar_display import RadarDisplay
#       radar_view = RadarDisplay(view_range_m=150.0, update_every=5)
#
#       def to_body(v):   # Phobos frame -> shuttle body frame (same call as the
#                         # wiki's beeline example uses)
#           return np.asarray(
#               st.frames.FramedVector(np.asarray(v, dtype=float), phobosFrame)
#                 .ExprIn(shuttleFrame), dtype=float)
#
# 2) Inside the loop, under "YOUR CONTROLLER CODE GOES HERE":
#
#       debris = GetNearestDebrisLocations(200)
#       target_loc = target.getLocVelAcc().loc_WRT_ExprIn(phobosFrame)
#       radar_view.update(loc, debris, target_loc, iteration, to_body=to_body)
# ------------------------------------------------------------------------------
