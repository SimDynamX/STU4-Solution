# The licensed runtime must connect before the later numeric/controller imports.
# ruff: noqa: E402
#############################################################################################
# DON'T CHANGE ANY OF THE BELOW. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

import datetime
import os
import sys
import time
import traceback

import spaceteams as st


def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)


sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)

import numpy as np
import numpy.typing as npt

this = st.GetThisSystem()

doSpawning = st.SimGlobals.GetSimEntity().GetParam(st.VarType.bool, "DoSpawning")
if doSpawning:
    while True:
        time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

phobos: st.Entity = this.GetParam(st.VarType.entityRef, "Phobos")
phobosFrame = phobos.GetBodyFixedFrame()
shuttle: st.Entity = this.GetParam(st.VarType.entityRef, "Shuttle")
shuttleFrame = shuttle.GetBodyFixedFrame()
target: st.Entity = this.GetParam(st.VarType.entityRef, "Target")

topCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopCamera")
topCameraFrame = topCamera.GetBodyFixedFrame()
bottomCamera: st.Entity = this.GetParam(st.VarType.entityRef, "BottomCamera")
bottomCameraFrame = bottomCamera.GetBodyFixedFrame()


def GetNearestDebrisLocations(N: int) -> list[npt.NDArray]:
    allDebrisInfo: list[npt.NDArray] = shuttle.GetParamArray(
        st.VarType.doubleM3x3, ["Radar", "SortedDebrisData"]
    )
    nearest = allDebrisInfo[: min(N, len(allDebrisInfo))]
    return nearest


def PanTiltCamera(camera: st.Entity, yaw_deg: float, pitch_deg: float) -> None:
    if camera == topCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
    elif camera == bottomCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
    camera.setRotation_DCM(
        st.math.Euler321_to_DCM([np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0.0]), shuttleFrame
    )


def SetCommand(accelCommand: npt.NDArray, angAccelCommand: npt.NDArray) -> None:
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], accelCommand)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], angAccelCommand)


#############################################################################################
# DON'T CHANGE ANY OF THE ABOVE. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################


#########################################################
# USER CODE                                             #
#########################################################

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvp import config as cfg
from mvp.attitude import Reorienter
from mvp.context import init as mvp_init
from mvp.context import phobos_to_body
from mvp.display import Hud
from mvp.guidance import Guidance
from mvp.state_estimator import DebrisTracks, ShuttleState

mvp_init(phobosFrame=phobosFrame, shuttleFrame=shuttleFrame)

LOOP_FREQ = this.GetParam(st.VarType.double, "LoopFreqHz")

t0_sim: st.timestamp = st.SimGlobals.GetSimEntity().GetParam(
    st.VarType.timestamp, ["Time", "StartDateTAI"]
)
t0: datetime.datetime = t0_sim.as_datetime()

shuttle_state = ShuttleState()
tracks = DebrisTracks()
guidance = Guidance()
hud = Hud()
reorienter = Reorienter()

last_damage = 0.0
iteration = 0
exit_flag: bool = False
while not exit_flag:
    time.sleep(1.0 / LOOP_FREQ)

    t = (st.SimGlobals.SimClock.GetTimeNow().as_datetime() - t0).total_seconds()

    # --- Shuttle state ---
    s = shuttle.getLocVelAcc()
    shuttle_state.update(t, s.loc_WRT_ExprIn(phobosFrame), s.vel_WRT_ExprIn(phobosFrame))
    target_pos = np.asarray(target.getLocVelAcc().loc_WRT_ExprIn(phobosFrame), dtype=float)

    # --- Radar -> debris tracks ---
    tracks.ingest(GetNearestDebrisLocations(cfg.RADAR_N), shuttle_state)
    snap = tracks.snapshot(t)

    # --- Guidance: acceleration command in Phobos-frame axes ---
    accel_phobos, tel = guidance.update(t, shuttle_state, snap, target_pos)

    # --- Track shuttle damage to see if radar is avoidance is working in case of a collison
    damage = shuttle.GetParam(st.VarType.double, ["Stats", "Damage"])
    if damage > last_damage + 1e-9 and len(snap.pos):
        dd = np.linalg.norm(snap.pos - shuttle_state.pos, axis=1)
        j = int(np.argmin(dd))
        st.OnScreenLogMessage(
            f"HIT t={t:.1f}s dmg {last_damage:.2f}->{damage:.2f} | "
            f"nearest #{snap.ids[j]} {dd[j]:.1f} m, "
            f"n={snap.n[j]}, sig_v={snap.sig_v[j]:.2f}, "
            f"rel speed {np.linalg.norm(snap.vel[j] - shuttle_state.vel):.1f} m/s "
            f"| mode {tel['mode']}",
            "HitLog",
            st.Severity.Info,
        )
    last_damage = damage

    # --- Commands (the shuttle expects body-frame axes) ---
    ang_cmd = np.zeros(3)
    if cfg.REORIENT_SHUTTLE:
        d_body = phobos_to_body(target_pos - shuttle_state.pos)
        angvel_body = phobos_to_body(shuttle.getAngVelocity().WRT_ExprIn(phobosFrame))
        ang_cmd = reorienter.command(d_body, angvel_body)
    SetCommand(phobos_to_body(accel_phobos), ang_cmd)

    # --- Display ---
    if iteration % cfg.HUD_EVERY_N == 0:
        hud.draw(shuttle_state, snap, tel, target_pos, tracks.scans)

    iteration += 1

st.leave_sim()
