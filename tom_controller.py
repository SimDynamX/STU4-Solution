# Keep imports after the SDK bootstrap and example setup in their original order.
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
# EXAMPLE USER CODE (NOTE: RUNNING A LOOP IS NECESSARY) #
#########################################################


import copy

from PIL import Image
from radar_display import RadarDisplay

###############################
# Get initial simulation time #
###############################

t0_sim: st.timestamp = st.SimGlobals.GetSimEntity().GetParam(
    st.VarType.timestamp, ["Time", "StartDateTAI"]
)
t0: datetime.datetime = t0_sim.as_datetime()

###################
# Make a log file #
###################
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_log.txt")
# Keep the existing log open for the process lifetime; line-buffered, fresh each run.
log_file = open(LOG_PATH, "w", buffering=1)  # noqa: SIM115


def status(msg: str, key: str = "Status") -> None:
    print(msg, flush=True)
    st.OnScreenLogMessage(msg, key, st.Severity.Info)
    log_file.write(msg + "\n")


"""
Get angular velocity from two quaternions representing the orientation of 
an object at time (t0) and (t0 + dt). Assumes scalar-last quaternions.
"""


def AngVelFromQuats(q1: npt.NDArray, q2: npt.NDArray, dt: float):
    return (2.0 / dt) * np.array(
        [
            q1[3] * q2[0] - q1[0] * q2[3] - q1[1] * q2[2] + q1[2] * q2[1],
            q1[3] * q2[1] + q1[0] * q2[2] - q1[1] * q2[3] - q1[2] * q2[0],
            q1[3] * q2[2] - q1[0] * q2[1] + q1[1] * q2[0] - q1[2] * q2[3],
        ]
    )


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


DIRS = fib_sphere(400)


def pick_safe_velocity(v_des, debris_info, margin=8.0, horizon=15.0):
    speed = np.linalg.norm(v_des)
    if speed < 1e-3 or len(debris_info) == 0:
        return v_des
    P = np.array([d[0, :] for d in debris_info])  # relative to shuttle
    R = np.array([d[1, 2] / 2 for d in debris_info]) + margin  # inflated radii
    order = np.argsort(-(DIRS @ (v_des / speed)))  # best-aligned first
    for scale in (1.0, 0.5, 0.25):  # then try slower
        V = DIRS[order] * speed * scale
        t = np.clip((P @ V.T) / (speed * scale) ** 2, 0, horizon)
        closest = np.linalg.norm(P[:, None, :] - t[:, :, None] * V[None], axis=2)
        safe = np.all(closest > R[:, None], axis=0)
        if safe.any():
            return V[np.argmax(safe)]
    return np.zeros(3)  # nowhere safe: stop


##################
# Start the loop #
##################
radar_view = RadarDisplay(view_range_m=50, update_every=1)


def to_body(v):
    return np.asarray(
        st.frames.FramedVector(np.asarray(v, dtype=float), phobosFrame).ExprIn(shuttleFrame),
        dtype=float,
    )


###################
# Debris Tracking #
###################
past_positions = []

from debris_tracker import DebrisTracker

tracker = DebrisTracker()
from avoidance import Avoider, BurnGovernor

avoider = Avoider()
governor = BurnGovernor(gain=2.0)


iteration = 0
imageNumber = 0
exit_flag: bool = False
while not exit_flag:
    ################################
    # Wait between loop iterations #
    ################################

    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

    ###################################
    # Get elapsed time from sim start #
    ###################################

    t_now: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    t_from_start: float = (t_now - t0).total_seconds()

    #####################################
    # Example for getting vehicle state #
    #####################################

    locVelAcc = shuttle.getLocVelAcc()
    loc = locVelAcc.loc_WRT_ExprIn(phobosFrame)
    vel = locVelAcc.vel_WRT_ExprIn(phobosFrame)
    rot = shuttle.getRotation().Quat_WRT(phobosFrame)
    angvel = shuttle.getAngVelocity().WRT_ExprIn(phobosFrame)

    #####################################################
    # Example for getting additional vehicle properties #
    #####################################################

    mass = shuttle.getMass()
    inertiaBodyFrame = shuttle.getInertiaTensor(shuttleFrame)
    damage: float = shuttle.GetParam(st.VarType.double, ["Stats", "Damage"])
    fuelUsed: float = shuttle.GetParam(st.VarType.double, ["Stats", "FuelUsed"])

    ##################################################################
    # Example for rotating the top camera 30 deg left and 10 deg up, #
    # and the bottom camera 20 deg right and 15 deg down.            #
    ##################################################################

    if iteration % 20 == 0:
        PanTiltCamera(topCamera, 30.0, -10.0)
        PanTiltCamera(bottomCamera, -20.0, 15.0)

    ####################################################################################
    # Example for obtaining an image from the top camera with FOV = 60 deg and EV = 12 #
    ####################################################################################

    if iteration % 20 == 0:  # Don't hot-loop the image captures
        cameraParams = st.CaptureImageProperties()
        cameraParams.nonphys_EV = 10.0
        cameraParams.FOV = 60.0
        cameraParams.ResolutionX = 512
        cameraParams.ResolutionY = 512
        image = st.camera.CaptureImageBlocking(topCamera, cameraParams)

        if image is not None:
            imageR = np.asarray(image.as_RGB8().PixelsR).reshape(
                (cameraParams.ResolutionX, cameraParams.ResolutionY)
            )
            imageG = np.asarray(image.as_RGB8().PixelsG).reshape(
                (cameraParams.ResolutionX, cameraParams.ResolutionY)
            )
            imageB = np.asarray(image.as_RGB8().PixelsB).reshape(
                (cameraParams.ResolutionX, cameraParams.ResolutionY)
            )
            imageRGB = Image.fromarray(np.dstack((imageR, imageG, imageB)).astype(np.uint8))

            imageNumber += 1
            from camera_overlay import overlay_contacts  # with your other imports

            contacts = [(to_body(d[0, :]), d[1, 2]) for d in GetNearestDebrisLocations(100)]
            marked = overlay_contacts(
                imageRGB, contacts, "top", yaw_deg=30.0, pitch_deg=-10.0, fov_deg=cameraParams.FOV
            )
            marked.save(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), f"cam_{imageNumber:04d}.png"
                )
            )

        else:
            st.OnScreenLogMessage(
                "Failed to capture camera image :(", "CameraNoCapture", st.Severity.Info
            )

    ###################################################################################
    # Example for getting the positions, closest points, range rates, and diameters   #
    # of the nearest 4 pieces of debris and printing the distance to the closest one. #
    ###################################################################################

    debrisInfo = GetNearestDebrisLocations(4)
    if len(debrisInfo) > 0:
        closestDebris = debrisInfo[0]
        closestDebris_loc = closestDebris[0, :]
        closestDebris_closestPoint = closestDebris[1, 0]
        closestDebris_rangeRate = closestDebris[1, 1]
        closestDebris_diameter = closestDebris[1, 2]
        st.OnScreenLogMessage(
            f"Nearest debris: {np.linalg.norm(closestDebris_loc):.1f} m",
            "UserAPIPrintout",
            st.Severity.Info,
        )

    ##################################
    # YOUR CONTROLLER CODE GOES HERE #
    ##################################

    debris = GetNearestDebrisLocations(200)
    target_loc = target.getLocVelAcc().loc_WRT_ExprIn(phobosFrame)
    radar_view.update(loc, debris, target_loc, iteration, to_body=to_body)

    # DEBRIS TRACKING #
    debris = GetNearestDebrisLocations(50)
    tracks = tracker.update(loc, vel, debris)  # loc, vel: shuttle state in the Phobos frame
    snap = tracker.snapshot(t_from_start, loc, vel)

    for tr in tracks:
        p_rel, v_rel = tracker.relative_state(tr, loc, vel)

    ##################

    # locVelAcc = shuttle.getLocVelAcc()
    # loc = locVelAcc.loc_WRT_ExprIn(phobosFrame)
    # vel = locVelAcc.vel_WRT_ExprIn(phobosFrame)
    # rot = shuttle.getRotation().Quat_WRT(phobosFrame)
    # angvel = shuttle.getAngVelocity().WRT_ExprIn(phobosFrame)
    #
    # st.OnScreenLogMessage(f"Rotation: {rot}", "UserAPIPrintout", st.Severity.Info)
    # st.OnScreenLogMessage(f"Velocity: {vel}", "UserAPIPrintout", st.Severity.Info)
    # linearAcceleration_m_s2 = np.zeros(3)
    # angularAcceleration_rad_s2 = np.zeros(3)
    # if vel[0] < 13 :
    #
    #    linearAcceleration_m_s2[0] = 5
    #
    ##if rot[1] > -np.pi/4 :
    ##    angularAcceleration_rad_s2[1] = -0.5
    ##elif rot[1] > -np.pi/2 :
    ##    angularAcceleration_rad_s2[1] = 0.49999
    ##elif rot[1] < -np.pi/2 :
    ##    angularAcceleration_rad_s2[1] = 0
    ## Send acceleration and angular acceleration commands
    # SetCommand(linearAcceleration_m_s2, angularAcceleration_rad_s2)

    ####################################################
    # Example for commanding a beeline acceleration    #
    # toward the target (and facing it the entire way) #
    ####################################################

    # Gains
    accel_gain: float = 5  # 1.0
    angvel_gain: float = 0.2  # 0.2
    kp_pitch = 2.0
    kp_yaw = 2.0
    kp_roll = 2.0

    # Translational control
    targetState: st.frames.FramedLocVelAcc = target.getLocVelAcc()
    targetLoc: npt.NDArray = targetState.loc_WRT_ExprIn(phobosFrame)
    targetVel: npt.NDArray = targetState.vel_WRT_ExprIn(phobosFrame)

    goalDirection: npt.NDArray = (targetLoc - loc) / np.linalg.norm(targetLoc - loc)
    goalSpeed_m_s: float = 17.0  # just keep accelerating
    goalVelocity: npt.NDArray = goalSpeed_m_s * goalDirection

    v_des = goalVelocity
    v_safe = avoider.pick(v_des, snap, vel)  # note: vel is now an argument
    linearAcceleration_m_s2 = governor.accel(v_safe, vel)
    linearAcceleration_m_s2 = st.frames.FramedVector(linearAcceleration_m_s2, phobosFrame).ExprIn(
        shuttleFrame
    )

    # Rotational control
    # Align shuttle x-axis to target and shuttle y-axis to Phobos y-axis
    x_hat = copy.deepcopy(goalDirection)
    y_hat = np.array([0.0, -1.0, 0.0])
    z_hat = np.cross(x_hat, y_hat)
    z_hat = z_hat / np.linalg.norm(z_hat)
    T_shuttlePhobos = np.vstack((x_hat, y_hat, z_hat))
    q_shuttlePhobos = st.math.DCM_to_Quat(T_shuttlePhobos)

    desired_AngVel = np.zeros(3)
    if angvel_gain != 0.0:
        desired_AngVel = AngVelFromQuats(rot, q_shuttlePhobos, 1.0 / angvel_gain)
    desired_AngVel = (
        st.frames.FramedAngVel(desired_AngVel, phobosFrame).WRT(phobosFrame).ExprIn(shuttleFrame)
    )

    desired_AngVel_x = desired_AngVel[0]
    desired_AngVel_y = desired_AngVel[1]
    desired_AngVel_z = desired_AngVel[2]

    # get current angular velocity values
    current_AngVel = shuttle.getAngVelocity().WRT(phobosFrame).ExprIn(shuttleFrame)
    current_AngVel_x: float = current_AngVel[0]
    current_AngVel_y: float = current_AngVel[1]
    current_AngVel_z: float = current_AngVel[2]

    # calculate angular velocity error
    AngVel_error_x = desired_AngVel_x - current_AngVel_x
    AngVel_error_y = desired_AngVel_y - current_AngVel_y
    AngVel_error_z = desired_AngVel_z - current_AngVel_z

    # if AngVel_error_x > 0.1 or AngVel_error_y > 0.1 or AngVel_error_z > 0.1 :

    strength_torque = shuttle.GetParam(st.VarType.double, ["Thrusters", "TorqueStrength_Nm"])
    torque_x = kp_roll * AngVel_error_x * strength_torque
    torque_y = kp_pitch * AngVel_error_y * strength_torque
    torque_z = kp_yaw * AngVel_error_z * strength_torque

    torque_Nm = np.array([torque_x, torque_y, torque_z])
    angularAcceleration_rad_s2 = np.array(
        [0, 0, 0]
    )  # np.linalg.inv(shuttle.getInertiaTensor(shuttleFrame)) @ torque_Nm  #
    # Send acceleration and angular acceleration commands
    SetCommand(linearAcceleration_m_s2, angularAcceleration_rad_s2)

    debris = GetNearestDebrisLocations(5)
    if debris:
        P = np.array([d[0, :] for d in debris])  # (N,3) centres relative to shuttle, Phobos axes
        rc = np.array([d[1, 0] for d in debris])  # closest-surface range
        vr = np.array([d[1, 1] for d in debris])  # range rate (negative = closing)
        D = np.array([d[1, 2] for d in debris])  # diameters
        ts = np.array([d[2, 0] for d in debris])  # measurement times
        age = t_from_start - ts
        closing = vr < 0

    if iteration % 2 == 0:
        dist = np.linalg.norm(target_loc - loc)
        closest = debris[0][1, 0] if len(debris) else float("nan")
        status(
            f"t={t_from_start:6.1f}s  dist={dist:7.1f}m  speed={np.linalg.norm(vel):5.2f}m/s  "
            f"nearest={closest:6.1f}m  damage={damage:.2f}  fuel={fuelUsed:.1f}  "
            f"velocity={vel}"
            f"avoid={avoider.last_info}"
        )

    iteration += 1

st.leave_sim()
