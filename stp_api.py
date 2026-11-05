"""Optional adapter for the real, licensed Space Teams Pro Python SDK.

Bindings follow the pinned official UserAPI.py and Controller.py below. Importing
this module does not import ``spaceteams`` or connect to a simulation. Call
``connect_from_argv`` only inside the competition's ControllerSystem process,
with the interpreter and arguments supplied by the licensed runtime.

Required current-system parameters are Phobos, Shuttle, Target, TopCamera and
BottomCamera entity references, plus a positive LoopFreqHz. The simulation must
have DoSpawning=false and Time/StartDateTAI. This is not a Bevy/Rapier adapter and
does not establish equivalent physics or SDK/interpreter compatibility.
"""

from __future__ import annotations

import importlib
import platform
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType, TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

OFFICIAL_SOLUTION_COMMIT = "8545c085426a71c39cbe1d544a844b3edb757007"
OFFICIAL_SIM_COMMIT = "c1e3fc8f9ee3b189bba64d8bd87818a8003639cc"
OFFICIAL_USER_API_URL = (
    "https://raw.githubusercontent.com/SimDynamX/STU4-Solution/"
    f"{OFFICIAL_SOLUTION_COMMIT}/UserAPI.py"
)
OFFICIAL_CONTROLLER_URL = (
    "https://raw.githubusercontent.com/SimDynamX/STU4-Solution/"
    f"{OFFICIAL_SOLUTION_COMMIT}/Controller.py"
)
OFFICIAL_RADAR_URL = (
    f"https://raw.githubusercontent.com/SimDynamX/STU4-Sim/{OFFICIAL_SIM_COMMIT}/Systems/Radar.py"
)
OFFICIAL_SCORING_URL = (
    f"https://raw.githubusercontent.com/SimDynamX/STU4-Sim/{OFFICIAL_SIM_COMMIT}/Systems/Scoring.py"
)

__all__ = [
    "OFFICIAL_CONTROLLER_URL",
    "OFFICIAL_RADAR_URL",
    "OFFICIAL_SCORING_URL",
    "OFFICIAL_SIM_COMMIT",
    "OFFICIAL_SOLUTION_COMMIT",
    "OFFICIAL_USER_API_URL",
    "STPClient",
    "STPState",
    "connect_from_argv",
]


@dataclass(frozen=True, slots=True)
class STPState:
    """Adapter-local numeric report, not an SDK type or atomic snapshot.

    Position, velocity and angular velocity are relative to and expressed in
    P = Phobos.GetBodyFixedFrame(). Inertia is expressed in
    B = Shuttle.GetBodyFixedFrame(). The quaternion is the SDK's Hamilton xyzw
    passive P-to-shuttle-body rotation, copied without normalization, sign flips
    or conversion to the active rotations used by Bevy/Rapier.

    Fuel and damage are the official Stats counters, labelled kg and J by the
    published scoring code. elapsed_time_s is the report's clock read, not a
    replacement for radar measurement timestamps. SDK reads occur sequentially.
    """

    elapsed_time_s: float
    position_phobos_m: npt.NDArray[np.float64]
    velocity_phobos_m_s: npt.NDArray[np.float64]
    passive_quaternion_phobos_to_body_xyzw: npt.NDArray[np.float64]
    angular_velocity_phobos_rad_s: npt.NDArray[np.float64]
    target_position_phobos_m: npt.NDArray[np.float64]
    mass_kg: float
    inertia_body_kg_m2: npt.NDArray[np.float64]
    fuel_used_kg: float
    damage_j: float


def _finite_real_vector(value: npt.ArrayLike, length: int, name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(value)
    message = f"{name} must be a finite real vector of shape ({length},)"
    if (
        vector.shape != (length,)
        or not np.issubdtype(vector.dtype, np.number)
        or np.iscomplexobj(vector)
    ):
        raise ValueError(message)
    vector = np.array(vector, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(vector)):
        raise ValueError(message)
    return vector


class STPClient:
    """Connected SDK handles. Construct through ``connect_from_argv``.

    SetCommand, GetNearestDebrisLocations and PanTiltCamera retain the official
    helper names. read_state, elapsed_time_s, loop_frequency_hz, capture_image,
    read_only_probe and close are adapter conveniences, not SDK methods. Entity and
    frame attributes retain the starter's names and are real SDK objects.
    """

    def __init__(self, st: ModuleType) -> None:
        self._st = st
        self._closed = False
        self.this = st.GetThisSystem()
        sim_entity = st.SimGlobals.GetSimEntity()
        if sim_entity.GetParam(st.VarType.bool, "DoSpawning"):
            raise RuntimeError(
                "Space Teams Pro has DoSpawning=true and is not controller-ready. "
                "The official starter idles in this phase; this adapter rejects it. "
                "Launch the ControllerSystem in a controller-ready simulation."
            )

        self.phobos = self.this.GetParam(st.VarType.entityRef, "Phobos")
        self.phobosFrame = self.phobos.GetBodyFixedFrame()
        self.shuttle = self.this.GetParam(st.VarType.entityRef, "Shuttle")
        self.shuttleFrame = self.shuttle.GetBodyFixedFrame()
        self.target = self.this.GetParam(st.VarType.entityRef, "Target")
        self.topCamera = self.this.GetParam(st.VarType.entityRef, "TopCamera")
        self.topCameraFrame = self.topCamera.GetBodyFixedFrame()
        self.bottomCamera = self.this.GetParam(st.VarType.entityRef, "BottomCamera")
        self.bottomCameraFrame = self.bottomCamera.GetBodyFixedFrame()
        self._start_date_tai = sim_entity.GetParam(
            st.VarType.timestamp, ["Time", "StartDateTAI"]
        ).as_datetime()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Space Teams Pro client is closed")

    @property
    def loop_frequency_hz(self) -> float:
        """Adapter-local read of the current system's official LoopFreqHz."""
        self._ensure_open()
        frequency = float(self.this.GetParam(self._st.VarType.double, "LoopFreqHz"))
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("ControllerSystem LoopFreqHz must be finite and positive")
        return frequency

    def SetCommand(self, accelCommand: npt.ArrayLike, angAccelCommand: npt.ArrayLike) -> None:
        """Forward body-frame m/s² and rad/s² through the official SetParam calls.

        Both real 3-vectors must be finite before either write occurs. Values
        are not rescaled or locally clamped. At OFFICIAL_SIM_COMMIT the published
        competition actuator independently caps their norms at 0.5 m/s² and
        0.05 rad/s² and uses DirectCommand=false jet allocation. This method
        changes none of those settings. A local practice boost is not an
        official actuator setting. The two SDK writes are sequential, not an
        atomic transaction.
        """
        self._ensure_open()
        accel = _finite_real_vector(accelCommand, 3, "accelCommand")
        angular_accel = _finite_real_vector(angAccelCommand, 3, "angAccelCommand")
        self.shuttle.SetParam(self._st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], accel)
        self.shuttle.SetParam(
            self._st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], angular_accel
        )

    def GetNearestDebrisLocations(self, N: int) -> list[npt.NDArray[np.float64]]:
        """Return up to N public radar matrices without changing rows or times.

        Row 0 is relative position in P. Row 1 holds closest range, scalar
        range rate and diameter. Row 2 holds measurement time, zero, zero.
        Positive range rate means receding. These are not actor IDs or full
        velocities; ordering is not a tracking identity. Measurement times use
        radar-loop startup, not necessarily the mission elapsed-time origin.
        """
        self._ensure_open()
        if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 0:
            raise ValueError("N must be a non-negative integer")
        all_debris = self.shuttle.GetParamArray(
            self._st.VarType.doubleM3x3, ["Radar", "SortedDebrisData"]
        )
        return all_debris[: min(N, len(all_debris))]

    def PanTiltCamera(self, camera: Any, yaw_deg: float, pitch_deg: float) -> None:
        """Aim topCamera or bottomCamera with the official degree limits/DCM."""
        self._ensure_open()
        yaw_deg, pitch_deg = _finite_real_vector([yaw_deg, pitch_deg], 2, "camera angles")
        if camera == self.topCamera:
            pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
        elif camera == self.bottomCamera:
            pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
        else:
            raise ValueError("camera must be this client's topCamera or bottomCamera")
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        camera.setRotation_DCM(
            self._st.math.Euler321_to_DCM([np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0.0]),
            self.shuttleFrame,
        )

    def elapsed_time_s(self) -> float:
        """Adapter-local seconds from official StartDateTAI and SimClock reads."""
        self._ensure_open()
        now = self._st.SimGlobals.SimClock.GetTimeNow().as_datetime()
        return (now - self._start_date_tai).total_seconds()

    def read_state(self) -> STPState:
        """Collect sequential, frame-qualified official SDK reads into STPState."""
        self._ensure_open()
        elapsed = self.elapsed_time_s()
        loc_vel_acc = self.shuttle.getLocVelAcc()
        return STPState(
            elapsed_time_s=elapsed,
            position_phobos_m=np.array(
                loc_vel_acc.loc_WRT_ExprIn(self.phobosFrame), dtype=np.float64, copy=True
            ),
            velocity_phobos_m_s=np.array(
                loc_vel_acc.vel_WRT_ExprIn(self.phobosFrame), dtype=np.float64, copy=True
            ),
            passive_quaternion_phobos_to_body_xyzw=np.array(
                self.shuttle.getRotation().Quat_WRT(self.phobosFrame),
                dtype=np.float64,
                copy=True,
            ),
            angular_velocity_phobos_rad_s=np.array(
                self.shuttle.getAngVelocity().WRT_ExprIn(self.phobosFrame),
                dtype=np.float64,
                copy=True,
            ),
            target_position_phobos_m=np.array(
                self.target.getLocVelAcc().loc_WRT_ExprIn(self.phobosFrame),
                dtype=np.float64,
                copy=True,
            ),
            mass_kg=float(self.shuttle.getMass()),
            inertia_body_kg_m2=np.array(
                self.shuttle.getInertiaTensor(self.shuttleFrame), dtype=np.float64, copy=True
            ),
            fuel_used_kg=float(
                self.shuttle.GetParam(self._st.VarType.double, ["Stats", "FuelUsed"])
            ),
            damage_j=float(self.shuttle.GetParam(self._st.VarType.double, ["Stats", "Damage"])),
        )

    def capture_image(self, camera: Any, properties: Any) -> Any:
        """Adapter-local delegation to st.camera.CaptureImageBlocking.

        Supply the real SDK's st.CaptureImageProperties. Return its actual
        CapturedImage or None unchanged. No RGB data, timestamps or success
        results are fabricated. Capture properties are caller-selected, not
        claims about mandatory competition optics.
        """
        self._ensure_open()
        return self._st.camera.CaptureImageBlocking(camera, properties)

    def read_only_probe(
        self,
        *,
        radar_clock_offset_s: float | None = None,
        max_radar_age_s: float = 1.5,
    ) -> dict[str, Any]:
        """Read actual runtime data without camera aiming or jet commands.

        The report includes no argv, server URL, tokens or raw SDK handle reprs.
        Unknown SDK versions and running source revisions remain unknown. Source
        pins document the reviewed interface, not the installed server version.
        This method is project reporting, not an invented SDK probe endpoint.
        """
        self._ensure_open()
        if __package__:
            from .control_contract import STP_PHOBOS_FRAME, radar_rows, scalar
        else:
            from control_contract import STP_PHOBOS_FRAME, radar_rows, scalar

        maximum_age = scalar(max_radar_age_s, "max_radar_age_s")
        if radar_clock_offset_s is not None:
            offset = _finite_real_vector([radar_clock_offset_s], 1, "radar_clock_offset_s")[0]
            radar_clock_offset_s = float(offset)
        sdk_version = getattr(self._st, "__version__", None)
        sdk_version_source = "spaceteams.__version__"
        if not isinstance(sdk_version, str) or not sdk_version.strip():
            try:
                sdk_version = version("spaceteams")
                sdk_version_source = "installed spaceteams distribution metadata"
            except PackageNotFoundError:
                sdk_version = None
                sdk_version_source = None
        # Radar first reduces future row times relative to the later state read.
        # It does not turn either sequential report into an atomic snapshot.
        debris = self.GetNearestDebrisLocations(500)
        state = self.read_state()
        now = self.elapsed_time_s()
        frequency = self.loop_frequency_hz
        radar_report: dict[str, Any] = {
            "configured_max_radar_age_s": maximum_age,
            "declared_radar_clock_offset_s": radar_clock_offset_s,
            "observed_min_radar_age_s": None,
            "observed_max_radar_age_s": None,
            "source_clock": "radar-loop startup, not necessarily mission StartDateTAI",
            "atomic": False,
            "failure_reasons": [],
        }
        try:
            rows = radar_rows(debris)
            radar_report["rows"] = len(rows)
            if not len(rows):
                radar_report["failure_reasons"].append("Empty radar is not evidence of clear space")
            else:
                radar_report["oldest_source_radar_time_s"] = float(np.min(rows[:, 2, 0]))
                radar_report["newest_source_radar_time_s"] = float(np.max(rows[:, 2, 0]))
                radar_report["source_time_spread_s"] = float(np.ptp(rows[:, 2, 0]))
                if radar_clock_offset_s is None:
                    radar_report["failure_reasons"].append(
                        "Radar age is unknown without an independently declared clock offset"
                    )
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        times = rows[:, 2, 0] + radar_clock_offset_s
                        ages = now - times
                    if not np.isfinite(times).all() or not np.isfinite(ages).all():
                        raise ValueError("Radar clock conversion is not representable")
                    radar_report["observed_min_radar_age_s"] = float(np.min(ages))
                    radar_report["observed_max_radar_age_s"] = float(np.max(ages))
                    if np.any(times < 0.0) or np.any(times > state.elapsed_time_s):
                        radar_report["failure_reasons"].append("Negative or future radar timestamp")
                    if np.any(ages > maximum_age):
                        radar_report["failure_reasons"].append(
                            "Radar exceeds configured maximum age"
                        )
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            radar_report["failure_reasons"].append(str(error))
        return {
            "schema": "wsorb.stp_read_only_probe",
            "schema_version": 1,
            "commands_written": False,
            "runtime_versions": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "operating_system": platform.system(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "sdk": sdk_version,
                "sdk_version_source": sdk_version_source,
            },
            "reviewed_source_pins": {
                "solution": OFFICIAL_SOLUTION_COMMIT,
                "simulation": OFFICIAL_SIM_COMMIT,
                "user_api_url": OFFICIAL_USER_API_URL,
                "controller_url": OFFICIAL_CONTROLLER_URL,
                "radar_url": OFFICIAL_RADAR_URL,
                "scoring_url": OFFICIAL_SCORING_URL,
                "running_server_revision": None,
            },
            "resolved_handles": {
                name: getattr(self, name) is not None
                for name in (
                    "this",
                    "phobos",
                    "phobosFrame",
                    "shuttle",
                    "shuttleFrame",
                    "target",
                    "topCamera",
                    "topCameraFrame",
                    "bottomCamera",
                    "bottomCameraFrame",
                )
            },
            "clocks": {
                "mission_start_date_tai": self._start_date_tai.isoformat(),
                "state_elapsed_time_s": state.elapsed_time_s,
                "later_elapsed_time_s": now,
                "loop_frequency_hz": frequency,
                "reads_are_sequential": True,
            },
            "state_source_kind": "stp_adapter_numeric_report",
            "state_frame": STP_PHOBOS_FRAME,
            "rotation_convention": "passive_xyzw",
            "sdk_surface_available": {
                "Shuttle.SetParam": callable(getattr(self.shuttle, "SetParam", None)),
                "Shuttle.GetParamArray": callable(getattr(self.shuttle, "GetParamArray", None)),
                "Shuttle.getLocVelAcc": callable(getattr(self.shuttle, "getLocVelAcc", None)),
                "write_path_tested": False,
                "scope": "Actual bound attributes only; no command was written by this probe",
            },
            "state_measurements": {
                "mass_kg": state.mass_kg,
                "fuel_used_kg": state.fuel_used_kg,
                "damage_j": state.damage_j,
            },
            "radar": radar_report,
            "limitations": (
                "No command-path, physics-parity or full-flight validation; no published "
                "mission.status read; radius and sensor-health evidence are unavailable here"
            ),
        }

    def close(self) -> None:
        """Leave the real simulation once, without implicitly changing commands."""
        if not self._closed:
            self._closed = True
            self._st.leave_sim()

    def __enter__(self) -> STPClient:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def connect_from_argv(argv: Sequence[str] | None = None) -> STPClient:
    """Load the real SDK, use its official bootstrap, then resolve controller handles.

    None forwards sys.argv unchanged. Otherwise supply the full argument
    sequence including the program name, exactly as the licensed runtime
    provides it. No server address, port or join arguments are invented here.
    If handle setup fails after connecting, attempt leave_sim before propagating.
    """
    try:
        st = importlib.import_module("spaceteams")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The real 'spaceteams' SDK could not be loaded. It ships with licensed "
            "Space Teams Pro. Launch the ControllerSystem with that runtime's "
            "supported Python interpreter. The Bevy/Rapier browser and Nix "
            "environment do not supply this SDK."
        ) from error
    try:
        st.connect_to_sim(sys.argv if argv is None else list(argv))
    except Exception as error:
        raise RuntimeError(
            "Space Teams Pro connection failed. Run inside the licensed competition "
            "ControllerSystem with its runtime-provided argv; this adapter does "
            "not connect to the local browser simulation."
        ) from error
    try:
        return STPClient(st)
    except BaseException:
        with suppress(Exception):
            st.leave_sim()
        raise
