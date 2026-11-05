"""Opt-in portable control math, not a flight-safety or SDK compatibility claim.

Uses only STPState and the public anonymous radar matrices. There is no Phobos
terrain, shuttle clearance, target velocity, vision, route search, independent
navigation sensor or SDK watchdog here. Radial repulsion cannot resolve unknown
tangential debris motion or guarantee collision avoidance. Target PD damps
velocity relative to Phobos, not relative to a moving target. There is no
official arrival-region geometry or mission-complete signal in these inputs.
Checkpoint control uses an explicitly reviewed radius and an interior waypoint.
Its within-radius measurement is not the published scoring system's alert.

Radar time uses a different startup origin. The operator must declare
``mission_measurement_time = source_radar_time + radar_clock_offset_s`` from
the runtime's clock configuration. No offset is estimated from observations.
Source rows and timestamps remain unchanged. State reads are sequential, and
neither their report time nor a later clock read makes them atomic.

Importing this module loads no SDK, connects nowhere and reads no configuration.
It also supports direct imports beside stp_api.py in the licensed entry point.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

if __package__:
    from .control_contract import (
        STP_PHOBOS_FRAME,
        ControlSample,
        body_to_reference,
        bounded,
        checked_sample,
        coast_command,
        radar_rows,
        reference_to_body,
        scalar,
        vector,
    )
    from .stp_api import STPState
else:
    # Direct sibling imports avoid the historical package and its Python 3.12+ syntax.
    from control_contract import (
        STP_PHOBOS_FRAME,
        ControlSample,
        body_to_reference,
        bounded,
        checked_sample,
        coast_command,
        radar_rows,
        reference_to_body,
        scalar,
        vector,
    )
    from stp_api import STPState

CONFIG_SCHEMA = "wsorb.stp_policy"
CONFIG_SCHEMA_VERSION = 1
CONFIG_FILE_ENV = "WSORB_STP_POLICY_FILE"
CONFIG_JSON_ENV = "WSORB_STP_POLICY_JSON"
MAX_CONFIG_BYTES = 16 * 1024
_TIME_TOLERANCE_S = 1.0e-12  # Absorbs only fixed-step summation noise, never a sensor offset.
_AXIS_MIN_PROGRESS_FRACTION = 0.9  # Chosen guard, not a calibrated meeting parameter.
_NORM_HEADROOM_REL = 8.0 * np.finfo(np.float64).eps
# Live diagnostic budgets, not a change to raw-radar control coverage. Association
# is conditional on the declared rotating subset, never all publication rivals.
_TRACKING_LIVE_BUDGET = {
    "max_tracks": 32,
    "max_input_rows": 4096,
    "max_rows_per_update": 16,
    "max_candidate_pairs": 512,
    "max_fingerprints": 1024,
    "report_limit": 4,
}

# Review/input budgets for this baseline, not properties of the licensed SDK.
_BOUNDS = {
    "duration_wall_s": (0.0, 3600.0),
    "max_report_age_s": (0.0, 5.4),
    "max_radar_age_s": (0.0, 5.4),
    "position_gain_s2": (0.0, 1.0),
    "velocity_gain_s": (0.0, 10.0),
    "angular_velocity_gain_s": (0.0, 10.0),
    "avoidance_range_m": (0.0, 1e6),
    "avoidance_horizon_s": (0.0, 30.0),
    "avoidance_accel_mps2": (0.0, 0.5),
    "max_accel_mps2": (0.0, 0.5),
    "max_angular_accel_rad_s2": (0.0, 0.05),
    "radar_clock_offset_s": (-1e9, 1e9),
    "checkpoint_radius_m": (0.0, 1e6),
    "checkpoint_inset_m": (0.0, 1e6),
    "max_speed_mps": (0.0, 1000.0),
    "control_horizon_s": (0.01, 10.0),
    "axis_velocity_error_slack_mps": (0.0, 1.0),
    "checkpoint_speed_tolerance_mps": (0.0, 10.0),
    "radar_clearance_m": (0.0, 1000.0),
    "radar_position_allowance_m": (0.0, 1000.0),
    "radar_rate_allowance_mps": (0.0, 1000.0),
    "unknown_tangential_speed_mps": (0.0, 1000.0),
}


def _number(value: Any, name: str) -> float:
    low, high = _BOUNDS[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [{low}, {high}]")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not a representable finite real number") from error
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite real number in [{low}, {high}]")
    if (
        name
        in (
            "duration_wall_s",
            "avoidance_range_m",
            "checkpoint_radius_m",
            "checkpoint_inset_m",
            "max_speed_mps",
        )
        and value == 0.0
    ):
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class STPPolicyConfig:
    """Immutable, validated policy data. Construction alone never commands jets.

    Enabling requires a declared radar offset, even when that offset is zero.
    Gains and repulsion distances are operator-reviewed tuning assumptions, not
    geometry or calibrated sensor bounds. Norm limits can only lower the public
    actuator caps. duration_wall_s limits the example's wall-clock loop, not SDK
    call duration, command persistence or process-death cleanup.
    """

    enabled: bool = False
    radar_clock_offset_s: float | None = None
    duration_wall_s: float = 30.0
    max_report_age_s: float = 1.0
    max_radar_age_s: float = 1.5
    position_gain_s2: float = 0.0012
    velocity_gain_s: float = 0.07
    angular_velocity_gain_s: float = 0.1
    avoidance_range_m: float = 70.0
    avoidance_horizon_s: float = 3.0
    avoidance_accel_mps2: float = 0.5
    max_accel_mps2: float = 0.5
    max_angular_accel_rad_s2: float = 0.05
    policy: str = "legacy_pd_radar"
    body_axis_preference: bool = False
    tracking_diagnostics: bool = True
    max_speed_mps: float = 8.0
    checkpoint_radius_m: float = 50.0
    checkpoint_inset_m: float = 1.0
    control_horizon_s: float = 0.75
    axis_velocity_error_slack_mps: float = 0.02
    checkpoint_speed_tolerance_mps: float = 0.2
    radar_clearance_m: float = 10.0
    radar_position_allowance_m: float = 3.0
    radar_rate_allowance_mps: float = 0.5
    unknown_tangential_speed_mps: float = 2.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be an explicit boolean")
        if self.policy not in ("legacy_pd_radar", "checkpoint_velocity"):
            raise ValueError("policy must be legacy_pd_radar or checkpoint_velocity")
        for name in ("body_axis_preference", "tracking_diagnostics"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit boolean")
        for name in _BOUNDS:
            value = getattr(self, name)
            if name == "radar_clock_offset_s" and value is None:
                if self.enabled:
                    raise ValueError("An enabled policy requires explicit radar_clock_offset_s")
                continue
            object.__setattr__(self, name, _number(value, name))
        if self.policy == "legacy_pd_radar" and self.duration_wall_s > 300.0:
            raise ValueError("legacy_pd_radar duration_wall_s must be at most 300 seconds")
        if self.checkpoint_inset_m >= self.checkpoint_radius_m:
            raise ValueError("checkpoint_inset_m must be less than checkpoint_radius_m")
        if (
            self.control_horizon_s > self.avoidance_horizon_s
            and self.policy == "checkpoint_velocity"
        ):
            raise ValueError("control_horizon_s must not exceed avoidance_horizon_s")

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> STPPolicyConfig:
        """Parse versioned data with explicit enable, rejecting unknown fields."""
        if not isinstance(document, Mapping) or any(type(key) is not str for key in document):
            raise ValueError("STP policy configuration must be an object with string keys")
        required = {"schema", "schema_version", "enabled"}
        allowed = required | {item.name for item in fields(cls)}
        if required - document.keys() or document.keys() - allowed:
            raise ValueError("STP policy configuration has missing or unknown fields")
        if document["schema"] != CONFIG_SCHEMA:
            raise ValueError(f"schema must be {CONFIG_SCHEMA!r}")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != CONFIG_SCHEMA_VERSION
        ):
            raise ValueError("schema_version must be the integer 1")
        return cls(
            **{
                name: value
                for name, value in document.items()
                if name not in ("schema", "schema_version")
            }
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate STP policy JSON key: {name}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"STP policy JSON cannot contain {value}")


def parse_config(source: str | bytes) -> STPPolicyConfig:
    """Parse at most 16 KiB of strict UTF-8 JSON without executing any code."""
    if not isinstance(source, (str, bytes)) or len(source) > MAX_CONFIG_BYTES:
        raise ValueError("STP policy configuration must be at most 16384 UTF-8 bytes")
    try:
        data = source.encode("utf-8") if isinstance(source, str) else source
        if len(data) > MAX_CONFIG_BYTES:
            raise ValueError("STP policy configuration must be at most 16384 UTF-8 bytes")
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("STP policy configuration must be a valid UTF-8 JSON object") from error
    return STPPolicyConfig.from_mapping(document)


def load_config_from_environment(environ: Mapping[str, str] | None = None) -> STPPolicyConfig:
    """Load one local file or inline JSON, never SDK bootstrap arguments.

    Select WSORB_STP_POLICY_FILE or WSORB_STP_POLICY_JSON, not both. Missing
    configuration disables the policy; present but invalid configuration raises
    before the example connects. Configuration is read once, not hot-reloaded.
    """
    environ = os.environ if environ is None else environ
    present = [name for name in (CONFIG_FILE_ENV, CONFIG_JSON_ENV) if name in environ]
    if not present:
        return STPPolicyConfig()
    if len(present) != 1:
        raise ValueError("Select only one of WSORB_STP_POLICY_FILE and WSORB_STP_POLICY_JSON")
    value = environ[present[0]]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{present[0]} must be a nonempty string")
    if present[0] == CONFIG_JSON_ENV:
        return parse_config(value)
    try:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError("WSORB_STP_POLICY_FILE must name a local regular JSON file")
        with path.open("rb") as handle:
            source = handle.read(MAX_CONFIG_BYTES + 1)
    except OSError as error:
        raise ValueError("Could not read WSORB_STP_POLICY_FILE") from error
    return parse_config(source)


def sample_from_stp_state(state: STPState) -> ControlSample:
    """Copy the adapter report with its actual P frame and unchanged SDK xyzw.

    No SDK call, quaternion conjugation or local-physics relabelling occurs here.
    This conversion is only for real adapter-local STPState objects. The local
    API runner constructs its own ControlSample with its actual source/frame.
    """
    if not isinstance(state, STPState):
        raise ValueError("state must be an actual adapter-local STPState")
    return ControlSample(
        elapsed_time_s=state.elapsed_time_s,
        position_world_m=state.position_phobos_m,
        velocity_world_mps=state.velocity_phobos_m_s,
        quaternion_xyzw=state.passive_quaternion_phobos_to_body_xyzw,
        rotation_convention="passive_xyzw",
        angular_velocity_world_rad_s=state.angular_velocity_phobos_rad_s,
        target_position_world_m=state.target_position_phobos_m,
        mass_kg=state.mass_kg,
        inertia_body_kg_m2=state.inertia_body_kg_m2,
        fuel_used_kg=state.fuel_used_kg,
        damage_j=state.damage_j,
        frame=STP_PHOBOS_FRAME,
        source_kind="stp_adapter_numeric_report",
    )


def _checked_state(
    state: STPState,
) -> tuple[
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    sample = checked_sample(sample_from_stp_state(state))
    return (
        sample.elapsed_time_s,
        sample.position_world_m,
        sample.velocity_world_mps,
        sample.quaternion_xyzw,
        sample.angular_velocity_world_rad_s,
        sample.target_position_world_m,
    )


def decide_command(
    state: STPState,
    debris: npt.ArrayLike,
    config: STPPolicyConfig | None = None,
    *,
    now_elapsed_time_s: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Legacy stateless PD/radial baseline, retaining its original numeric API.

    This wrapper remains the legacy baseline even with a checkpoint config.
    Use STPCheckpointController for explicit persistent checkpoint guidance.

    With enable and valid data, combine target PD with bounded repulsion away
    from each public radar direction. Only closing scalar range rate advances
    the closest range over sample age plus the configured radial lookahead.
    Negative noisy ranges count as zero clearance. Diameter is validated, not
    subtracted from a range that already measures the closest point. Increasing
    proximity suppresses target PD. Angular control only damps relative spin.

    A fresh, separately read adapter elapsed clock is required. Any invalid
    state, missing/empty/malformed radar, stale sample/report, future timestamp
    or unrepresentable calculation returns two actual zero vectors. Radar must
    not be future-dated even relative to the earlier state report clock. Only a
    1e-12 s float-accumulation tolerance is used; no timestamp clipping, inferred
    offset or backdating is used.
    These checks are data checks, not independent sensor-health evidence or
    proof that the public radar covers every hazard.
    """
    if config is None:
        return coast_command()
    if not isinstance(config, STPPolicyConfig):
        raise ValueError("config must be STPPolicyConfig")
    if not config.enabled:
        return coast_command()
    try:
        elapsed, position, velocity, quaternion, angular_velocity, target = _checked_state(state)
        now = scalar(now_elapsed_time_s, "now_elapsed_time_s")
        if now < elapsed or now - elapsed > config.max_report_age_s:
            raise ValueError("Future or stale state report")
        rows = radar_rows(debris)
        if len(rows) == 0:
            raise ValueError("Empty radar is not evidence of clear space")
        with np.errstate(over="ignore", invalid="ignore"):
            # Keep the original rows, especially row 2, unchanged.
            times = rows[:, 2, 0] + config.radar_clock_offset_s
            ages = now - times
        if (
            not np.isfinite(times).all()
            or not np.isfinite(ages).all()
            or np.any(times < -_TIME_TOLERANCE_S)
            or np.any(times > elapsed + _TIME_TOLERANCE_S)
            or np.any(ages > config.max_radar_age_s + _TIME_TOLERANCE_S)
        ):
            raise ValueError("Radar timestamps are invalid, future or stale after declared offset")
        with np.errstate(over="ignore", invalid="ignore"):
            pd = (target - position) * config.position_gain_s2 - velocity * config.velocity_gain_s
            radial_ranges = rows[:, 1, 0] + np.minimum(rows[:, 1, 1], 0.0) * (
                ages + config.avoidance_horizon_s
            )
        if not np.isfinite(radial_ranges).all():
            raise ValueError("Radial lookahead is not representable")
        # Divide only inside the response range. Avoid overflowing a quotient
        # for distant rows whose repulsion weight is already known to be zero.
        weights = np.zeros(len(rows), dtype=np.float64)
        nearby = radial_ranges < config.avoidance_range_m
        weights[nearby] = 1.0 - np.maximum(radial_ranges[nearby], 0.0) / config.avoidance_range_m
        lengths = np.hypot(np.hypot(rows[:, 0, 0], rows[:, 0, 1]), rows[:, 0, 2])
        directions = rows[:, 0, :] / lengths[:, None]
        repulsion = bounded(
            -np.sum(directions * weights[:, None], axis=0) * config.avoidance_accel_mps2,
            config.avoidance_accel_mps2,
        )
        acceleration = bounded(
            bounded(pd, config.max_accel_mps2) * (1.0 - float(np.max(weights))) + repulsion,
            config.max_accel_mps2,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            angular_acceleration = vector(
                -angular_velocity * config.angular_velocity_gain_s, "angular acceleration"
            )
        return (
            bounded(
                reference_to_body(acceleration, quaternion, "passive_xyzw"),
                config.max_accel_mps2,
            ),
            bounded(
                reference_to_body(angular_acceleration, quaternion, "passive_xyzw"),
                config.max_angular_accel_rad_s2,
            ),
        )
    except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
        return coast_command()


def _fresh_radar(
    debris: npt.ArrayLike,
    config: STPPolicyConfig,
    elapsed: float,
    now: float,
    freshness: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rows = radar_rows(debris)
    freshness["radar_rows"] = len(rows)
    if not len(rows):
        raise ValueError("Empty radar is not evidence of clear space")
    with np.errstate(over="ignore", invalid="ignore"):
        times = rows[:, 2, 0] + config.radar_clock_offset_s
        ages = now - times
    if not np.isfinite(times).all() or not np.isfinite(ages).all():
        raise ValueError("Radar clock conversion is not representable")
    freshness.update(
        oldest_source_radar_time_s=float(np.min(rows[:, 2, 0])),
        newest_source_radar_time_s=float(np.max(rows[:, 2, 0])),
        observed_min_radar_age_s=float(np.min(ages)),
        observed_max_radar_age_s=float(np.max(ages)),
        measurement_time_spread_s=float(np.max(times) - np.min(times)),
    )
    if np.any(times < -_TIME_TOLERANCE_S) or np.any(times > elapsed + _TIME_TOLERANCE_S):
        raise ValueError("Radar timestamps are negative or future after the declared offset")
    if np.any(ages > config.max_radar_age_s + _TIME_TOLERANCE_S):
        raise ValueError("Stale radar exceeds configured max_radar_age_s")
    return rows, ages


def decision_is_current(report: Mapping[str, Any], *, now_elapsed_time_s: float) -> bool:
    """Pure post-compute deadline check, not an SDK decision lease or watchdog.

    The example must supply a newly read real SDK clock after all computation.
    Do not extend the original report/radar deadline by retiming old samples.
    """
    try:
        now = scalar(now_elapsed_time_s, "now_elapsed_time_s")
        deadline = scalar(report["decision_valid_until_elapsed_s"], "decision deadline")
        checked_at = scalar(
            report["source_freshness"]["now_elapsed_time_s"], "decision input clock"
        )
        return checked_at <= now <= deadline
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def _radial_response(
    guidance: npt.ArrayLike,
    rows: npt.NDArray[np.float64],
    ages: npt.NDArray[np.float64],
    config: STPPolicyConfig,
) -> npt.NDArray[np.float64]:
    """Legacy radial response, shared by the persistent guidance modes."""
    with np.errstate(over="ignore", invalid="ignore"):
        ranges = rows[:, 1, 0] + np.minimum(rows[:, 1, 1], 0.0) * (
            ages + config.avoidance_horizon_s
        )
    if not np.isfinite(ranges).all():
        raise ValueError("Radial lookahead is not representable")
    weights = np.zeros(len(rows), dtype=np.float64)
    nearby = ranges < config.avoidance_range_m
    weights[nearby] = 1.0 - np.maximum(ranges[nearby], 0.0) / config.avoidance_range_m
    lengths = np.hypot(np.hypot(rows[:, 0, 0], rows[:, 0, 1]), rows[:, 0, 2])
    repulsion = bounded(
        -np.sum(rows[:, 0, :] / lengths[:, None] * weights[:, None], axis=0)
        * config.avoidance_accel_mps2,
        config.avoidance_accel_mps2,
    )
    return bounded(
        bounded(guidance, config.max_accel_mps2) * (1.0 - float(np.max(weights))) + repulsion,
        config.max_accel_mps2,
    )


def _command_bounded(value: npt.ArrayLike, maximum: float) -> npt.NDArray[np.float64]:
    """Leave float64 norm headroom BEFORE checking the actual candidate.

    Real Node Math.hypot can round a body vector above 0.5 when Python's hypot
    rounds the same components to exactly 0.5. This is stricter than the cap,
    not a transport retry, SDK rescale or relaxation of any stopping criterion.
    """
    ceiling = float(np.nextafter(maximum * (1.0 - _NORM_HEADROOM_REL), 0.0))
    return bounded(value, ceiling)


class STPCheckpointController:
    """Persistent project controller with no connection or command-writing effects.

    Both real STP and local API producers pass their own ControlSample into
    this exact math. Neither producer may claim the other's SDK or physics.
    Tracking is diagnostic only. Every proposed body-axis command is converted
    back to the declared reference frame and checked before selection; projecting
    an already approved diagonal command onto an axis is not permitted.
    """

    def __init__(self, config: STPPolicyConfig) -> None:
        if not isinstance(config, STPPolicyConfig):
            raise ValueError("config must be STPPolicyConfig")
        if __package__:
            from .tracking import RadarTracker, TrackerConfig
        else:
            from tracking import RadarTracker, TrackerConfig

        self.config = config
        self._tracker = (
            RadarTracker(TrackerConfig(**_TRACKING_LIVE_BUDGET))
            if config.tracking_diagnostics
            else None
        )
        self._diagnostic_cursor = 0
        self._previous_axis: int | None = None
        self._approach_direction: npt.NDArray[np.float64] | None = None
        self._source: tuple[str, str, str] | None = None
        self._last_now: float | None = None
        self._calls = 0
        self._coast_calls = 0
        self._report = self._new_report()

    def _new_report(self) -> dict[str, Any]:
        return {
            "schema": "wsorb.checkpoint_control_report",
            "schema_version": 1,
            "policy": self.config.policy,
            "body_axis_preference": self.config.body_axis_preference,
            "tracking_diagnostics": self.config.tracking_diagnostics,
            "enabled": self.config.enabled,
            "config": asdict(self.config),
            "calls": self._calls,
            "coast_calls": self._coast_calls,
            "failure_reasons": [],
            "decision_valid_until_elapsed_s": None,
            "source_freshness": {
                "declared_radar_clock_offset_s": self.config.radar_clock_offset_s,
                "configured_max_radar_age_s": self.config.max_radar_age_s,
                "configured_max_report_age_s": self.config.max_report_age_s,
                "observed_min_radar_age_s": None,
                "observed_max_radar_age_s": None,
                "sample_atomicity": "not guaranteed by this numeric contract",
            },
            "checkpoint": None,
            "safety_filter": None,
            "candidate_choice": {"selected_kind": None, "candidates": []},
            "track_summary": None if self.config.tracking_diagnostics else {"enabled": False},
            "tracking_work": {
                "enabled": self.config.tracking_diagnostics,
                "limits": dict(_TRACKING_LIVE_BUDGET),
                "raw_control_radar_rows": 0,
                "tracker_selected_rows": 0,
                "publication_subsampled": False,
                "association_scope": (
                    "rotating public-row subset only; omitted rows may contain competing rivals; "
                    "no full-publication tracking or physical identity claim"
                ),
            },
            "command": None,
            "model_assumptions": {
                "reference_frame": "producer-declared, not assumed inertial",
                "target_velocity": "unavailable; guidance damps velocity relative to named frame",
                "checkpoint_geometry": (
                    "operator radius must be reviewed against server Scoring KeepOutRadius_m; "
                    "target hull, shuttle hull and terrain are unavailable"
                ),
                "official_scoring": "not observable here; published Scoring uses OnScreenAlert",
                "radar_publication": (
                    "anonymous, mixed row times, not atomic; pinned 200-row 120 Hz scan "
                    "can take about 1.67 s, so age bound is an explicit operator choice"
                ),
                "radar_prediction": (
                    "radial constant velocity plus declared unknown tangential/rate/position "
                    "allowances; constant observer velocity backdating with 0.5 m/s2 allowance"
                ),
                "candidate_prediction": (
                    "constant current attitude and candidate reference acceleration over short "
                    "horizon; chord inflated by acceleration sag; no gravity/contact model"
                ),
                "command_norm_headroom": (
                    "8 float64 epsilons plus one ulp below configured caps before candidate "
                    "checks, for cross-runtime norm rounding; never post-approval rescaling"
                ),
                "stopping_check": (
                    "checkpoint inward-speed reserve at half configured acceleration, "
                    "not a terrain or moving-debris braking guarantee"
                ),
                "braking_filter": (
                    "before axis ranking, a nominal stopping-reserve failure may use bounded "
                    "full velocity-opposing braking only after the SAME raw-radar and next-hold "
                    "stopping checks pass; no constraint relaxation or source retiming"
                ),
                "tracker": "bounded estimates and ambiguous associations, never obstacle clearance",
                "axis_progress": (
                    "retain at least 90 percent of baseline one-hold velocity-error reduction; "
                    "effective slack is also capped by configured absolute slack; chosen assumption"
                ),
                "angular_control": "bounded relative-spin damping only; no initial reorientation",
                "fuel_comparison": (
                    "body-command L1 is an allocation proxy, not official jet fuel; "
                    "local CoM fuel uses L2, so no 41 percent savings claim"
                ),
                "safety": (
                    "input/prediction checks only; coast does not stop motion or ensure safety"
                ),
            },
        }

    @property
    def report(self) -> dict[str, Any]:
        """Detached strict-JSON data. Mutating it cannot change controller state."""
        return json.loads(json.dumps(self._report, allow_nan=False))

    def _coast(
        self, reason: str, now: float | None, frame: str | None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self._coast_calls += 1
        self._report["coast_calls"] = self._coast_calls
        self._report["failure_reasons"].append(reason)
        self._report["command"] = {
            "accel_body_mps2": [0.0, 0.0, 0.0],
            "angular_accel_body_rad_s2": [0.0, 0.0, 0.0],
            "kind": "coast",
            "written_to_any_api": False,
        }
        if self._tracker is not None and now is not None and frame is not None:
            try:
                self._report["track_summary"] = self._tracker.coast(now, frame=frame)
            except (
                ValueError,
                TypeError,
                OverflowError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ):
                self._report["track_summary"] = {
                    "failure_reasons": ["Diagnostic tracker coast failed"]
                }
        return coast_command()

    def _checkpoint_guidance(
        self, sample: ControlSample
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        config = self.config
        position = sample.position_world_m
        target = sample.target_position_world_m
        with np.errstate(over="ignore", invalid="ignore"):
            approach = vector(position - target, "checkpoint approach")
        distance = math.hypot(*approach)
        if not math.isfinite(distance):
            raise ValueError("Target distance is not representable")
        if self._approach_direction is None:
            self._approach_direction = (
                approach / distance if distance else np.array([1.0, 0.0, 0.0])
            )
        with np.errstate(over="ignore", invalid="ignore"):
            goal = vector(
                target
                + self._approach_direction
                * (config.checkpoint_radius_m - config.checkpoint_inset_m),
                "interior checkpoint waypoint",
            )
            delta = vector(goal - position, "checkpoint waypoint error")
        remaining = math.hypot(*delta)
        if not math.isfinite(remaining):
            raise ValueError("Waypoint distance is not representable")
        # Reserve half the command cap and one P-loop time constant. These are
        # declared tuning assumptions, not measured actuator/gravity authority.
        authority = config.max_accel_mps2 * 0.5
        speed = 0.0
        if authority > 0.0 and config.velocity_gain_s > 0.0 and remaining > 0.0:
            lag_speed = authority / config.velocity_gain_s
            speed = min(
                config.max_speed_mps,
                2.0
                * authority
                * remaining
                / (math.hypot(lag_speed, math.sqrt(2.0 * authority * remaining)) + lag_speed),
            )
        desired_velocity = delta / remaining * speed if remaining else np.zeros(3)
        with np.errstate(over="ignore", invalid="ignore"):
            guidance = bounded(
                (desired_velocity - sample.velocity_world_mps) * config.velocity_gain_s,
                config.max_accel_mps2,
            )
        frame_speed = math.hypot(*sample.velocity_world_mps)
        if not math.isfinite(frame_speed):
            raise ValueError("Frame-relative speed is not representable")
        self._report["checkpoint"] = {
            "radius_m": config.checkpoint_radius_m,
            "inset_m": config.checkpoint_inset_m,
            "interior_waypoint_world_m": goal.tolist(),
            "distance_to_target_m": distance,
            "distance_to_interior_waypoint_m": remaining,
            "within_configured_radius": distance <= config.checkpoint_radius_m,
            "frame_relative_speed_mps": frame_speed,
            "configured_speed_tolerance_mps": config.checkpoint_speed_tolerance_mps,
            "within_radius_and_low_frame_speed": (
                distance <= config.checkpoint_radius_m
                and frame_speed <= config.checkpoint_speed_tolerance_mps
            ),
            "threshold_kind": "derived policy measurement, not official scoring or a flight freeze",
            "official_scoring_status": None,
            "geometry_source": "operator configuration, not an SDK geometry read",
            "desired_velocity_world_mps": desired_velocity.tolist(),
            "desired_speed_mps": speed,
        }
        return guidance, desired_velocity

    def _candidate_guard(
        self,
        accel_body: npt.ArrayLike,
        sample: ControlSample,
        rows: npt.NDArray[np.float64],
        ages: npt.NDArray[np.float64],
    ) -> dict[str, Any]:
        """Check the actual candidate, never an unprojected parent command.

        Raw closest-point and diameter reports both contribute to the envelope;
        diameter is not subtracted again from closest range. Tracker estimates
        cannot shrink any radius, age or missing-motion allowance in this check.
        """
        config = self.config
        body = vector(accel_body, "candidate body acceleration")
        if math.hypot(*body) > config.max_accel_mps2:
            return {"accepted": False, "failure_reasons": ["Candidate exceeds acceleration cap"]}
        acceleration = body_to_reference(body, sample.quaternion_xyzw, sample.rotation_convention)
        horizon = config.avoidance_horizon_s
        lengths = np.hypot(np.hypot(rows[:, 0, 0], rows[:, 0, 1]), rows[:, 0, 2])
        directions = rows[:, 0, :] / lengths[:, None]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            # Public range rate determines only one component of debris motion.
            # Zero nominal tangential motion is an assumption with an explicit
            # expanding allowance, not a recovered full velocity measurement.
            radial_world_velocity = (
                directions * (rows[:, 1, 1] + directions @ sample.velocity_world_mps)[:, None]
            )
            relative_velocity = radial_world_velocity - sample.velocity_world_mps
            starts = rows[:, 0, :] + relative_velocity * ages[:, None]
            ends = starts + relative_velocity * horizon - 0.5 * acceleration * horizon**2
            uncertainty = (
                config.radar_clearance_m
                + config.radar_position_allowance_m
                + (config.radar_rate_allowance_mps + config.unknown_tangential_speed_mps)
                * (ages + horizon)
                + 0.25 * ages**2
                + 0.125 * math.hypot(*acceleration) * horizon**2
            )
            radii = np.maximum(rows[:, 1, 2] * 0.5, lengths - rows[:, 1, 0]) + uncertainty
            segments = ends - starts
            squared_lengths = np.sum(segments * segments, axis=1)
            fractions = np.zeros(len(rows))
            moving = squared_lengths > 0.0
            fractions[moving] = np.clip(
                -np.sum(starts[moving] * segments[moving], axis=1) / squared_lengths[moving],
                0.0,
                1.0,
            )
            closest = starts + segments * fractions[:, None]
            margins = np.hypot(np.hypot(closest[:, 0], closest[:, 1]), closest[:, 2]) - radii
        if not all(
            np.isfinite(value).all()
            for value in (starts, ends, radii, squared_lengths, fractions, margins)
        ):
            raise ValueError("Candidate radar prediction is not representable")
        reasons = []
        radar_margin = float(np.min(margins))
        if radar_margin <= 0.0:
            reasons.append("Candidate intersects a raw-radar prediction envelope")
        stopping_margin = None
        if self._report["checkpoint"] is not None:
            waypoint = np.array(self._report["checkpoint"]["interior_waypoint_world_m"])
            duration = config.control_horizon_s
            with np.errstate(over="ignore", invalid="ignore"):
                end_velocity = vector(
                    sample.velocity_world_mps + acceleration * duration, "candidate end velocity"
                )
                end_error = vector(
                    waypoint
                    - sample.position_world_m
                    - sample.velocity_world_mps * duration
                    - 0.5 * acceleration * duration**2,
                    "candidate waypoint error",
                )
            initial_error = vector(waypoint - sample.position_world_m, "initial waypoint error")
            initial_distance = math.hypot(*initial_error)
            direction = (
                initial_error / initial_distance if initial_distance else -self._approach_direction
            )
            # Keep the ORIGINAL approach direction. Flipping it after a fast
            # candidate has crossed the waypoint would falsely erase overshoot.
            signed_remaining = float(end_error @ direction)
            inward_speed = max(0.0, float(end_velocity @ direction))
            authority = config.max_accel_mps2 * 0.5
            if authority == 0.0 and inward_speed > 0.0:
                reasons.append("No configured acceleration authority for checkpoint stopping")
            else:
                stop_distance = inward_speed**2 / (2.0 * authority) if authority else 0.0
                # The interior point leaves a configured margin to the boundary.
                stopping_margin = signed_remaining + config.checkpoint_inset_m - stop_distance
                if not math.isfinite(stopping_margin):
                    raise ValueError("Candidate checkpoint stopping margin is not representable")
                if stopping_margin < 0.0:
                    reasons.append("Candidate lacks configured checkpoint stopping reserve")
        return {
            "accepted": not reasons,
            "failure_reasons": reasons,
            "radar_rows_checked": len(rows),
            "min_raw_radar_prediction_margin_m": radar_margin,
            "checkpoint_stopping_margin_m": stopping_margin,
            "accel_world_mps2_checked": acceleration.tolist(),
        }

    def _filtered_baseline(
        self,
        nominal: npt.NDArray[np.float64],
        sample: ControlSample,
        rows: npt.NDArray[np.float64],
        ages: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64] | None, dict[str, Any]]:
        """Keep the reserve criterion and strengthen braking, not coast into it.

        The nominal P loop may decelerate too weakly to keep the next hold inside
        the declared stopping reserve. An actual velocity-opposing command can
        still preserve that reserve. This bounded two-option filter checks that
        command before it becomes the baseline seen by body-axis ranking.
        Failure proves neither global infeasibility nor safety of coasting.
        """
        nominal_check = self._candidate_guard(nominal, sample, rows, ages)
        report = {
            "activated": False,
            "selected_source": "nominal",
            "criteria_relaxed": False,
            "attempts": [
                {
                    "source": "nominal",
                    "accel_body_mps2": nominal.tolist(),
                    **nominal_check,
                }
            ],
        }
        self._report["safety_filter"] = report
        margin = nominal_check.get("checkpoint_stopping_margin_m")
        if margin is None or margin >= 0.0:
            # Radar-only failures still go through the existing candidate guards;
            # they never count as approved merely because the stop filter is idle.
            return nominal, nominal_check
        report["activated"] = True
        speed = math.hypot(*sample.velocity_world_mps)
        if not math.isfinite(speed):
            raise ValueError("Braking velocity norm is not representable")
        braking_world = (
            -sample.velocity_world_mps / speed * self.config.max_accel_mps2
            if speed
            else np.zeros(3)
        )
        braking = _command_bounded(
            reference_to_body(braking_world, sample.quaternion_xyzw, sample.rotation_convention),
            self.config.max_accel_mps2,
        )
        braking_check = self._candidate_guard(braking, sample, rows, ages)
        report["attempts"].append(
            {
                "source": "bounded_velocity_braking",
                "accel_body_mps2": braking.tolist(),
                **braking_check,
            }
        )
        if braking_check["accepted"]:
            report["selected_source"] = "bounded_velocity_braking"
            return braking, braking_check
        report["selected_source"] = None
        report["failure_reason"] = (
            "Neither tested nominal nor bounded braking command passed all unchanged guards; "
            "coast is not a claim of safety or proof that no other control exists"
        )
        return None, braking_check

    def command(
        self,
        sample: ControlSample,
        debris: npt.ArrayLike,
        *,
        now_elapsed_time_s: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return bounded body vectors. Invalid/missing/future/stale data coasts.

        This function writes to no API. In particular a disabled controller must
        not be used as permission to write its zero return to a licensed shuttle.
        A fresh clock and independently declared radar offset remain mandatory.
        """
        self._calls += 1
        self._report = self._new_report()
        now, frame = None, None
        if not self.config.enabled:
            return self._coast("Policy is disabled; caller must remain read-only", now, frame)
        try:
            checked_sample(sample)
            now = scalar(now_elapsed_time_s, "now_elapsed_time_s")
            frame = sample.frame
            elapsed = scalar(sample.elapsed_time_s, "elapsed_time_s")
            freshness = self._report["source_freshness"]
            freshness.update(
                frame=frame,
                source_kind=sample.source_kind,
                rotation_convention=sample.rotation_convention,
                sample_elapsed_time_s=elapsed,
                now_elapsed_time_s=now,
                report_age_s=now - elapsed,
            )
            if now < elapsed or now - elapsed > self.config.max_report_age_s:
                raise ValueError("Future or stale state report")
            if self._last_now is not None and now < self._last_now:
                raise ValueError("Elapsed clock moved backwards; construct a new controller")
            source = (sample.frame, sample.source_kind, sample.rotation_convention)
            if self._source is not None and source != self._source:
                raise ValueError(
                    "Sample frame/source/convention changed; construct a new controller"
                )
            rows, ages = _fresh_radar(debris, self.config, elapsed, now, freshness)
            self._report["tracking_work"]["raw_control_radar_rows"] = len(rows)
            self._report["decision_valid_until_elapsed_s"] = min(
                elapsed + self.config.max_report_age_s,
                freshness["oldest_source_radar_time_s"]
                + self.config.radar_clock_offset_s
                + self.config.max_radar_age_s,
            )
            self._source = source
            self._last_now = now
            if self._tracker is not None:
                try:
                    count = min(len(rows), _TRACKING_LIVE_BUDGET["max_rows_per_update"])
                    indices = (np.arange(count) + self._diagnostic_cursor) % len(rows)
                    selected_rows = rows[indices]
                    self._diagnostic_cursor = int((self._diagnostic_cursor + count) % len(rows))
                    self._report["tracking_work"].update(
                        tracker_selected_rows=count,
                        publication_subsampled=count < len(rows),
                        omitted_publication_rows=len(rows) - count,
                    )
                    # These ego coordinates belong to the original state report,
                    # not the later clock read. Preserve that epoch in tracking.
                    self._report["track_summary"] = self._tracker.update(
                        elapsed,
                        sample.position_world_m,
                        sample.velocity_world_mps,
                        selected_rows,
                        frame=frame,
                        radar_clock_offset_s=self.config.radar_clock_offset_s,
                    )
                    self._report["track_summary"]["control_read_time_s"] = now
                    self._report["track_summary"]["ego_state_time_s"] = elapsed
                except (
                    ValueError,
                    TypeError,
                    OverflowError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ):
                    # Diagnostic estimates never decide raw obstacle clearance.
                    self._report["track_summary"] = {
                        "failure_reasons": ["Diagnostic tracker update failed"]
                    }
            with np.errstate(over="ignore", invalid="ignore"):
                if self.config.policy == "checkpoint_velocity":
                    guidance, desired_velocity = self._checkpoint_guidance(sample)
                else:
                    guidance = vector(
                        (sample.target_position_world_m - sample.position_world_m)
                        * self.config.position_gain_s2
                        - sample.velocity_world_mps * self.config.velocity_gain_s,
                        "legacy guidance",
                    )
                    desired_velocity = np.zeros(3)
                acceleration = _radial_response(guidance, rows, ages, self.config)
                baseline = _command_bounded(
                    reference_to_body(
                        acceleration, sample.quaternion_xyzw, sample.rotation_convention
                    ),
                    self.config.max_accel_mps2,
                )
                baseline, baseline_check = self._filtered_baseline(baseline, sample, rows, ages)
                if baseline is None:
                    return self._coast(
                        "No tested braking baseline passed unchanged radar/stopping guards",
                        now,
                        frame,
                    )
                angular = _command_bounded(
                    reference_to_body(
                        -sample.angular_velocity_world_rad_s * self.config.angular_velocity_gain_s,
                        sample.quaternion_xyzw,
                        sample.rotation_convention,
                    ),
                    self.config.max_angular_accel_rad_s2,
                )
            if __package__:
                from .maneuvers import axis_preferred_candidates
            else:
                from maneuvers import axis_preferred_candidates

            velocity_body = reference_to_body(
                sample.velocity_world_mps, sample.quaternion_xyzw, sample.rotation_convention
            )
            desired_body = reference_to_body(
                desired_velocity, sample.quaternion_xyzw, sample.rotation_convention
            )
            with np.errstate(over="ignore", invalid="ignore"):
                current_error = math.hypot(*vector(desired_body - velocity_body, "velocity error"))
                baseline_error = math.hypot(
                    *vector(
                        desired_body - velocity_body - baseline * self.config.control_horizon_s,
                        "baseline predicted velocity error",
                    )
                )
            if not math.isfinite(current_error) or not math.isfinite(baseline_error):
                raise ValueError("Predicted velocity-error reduction is not representable")
            reduction = max(0.0, current_error - baseline_error)
            # A fixed absolute slack can discard nearly ALL of a small velocity
            # correction. Limit that loss relative to the baseline's predicted
            # progress as well, especially during the final slow approach.
            effective_slack = min(
                self.config.axis_velocity_error_slack_mps,
                (1.0 - _AXIS_MIN_PROGRESS_FRACTION) * reduction,
            )
            self._report["candidate_choice"].update(
                current_velocity_error_mps=current_error,
                baseline_predicted_velocity_error_mps=baseline_error,
                baseline_error_reduction_mps=reduction,
                effective_axis_error_slack_mps=effective_slack,
                min_axis_progress_fraction=_AXIS_MIN_PROGRESS_FRACTION,
            )
            candidates = axis_preferred_candidates(
                baseline,
                velocity_body,
                desired_body,
                self.config.control_horizon_s,
                max_accel_mps2=self.config.max_accel_mps2,
                velocity_error_slack_mps=effective_slack,
                previous_axis=self._previous_axis,
            )
            baseline_candidate = next(
                item
                for item in candidates
                if item.axis_index is None and np.array_equal(item.accel_body_mps2, baseline)
            )
            checks = []
            accepted = []
            for candidate in candidates:
                is_baseline = candidate is baseline_candidate
                check = (
                    baseline_check
                    if is_baseline
                    else self._candidate_guard(candidate.accel_body_mps2, sample, rows, ages)
                )
                eligible = is_baseline or (
                    self.config.body_axis_preference
                    and candidate.predicted_velocity_error_mps
                    <= baseline_candidate.predicted_velocity_error_mps + effective_slack
                    and candidate.allocation_effort_l1_mps2
                    <= baseline_candidate.allocation_effort_l1_mps2
                )
                checks.append(
                    {
                        "kind": candidate.kind.value,
                        "axis_index": candidate.axis_index,
                        "accel_body_mps2": list(candidate.accel_body_mps2),
                        "predicted_velocity_error_mps": candidate.predicted_velocity_error_mps,
                        "allocation_effort_l1_mps2": candidate.allocation_effort_l1_mps2,
                        "resultant_effort_l2_mps2": candidate.resultant_effort_l2_mps2,
                        "eligible_for_maneuver_policy": eligible,
                        **check,
                    }
                )
                if eligible and check["accepted"]:
                    accepted.append(candidate)
            self._report["candidate_choice"]["candidates"] = checks
            if not accepted:
                return self._coast(
                    "No eligible candidate passed raw radar and stopping checks", now, frame
                )
            selected = accepted[0]  # Preserve the maneuver module's ranking/hysteresis.
            self._previous_axis = selected.axis_index
            self._report["candidate_choice"].update(
                selected_kind=selected.kind.value,
                selected_axis_index=selected.axis_index,
                baseline_fallback=(
                    selected is baseline_candidate and self.config.body_axis_preference
                ),
            )
            if self._report["candidate_choice"]["baseline_fallback"]:
                self._report["candidate_choice"]["fallback_reason"] = (
                    "No ranked body-axis alternative passed eligibility "
                    "and raw-radar/stopping checks"
                )
            # Return the exact tested candidate. No post-approval projection or
            # rescaling can invalidate the candidate-specific prediction check.
            linear = vector(selected.accel_body_mps2, "selected acceleration")
            self._report["command"] = {
                "accel_body_mps2": linear.tolist(),
                "angular_accel_body_rad_s2": angular.tolist(),
                "kind": selected.kind.value,
                "written_to_any_api": False,
            }
            return linear, angular
        except (
            ValueError,
            TypeError,
            OverflowError,
            FloatingPointError,
            np.linalg.LinAlgError,
            StopIteration,
        ) as error:
            return self._coast(str(error) or type(error).__name__, now, frame)
