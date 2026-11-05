"""Source port of the published official-profile beeline for STP, not evidence.

This is a source-level port of the official-profile behavior used by Ali's
example. It has not been executed against a licensed Space Teams Pro runtime.
A source port is not licensed-runtime evidence: the operator must run the
disabled-by-default probe and the team's validation on the real STP server
before relying on any command produced here.

The port keeps the published command shape: a 10 m/s target velocity, unity
velocity-error gain, body +X alignment, and the published actuator caps of
0.5 m/s2 and 0.05 rad/s2. The only addition is a conservative radial potential
built from the public anonymous radar rows. The controller consumes a
ControlSample produced by ``sample_from_stp_state`` from the STP adapter's
STPState, plus the unchanged public radar matrices. It preserves radar
measurement times and never invents debris IDs, covariances, full debris
velocities, target velocity, terrain, hull geometry, mission status or scoring.
Radial repulsion cannot prove collision avoidance.

Importing this module loads no SDK, connects nowhere and reads no configuration.
It also supports direct imports beside stp_api.py and control_contract.py in the
licensed entry point.
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
    from .stp_api import (
        OFFICIAL_CONTROLLER_URL,
        OFFICIAL_RADAR_URL,
        OFFICIAL_SCORING_URL,
        OFFICIAL_SIM_COMMIT,
        OFFICIAL_SOLUTION_COMMIT,
        OFFICIAL_USER_API_URL,
    )
else:
    # Direct sibling imports avoid the historical package and its newer syntax.
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
    from stp_api import (
        OFFICIAL_CONTROLLER_URL,
        OFFICIAL_RADAR_URL,
        OFFICIAL_SCORING_URL,
        OFFICIAL_SIM_COMMIT,
        OFFICIAL_SOLUTION_COMMIT,
        OFFICIAL_USER_API_URL,
    )

CONFIG_SCHEMA = "wsorb.stp_official"
CONFIG_SCHEMA_VERSION = 1
CONFIG_FILE_ENV = "WSORB_STP_OFFICIAL_FILE"
CONFIG_JSON_ENV = "WSORB_STP_OFFICIAL_JSON"
MAX_CONFIG_BYTES = 16 * 1024

OFFICIAL_BEELINE_SPEED_MPS = 10.0
MAX_ACCEL_MPS2 = 0.5
MAX_ANGULAR_ACCEL_RAD_S2 = 0.05
_STP_STATE_SOURCE = "stp_adapter_numeric_report"
_STP_ROTATION_CONVENTION = "passive_xyzw"
_EPSILON = 1.0e-12
_TIME_TOLERANCE_S = 1.0e-12  # Fixed-step summation noise, never a sensor offset.
_NORM_HEADROOM_REL = 8.0 * np.finfo(np.float64).eps

# Review/input budgets for this source port, not properties of the licensed SDK.
_NUMERIC_BOUNDS = {
    "duration_wall_s": (0.0, 3600.0),
    "max_report_age_s": (0.0, 5.4),
    "max_radar_age_s": (0.0, 5.4),
    "attitude_response_time_s": (0.01, 10.0),
    "avoidance_range_m": (0.0, 1e6),
    "avoidance_accel_mps2": (0.0, MAX_ACCEL_MPS2),
    "radar_margin_m": (0.0, 1e6),
    "avoidance_horizon_s": (0.0, 30.0),
    "radar_clock_offset_s": (-1e9, 1e9),
}

__all__ = [
    "CONFIG_FILE_ENV",
    "CONFIG_JSON_ENV",
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_VERSION",
    "MAX_ACCEL_MPS2",
    "MAX_ANGULAR_ACCEL_RAD_S2",
    "MAX_CONFIG_BYTES",
    "OFFICIAL_BEELINE_SPEED_MPS",
    "STPOfficialConfig",
    "STPOfficialController",
    "decision_is_current",
    "load_config_from_environment",
    "parse_config",
]


def _number(value: Any, name: str) -> float:
    low, high = _NUMERIC_BOUNDS[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [{low}, {high}]")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not a representable finite real number") from error
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be a finite real number in [{low}, {high}]")
    return number


@dataclass(frozen=True, slots=True)
class STPOfficialConfig:
    """Small, immutable operator data. Construction alone never commands jets.

    ``enabled`` and ``probe`` are explicit booleans. Enabling requires an
    independently established ``radar_clock_offset_s``; zero is valid only when
    the operator knows the two clock origins coincide. This offset is never
    inferred from a reading. ``probe`` may be true without an offset because a
    read-only report can say that radar age is unknown.

    ``attitude_response_time_s``, the repulsion range, margin, horizon and
    acceleration share are reviewed tuning assumptions, not published geometry
    or measured actuator authority. The linear and angular caps themselves are
    fixed at the published limits and cannot be raised by configuration.
    """

    enabled: bool = False
    radar_clock_offset_s: float | None = None
    probe: bool = False
    duration_wall_s: float = 30.0
    max_report_age_s: float = 1.0
    max_radar_age_s: float = 1.5
    attitude_response_time_s: float = 0.75
    avoidance_range_m: float = 70.0
    avoidance_accel_mps2: float = 0.5
    radar_margin_m: float = 10.0
    avoidance_horizon_s: float = 1.5

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be an explicit boolean")
        if type(self.probe) is not bool:
            raise ValueError("probe must be an explicit boolean")
        for name in _NUMERIC_BOUNDS:
            value = getattr(self, name)
            if name == "radar_clock_offset_s" and value is None:
                if self.enabled:
                    raise ValueError("An enabled controller requires explicit radar_clock_offset_s")
                continue
            object.__setattr__(self, name, _number(value, name))
        for name in ("duration_wall_s", "avoidance_range_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> STPOfficialConfig:
        """Parse versioned data with explicit enable, rejecting unknown fields."""
        if not isinstance(document, Mapping) or any(type(key) is not str for key in document):
            raise ValueError("STP official configuration must be an object with string keys")
        required = {"schema", "schema_version", "enabled"}
        allowed = required | {item.name for item in fields(cls)}
        if required - document.keys() or document.keys() - allowed:
            raise ValueError("STP official configuration has missing or unknown fields")
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
            raise ValueError(f"Duplicate STP official JSON key: {name}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"STP official JSON cannot contain {value}")


def parse_config(source: str | bytes) -> STPOfficialConfig:
    """Parse at most 16 KiB of strict UTF-8 JSON without executing any code."""
    if not isinstance(source, (str, bytes)) or len(source) > MAX_CONFIG_BYTES:
        raise ValueError("STP official configuration must be at most 16384 UTF-8 bytes")
    try:
        data = source.encode("utf-8") if isinstance(source, str) else source
        if len(data) > MAX_CONFIG_BYTES:
            raise ValueError("STP official configuration must be at most 16384 UTF-8 bytes")
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("STP official configuration must be a valid UTF-8 JSON object") from error
    return STPOfficialConfig.from_mapping(document)


def load_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> STPOfficialConfig:
    """Load one local file or inline JSON, never SDK bootstrap arguments.

    Select WSORB_STP_OFFICIAL_FILE or WSORB_STP_OFFICIAL_JSON, not both. Missing
    configuration means disabled with the probe off, so the entry point does not
    connect. Present but invalid configuration raises before connecting.
    """
    environ = os.environ if environ is None else environ
    present = [name for name in (CONFIG_FILE_ENV, CONFIG_JSON_ENV) if name in environ]
    if not present:
        return STPOfficialConfig()
    if len(present) != 1:
        raise ValueError("Select only one of WSORB_STP_OFFICIAL_FILE and WSORB_STP_OFFICIAL_JSON")
    value = environ[present[0]]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{present[0]} must be a nonempty string")
    if present[0] == CONFIG_JSON_ENV:
        return parse_config(value)
    try:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError("WSORB_STP_OFFICIAL_FILE must name a local regular JSON file")
        with path.open("rb") as handle:
            source = handle.read(MAX_CONFIG_BYTES + 1)
    except OSError as error:
        raise ValueError("Could not read WSORB_STP_OFFICIAL_FILE") from error
    return parse_config(source)


def _command_bounded(value: npt.ArrayLike, maximum: float) -> npt.NDArray[np.float64]:
    """Leave float64 norm headroom below the public cap before the actual check."""
    ceiling = float(np.nextafter(maximum * (1.0 - _NORM_HEADROOM_REL), 0.0))
    return bounded(value, ceiling)


def _normalize(value: npt.ArrayLike) -> npt.NDArray[np.float64] | None:
    array = np.asarray(value, dtype=np.float64)
    length = math.hypot(*array)
    if not math.isfinite(length) or length <= _EPSILON:
        return None
    return array / length


def _perpendicular(
    direction: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """A deterministic lateral axis, not a fixed mission corridor."""
    side = np.cross(direction, velocity)
    length = math.hypot(*side)
    if not math.isfinite(length) or length <= _EPSILON:
        side = np.cross(direction, np.asarray((0.0, 1.0, 0.0)))
        length = math.hypot(*side)
    if not math.isfinite(length) or length <= _EPSILON:
        side = np.cross(direction, np.asarray((0.0, 0.0, 1.0)))
        length = math.hypot(*side)
    if not math.isfinite(length) or length <= _EPSILON:
        return np.asarray((0.0, 1.0, 0.0))
    return side / length


class STPOfficialController:
    """Pure source-port controller; it has no SDK handle and writes no command.

    The caller supplies a ControlSample produced by ``sample_from_stp_state``
    from the real adapter's STPState, plus the unchanged public radar matrix.
    Only the adapter's P-frame state and anonymous radar rows are used. Radar
    source times remain in the report; the declared offset is applied only to a
    private copy. Missing, malformed, negative, future or stale data, an empty
    radar publication, a changed source/frame/convention, a backwards clock or
    an unrepresentable calculation returns two actual zero vectors.

    The published beeline is preserved before the public-radar potential:
    target speed is 10 m/s, velocity-error gain is unity, body +X is aligned by
    a bounded shortest-arc law, and both returned body commands stay below the
    published 0.5 m/s2 and 0.05 rad/s2 caps. The radial potential is a
    conservative steering aid, not a collision or stopping certificate.
    """

    def __init__(self, config: STPOfficialConfig) -> None:
        if not isinstance(config, STPOfficialConfig):
            raise ValueError("config must be STPOfficialConfig")
        self.config = config
        self._source: tuple[str, str, str] | None = None
        self._last_now: float | None = None
        self._calls = 0
        self._coast_calls = 0
        self._report = self._new_report()

    def _new_report(self) -> dict[str, Any]:
        return {
            "schema": "wsorb.stp_official_report",
            "schema_version": 1,
            "policy": "official_competition_beeline_source_port",
            "enabled": self.config.enabled,
            "config": asdict(self.config),
            "calls": self._calls,
            "coast_calls": self._coast_calls,
            "failure_reasons": [],
            "decision_valid_until_elapsed_s": None,
            "source_freshness": {
                "declared_radar_clock_offset_s": self.config.radar_clock_offset_s,
                "configured_max_report_age_s": self.config.max_report_age_s,
                "configured_max_radar_age_s": self.config.max_radar_age_s,
                "sample_elapsed_time_s": None,
                "now_elapsed_time_s": None,
                "report_age_s": None,
                "observed_min_radar_age_s": None,
                "observed_max_radar_age_s": None,
                "radar_source_times_s": [],
                "radar_measurement_times_s": [],
                "source_time_spread_s": None,
                "sample_atomicity": "not guaranteed; state and radar reads are sequential",
            },
            "provenance": {
                "kind": "source_port_of_published_official_profile",
                "licensed_runtime_evidence": False,
                "probe_validated_on_real_server": False,
                "official_solution_commit": OFFICIAL_SOLUTION_COMMIT,
                "official_sim_commit": OFFICIAL_SIM_COMMIT,
                "user_api_url": OFFICIAL_USER_API_URL,
                "controller_url": OFFICIAL_CONTROLLER_URL,
                "radar_url": OFFICIAL_RADAR_URL,
                "scoring_url": OFFICIAL_SCORING_URL,
                "state_source_kind": _STP_STATE_SOURCE,
                "state_frame": STP_PHOBOS_FRAME,
                "rotation_convention": _STP_ROTATION_CONVENTION,
                "radar_publication": (
                    "anonymous closest-point rows; publication order is not a tracking identity"
                ),
                "identities": "not published; this controller invents none",
                "covariances": "not published; this controller invents none",
            },
            "model_assumptions": {
                "reference_frame": "P = Phobos.GetBodyFixedFrame(), declared by the adapter",
                "rotation": "passive Hamilton xyzw, copied without conjugation",
                "target_velocity": "unavailable; velocity error is relative to P",
                "radar": "anonymous rows; times use radar-loop startup",
                "radar_motion": "only scalar range rate is applied; tangential motion is unknown",
                "repulsion": (
                    "configured margin plus closing-rate lookahead; not a clearance certificate"
                ),
                "attitude": "shortest-arc body +X alignment with a configured response time",
                "command": "bounded below public caps with float64 norm headroom",
            },
            "guidance": {
                "beeline_speed_mps": OFFICIAL_BEELINE_SPEED_MPS,
                "velocity_error_gain_s": 1.0,
                "accel_limit_mps2": MAX_ACCEL_MPS2,
                "target_distance_m": None,
                "desired_velocity_phobos_mps": None,
                "nominal_accel_phobos_mps2": None,
                "repulsion_accel_phobos_mps2": None,
                "beeline_suppression": None,
                "accel_before_body_rotation_phobos_mps2": None,
                "law": "published_10_mps_target_velocity_with_unity_velocity_error_gain",
            },
            "radar": {
                "rows": 0,
                "anonymous": True,
                "in_range_rows": 0,
                "max_proximity_weight": None,
                "effective_range_rule": (
                    "closest_range + radar_margin_m + "
                    "min(range_rate, 0) * (age + avoidance_horizon_s)"
                ),
                "source_times_preserved": True,
            },
            "attitude": {
                "body_plus_x_phobos": None,
                "desired_body_plus_x_phobos": None,
                "alignment_angle_rad": None,
                "attitude_response_time_s": self.config.attitude_response_time_s,
                "angular_accel_limit_rad_s2": MAX_ANGULAR_ACCEL_RAD_S2,
                "angular_velocity_phobos_rad_s": None,
                "law": "shortest_arc_bounded_body_plus_x_alignment",
            },
            "command": None,
            "limitations": (
                "Source port only, not licensed-runtime evidence. Run the disabled-by-default "
                "probe and validation on the real STP server. Radar is anonymous and non-atomic, "
                "publishes no identities or covariances, and radial repulsion cannot prove "
                "collision avoidance. Target velocity, terrain, hull geometry, vision, mission "
                "status and official scoring are unavailable here."
            ),
        }

    @property
    def report(self) -> dict[str, Any]:
        """Detached strict-JSON data. Mutating it cannot change controller state."""
        return json.loads(json.dumps(self._report, allow_nan=False))

    def _coast(self, reason: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self._coast_calls += 1
        self._report["coast_calls"] = self._coast_calls
        self._report["failure_reasons"].append(reason)
        self._report["command"] = {
            "accel_body_mps2": [0.0, 0.0, 0.0],
            "angular_accel_body_rad_s2": [0.0, 0.0, 0.0],
            "kind": "coast",
            "written_to_any_api": False,
        }
        return coast_command()

    def _attitude_command(
        self, sample: ControlSample, goal_direction: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        body_x = body_to_reference(
            np.asarray((1.0, 0.0, 0.0)),
            sample.quaternion_xyzw,
            sample.rotation_convention,
        )
        cross = np.cross(body_x, goal_direction)
        sine = float(math.hypot(*cross))
        cosine = float(np.clip(np.dot(body_x, goal_direction), -1.0, 1.0))
        angle = math.atan2(sine, cosine)
        if not math.isfinite(angle):
            raise ValueError("Attitude alignment angle is not representable")
        lease = self.config.attitude_response_time_s
        limit = MAX_ANGULAR_ACCEL_RAD_S2
        if angle <= _EPSILON:
            desired_rate = np.zeros(3, dtype=np.float64)
        else:
            magnitude = min(angle / lease, math.sqrt(2.0 * limit * angle))
            axis = (
                _perpendicular(body_x, sample.angular_velocity_world_rad_s)
                if sine <= _EPSILON
                else cross / sine
            )
            desired_rate = axis * magnitude
        with np.errstate(over="ignore", invalid="ignore"):
            rate_error = vector(
                desired_rate - sample.angular_velocity_world_rad_s, "attitude rate error"
            )
            angular_reference = vector(rate_error / lease, "attitude acceleration")
            angular_body = _command_bounded(
                reference_to_body(
                    angular_reference,
                    sample.quaternion_xyzw,
                    sample.rotation_convention,
                ),
                limit,
            )
        return angular_body, {
            "body_plus_x_phobos": body_x.tolist(),
            "desired_body_plus_x_phobos": goal_direction.tolist(),
            "alignment_angle_rad": angle,
            "angular_velocity_phobos_rad_s": sample.angular_velocity_world_rad_s.tolist(),
        }

    def command(
        self,
        sample: ControlSample,
        debris: npt.ArrayLike,
        *,
        now_elapsed_time_s: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return bounded body vectors. Invalid/missing/stale data returns zeros.

        This function writes to no API. A disabled controller must not be used as
        permission to write its zero return to a licensed shuttle. A fresh clock
        and an independently declared radar offset remain mandatory.
        """
        self._calls += 1
        self._report = self._new_report()
        if not self.config.enabled:
            return self._coast("Controller is disabled; caller must remain read-only")
        try:
            checked_sample(sample)
            if sample.frame != STP_PHOBOS_FRAME:
                raise ValueError("sample frame is not the STP Phobos frame")
            if sample.source_kind != _STP_STATE_SOURCE:
                raise ValueError("sample is not an STP adapter numeric report")
            if sample.rotation_convention != _STP_ROTATION_CONVENTION:
                raise ValueError("sample does not use the STP passive xyzw convention")
            now = scalar(now_elapsed_time_s, "now_elapsed_time_s")
            elapsed = scalar(sample.elapsed_time_s, "elapsed_time_s")
            freshness = self._report["source_freshness"]
            freshness.update(
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
            rows = radar_rows(debris)
            if len(rows) == 0:
                raise ValueError("Empty radar is not evidence of clear space")
            source_times = rows[:, 2, 0].copy()
            with np.errstate(over="ignore", invalid="ignore"):
                measurement_times = source_times + self.config.radar_clock_offset_s
                ages = now - measurement_times
            if not np.isfinite(measurement_times).all() or not np.isfinite(ages).all():
                raise ValueError("Radar clock conversion is not representable")
            freshness.update(
                radar_source_times_s=[float(value) for value in source_times],
                radar_measurement_times_s=[float(value) for value in measurement_times],
                observed_min_radar_age_s=float(np.min(ages)),
                observed_max_radar_age_s=float(np.max(ages)),
                source_time_spread_s=float(np.max(source_times) - np.min(source_times)),
            )
            if np.any(measurement_times < -_TIME_TOLERANCE_S) or np.any(
                measurement_times > elapsed + _TIME_TOLERANCE_S
            ):
                raise ValueError(
                    "Radar timestamps are negative or future after the declared offset"
                )
            if np.any(ages > self.config.max_radar_age_s + _TIME_TOLERANCE_S):
                raise ValueError("Stale radar exceeds configured max_radar_age_s")
            self._report["decision_valid_until_elapsed_s"] = min(
                elapsed + self.config.max_report_age_s,
                float(np.min(measurement_times)) + self.config.max_radar_age_s,
            )
            self._source = source
            self._last_now = now
            with np.errstate(over="ignore", invalid="ignore"):
                delta = vector(
                    sample.target_position_world_m - sample.position_world_m,
                    "target displacement",
                )
            distance = float(math.hypot(*delta))
            if not math.isfinite(distance):
                raise ValueError("Target distance is not representable")
            if distance <= _EPSILON:
                raise ValueError("Target coincides with the shuttle position; no beeline direction")
            goal_direction = delta / distance
            desired_velocity = OFFICIAL_BEELINE_SPEED_MPS * goal_direction
            with np.errstate(over="ignore", invalid="ignore"):
                nominal = bounded(
                    desired_velocity - sample.velocity_world_mps,
                    MAX_ACCEL_MPS2,
                )
                closing = np.minimum(rows[:, 1, 1], 0.0) * (ages + self.config.avoidance_horizon_s)
                effective_ranges = rows[:, 1, 0] + self.config.radar_margin_m + closing
            if not np.isfinite(effective_ranges).all():
                raise ValueError("Radar effective range is not representable")
            weights = np.zeros(len(rows), dtype=np.float64)
            nearby = effective_ranges < self.config.avoidance_range_m
            weights[nearby] = 1.0 - (
                np.maximum(effective_ranges[nearby], 0.0) / self.config.avoidance_range_m
            )
            lengths = np.hypot(np.hypot(rows[:, 0, 0], rows[:, 0, 1]), rows[:, 0, 2])
            directions = rows[:, 0, :] / lengths[:, None]
            with np.errstate(over="ignore", invalid="ignore"):
                combined = (
                    -np.sum(directions * weights[:, None], axis=0)
                    * self.config.avoidance_accel_mps2
                )
            repulsion = bounded(combined, self.config.avoidance_accel_mps2)
            max_weight = float(np.max(weights))
            with np.errstate(over="ignore", invalid="ignore"):
                guidance = bounded(nominal, MAX_ACCEL_MPS2) * (1.0 - max_weight)
                acceleration = bounded(guidance + repulsion, MAX_ACCEL_MPS2)
                accel_body = _command_bounded(
                    reference_to_body(
                        acceleration,
                        sample.quaternion_xyzw,
                        sample.rotation_convention,
                    ),
                    MAX_ACCEL_MPS2,
                )
            angular_body, attitude = self._attitude_command(sample, goal_direction)
            self._report["guidance"].update(
                target_distance_m=distance,
                desired_velocity_phobos_mps=desired_velocity.tolist(),
                nominal_accel_phobos_mps2=nominal.tolist(),
                repulsion_accel_phobos_mps2=repulsion.tolist(),
                beeline_suppression=max_weight,
                accel_before_body_rotation_phobos_mps2=acceleration.tolist(),
            )
            self._report["radar"].update(
                rows=len(rows),
                in_range_rows=int(np.count_nonzero(weights > 0.0)),
                max_proximity_weight=max_weight,
            )
            self._report["attitude"].update(attitude)
            self._report["command"] = {
                "accel_body_mps2": accel_body.tolist(),
                "angular_accel_body_rad_s2": angular_body.tolist(),
                "kind": "official_beeline",
                "written_to_any_api": False,
            }
            return accel_body, angular_body
        except (
            ValueError,
            TypeError,
            OverflowError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            return self._coast(str(error) or type(error).__name__)


def decision_is_current(report: Mapping[str, Any], *, now_elapsed_time_s: float) -> bool:
    """Pure post-compute deadline check, not an SDK decision lease or watchdog.

    The entry point must supply a newly read real SDK clock after computation.
    This does not extend the original report or radar deadline.
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
