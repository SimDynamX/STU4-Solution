"""Rank BODY-frame component projections without making a flight-safety claim.

All input vectors must already share the shuttle's BODY frame. The caller owns
the actual quaternion conversion, collision, stopping and trajectory checks for
each candidate, and fallback to the original command if an alternative is unsafe.
This module changes neither attitude nor angular commands and imports no SDK or
native flight package. It also works as a direct sibling import in an STP bundle.

L1 is an allocation-effort proxy, not measured fuel. The native centre-of-mass API
burns by resultant L2 acceleration. Multiple allocated jets may instead track L1
and other moment demands. Two equal-magnitude orthogonal components give
L1/L2 = sqrt(2). At equal resultant, only this proxy is sqrt(2) times the cardinal
proxy. This establishes neither fuel savings nor a universal flight penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt

__all__ = ["ManeuverKind", "BodyCommandCandidate", "axis_preferred_candidates"]

_ROUNDING_REL_TOL = 8.0 * math.ulp(1.0)
_TIE_ABS_LIMIT = 1e-12


class ManeuverKind(Enum):
    """An unchanged command or a projection onto one declared BODY axis."""

    BASELINE = "baseline"
    BODY_AXIS = "body_axis"


class _PredictionOverflow(ValueError):
    """A finite input produced an unrepresentable prediction or error norm."""


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a representable finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a representable finite real number")
    return result


def _vector(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    # Object conversion preserves mixed booleans/complex values before coercion.
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real vector of shape (3,)") from error
    if raw.shape != (3,):
        raise ValueError(f"{name} must be a finite real vector of shape (3,)")
    return np.array([_finite_real(item, name) for item in raw], dtype=np.float64)


def _axis_index(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= 2
    ):
        raise ValueError(f"{name} must be None or an integer in [0, 2], not a boolean")
    return int(value)


@dataclass(frozen=True, slots=True, eq=False)
class BodyCommandCandidate:
    """One numeric option, with an independent, read-only float64 command copy.

    These metrics predict constant acceleration for the declared duration only.
    Neither the kind nor the rank certifies that the command is safe to execute.
    """

    accel_body_mps2: npt.NDArray[np.float64]
    kind: ManeuverKind
    axis_index: int | None
    predicted_velocity_error_mps: float
    allocation_effort_l1_mps2: float
    resultant_effort_l2_mps2: float

    def __post_init__(self) -> None:
        command = _vector(self.accel_body_mps2, "accel_body_mps2")
        # A bytes-backed copy cannot be made writable again with setflags.
        readonly = np.frombuffer(command.tobytes(), dtype=np.float64)
        object.__setattr__(self, "accel_body_mps2", readonly)
        if not isinstance(self.kind, ManeuverKind):
            raise ValueError("kind must be a ManeuverKind")
        object.__setattr__(self, "axis_index", _axis_index(self.axis_index, "axis_index"))
        for name in (
            "predicted_velocity_error_mps",
            "allocation_effort_l1_mps2",
            "resultant_effort_l2_mps2",
        ):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)


def _predicted_error(
    command: npt.NDArray[np.float64],
    velocity: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    duration: float,
) -> float:
    # Python float arithmetic avoids NumPy overflow warnings. Check the results
    # explicitly, and use hypot rather than squaring huge or tiny coordinates.
    after = tuple(float(v) + float(a) * duration for v, a in zip(velocity, command, strict=True))
    if not all(math.isfinite(value) for value in after):
        raise _PredictionOverflow("predicted BODY velocity must remain finite")
    error = math.hypot(*(v - float(t) for v, t in zip(after, target, strict=True)))
    if not math.isfinite(error):
        raise _PredictionOverflow("predicted velocity error must remain finite")
    return error


def _candidate(
    command: npt.NDArray[np.float64],
    kind: ManeuverKind,
    axis: int | None,
    velocity: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    duration: float,
) -> BodyCommandCandidate:
    return BodyCommandCandidate(
        accel_body_mps2=command,
        kind=kind,
        axis_index=axis,
        predicted_velocity_error_mps=_predicted_error(command, velocity, target, duration),
        allocation_effort_l1_mps2=math.fsum(abs(float(value)) for value in command),
        resultant_effort_l2_mps2=math.hypot(*command),
    )


def _numerical_tie(left: float, right: float) -> bool:
    tolerance = min(_TIE_ABS_LIMIT, _ROUNDING_REL_TOL * min(left, right))
    return abs(left - right) <= tolerance


def axis_preferred_candidates(
    accel_body: npt.ArrayLike,
    velocity_body: npt.ArrayLike,
    target_velocity_body: npt.ArrayLike,
    duration_s: float,
    *,
    max_accel_mps2: float = 0.5,
    velocity_error_slack_mps: float = 0.02,
    previous_axis: int | None = None,
) -> tuple[BodyCommandCandidate, ...]:
    """Return eligible signed component projections, then the original baseline.

    Inputs are finite real, non-boolean three-vectors in the same BODY frame.
    Duration and acceleration cap must be positive; the cap cannot exceed 0.5.
    An over-cap baseline is rejected, not renormalized. Only relative float64
    rounding up to eight epsilons is tolerated, with the original values retained.
    An unrepresentable baseline prediction or error is rejected. An alternative
    with an unrepresentable prediction is omitted, preserving the valid baseline.

    Projected commands retain each original signed component, not full thrust.
    Exact zero projections and projections identical to the baseline are omitted.
    No absolute zero threshold removes tiny nonzero commands. The baseline always
    remains last, even when there are no alternatives or it is already cardinal.

    Each option predicts ``v_after = velocity_body + accel * duration_s``.
    An axis is eligible only if its error is at most baseline error plus the
    declared nonnegative slack. Eligible axes sort by L1 proxy, then error, then
    axis index. The previous eligible axis can move first only when both metrics
    tie with the best within eight relative float64 epsilons AND 1e-12 absolute.
    Hysteresis never relaxes eligibility or overrides a material metric difference.

    This is a ranking, not an action selector. The caller must run its actual
    collision, stopping and trajectory guard on each command it considers, and
    retain its original baseline fallback. No PID or attitude changes are made.
    """
    command = _vector(accel_body, "accel_body")
    velocity = _vector(velocity_body, "velocity_body")
    target = _vector(target_velocity_body, "target_velocity_body")
    duration = _finite_real(duration_s, "duration_s")
    cap = _finite_real(max_accel_mps2, "max_accel_mps2")
    slack = _finite_real(velocity_error_slack_mps, "velocity_error_slack_mps")
    previous = _axis_index(previous_axis, "previous_axis")
    if duration <= 0.0:
        raise ValueError("duration_s must be positive")
    if not 0.0 < cap <= 0.5:
        raise ValueError("max_accel_mps2 must be positive and at most 0.5")
    if slack < 0.0:
        raise ValueError("velocity_error_slack_mps must be nonnegative")
    norm = math.hypot(*command)
    if norm > cap and not math.isclose(norm, cap, rel_tol=_ROUNDING_REL_TOL, abs_tol=0.0):
        raise ValueError("accel_body resultant exceeds max_accel_mps2")

    baseline = _candidate(command, ManeuverKind.BASELINE, None, velocity, target, duration)
    axes = []
    for axis, component in enumerate(command):
        if component == 0.0:
            continue
        projection = np.zeros(3, dtype=np.float64)
        projection[axis] = component
        if np.array_equal(projection, command):
            continue
        try:
            candidate = _candidate(
                projection, ManeuverKind.BODY_AXIS, axis, velocity, target, duration
            )
        except _PredictionOverflow:
            continue
        error = candidate.predicted_velocity_error_mps
        # Subtract only when positive, avoiding overflow in baseline_error + slack.
        if (
            error <= baseline.predicted_velocity_error_mps
            or error - baseline.predicted_velocity_error_mps <= slack
        ):
            axes.append(candidate)

    axes.sort(
        key=lambda item: (
            item.allocation_effort_l1_mps2,
            item.predicted_velocity_error_mps,
            item.axis_index,
        )
    )
    if axes and previous is not None:
        best = axes[0]
        for index, candidate in enumerate(axes):
            if (
                candidate.axis_index == previous
                and _numerical_tie(
                    candidate.allocation_effort_l1_mps2, best.allocation_effort_l1_mps2
                )
                and _numerical_tie(
                    candidate.predicted_velocity_error_mps, best.predicted_velocity_error_mps
                )
            ):
                axes.insert(0, axes.pop(index))
                break
    return (*axes, baseline)
