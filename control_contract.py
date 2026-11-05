"""Portable numeric control data and math, independent of SDK and native flight.

This Python 3.10+ subset needs NumPy, not the historical ``wsorb`` package entry
point. A ControlSample is project data, never an SDK type or atomic observation.
The producer must name its real coordinate frame and quaternion convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

Vec3 = tuple[float, float, float]
RotationConvention = Literal["active_xyzw", "passive_xyzw"]
MAX_RADAR_ROWS = 4096
STP_PHOBOS_FRAME = "P = Phobos.GetBodyFixedFrame()"


@dataclass(frozen=True, slots=True)
class ControlSample:
    """Mandatory, frame-labelled numeric report consumed by the same controller.

    World means the producer's named reference frame, not necessarily inertial.
    STP samples retain raw passive Hamilton xyzw components and P coordinates.
    Local API samples must declare their actual active quaternion and frame.
    The frozen object copies array inputs read-only but does not certify source,
    freshness or physical validity. Control validates those separately.
    """

    elapsed_time_s: float
    position_world_m: npt.NDArray[np.float64]
    velocity_world_mps: npt.NDArray[np.float64]
    quaternion_xyzw: npt.NDArray[np.float64]
    rotation_convention: RotationConvention
    angular_velocity_world_rad_s: npt.NDArray[np.float64]
    target_position_world_m: npt.NDArray[np.float64]
    mass_kg: float
    inertia_body_kg_m2: npt.NDArray[np.float64]
    fuel_used_kg: float
    damage_j: float
    frame: str
    source_kind: str

    def __post_init__(self) -> None:
        for name in (
            "position_world_m",
            "velocity_world_mps",
            "quaternion_xyzw",
            "angular_velocity_world_rad_s",
            "target_position_world_m",
            "inertia_body_kg_m2",
        ):
            value = np.array(getattr(self, name), copy=True)
            if value.dtype.kind in "iuf":
                # Valid control arithmetic uses float64, never wrapping integer
                # subtraction. Invalid complex/string/bool inputs stay invalid.
                value = np.array(value, dtype=np.float64, copy=True)
            if not value.dtype.hasobject:
                # An immutable buffer prevents a consumer from re-enabling
                # writes, including through a returned array's base object.
                value = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            else:
                value.flags.writeable = False
            object.__setattr__(self, name, value)


def scalar(value: Any, name: str, *, minimum: float = 0.0, positive: bool = False) -> float:
    message = f"{name} must be finite and {'positive' if positive else f'>= {minimum}'}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(message) from error
    if not math.isfinite(number) or number < minimum or (positive and number <= 0.0):
        raise ValueError(message)
    return number


def real_array(value: Any, name: str) -> npt.NDArray[np.float64]:
    message = f"{name} must contain finite real numbers"
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError(message)
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not np.isfinite(array).all():
        raise ValueError(message)
    return array


def vector(value: Any, name: str, *, length: int = 3) -> npt.NDArray[np.float64]:
    result = real_array(value, name)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return result


def rotation_reference_to_body(
    rotation_xyzw: npt.ArrayLike, convention: str
) -> npt.NDArray[np.float64]:
    """Return R(q).T for either explicitly declared supported xyzw convention.

    Local active q maps body to reference. The pinned STP passive Hamilton q
    defines the same reference-to-body coordinate matrix, R(q).T. No conjugation
    of raw SDK components is warranted. Only a private numeric copy is normalized.
    """
    quaternion = vector(rotation_xyzw, "rotation", length=4)
    if convention not in ("active_xyzw", "passive_xyzw"):
        raise ValueError("rotation_convention must be active_xyzw or passive_xyzw")
    scale = float(np.max(np.abs(quaternion)))
    if scale == 0.0:
        raise ValueError("rotation must be a nonzero quaternion")
    quaternion /= scale
    quaternion /= math.hypot(*quaternion)
    x, y, z, w = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
            [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
            [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def reference_to_body(
    value: npt.ArrayLike, rotation_xyzw: npt.ArrayLike, convention: str
) -> npt.NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        result = rotation_reference_to_body(rotation_xyzw, convention) @ vector(
            value, "reference vector"
        )
    return vector(result, "body vector")


def body_to_reference(
    value: npt.ArrayLike, rotation_xyzw: npt.ArrayLike, convention: str
) -> npt.NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        result = rotation_reference_to_body(rotation_xyzw, convention).T @ vector(
            value, "body vector"
        )
    return vector(result, "reference vector")


def bounded(value: npt.ArrayLike, maximum: float) -> npt.NDArray[np.float64]:
    maximum = scalar(maximum, "maximum")
    result = vector(value, "command")
    scale = float(np.max(np.abs(result)))
    if scale == 0.0 or maximum == 0.0:
        return np.zeros(3, dtype=np.float64)
    unit = result / scale
    length = math.hypot(*unit)
    if scale > maximum / length or math.hypot(*result) > maximum:
        result = unit / length * np.nextafter(maximum, 0.0)
        if math.hypot(*result) > maximum:
            result = np.nextafter(result, 0.0)
    return result


def coast_command() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two actual zero requests, not braking, a command write or an SDK watchdog."""
    return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)


def radar_rows(value: Any) -> npt.NDArray[np.float64]:
    """Copy and check public Nx3x3 rows without changing source times or order.

    Row 0 is relative position in the producer's frame. Row 1 is closest range,
    receding-positive scalar range rate and diameter. Row 2 is time, zero, zero.
    These rows contain neither identities nor full debris velocities.
    """
    rows = real_array(value, "radar")
    if rows.shape == (0,):
        rows = rows.reshape(0, 3, 3)
    if rows.ndim != 3 or rows.shape[1:] != (3, 3) or len(rows) > MAX_RADAR_ROWS:
        raise ValueError(f"radar must have shape (N, 3, 3), with N <= {MAX_RADAR_ROWS}")
    if np.any(rows[:, 1, 2] <= 0.0) or np.any(rows[:, 2, 0] < 0.0) or np.any(rows[:, 2, 1:] != 0.0):
        raise ValueError(
            "radar requires nonnegative time, positive diameter and zero spare columns"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.hypot(np.hypot(rows[:, 0, 0], rows[:, 0, 1]), rows[:, 0, 2])
    if np.any(lengths <= 0.0) or not np.isfinite(lengths).all():
        raise ValueError("radar row-0 relative vectors must have finite, positive lengths")
    return rows


def checked_sample(sample: ControlSample) -> ControlSample:
    """Validate project data, including positive-definite body inertia."""
    if not isinstance(sample, ControlSample):
        raise ValueError("sample must be a project ControlSample, not an SDK or browser object")
    scalar(sample.elapsed_time_s, "elapsed_time_s")
    for name in (
        "position_world_m",
        "velocity_world_mps",
        "angular_velocity_world_rad_s",
        "target_position_world_m",
    ):
        vector(getattr(sample, name), name)
    rotation_reference_to_body(sample.quaternion_xyzw, sample.rotation_convention)
    for name in ("frame", "source_kind"):
        value = getattr(sample, name)
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError(f"{name} must be a nonempty label of at most 256 characters")
    scalar(sample.mass_kg, "mass_kg", positive=True)
    scalar(sample.fuel_used_kg, "fuel_used_kg")
    scalar(sample.damage_j, "damage_j")
    inertia = real_array(sample.inertia_body_kg_m2, "inertia_body_kg_m2")
    if inertia.shape != (3, 3):
        raise ValueError("inertia_body_kg_m2 must have shape (3, 3)")
    scale = float(np.max(np.abs(inertia)))
    if scale == 0.0:
        raise ValueError("inertia_body_kg_m2 must be positive definite")
    inertia /= scale
    if not np.allclose(inertia, inertia.T, rtol=1e-8, atol=1e-12):
        raise ValueError("inertia_body_kg_m2 must be symmetric")
    if np.linalg.eigvalsh(inertia)[0] <= 0.0:
        raise ValueError("inertia_body_kg_m2 must be positive definite")
    return sample
