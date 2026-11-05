"""Bounded, diagnostic-only anonymous point tracking, using NumPy and Python 3.10+.

STP radar rows are [relative_position_P; closest_range, range_rate, diameter;
raw_time, 0, 0]. Their vectors already use the Phobos reference frame, not the
body or camera frame. The caller supplies ownship position/velocity in that same
frame at now_s. An old ownship pose is NOT available here: CV backdating and its
uncertainty are explicit modelling assumptions. Mission source time is exactly
raw time plus the caller's offset. The native API has offset zero; the licensed
SDK's different startup origin needs an operator-configured offset.

Defaults for sensor noise are public floors from official Radar.py at c1e3fc8:
independent 2 m Gaussian XYZ and closest range, 0.02 m/s scalar range rate, and
5 percent diameter. They are not universal sensor specifications. Other noise,
gates and lifetimes are chosen models, not calibrated confidence or tail bounds.
Closest range is not centre range, so range and diameter are reported, not fused
as additional centre measurements. There is no injected noise or truth input.

Track IDs name local hypotheses only. Empty radar, confirmation, reappearance
and covariance never establish clearance or physical identity. This module does
not import wsorb, flight, an SDK or a native wrapper, and does not change the raw
policy's obstacle envelopes. It can be imported directly beside stp_policy.py.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import groupby
from numbers import Real
from typing import Any

import numpy as np

__all__ = ["TrackerConfig", "RadarTracker"]


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [{low}, {high}]")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be a finite real number in [{low}, {high}]")
    return result


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Immutable assumptions and hard budgets, not a sensor calibration."""

    position_noise_std_m: float = 2.0
    closest_range_noise_std_m: float = 2.0
    range_rate_noise_std_mps: float = 0.02
    diameter_noise_fraction: float = 0.05
    acceleration_psd_m2_s3: float = 0.25
    ego_position_std_m: float = 1.0
    ego_velocity_std_mps: float = 0.5
    ego_acceleration_psd_m2_s3: float = 0.25
    initial_position_std_m: float = 10000.0
    initial_velocity_std_mps: float = 50.0
    max_source_age_s: float = 5.0
    coast_after_s: float = 1.0
    delete_after_s: float = 10.0
    fingerprint_history_s: float = 15.0
    max_association_distance_m: float = 250.0
    association_nis_gate: float = 25.0
    confirm_hits: int = 3
    max_tracks: int = 256
    max_input_rows: int = 1024
    max_rows_per_update: int = 256
    max_candidate_pairs: int = 8192
    max_fingerprints: int = 8192
    report_limit: int = 16

    def __post_init__(self) -> None:
        bounds = {
            "position_noise_std_m": (2.0, 1e4),
            "closest_range_noise_std_m": (2.0, 1e4),
            "range_rate_noise_std_mps": (0.02, 1e4),
            "diameter_noise_fraction": (0.05, 1.0),
            "acceleration_psd_m2_s3": (0.0, 1e6),
            "ego_position_std_m": (0.0, 1e4),
            "ego_velocity_std_mps": (0.0, 1e4),
            "ego_acceleration_psd_m2_s3": (0.0, 1e6),
            "initial_position_std_m": (1e4, 1e6),
            "initial_velocity_std_mps": (10.0, 1e4),
            "max_source_age_s": (0.001, 3600.0),
            "coast_after_s": (0.0, 3600.0),
            "delete_after_s": (0.001, 3600.0),
            "fingerprint_history_s": (0.001, 7200.0),
            "max_association_distance_m": (0.001, 1e6),
            "association_nis_gate": (0.001, 1e4),
        }
        for name, (low, high) in bounds.items():
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high))
        for name, high in {
            "confirm_hits": 100,
            "max_tracks": 256,
            "max_input_rows": 4096,
            "max_rows_per_update": 1024,
            "max_candidate_pairs": 65536,
            "max_fingerprints": 16384,
            "report_limit": 16,
        }.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= high:
                raise ValueError(f"{name} must be an integer in [1, {high}]")
        if self.confirm_hits < 2:
            raise ValueError("confirmation requires multiple genuinely new source timestamps")
        if not self.coast_after_s < self.delete_after_s:
            raise ValueError("coast_after_s must be below delete_after_s")
        if not self.max_source_age_s < self.delete_after_s:
            raise ValueError("max_source_age_s must be below delete_after_s")
        if self.fingerprint_history_s < self.max_source_age_s:
            raise ValueError("fingerprint_history_s must cover max_source_age_s")
        if self.max_rows_per_update > self.max_input_rows:
            raise ValueError("max_rows_per_update must not exceed max_input_rows")


def _array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Reject bools even in mixed Python lists before NumPy coerces them."""
    if isinstance(value, np.ndarray):
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real array of shape {shape}, without bools")
    elif isinstance(value, (list, tuple)) and shape and len(value) == shape[0]:
        if len(shape) > 1:
            return np.stack([_array(item, name, shape[1:]) for item in value])
        for item in value:
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
                raise ValueError(f"{name} must contain real numbers, without bools")
    else:
        raise ValueError(f"{name} must be a real array of shape {shape}, without bools")
    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.array(value, dtype=np.float64, copy=True)
    except (OverflowError, ValueError, FloatingPointError) as error:
        raise ValueError(f"{name} must contain representable real numbers") from error


def _vector(value: Any, name: str, limit: float) -> np.ndarray:
    result = _array(value, name, (3,))
    if not np.isfinite(result).all() or np.any(np.abs(result) > limit):
        raise ValueError(f"{name} must be finite with components bounded by {limit}")
    return result


def _radar(value: Any, limit: int) -> np.ndarray:
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise ValueError("radar must have shape (N, 3, 3)")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        raise ValueError("radar must have shape (N, 3, 3)")
    if len(value) > limit:
        raise ValueError(f"radar exceeds max_input_rows={limit}")
    if len(value) == 0:
        if isinstance(value, np.ndarray) and value.shape not in ((0,), (0, 3, 3)):
            raise ValueError("radar must have shape (N, 3, 3)")
        return np.empty((0, 3, 3))
    return _array(value, "radar", (len(value), 3, 3))


def _transition(dt: float) -> np.ndarray:
    result = np.eye(6)
    result[:3, 3:] = np.eye(3) * dt
    return result


def _process_covariance(dt: float, psd: float) -> np.ndarray:
    """Continuous white acceleration, including p/v cross covariance."""
    return np.kron(np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]), np.eye(3)) * psd


def _symmetric_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    if values[0] < 0.0:
        matrix = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return (matrix + matrix.T) * 0.5


def _predict(state: np.ndarray, covariance: np.ndarray, dt: float, psd: float):
    transition = _transition(dt)
    return transition @ state, _symmetric_psd(
        transition @ covariance @ transition.T + _process_covariance(dt, psd)
    )


@dataclass(slots=True)
class _Measurement:
    row: np.ndarray
    source_time: float
    ego_position: np.ndarray
    ego_velocity: np.ndarray
    ego_covariance: np.ndarray
    position: np.ndarray
    lag: float


@dataclass(slots=True)
class _Track:
    serial: int
    state: np.ndarray
    covariance: np.ndarray
    source_time: float
    born_time: float
    hits: int
    confirmed_time: float | None
    ambiguous: bool
    diagnostic: dict[str, Any]


class RadarTracker:
    """One bounded local tracking generation; update/coast return detached JSON data.

    Malformed shapes, bools, ownship data and scalar arguments raise ValueError
    before mutation. Invalid individual numeric radar rows are counted/discarded.
    Future/stale rows are never shifted or clipped into the acceptance window.
    Reports predict copies to now; stored posteriors remain at accepted source
    times. A late row can be gated for rejection, never used to rewind a posterior.
    This object is not intended for simultaneous calls from multiple threads.
    """

    def __init__(self, config: TrackerConfig | None = None):
        if config is not None and not isinstance(config, TrackerConfig):
            raise ValueError("config must be TrackerConfig or None")
        self.config = TrackerConfig() if config is None else config
        self._tracks: list[_Track] = []
        self._history: dict[bytes, float] = {}
        self._forgotten_through = -1.0
        self._frame: str | None = None
        self._now: float | None = None
        self._offset = 0.0
        self._generation = 0
        self._serial = 0
        self._created = 0
        self._deleted = 0

    def _begin(self, now: float, frame: str, offset: float) -> dict[str, Any]:
        if not isinstance(frame, str) or not frame.strip() or len(frame) > 128:
            raise ValueError("frame must be a nonempty frame name of at most 128 characters")
        reasons = []
        if self._frame is None:
            reasons.append("initialization")
        elif frame != self._frame:
            reasons.append("frame_changed")
        if self._now is not None and now < self._now:
            reasons.append("mission_clock_rewind")
        if self._frame is not None and offset != self._offset:
            reasons.append("radar_clock_offset_changed")
        reset_count = len(self._tracks) if reasons else 0
        if reasons:
            self._generation += 1
            self._tracks.clear()
            self._history.clear()
            self._forgotten_through = -1.0
        self._frame, self._now, self._offset = frame, now, offset
        deleted = [t for t in self._tracks if now >= t.source_time + self.config.delete_after_s]
        self._tracks = [t for t in self._tracks if now < t.source_time + self.config.delete_after_s]
        self._deleted += len(deleted)
        self._history = {
            key: time
            for key, time in self._history.items()
            if time >= now - self.config.fingerprint_history_s
        }
        return {
            "reset_reasons": reasons,
            "reset_track_count": reset_count,
            "deleted_count": len(deleted),
            "deleted_ids": [self._identifier(t) for t in deleted[: self.config.report_limit]],
            "input_rows": 0,
            "candidate_pairs": 0,
            "associated_rows": 0,
            "created_tracks": 0,
            "ambiguous_rows": 0,
            "discarded": Counter(),
        }

    def _identifier(self, track: _Track) -> str:
        return f"g{self._generation}-t{track.serial}"

    def _measurement(self, row: np.ndarray, now: float, position, velocity) -> _Measurement:
        time = float(row[2, 0] + self._offset)
        lag = now - time
        backward = _transition(-lag)
        ego_prior = np.diag(
            [self.config.ego_position_std_m**2] * 3 + [self.config.ego_velocity_std_mps**2] * 3
        )
        # Backward CV has negative p/v correlations. This is an uncertainty model,
        # not a historical pose measurement or a smoothing result.
        reverse_sign = np.diag([1.0] * 3 + [-1.0] * 3)
        ego_covariance = (
            backward @ ego_prior @ backward.T
            + reverse_sign
            @ (_process_covariance(lag, self.config.ego_acceleration_psd_m2_s3))
            @ reverse_sign
        )
        ego_position = position - lag * velocity
        return _Measurement(
            row, time, ego_position, velocity.copy(), ego_covariance, ego_position + row[0], lag
        )

    def _innovation(self, state, covariance, measurement: _Measurement):
        relative = state[:3] - measurement.ego_position
        length = float(np.linalg.norm(relative))
        if length < 1e-9:
            return None
        direction = relative / length
        relative_velocity = state[3:] - measurement.ego_velocity
        rate = float(direction @ relative_velocity)
        gradient = (relative_velocity - direction * rate) / length
        jacobian = np.zeros((4, 6))
        jacobian[:3, :3] = np.eye(3)
        jacobian[3, :3], jacobian[3, 3:] = gradient, direction
        ego_jacobian = jacobian.copy()
        noise = np.diag(
            [self.config.position_noise_std_m**2] * 3 + [self.config.range_rate_noise_std_mps**2]
        )
        noise += ego_jacobian @ measurement.ego_covariance @ ego_jacobian.T
        observed = np.append(measurement.position, measurement.row[1, 1])
        expected = np.append(state[:3], rate)
        residual = observed - expected
        innovation_covariance = _symmetric_psd(jacobian @ covariance @ jacobian.T + noise)
        try:
            nis = float(residual @ np.linalg.solve(innovation_covariance, residual))
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(nis):
            return None
        return residual, jacobian, noise, innovation_covariance, max(0.0, nis)

    def _correct(self, state, covariance, measurement: _Measurement, innovation, dt: float):
        residual, jacobian, noise, innovation_covariance, nis = innovation
        gain = np.linalg.solve(innovation_covariance, jacobian @ covariance).T
        corrected = state + gain @ residual
        complement = np.eye(6) - gain @ jacobian
        posterior = _symmetric_psd(complement @ covariance @ complement.T + gain @ noise @ gain.T)
        post = self._innovation(corrected, posterior, measurement)
        diagnostic = {
            "raw_source_time_s": float(measurement.row[2, 0]),
            "aligned_source_time_s": measurement.source_time,
            "raw_radar_row": measurement.row.tolist(),
            "ownship_backdating_s": measurement.lag,
            "ownship_position_cv_estimate_world_m": measurement.ego_position.tolist(),
            "ownship_velocity_cv_estimate_world_mps": measurement.ego_velocity.tolist(),
            "ownship_covariance_at_source": measurement.ego_covariance.tolist(),
            "prediction_dt_s": dt,
            "H": jacobian.tolist(),
            "R": noise.tolist(),
            "innovation": residual.tolist(),
            "innovation_covariance": innovation_covariance.tolist(),
            "innovation_nis_not_probability": nis,
            "position_residual_norm_m": float(np.linalg.norm(residual[:3])),
            "radial_rate_residual_mps": float(residual[3]),
            "post_update_residual": None if post is None else post[0].tolist(),
        }
        return corrected, posterior, diagnostic

    def _remember(self, rows: list[tuple[float, bytes, np.ndarray]]) -> None:
        self._history.update((key, time) for time, key, _ in rows)
        if len(self._history) > self.config.max_fingerprints:
            # One bounded sort, not a full history scan for every evicted row.
            # Closing the evicted prefix prevents reusing forgotten evidence.
            ordered = sorted(self._history.items(), key=lambda item: (item[1], item[0]))
            removed = ordered[: -self.config.max_fingerprints]
            self._forgotten_through = max(self._forgotten_through, removed[-1][1])
            self._history = dict(ordered[-self.config.max_fingerprints :])

    def update(
        self,
        now_s,
        position_world_m,
        velocity_world_mps,
        radar,
        *,
        frame,
        radar_clock_offset_s=0.0,
    ) -> dict[str, Any]:
        """Consume heterogeneous anonymous source rows, without modifying inputs."""
        now = _number(now_s, "now_s", 0.0, 1e12)
        offset = _number(radar_clock_offset_s, "radar_clock_offset_s", -1e12, 1e12)
        position = _vector(position_world_m, "position_world_m", 1e12)
        velocity = _vector(velocity_world_mps, "velocity_world_mps", 1e6)
        rows = _radar(radar, self.config.max_input_rows)
        event = self._begin(now, frame, offset)
        event["input_rows"] = len(rows)
        new = []
        in_call = set()
        for row in rows:
            if (
                not np.isfinite(row).all()
                or np.any(np.abs(row) > 1e12)
                or row[1, 2] <= 0.0
                or abs(row[1, 1]) > 1e6
                or row[2, 0] < 0.0
                or np.any(row[2, 1:] != 0.0)
                or np.linalg.norm(row[0]) == 0.0
            ):
                event["discarded"]["invalid_row"] += 1
                continue
            time = float(row[2, 0] + offset)
            if time < 0.0 or time > now:
                event["discarded"]["invalid_or_future_source_time"] += 1
                continue
            if now - time > self.config.max_source_age_s:
                event["discarded"]["stale_source_time"] += 1
                continue
            # Canonical fingerprint only. The original numeric row is not edited.
            key = np.where(row == 0.0, 0.0, row).astype("<f8", copy=False).tobytes()
            if key in self._history or key in in_call:
                event["discarded"]["repeated_raw_row"] += 1
                continue
            if time <= self._forgotten_through:
                event["discarded"]["history_closed_source_time"] += 1
                continue
            in_call.add(key)
            new.append((time, key, row))
        new.sort(key=lambda item: (item[0], item[1]))
        excess = max(0, len(new) - self.config.max_rows_per_update)
        # Prefer recent source rows, then process them in source-time order.
        selected = new[excess:]
        if excess and selected[0][0] == new[excess - 1][0]:
            # Do not force associations from a partially inspected timestamp.
            cut_time = selected[0][0]
            selected = [item for item in selected if item[0] != cut_time]
        event["discarded"]["row_budget"] += len(new) - len(selected)
        self._remember(new)
        for _, group in groupby(selected, key=lambda item: item[0]):
            measurements = [self._measurement(item[2], now, position, velocity) for item in group]
            self._associate(measurements, event)
        return self._report(now, event)

    def _associate(self, measurements: list[_Measurement], event: dict[str, Any]) -> None:
        if event["candidate_pairs"] + len(self._tracks) * len(measurements) > (
            self.config.max_candidate_pairs
        ):
            # Never assign an examined row while ignoring its unexamined rivals
            # at the same timestamp. Leave the entire source cohort unresolved.
            event["discarded"]["candidate_budget"] += len(measurements)
            return
        edges: dict[int, list[tuple[int, Any, Any, Any]]] = {}
        track_degree: Counter = Counter()
        for row_index, measurement in enumerate(measurements):
            edges[row_index] = []
            for index, track in enumerate(self._tracks):
                event["candidate_pairs"] += 1
                dt = measurement.source_time - track.source_time
                if dt < 0.0:
                    # Diagnostic backward gate ONLY. No old row changes state.
                    backward = _transition(dt)
                    reverse_sign = np.diag([1.0] * 3 + [-1.0] * 3)
                    state = backward @ track.state
                    covariance = _symmetric_psd(
                        backward @ track.covariance @ backward.T
                        + reverse_sign
                        @ _process_covariance(-dt, self.config.acceleration_psd_m2_s3)
                        @ reverse_sign
                    )
                else:
                    state, covariance = _predict(
                        track.state, track.covariance, dt, self.config.acceleration_psd_m2_s3
                    )
                if np.linalg.norm(measurement.position - state[:3]) > (
                    self.config.max_association_distance_m
                ):
                    continue
                innovation = self._innovation(state, covariance, measurement)
                if innovation is not None and innovation[-1] <= self.config.association_nis_gate:
                    edges[row_index].append((index, state, covariance, innovation))
                    track_degree[index] += 1
        ambiguous_tracks = {index for index, degree in track_degree.items() if degree > 1}
        for candidates in edges.values():
            if len(candidates) > 1:
                ambiguous_tracks.update(candidate[0] for candidate in candidates)
        for index in ambiguous_tracks:
            self._tracks[index].ambiguous = True
        births = []
        for row_index, candidates in edges.items():
            measurement = measurements[row_index]
            if any(candidate[0] in ambiguous_tracks for candidate in candidates):
                event["ambiguous_rows"] += 1
                continue
            if not candidates:
                births.append(measurement)
                continue
            index, state, covariance, innovation = candidates[0]
            track = self._tracks[index]
            dt = measurement.source_time - track.source_time
            if dt <= 0.0:
                reason = "late_source_time" if dt < 0.0 else "same_source_time"
                event["discarded"][reason] += 1
                continue
            track.state, track.covariance, track.diagnostic = self._correct(
                state, covariance, measurement, innovation, dt
            )
            track.source_time = measurement.source_time
            track.hits += 1
            track.ambiguous = False
            if track.hits >= self.config.confirm_hits and track.confirmed_time is None:
                track.confirmed_time = measurement.source_time
            event["associated_rows"] += 1
        for measurement in births:
            if len(self._tracks) >= self.config.max_tracks:
                event["discarded"]["track_budget"] += 1
                continue
            # A diffuse position prior centred at the detection is initialization,
            # not a second observation. Zero velocity is a broad PRIOR mean only.
            state = np.append(measurement.position, np.zeros(3))
            covariance = np.diag(
                [self.config.initial_position_std_m**2] * 3
                + [self.config.initial_velocity_std_mps**2] * 3
            )
            innovation = self._innovation(state, covariance, measurement)
            if innovation is None:
                event["discarded"]["unusable_measurement_model"] += 1
                continue
            state, covariance, diagnostic = self._correct(
                state, covariance, measurement, innovation, 0.0
            )
            self._serial += 1
            self._created += 1
            event["created_tracks"] += 1
            self._tracks.append(
                _Track(
                    self._serial,
                    state,
                    covariance,
                    measurement.source_time,
                    measurement.source_time,
                    1,
                    None,
                    False,
                    diagnostic,
                )
            )

    def coast(self, now_s, *, frame) -> dict[str, Any]:
        """Age/delete by source-time deadlines, without adding measurement evidence."""
        now = _number(now_s, "now_s", 0.0, 1e12)
        return self._report(now, self._begin(now, frame, self._offset))

    def _report(self, now: float, event: dict[str, Any]) -> dict[str, Any]:
        tracks = []
        counts: Counter = Counter()
        maxima = Counter()
        for track in self._tracks:
            age = now - track.source_time
            state, covariance = _predict(
                track.state, track.covariance, age, self.config.acceleration_psd_m2_s3
            )
            confirmed = track.confirmed_time is not None
            coasting = age > self.config.coast_after_s
            counts["confirmed" if confirmed else "tentative"] += 1
            counts["coasting"] += int(coasting)
            counts["association_ambiguous"] += int(track.ambiguous)
            for key, value in {
                "position_std_m": float(
                    np.sqrt(max(0.0, np.linalg.eigvalsh(covariance[:3, :3])[-1]))
                ),
                "velocity_std_mps": float(
                    np.sqrt(max(0.0, np.linalg.eigvalsh(covariance[3:, 3:])[-1]))
                ),
                "source_age_s": age,
                "innovation_nis_not_probability": track.diagnostic[
                    "innovation_nis_not_probability"
                ],
                "position_residual_m": track.diagnostic["position_residual_norm_m"],
                "absolute_radial_rate_residual_mps": abs(
                    track.diagnostic["radial_rate_residual_mps"]
                ),
            }.items():
                maxima[key] = max(maxima[key], value)
            if len(tracks) >= self.config.report_limit:
                continue
            tracks.append(
                {
                    "id": self._identifier(track),
                    "state_time_s": now,
                    "state": state.tolist(),
                    "covariance": covariance.tolist(),
                    "last_aligned_source_time_s": track.source_time,
                    "last_raw_source_time_s": track.diagnostic["raw_source_time_s"],
                    "source_age_s": age,
                    "track_age_s": now - track.born_time,
                    "tentative_age_s": None if confirmed else now - track.born_time,
                    "genuinely_new_timestamp_hits": track.hits,
                    "confirmed": confirmed,
                    "status": (
                        "coasting" if coasting else ("confirmed" if confirmed else "tentative")
                    ),
                    "coasting_age_s": max(0.0, age - self.config.coast_after_s),
                    "confirmation_age_s": None if not confirmed else now - track.confirmed_time,
                    "deletion_deadline_s": track.source_time + self.config.delete_after_s,
                    "association_ambiguous": track.ambiguous,
                    "velocity_information": (
                        "radial_only; tangential_unobserved; zero_is_prior_mean_not_observation"
                        if track.hits == 1
                        else "radial_rate_and_associated_positions; no_full_velocity_sensor"
                    ),
                    "last_assimilation": deepcopy(track.diagnostic),
                }
            )
        event["discarded"] = dict(event["discarded"])
        event["association_search_limited"] = any(
            event["discarded"].get(reason, 0)
            for reason in (
                "candidate_budget",
                "row_budget",
                "track_budget",
                "history_closed_source_time",
            )
        )
        noise = np.diag(
            [self.config.position_noise_std_m**2] * 3 + [self.config.range_rate_noise_std_mps**2]
        )
        return {
            "schema": "wsorb.radar_tracking.diagnostic.v1",
            "diagnostic_only": True,
            "clearance_granted": False,
            "physical_identity_proven": False,
            "calibrated_probabilities": False,
            "raw_obstacle_envelopes_modified": False,
            "frame": self._frame,
            "now_s": now,
            "generation": self._generation,
            "radar_clock_offset_s": self._offset,
            "model": {
                "state_order": ["pX", "pY", "pZ", "vX", "vY", "vZ"],
                "state_units": ["m", "m", "m", "m/s", "m/s", "m/s"],
                "process": (
                    "CV; continuous_white_acceleration; Q=q*[[dt^3/3 I,dt^2/2 I],[dt^2/2 I,dt I]]"
                ),
                "F_at_1s": _transition(1.0).tolist(),
                "Q_at_1s": _process_covariance(1.0, self.config.acceleration_psd_m2_s3).tolist(),
                "measurement_order": [
                    "world_pX",
                    "world_pY",
                    "world_pZ",
                    "scalar_relative_range_rate",
                ],
                "radar_R_before_ego_uncertainty": noise.tolist(),
                "measurement": (
                    "h=[p,u dot (v-ego_v)]; u=(p-ego_p)/norm(p-ego_p); "
                    "EKF H includes position derivative"
                ),
                "R": (
                    "diag(XYZ_sigma^2,rate_sigma^2)+H*ego_source_covariance*H.T; "
                    "ego errors assumed independent across source times; bias not estimated"
                ),
                "ego_backdating": (
                    "p_then=p_now-age*v_now; reverse_CV_covariance with negative "
                    "p/v correlations and continuous acceleration Q"
                ),
                "initialization": (
                    "diffuse position prior centred at detection; broad zero-mean "
                    "velocity prior, not a full-velocity observation"
                ),
                "noise_provenance": (
                    "chosen assumptions with public sensor floors from official Radar.py "
                    "c1e3fc8; not universal or calibrated probabilities"
                ),
                "not_fused": (
                    "closest surface range and noisy diameter are retained raw, "
                    "not independent centre-position/identity observations"
                ),
                "association": (
                    "source-time gates; reject every conflicting gated association, "
                    "no forced match; late backward gates are rejection-only"
                ),
                "history": (
                    "exact raw fingerprints; capacity eviction closes a source-time prefix, "
                    "possibly dropping unseen old rows rather than reusing evidence"
                ),
                "identity": (
                    "local hypotheses only; crossings, confirmation and reappearance "
                    "do not prove physical identity"
                ),
                "configuration": asdict(self.config),
            },
            "counts": {
                "active": len(self._tracks),
                "reported": len(tracks),
                "unreported": len(self._tracks) - len(tracks),
                "tentative": counts["tentative"],
                "confirmed": counts["confirmed"],
                "coasting": counts["coasting"],
                "association_ambiguous": counts["association_ambiguous"],
                "created_lifetime": self._created,
                "deleted_lifetime": self._deleted,
                "fingerprints": len(self._history),
            },
            "maxima": dict(maxima),
            "event": event,
            "tracks": tracks,
        }
