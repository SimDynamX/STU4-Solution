"""ControllerSystem entry point for a licensed Space Teams Pro simulation.

Point the competition's ControllerSystem at this file using its existing
runtime configuration. Keep the sibling modules stp_api.py, stp_policy.py,
control_contract.py, maneuvers.py and tracking.py beside this file. This
portable subset targets Python 3.10+ syntax and needs compatible NumPy and the
real bundled spaceteams SDK. Actual SDK/interpreter support must be checked in
the licensed runtime; no SDK or interpreter is installed by this repository.
The historical practice-model package and flight/ are not imported.

Configuration comes only from WSORB_STP_POLICY_FILE or WSORB_STP_POLICY_JSON,
never argv. A valid disabled JSON document is:
{"schema": "wsorb.stp_policy", "schema_version": 1, "enabled": false}
To enable, explicitly set enabled=true and add radar_clock_offset_s using the
runtime's known clock origins. Mission measurement time equals the unchanged
source radar time plus this declared offset. Zero is valid only if independently
known to be correct. No offset is inferred from a reading. Other reviewed
settings and their limits are in STPPolicyConfig. Configuration is loaded once,
before connecting. Missing/disabled configuration prints a single read-only
probe and leaves. It NEVER writes even zero commands. Invalid config aborts.

Use exactly one active command writer for the shuttle. Do not run this alongside
the original Controller.py/UserAPI.py or another system commanding its jets.
The default legacy policy computes target PD, radial repulsion and spin damping.
Explicit policy=checkpoint_velocity uses bounded desired velocity and P velocity
error. Review checkpoint_radius_m against server Scoring's KeepOutRadius_m.
Defaults are a 50 m radius, 1 m interior inset and 8 m/s cruise cap. Approach
speed also includes a stopping reserve and P-loop lag allowance. Reports expose
frame-relative speed and a configured 0.2 m/s low-speed threshold; target-relative
velocity is unavailable. This derived radius/speed measurement is not official
success and NEVER freezes SDK flight. Optional body_axis_preference=true tests
each proposed axis command before selection. tracking_diagnostics can disable
diagnostic tracking; tracks cannot grant clearance. There is no terrain geometry,
vision or collision guarantee. State and
radar reads are sequential, not an atomic observation. DoSpawning=true is
explicitly rejected. Zero requests do not cancel momentum or gravity.

The loop runs for duration_wall_s, at most 300 seconds for legacy or an explicitly
configured 3600 seconds for checkpoint control, paced by the real
LoopFreqHz. This is a wall-clock budget, not a simulation-step API or an SDK
call timeout. A fresh real SDK elapsed clock after computation checks the
original state/radar validity deadline before each write. An expired decision is
discarded without retiming or retrying nonzero. There is no native SDK decision
lease, so this still cannot bound transport delay after the last clock read.
Enabled ownership attempts real zero commands in finally before leave_sim.
Python finally cannot guarantee cleanup after kernel/process death, blocked SDK
calls or failed writes. There is no native lease/watchdog parity here.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np


def main() -> None:
    # The colleague bundle runs outside this project's Python 3.13 environment.
    if sys.version_info < (3, 10):  # noqa: UP036
        raise RuntimeError("This controller bundle requires Python 3.10+ and a matching STP SDK")
    entry = Path(__file__).resolve()
    sys.path.insert(0, str(entry.parent))
    from stp_api import connect_from_argv
    from stp_policy import (
        STPCheckpointController,
        decision_is_current,
        load_config_from_environment,
        sample_from_stp_state,
    )

    config = load_config_from_environment()
    # No arguments are parsed or removed. The SDK receives original sys.argv.
    with connect_from_argv() as client:
        if not config.enabled:
            print(
                json.dumps(
                    client.read_only_probe(
                        radar_clock_offset_s=config.radar_clock_offset_s,
                        max_radar_age_s=config.max_radar_age_s,
                    ),
                    allow_nan=False,
                )
            )
            return  # No zero-on-exit writes in probe mode, including on errors.
        controller = None
        discard_event = None
        execution = {
            "schema": "wsorb.stp_controller_execution.v1",
            "read_only": False,
            "submission_attempts": 0,
            "command_pairs_submitted": 0,
            "nonzero_pairs_submitted": 0,
            "last_submission": None,
            "cleanup_zero_attempted": False,
            "cleanup_zero_pair_returned": False,
            "state_after_cleanup": None,
            "state_after_cleanup_error": None,
            "outcome_scope": (
                "SDK writes are sequential; a returned call is not proof of actuator application "
                "or official scoring success"
            ),
        }
        try:
            # Explicit enable gives this process command ownership. Other jet
            # writers must already be disabled by the operator.
            deadline = time.monotonic() + config.duration_wall_s
            controller = STPCheckpointController(config)
            print(
                json.dumps(
                    client.read_only_probe(
                        radar_clock_offset_s=config.radar_clock_offset_s,
                        max_radar_age_s=config.max_radar_age_s,
                    ),
                    allow_nan=False,
                )
            )
            while (remaining := deadline - time.monotonic()) > 0.0:
                # Wall-clock pacing only. sleep is NOT a simulation-step API.
                time.sleep(min(1.0 / client.loop_frequency_hz, remaining))
                if time.monotonic() >= deadline:
                    break
                # Request up to 500 public rows; the radar may expose fewer.
                # Read radar first to reduce future timestamps relative to state.
                # The reads still cannot form an atomic snapshot.
                debris = client.GetNearestDebrisLocations(500)
                state = client.read_state()
                now = client.elapsed_time_s()
                accel, angular_accel = controller.command(
                    sample_from_stp_state(state), debris, now_elapsed_time_s=now
                )
                # Report copying/tracking/candidate checks may take time. Only
                # a newly read real SDK clock can check the ORIGINAL deadline.
                report = controller.report
                fresh_now = client.elapsed_time_s()
                if time.monotonic() >= deadline:
                    break
                if not decision_is_current(report, now_elapsed_time_s=fresh_now):
                    # Do not refresh old samples or retry nonzero. The finally
                    # block attempts real zeros under enabled ownership.
                    discard_event = {
                        "event": "decision_discarded",
                        "reason": "Original source validity deadline expired or invalid",
                        "decision_valid_until_elapsed_s": report["decision_valid_until_elapsed_s"],
                        "post_compute_elapsed_time_s": fresh_now,
                    }
                    break
                if report["failure_reasons"]:
                    # Match the API validator: an unapproved coast return is
                    # not a successfully guarded control decision.
                    discard_event = {
                        "event": "decision_discarded",
                        "reason": "The policy did not approve an admissible command",
                        "policy_failure_reasons": report["failure_reasons"],
                    }
                    break
                execution["submission_attempts"] += 1
                client.SetCommand(accel, angular_accel)
                execution["command_pairs_submitted"] += 1
                execution["nonzero_pairs_submitted"] += int(np.any(accel) or np.any(angular_accel))
                execution["last_submission"] = {
                    "source_elapsed_time_s": state.elapsed_time_s,
                    "checked_elapsed_time_s": fresh_now,
                    "accel_body_mps2": accel.tolist(),
                    "angular_accel_body_rad_s2": angular_accel.tolist(),
                }
        finally:
            # This branch only exists after explicit enabled ownership. Best
            # effort real zeros do not erase momentum and are not a watchdog.
            execution["cleanup_zero_attempted"] = True
            with suppress(Exception):
                client.SetCommand(np.zeros(3), np.zeros(3))
                execution["cleanup_zero_pair_returned"] = True
            try:
                observed = client.read_state()
                observed_sample = sample_from_stp_state(observed)
                distance = float(
                    np.linalg.norm(observed.target_position_phobos_m - observed.position_phobos_m)
                )
                speed = float(np.linalg.norm(observed.velocity_phobos_m_s))
                execution["state_after_cleanup"] = {
                    "elapsed_time_s": observed.elapsed_time_s,
                    "frame": observed_sample.frame,
                    "source_kind": observed_sample.source_kind,
                    "distance_to_target_m": distance,
                    "frame_relative_speed_mps": speed,
                    "fuel_used_kg": observed.fuel_used_kg,
                    "damage_j": observed.damage_j,
                    "configured_radius_m": config.checkpoint_radius_m,
                    "within_configured_radius": distance <= config.checkpoint_radius_m,
                    "official_scoring_status": None,
                    "reads_are_sequential": True,
                }
            except Exception as error:
                execution["state_after_cleanup_error"] = type(error).__name__
            # Logging happens after cleanup so a blocked stdout cannot delay
            # this best-effort zero request on an expired decision.
            if discard_event is not None:
                print(json.dumps(discard_event, allow_nan=False))
            if controller is not None:
                print(json.dumps(controller.report, allow_nan=False))
            print(json.dumps(execution, allow_nan=False))
    # The context manager calls the real st.leave_sim even if the loop raises.


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
