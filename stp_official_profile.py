"""Portable ControllerSystem entry point for the official-profile source port.

This file is a source port of Ali's official-profile behavior, not licensed-
runtime evidence. It has not been run against the real Space Teams Pro SDK.
The operator must run the disabled-by-default probe and validation on the real
STP server before relying on any command.

Point the competition's ControllerSystem at this file using its existing
runtime configuration. Keep the sibling modules stp_official.py, stp_api.py,
stp_policy.py and control_contract.py beside this file. This portable subset
targets Python 3.10+ syntax and needs compatible NumPy and the real bundled
spaceteams SDK. The historical practice-model package and flight/ are not
imported.

Launch arguments are untouched: this file parses no argv and calls
``connect_from_argv()`` with no arguments, so the licensed runtime's original
sys.argv reaches the SDK unchanged. Configuration comes only from
WSORB_STP_OFFICIAL_FILE or WSORB_STP_OFFICIAL_JSON. A valid disabled document is::

    {"schema": "wsorb.stp_official", "schema_version": 1, "enabled": false}

The probe is off by default. Set probe=true for a read-only connection and one
JSON report; probe mode NEVER writes a command, including zero. Missing or
disabled configuration with probe=false exits without connecting to the SDK.
To command, set enabled=true and add radar_clock_offset_s using the runtime's
independently known clock origins. Mission measurement time equals the unchanged
source radar time plus that declared offset. Zero is valid only when the operator
knows the origins coincide; no offset is inferred from a reading. Other reviewed
settings and their limits are in STPOfficialConfig. Configuration is loaded once,
before connecting.

When enabled, this process is the single command writer for the shuttle. Do not
run it alongside the original Controller.py/UserAPI.py or any other system
commanding its jets. The loop is paced by the real LoopFreqHz and bounded by
duration_wall_s. It writes at most one acceleration/angular-acceleration pair per
iteration through the single ``_write_command`` call site. A fresh SDK elapsed
clock is read after computation and checked against the original decision
deadline before each write; an expired or failed decision is discarded without
retiming. A best-effort zero pair is attempted in finally after explicit enabled
ownership. Python finally cannot guarantee cleanup after process death, a blocked
SDK call or a failed write, and zero requests do not cancel momentum or gravity.
There is no native SDK lease, watchdog, terrain, vision, hull, target-velocity,
mission-status or scoring signal here. State and radar reads are sequential, not
an atomic observation.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _write_command(client: Any, accel: npt.ArrayLike, angular_accel: npt.ArrayLike) -> None:
    """The only ``SetCommand`` call site in this entry point, enabled mode only."""
    client.SetCommand(accel, angular_accel)


def _disabled_notice() -> dict[str, Any]:
    return {
        "schema": "wsorb.stp_official_disabled.v1",
        "enabled": False,
        "probe": False,
        "connected": False,
        "commands_written": False,
        "source_port_only": True,
        "licensed_runtime_evidence": False,
        "probe_validated_on_real_server": False,
        "message": (
            "Set WSORB_STP_OFFICIAL_JSON or WSORB_STP_OFFICIAL_FILE. Use probe=true "
            "for a read-only connection, or enabled=true with an independently "
            "established radar_clock_offset_s to take command ownership."
        ),
    }


def _run_enabled(
    client: Any,
    config: Any,
    controller: Any,
    *,
    sample_from_state: Any,
    decision_is_current: Any,
) -> None:
    """Take explicit command ownership and write through the single writer.

    The caller has already checked ``config.enabled`` and constructed the
    controller. Every write is preceded by a fresh SDK clock read and the
    original decision-deadline check. This is a source-port entry point, not a
    flight-safety or licensed-runtime validation.
    """
    execution = {
        "schema": "wsorb.stp_official_execution.v1",
        "source_port_only": True,
        "licensed_runtime_evidence": False,
        "probe_validated_on_real_server": False,
        "read_only": False,
        "single_command_writer": True,
        "submission_attempts": 0,
        "command_pairs_submitted": 0,
        "nonzero_pairs_submitted": 0,
        "last_submission": None,
        "cleanup_zero_attempted": False,
        "cleanup_zero_pair_returned": False,
        "outcome_scope": (
            "SDK writes are sequential; a returned call is not proof of actuator "
            "application or official scoring success"
        ),
    }
    discard_event = None
    try:
        # The probe is read-only and runs before command ownership begins. It
        # reports runtime facts and unknown ages; it never writes a command.
        print(
            json.dumps(
                client.read_only_probe(
                    radar_clock_offset_s=config.radar_clock_offset_s,
                    max_radar_age_s=config.max_radar_age_s,
                ),
                allow_nan=False,
            )
        )
        deadline = time.monotonic() + config.duration_wall_s
        while (remaining := deadline - time.monotonic()) > 0.0:
            # Wall-clock pacing only; sleep is not a simulation-step API.
            time.sleep(min(1.0 / client.loop_frequency_hz, remaining))
            if time.monotonic() >= deadline:
                break
            # Read radar first to reduce radar rows that are future-dated
            # relative to the later state report. The reads remain sequential.
            debris = client.GetNearestDebrisLocations(500)
            state = client.read_state()
            now = client.elapsed_time_s()
            accel, angular_accel = controller.command(
                sample_from_state(state),
                debris,
                now_elapsed_time_s=now,
            )
            report = controller.report
            fresh_now = client.elapsed_time_s()
            if time.monotonic() >= deadline:
                break
            if not decision_is_current(report, now_elapsed_time_s=fresh_now):
                discard_event = {
                    "event": "decision_discarded",
                    "reason": "Original source validity deadline expired or invalid",
                    "decision_valid_until_elapsed_s": report["decision_valid_until_elapsed_s"],
                    "post_compute_elapsed_time_s": fresh_now,
                }
                break
            if report["failure_reasons"]:
                discard_event = {
                    "event": "decision_discarded",
                    "reason": "The controller did not approve a current command",
                    "controller_failure_reasons": report["failure_reasons"],
                }
                break
            execution["submission_attempts"] += 1
            _write_command(client, accel, angular_accel)
            execution["command_pairs_submitted"] += 1
            execution["nonzero_pairs_submitted"] += int(np.any(accel) or np.any(angular_accel))
            execution["last_submission"] = {
                "source_elapsed_time_s": state.elapsed_time_s,
                "checked_elapsed_time_s": fresh_now,
                "accel_body_mps2": accel.tolist(),
                "angular_accel_body_rad_s2": angular_accel.tolist(),
            }
    finally:
        # This branch exists only after explicit enabled ownership. Best-effort
        # real zeros do not erase momentum and are not a watchdog.
        execution["cleanup_zero_attempted"] = True
        with suppress(Exception):
            _write_command(client, np.zeros(3), np.zeros(3))
            execution["cleanup_zero_pair_returned"] = True
        # Log after cleanup so a blocked stdout cannot delay the zero attempt.
        if discard_event is not None:
            print(json.dumps(discard_event, allow_nan=False))
        print(json.dumps(controller.report, allow_nan=False))
        print(json.dumps(execution, allow_nan=False))


def main() -> None:
    # The colleague bundle runs outside this project's Python 3.13 environment.
    if sys.version_info < (3, 10):  # noqa: UP036
        raise RuntimeError("This controller bundle requires Python 3.10+ and a matching STP SDK")
    entry = Path(__file__).resolve()
    sys.path.insert(0, str(entry.parent))
    from stp_api import connect_from_argv
    from stp_official import (
        STPOfficialController,
        decision_is_current,
        load_config_from_environment,
    )
    from stp_policy import sample_from_stp_state

    config = load_config_from_environment()
    # Missing/disabled configuration without an explicit probe does not even
    # connect. This path cannot write a command, including zero.
    if not config.enabled and not config.probe:
        print(json.dumps(_disabled_notice(), allow_nan=False))
        return
    with connect_from_argv() as client:
        # Probe mode is explicitly read-only and returns before any write path.
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
            return
        controller = STPOfficialController(config)
        _run_enabled(
            client,
            config,
            controller,
            sample_from_state=sample_from_stp_state,
            decision_is_current=decision_is_current,
        )
    # The context manager calls the real st.leave_sim even if the loop raises.


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
