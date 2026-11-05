# WSO Solution

Western Sydney Orbital's controller candidates for **Escape from Phobos**, the
Fall 2026 Space Teams PRO challenge. The mission is to fly the shuttle from
Phobos to the return ship through impact debris while balancing time, fuel and
damage.

This repository is a fork of
[SimDynamX/STU4-Solution](https://github.com/SimDynamX/STU4-Solution). It keeps
the starter files at the root and adds the team's controller entrypoints. The
main controller has not been selected yet, so every candidate is kept runnable
for comparison.

Local practice worlds, the simulator, sweeps and recorded comparisons remain in
[western_sydney_orbital](https://github.com/koalazub/western_sydney_orbital).

## Controller candidates

| Entrypoint | Owner | Supporting files | Notes |
| --- | --- | --- | --- |
| [`stp_controller.py`](stp_controller.py) | WSO | `stp_api.py`, `stp_policy.py`, `control_contract.py`, `maneuvers.py`, `tracking.py` | Portable checkpoint controller. Configure with `WSORB_STP_POLICY_JSON` or `WSORB_STP_POLICY_FILE`. A disabled document runs a read-only probe. |
| [`stp_official_profile.py`](stp_official_profile.py) | Ali | `stp_official.py` plus the WSO modules above | Source port of the local official-profile beeline. Configure with `WSORB_STP_OFFICIAL_JSON` or `WSORB_STP_OFFICIAL_FILE`. A disabled probe is read-only. |
| [`bui_controller.py`](bui_controller.py) | Bui | [`mvp/`](mvp/) | Alpha-beta tracking with cruise and sidestep guidance. Needs OpenCV (`cv2`). |
| [`tom_controller.py`](tom_controller.py) | Tom | `avoidance.py`, `debris_tracker.py`, `radar_display.py`, `camera_overlay.py`, `live_map.py` | Kalman tracker with velocity-candidate avoidance. Needs matplotlib and Pillow. |

`Controller.py`, `BasicController.py` and `UserAPI.py` are the upstream starter
examples, unchanged. `Controller.py` is not the team's main controller.

## Running a controller

Point the licensed Space Teams PRO `ControllerSystem` at one entrypoint. Keep
the supporting files beside it; the portable entrypoints add their own directory
to the import path, and Python adds the script directory for the teammate
entrypoints.

Use exactly one command writer per flight. Do not run two entrypoints against
the same shuttle.

For the portable controllers, start with a read-only configuration:

```json
{"schema": "wsorb.stp_policy", "schema_version": 1, "enabled": false}
```

```json
{"schema": "wsorb.stp_official", "schema_version": 1, "enabled": false, "probe": true}
```

An enabled run needs an independently established `radar_clock_offset_s`, a
reviewed checkpoint radius and sole command ownership. The configuration
reference lives in `STPPolicyConfig` and `STPOfficialConfig`.

This repository contains source only. It does not include a licensed-runtime
result, and no candidate should be described as an official STP score until it
has been probed and flown on the real server.

## Dependencies

The WSO and Ali modules need Python 3.10+ and NumPy. Bui's controller also needs
OpenCV; Tom's controller also needs matplotlib and Pillow. `spaceteams` is
supplied by the licensed runtime.

## Upstream

```sh
git remote add upstream https://github.com/SimDynamX/STU4-Solution.git
git fetch upstream
```

The fork uses Jujutsu locally. Push with `jj git push --bookmark main`.

## Competition resources

- [Rules and submissions](https://github.com/SimDynamX/SpaceTeamsPRO/discussions/218)
- [Official announcements and discussions](https://github.com/SimDynamX/SpaceTeamsPRO/discussions)
- [Starter code and wiki](https://github.com/SimDynamX/STU4-Solution/wiki)
- [Competition simulation source](https://github.com/SimDynamX/STU4-Sim)
- [Python API documentation](https://simdynamx.github.io/stp-python-api-docs/)
