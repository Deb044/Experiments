"""Teacher-forced action diagnostics and paired closed-loop simulator episodes."""

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .actions import hold_action
from .data import Windows, build_manifest
from .detector import load_detector
from .gating import Gate
from .io import read_json, write_json
from .simulator import hold_sanity, observation_window, quaternion_distance
from .training import ARMS, load_adapter, synchronize


def csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_episode(cfg, backend, simulator, task, trial, mode, detector, requested, deadline, video_path=None):
    env, obs = simulator.environment(task, trial)
    gate = Gate(mode, detector.threshold if detector is not None else 0.5, cfg.gate_smoothing)
    traces, frames = [], []
    started, success, error, truncated = time.monotonic(), False, None, False
    try:
        for _ in range(cfg.settle_steps):
            deadline.check()
            obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                raise RuntimeError("task already completed during settling; invalid evaluation state")
        first_position = obs["robot0_eef_pos"].copy()
        first_orientation = obs["robot0_eef_quat"].copy()
        first_gripper = obs["robot0_gripper_qpos"].copy()
        previous_action, previous_active, previous_gripper = None, None, None
        max_position, max_rotation, max_gripper = 0.0, 0.0, 0.0
        for step in range(simulator.horizon):
            deadline.check()
            window = observation_window(obs, task["instruction"])
            synchronize()
            then = time.monotonic()
            if mode == "oracle":
                probability = float(requested)
            elif mode in {"base", "always"}:
                probability = 0.0 if mode == "base" else 1.0
            else:
                visual, text = backend.features(window)
                probability = detector.probability(visual, text)
            value = gate.step(probability)
            if mode in {"direct", "oracle"} and value == 1:
                action = hold_action(window["gripper_qpos"], cfg.gripper_open_threshold)
            else:
                action = backend.action(window, value if mode not in {"direct", "oracle"} else 0)
            synchronize()
            elapsed = time.monotonic() - then
            if np.asarray(action).shape != (7,) or not np.all(np.isfinite(action)):
                raise ValueError("invalid policy action")
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            displacement = float(np.linalg.norm(obs["robot0_eef_pos"] - first_position))
            angle = quaternion_distance(first_orientation, obs["robot0_eef_quat"])
            gripper_displacement = float(np.linalg.norm(obs["robot0_gripper_qpos"] - first_gripper))
            max_gripper = max(max_gripper, gripper_displacement)
            max_position, max_rotation = max(max_position, displacement), max(max_rotation, angle)
            active = value >= gate.threshold
            trace = {
                "step": step,
                "probability": probability,
                "gate": value,
                "latency_seconds": elapsed,
                "position_displacement_m": displacement,
                "rotation_displacement_rad": angle,
                "translation_command_norm": float(np.linalg.norm(action[:3])),
                "gripper_displacement": gripper_displacement,
                "rotation_command_norm": float(np.linalg.norm(action[3:6])),
                "gripper_command": float(action[6]),
                "translation_command_jump": None
                if previous_action is None
                else float(np.linalg.norm(action[:3] - previous_action[:3])),
                "rotation_command_jump": None
                if previous_action is None
                else float(np.linalg.norm(action[3:6] - previous_action[3:6])),
                "gate_threshold_crossing": int(previous_active is not None and active != previous_active),
                "gripper_transition": int(previous_gripper is not None and action[6] != previous_gripper),
            }
            traces.append(trace)
            previous_action, previous_active, previous_gripper = action.copy(), active, action[6]
            if video_path is not None:
                frames.append(np.asarray(window["image"]))
            if done:
                success = True
                break
        latencies = [row["latency_seconds"] for row in traces]
        discontinuity = {}
        for coordinate in ["translation", "rotation"]:
            values = [
                r[f"{coordinate}_command_jump"] for r in traces if r[f"{coordinate}_command_jump"] is not None
            ]
            discontinuity[f"mean_{coordinate}_command_jump"] = float(np.mean(values)) if values else 0.0
            discontinuity[f"p95_{coordinate}_command_jump"] = (
                float(np.percentile(values, 95)) if values else 0.0
            )
        result = {
            "task_id": task["id"],
            "trial": trial,
            "requested": requested,
            "mode": mode,
            "success": success,
            "valid": True,
            "truncated": False,
            "steps": len(traces),
            "wall_seconds": time.monotonic() - started,
            "max_position_displacement_m": max_position,
            "max_rotation_displacement_rad": max_rotation,
            "max_gripper_displacement": max_gripper,
            "hold_compliant": max_position <= cfg.hold_position_tolerance
            and (max_rotation <= cfg.hold_rotation_tolerance)
            and max_gripper <= cfg.hold_gripper_tolerance,
            "false_intervention": (not requested) and any(row["gate"] >= gate.threshold for row in traces),
            "mean_activation": float(np.mean([row["gate"] for row in traces])),
            "median_latency_seconds": float(np.median(latencies)),
            "p95_latency_seconds": float(np.percentile(latencies, 95)),
            "gate_crossings": sum(row["gate_threshold_crossing"] for row in traces),
            "gripper_transitions": sum(row["gripper_transition"] for row in traces),
            **discontinuity,
            "error": None,
        }
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        truncated = isinstance(exception, TimeoutError)
        result = {
            "task_id": task["id"],
            "trial": trial,
            "requested": requested,
            "mode": mode,
            "success": None,
            "valid": False,
            "truncated": truncated,
            "steps": len(traces),
            "wall_seconds": time.monotonic() - started,
            "error": error,
        }
    finally:
        env.close()
    if frames and video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        # Video failure is an artifact error, never silently a policy failure or success.
        imageio.mimsave(video_path, frames, fps=20)
    return result, traces


def prepare(cfg, backend, simulator, deadline):
    if (cfg.root / "manifest.json").exists():
        from .data import load_manifest

        return load_manifest(cfg, verify_files=True)
    tasks = simulator.tasks()
    selection, accepted = [], []
    smoke_dir = cfg.root / "selection"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    sanity = {
        "direct": hold_sanity(cfg, simulator, tasks[0]),
        "quantized": hold_sanity(cfg, simulator, tasks[0], backend.codec),
    }
    write_json(smoke_dir / "hold-sanity.json", sanity)
    # Codec consistency is checked in physical/simulator coordinates, including gripper endpoints.
    for gripper in [np.array([0.03, -0.03]), np.zeros(2)]:
        target = hold_action(gripper, cfg.gripper_open_threshold)
        decoded = backend.codec.decode(backend.codec.encode(target))
        if not np.all(np.abs(decoded[:6] - target[:6]) <= backend.codec.half_bin_tolerance[:6] * 2 + 1e-6):
            raise ValueError("hold target action-token round trip exceeds quantization tolerance")
        if decoded[6] != target[6]:
            raise ValueError("hold target gripper round trip failed")
    for task in tasks:
        deadline.check()
        episodes = []
        for trial in range(cfg.smoke_trials):
            path = smoke_dir / f"task-{task['id']}-trial-{trial}.json"
            if path.exists():
                result = read_json(path)
            else:
                result, _ = run_episode(cfg, backend, simulator, task, trial, "base", None, False, deadline)
                write_json(path, result)
            episodes.append(result)
        passed = all(row["valid"] for row in episodes) and any(row["success"] for row in episodes)
        selection.append({"task_id": task["id"], "accepted": passed, "episodes": episodes})
        write_json(smoke_dir / "selection-progress.json", selection)
        print(f"Task {task['id']}: baseline eligible={passed}", flush=True)
        if passed:
            accepted.append(task)
        if len(accepted) == 6:
            break
    if len(accepted) != 6:
        raise ValueError("fewer than six baseline-capable tasks; no behavior-suppression comparison is valid")
    return build_manifest(cfg, accepted, selection)


def offline(cfg, backend, manifest, deadline):
    windows = Windows(cfg)
    rows = []
    try:
        configurations = [("base", 0, "base")]
        configurations += [(arm, stage, "always") for arm in ARMS for stage in range(1, 4)]
        configurations += [("orthogonal", 3, mode) for mode in ["hard", "soft", "smoothed"]]
        detector = load_detector(cfg)
        for arm, stage, mode in configurations:
            output = cfg.root / "offline" / f"{arm}-stage-{stage}-{mode}.json"
            if output.exists():
                rows.extend(read_json(output))
                continue
            if arm == "base":
                backend.adapters.load(backend.initial)
            else:
                load_adapter(cfg, backend, manifest, arm, stage)
            grouped = {}
            for record in manifest["records"]:
                if record["split"] != "test":
                    continue
                grouped.setdefault((record["task_id"], record["trajectory"]), []).append(record)
            current_rows = []
            backend.training(False)
            for group in grouped.values():
                gate = Gate(mode, detector.threshold, cfg.gate_smoothing)
                # History is reset for each complete trajectory and traversed chronologically.
                for record in sorted(group, key=lambda row: row["frame"]):
                    deadline.check()
                    window = windows.get(record)
                    if mode in {"hard", "soft", "smoothed"}:
                        visual, text = backend.features(window)
                        probability = detector.probability(visual, text)
                    else:
                        probability = float(mode == "always")
                    value = gate.step(probability)
                    backend.adapters.gate(value)
                    target = (
                        hold_action(window["gripper_qpos"], cfg.gripper_open_threshold)
                        if (record["role"] == "forget")
                        else window["action"]
                    )
                    with torch.no_grad():
                        ce, predicted = backend.supervised(window, target)
                    # These are teacher-forced diagnostics, not autonomous policy performance.
                    current_rows.append(
                        {
                            "arm": arm,
                            "stage": stage,
                            "mode": mode,
                            "key": record["key"],
                            "task_id": record["task_id"],
                            "request": record["stage"],
                            "role": record["role"],
                            "trajectory": record["trajectory"],
                            "ce": float(ce),
                            "translation_error": float(np.linalg.norm(predicted[:3] - target[:3])),
                            "rotation_error": float(np.linalg.norm(predicted[3:6] - target[3:6])),
                            "gripper_error": float(predicted[6] != target[6]),
                            "gate": value,
                        }
                    )
            write_json(output, current_rows)
            rows.extend(current_rows)
            print(f"Offline {arm} stage {stage} {mode}: {len(current_rows)} windows", flush=True)
        csv_rows(cfg.root / "offline.csv", rows)
        return rows
    finally:
        windows.close()


def rollout_configurations(cfg):
    configs = [
        ("base", "base"),
        ("sequential", "always"),
        ("orthogonal", "always"),
        ("orthogonal", "smoothed"),
        ("base", "direct"),
    ]
    if cfg.rollout_optional_gates:
        configs += [("orthogonal", "hard"), ("orthogonal", "soft")]
    return configs


def rollouts(cfg, backend, simulator, manifest, deadline):
    configurations = rollout_configurations(cfg)
    detector = load_detector(cfg)
    base_smoke = [episode for row in manifest["selection"] for episode in row["episodes"] if episode["valid"]]
    # Requested-task failures may consume the full horizon even when base successes are quick.
    seconds = (
        float(
            np.percentile([episode["wall_seconds"] / max(episode["steps"], 1) for episode in base_smoke], 95)
        )
        * simulator.horizon
    )
    preferred = len(configurations) * 6 * cfg.rollout_trials
    # Conservative 2x factor accounts for non-completion horizons and detector overhead.
    # Runtime budget still takes precedence; completion is checked, not assumed.
    budget_seconds = min(deadline.remaining, cfg.rollout_budget_hours * 3600)
    trials = cfg.rollout_trials if preferred * seconds * 2 <= budget_seconds else cfg.rollout_min_trials
    planned = len(configurations) * 6 * trials
    plan_path = cfg.root / "rollout-plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        trials, planned = plan["trials_per_task"], plan["planned_episodes"]
    else:
        write_json(
            plan_path,
            {
                "trials_per_task": trials,
                "planned_episodes": planned,
                "estimated_seconds": planned * seconds * 2,
                "smoke_p95_seconds": seconds,
                "initial_state_indices": list(range(cfg.smoke_trials, cfg.smoke_trials + trials)),
                "note": "evaluation states excluded from smoke selection; not asserted unseen in pretraining",
            },
        )
    expected_names = [
        f"{arm}-{mode}-task-{task['id']}-trial-{trial}"
        for task in manifest["tasks"]
        for trial in range(cfg.smoke_trials, cfg.smoke_trials + trials)
        for arm, mode in configurations
    ]
    remaining = sum(
        not (
            (cfg.root / "episodes" / f"{name}.json").exists()
            and read_json(cfg.root / "episodes" / f"{name}.json").get("valid")
        )
        for name in expected_names
    )
    if remaining * seconds * 2 > budget_seconds:
        raise TimeoutError(
            "minimum complete paired rollout set does not fit; offline results remain available"
        )
    episode_dir, trace_dir = cfg.root / "episodes", cfg.root / "traces"
    rows = []
    # Round-robin paired comparisons: a cutoff cannot leave all episodes for only one method.
    for task in manifest["tasks"]:
        for trial in range(cfg.smoke_trials, cfg.smoke_trials + trials):
            for arm, mode in configurations:
                deadline.check()
                name = f"{arm}-{mode}-task-{task['id']}-trial-{trial}"
                path = episode_dir / f"{name}.json"
                if path.exists():
                    result = read_json(path)
                    if result.get("valid"):
                        rows.append(result)
                        continue
                    # Preserve failure before retrying it; never substitute a different initial state.
                    archive = path.with_name(
                        path.stem + ".failed-" + datetime.now(timezone.utc).strftime("%H%M%S") + ".json"
                    )
                    path.rename(archive)
                if arm == "base":
                    backend.adapters.load(backend.initial)
                else:
                    load_adapter(cfg, backend, manifest, arm, 3)
                requested = task in manifest["tasks"][:3]
                video = (
                    cfg.root / "videos" / f"{name}.mp4"
                    if (trial - cfg.smoke_trials < cfg.video_episodes_per_configuration)
                    else None
                )
                result, traces = run_episode(
                    cfg, backend, simulator, task, trial, mode, detector, requested, deadline, video
                )
                result.update(arm=arm, configuration=f"{arm}-{mode}")
                write_json(path, result)
                csv_rows(trace_dir / f"{name}.csv", traces)
                rows.append(result)
                csv_rows(cfg.root / "episodes.csv", rows)
                print(f"Episode {name}: valid={result['valid']} success={result['success']}", flush=True)
                if not result["valid"]:
                    raise RuntimeError(f"rollout failed; not counted as policy failure: {result['error']}")
    csv_rows(cfg.root / "episodes.csv", rows)
    write_json(
        cfg.root / "rollout-status.json",
        {
            "complete": len(rows) == planned,
            "completed_episodes": len(rows),
            "planned_episodes": planned,
            "expected_episode_files": [name + ".json" for name in expected_names],
        },
    )
    return rows
