"""Tables and standalone figures from actual artifacts; missing results stay missing."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import csv_rows
from .io import read_json, write_json


def mean(values):
    return float(np.mean(values)) if values else None


def group_mean(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault((row["task_id"], row["trajectory"]), []).append(row[key])
    return mean([mean(values) for values in groups.values()])


def figure_save(figure, directory, name, synthetic):
    if synthetic:
        figure.suptitle("SYNTHETIC CPU PLUMBING - NOT RESEARCH RESULTS", fontsize=10)
    figure.tight_layout()
    figure.savefig(directory / f"{name}.png", dpi=180)
    figure.savefig(directory / f"{name}.pdf")
    plt.close(figure)


def generate(cfg):
    out = cfg.root / "report"
    out.mkdir(parents=True, exist_ok=True)
    synthetic = cfg.backend == "synthetic"
    training_rows = []
    for path in sorted((cfg.root / "training").glob("*.jsonl")):
        arm, _, stage = path.stem.rpartition("-stage-")
        # A crash can leave log entries beyond the last saved optimizer update.
        # The last entry for an update is the one replayed by a resumed run.
        updates = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A terminated writer may leave a partial final log line.
            updates[row["update"]] = row
        training_rows.extend(row | {"arm": arm, "stage": int(stage)} for row in updates.values())
    csv_rows(out / "training-curves.csv", training_rows)
    if training_rows:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
        for stage in range(1, 4):
            for arm in ["sequential", "orthogonal"]:
                rows = sorted(
                    [r for r in training_rows if r["arm"] == arm and r["stage"] == stage],
                    key=lambda r: r["update"],
                )
                for index, loss in enumerate(["forget_ce", "retain_ce"]):
                    axes[index, stage - 1].plot(
                        [r["update"] for r in rows], [r[loss] for r in rows], label=arm
                    )
                    axes[index, stage - 1].set_title(f"Request {stage}: {loss}")
                    axes[index, stage - 1].set_xlabel("Optimizer update")
                    axes[index, stage - 1].ticklabel_format(axis="y", style="plain", useOffset=False)
                    axes[index, stage - 1].legend(fontsize=8)
        figure_save(fig, out, "training-curves", synthetic)
    telemetry_rows = []
    for path in sorted((cfg.root / "telemetry").glob("*.json")):
        entry = read_json(path)
        metrics = entry["metrics"]
        telemetry_rows.append(
            {
                "phase": entry["phase"],
                "wall_seconds": entry["wall_seconds"],
                "job_id": entry["job"].get("SLURM_JOB_ID"),
                "level": entry["job"].get("O3_JOB_LEVEL"),
                "peak_allocated_gib": entry.get("peak_allocated_gib"),
                "peak_reserved_gib": entry.get("peak_reserved_gib"),
                "device_used_gib_max": metrics["device_used_gib"]["max"],
                "gpu_utilization_mean": metrics["gpu_utilization_percent"]["mean"],
                "exception": entry["exception"],
            }
        )
    csv_rows(out / "hardware-phases.csv", telemetry_rows)
    windows = []
    for path in sorted((cfg.root / "offline").glob("*.json")):
        windows.extend(read_json(path))
    summary = []
    keys = sorted({(r["arm"], r["stage"], r["mode"], r["role"], r["request"]) for r in windows})
    for arm, stage, mode, role, request in keys:
        rows = [
            r
            for r in windows
            if (r["arm"], r["stage"], r["mode"], r["role"], r["request"]) == (arm, stage, mode, role, request)
        ]
        summary.append(
            {
                "arm": arm,
                "stage": stage,
                "mode": mode,
                "role": role,
                "request": request,
                "windows": len(rows),
                "trajectories": len({(r["task_id"], r["trajectory"]) for r in rows}),
                **{
                    key: group_mean(rows, key)
                    for key in ["ce", "translation_error", "rotation_error", "gripper_error", "gate"]
                },
            }
        )
    csv_rows(out / "offline-summary.csv", summary)
    recovery = []
    for arm in ["sequential", "orthogonal"]:
        for request in range(1, 4):
            entries = [
                r
                for r in summary
                if r["arm"] == arm
                and r["mode"] == "always"
                and (r["request"] == request and r["stage"] >= request)
            ]
            anchor = next((r for r in entries if r["stage"] == request), None)
            if anchor is not None:
                recovery.extend(
                    {
                        "arm": arm,
                        "request": request,
                        "stage": row["stage"],
                        "ce_recovery": row["ce"] - anchor["ce"],
                        "translation_error_recovery": row["translation_error"] - anchor["translation_error"],
                    }
                    for row in entries
                )
    csv_rows(out / "earlier-request-recovery.csv", recovery)
    heat_values = [
        r["ce"]
        for r in summary
        if r["arm"] in {"sequential", "orthogonal"}
        and r["mode"] == "always"
        and r["request"] > 0
        and r["stage"] >= r["request"]
    ]
    if heat_values:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, arm in zip(axes, ["sequential", "orthogonal"]):
            matrix = np.full((3, 3), np.nan)
            for row in summary:
                if (
                    row["arm"] == arm
                    and row["mode"] == "always"
                    and row["request"] > 0
                    and (row["stage"] >= row["request"])
                ):
                    matrix[row["request"] - 1, row["stage"] - 1] = row["ce"]
            image = ax.imshow(
                np.ma.masked_invalid(matrix), cmap="viridis", vmin=min(heat_values), vmax=max(heat_values)
            )
            ax.set(
                xticks=range(3),
                xticklabels=[1, 2, 3],
                yticks=range(3),
                yticklabels=[1, 2, 3],
                xlabel="Completed request stage",
                ylabel="Audited request",
                title=arm,
            )
            fig.colorbar(image, ax=ax, label="Trajectory-mean hold action CE")
        figure_save(fig, out, "request-history", synthetic)
    episodes = [
        read_json(path)
        for path in sorted((cfg.root / "episodes").glob("*.json"))
        if ".failed-" not in path.name
    ]
    rollout_summary, paired = [], []
    for configuration in sorted({row["configuration"] for row in episodes}):
        all_rows = [r for r in episodes if r["configuration"] == configuration]
        valid = [r for r in all_rows if r["valid"]]
        requested = [r for r in valid if r["requested"]]
        allowed = [r for r in valid if not r["requested"]]
        rollout_summary.append(
            {
                "configuration": configuration,
                "valid_episodes": len(valid),
                "invalid_episodes": len(all_rows) - len(valid),
                "requested_successes": sum(r["success"] for r in requested),
                "requested_episodes": len(requested),
                "requested_completion_rate": mean([r["success"] for r in requested]),
                "requested_hold_compliance": mean([r["hold_compliant"] for r in requested]),
                "allowed_successes": sum(r["success"] for r in allowed),
                "allowed_episodes": len(allowed),
                "allowed_success_rate": mean([r["success"] for r in allowed]),
                "false_intervention_episodes": sum(r["false_intervention"] for r in allowed),
                "false_intervention_rate": mean([r["false_intervention"] for r in allowed]),
                "requested_gripper_displacement_mean": mean(
                    [r["max_gripper_displacement"] for r in requested]
                ),
                "mean_activation": mean([r["mean_activation"] for r in valid]),
                **{
                    f"{role}_{coordinate}_jump_mean": mean(
                        [r[f"mean_{coordinate}_command_jump"] for r in group]
                    )
                    for role, group in [("requested", requested), ("allowed", allowed)]
                    for coordinate in ["translation", "rotation"]
                },
                "mean_episode_wall_seconds": mean([r["wall_seconds"] for r in valid]),
                "episode_median_latency_mean": mean([r["median_latency_seconds"] for r in valid]),
                "episode_p95_latency_mean": mean([r["p95_latency_seconds"] for r in valid]),
            }
        )
    base = {
        (r["task_id"], r["trial"]): r for r in episodes if r["configuration"] == "base-base" and r["valid"]
    }
    for row in episodes:
        reference = base.get((row["task_id"], row["trial"]))
        if row["valid"] and reference is not None and row["configuration"] != "base-base":
            paired.append(
                {
                    "configuration": row["configuration"],
                    "task_id": row["task_id"],
                    "trial": row["trial"],
                    "requested": row["requested"],
                    "success_difference": int(row["success"]) - int(reference["success"]),
                    "position_displacement_difference_m": row["max_position_displacement_m"]
                    - (reference["max_position_displacement_m"]),
                    "gate_crossings": row["gate_crossings"],
                    "gripper_transitions": row["gripper_transitions"],
                    **{
                        f"{coordinate}_jump_difference": row[f"mean_{coordinate}_command_jump"]
                        - reference[f"mean_{coordinate}_command_jump"]
                        for coordinate in ["translation", "rotation"]
                    },
                }
            )
    csv_rows(out / "rollout-summary.csv", rollout_summary)
    csv_rows(out / "paired-episodes.csv", paired)
    if rollout_summary:
        fig, ax = plt.subplots(figsize=(8, 5))
        for index, row in enumerate(rollout_summary):
            if row["requested_completion_rate"] is not None and row["allowed_success_rate"] is not None:
                x, y = 1 - row["requested_completion_rate"], row["allowed_success_rate"]
                ax.scatter(
                    x, y, label=row["configuration"], marker=["o", "s", "^", "D", "x", "v", "P"][index % 7]
                )
        ax.legend(loc="lower left", fontsize=8)
        ax.set(
            xlabel="Requested task non-completion (not proof of deletion)",
            ylabel="Allowed task success",
            xlim=(-0.05, 1.1),
            ylim=(-0.05, 1.1),
            title="Suppression / utility trade-off",
        )
        figure_save(fig, out, "suppression-utility", synthetic)
    traces = sorted((cfg.root / "traces").glob("orthogonal-smoothed-*.csv"))
    if traces:
        with traces[0].open(encoding="utf-8") as stream:
            trace = list(csv.DictReader(stream))
        if trace:
            fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
            axes[0].plot([float(r["probability"]) for r in trace], label="Detector probability")
            axes[0].plot([float(r["gate"]) for r in trace], label="Adapter activation")
            axes[0].legend()
            axes[0].set_ylabel("Activation")
            axes[0].set_ylim(-0.05, 1.05)
            axes[1].plot(
                [float(r["translation_command_norm"]) for r in trace], label="Translation command norm"
            )
            axes[1].plot(
                [float(r["rotation_command_norm"]) for r in trace],
                label="Rotation command norm",
                linestyle="--",
            )
            axes[1].legend()
            axes[1].set_ylabel("Simulator command norm")
            axes[2].plot([float(r["position_displacement_m"]) for r in trace], label="Position displacement")
            axes[2].set_ylabel("Displacement (m)")
            axes[2].legend()
            axes[2].set_xlabel("Control step")
            figure_save(fig, out, "activation-action-trace", synthetic)
    detectors = (
        read_json(cfg.root / "detector-metrics.json") if (cfg.root / "detector-metrics.json").exists() else []
    )
    csv_rows(
        out / "detector-summary.csv",
        [{k: v for k, v in row.items() if k != "trajectory_mean_probabilities"} for row in detectors],
    )
    status = (
        read_json(cfg.root / "rollout-status.json")
        if (cfg.root / "rollout-status.json").exists()
        else {"complete": False, "reason": "paired closed-loop evaluation not completed"}
    )
    if status.get("complete"):
        expected = status.get("expected_episode_files", [])
        complete = (
            len(expected) == status["planned_episodes"]
            and bool(expected)
            and all(
                (cfg.root / "episodes" / name).exists()
                and read_json(cfg.root / "episodes" / name).get("valid")
                for name in expected
            )
        )
        status = status | {"complete": complete}
        if not complete:
            status["reason"] = "expected paired episode artifacts are missing or invalid"
    payload = {
        "synthetic": synthetic,
        "scope": "conditional behavior suppression",
        "rollouts": status,
        "offline_summary_rows": len(summary),
        "valid_episode_count": sum(r["valid"] for r in episodes),
        "invalid_episode_count": sum(not r["valid"] for r in episodes),
        "hypotheses": {
            "H1": "inspect request-history and matched current/retained errors; not automatically accepted",
            "H2": "requires hard/soft/smoothed paired rollouts; optional gates absent means untested",
            "H3": "compare smoothed adapter against detector-plus-hold at matched suppression",
        },
    }
    write_json(out / "summary.json", payload)
    title = "SYNTHETIC PLUMBING REPORT - NOT RESEARCH RESULTS" if synthetic else "O3-VLA pilot results"
    lines = [
        f"# {title}",
        "",
        "Conditional behavior suppression; no certified data deletion claim.",
        "",
        f"Paired rollout set complete: **{status.get('complete', False)}**.",
        f"Valid episodes: {payload['valid_episode_count']}; invalid episodes: {payload['invalid_episode_count']}.",
        "",
        "One training seed. Initial-state trials are not independent training seeds. No publication-ready significance claim.",
        "Offline action diagnostics use teacher forcing; robot success comes exclusively from simulator rollouts.",
        "",
        "| Configuration | Requested completion | Allowed success | Invalid episodes |",
        "|---|---|---|---|",
    ]
    for row in rollout_summary:
        lines.append(
            f"| {row['configuration']} | {row['requested_successes']}/{row['requested_episodes']} | "
            f"{row['allowed_successes']}/{row['allowed_episodes']} | {row['invalid_episodes']} |"
        )
    lines += [
        "",
        "Inspect CSVs for trajectory means, stage recovery, detector calibration, and paired episode differences.",
        "Figures are exported as PNG and PDF. Telemetry includes unsupported MIG metrics as null.",
        "Temporal smoothing alone is not proof of better control. Missing comparisons remain untested.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
