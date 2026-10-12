"""Explicit stages and one-day orchestration; independent from VLA_WS."""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .config import load
from .io import Deadline, bind_run, read_json, seed_all, write_json


@contextmanager
def exclusive_run(root):
    stream = (root / ".writer.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def doctor(cfg):
    import json

    from .telemetry import hardware

    payload = hardware() | {
        "python": sys.executable,
        "source": str(Path(__file__).resolve()),
        "backend": cfg.backend,
        "output": str(cfg.root),
        "data_root": cfg.data_root,
        "libero_root": cfg.libero_root,
        "model_revision": cfg.model_revision,
        "code_revision": cfg.code_revision,
    }
    print(json.dumps(payload, indent=2), flush=True)
    write_json(cfg.root / "doctor.json", payload)


def execute(cfg, command):
    from .backend import make_backend
    from .data import load_manifest
    from .detector import cache_features, fit_detectors
    from .evaluation import offline, prepare, rollouts
    from .report import generate
    from .simulator import make_simulator
    from .telemetry import Telemetry
    from .training import benchmark, train

    bind_run(cfg)
    with exclusive_run(cfg.root):
        if command == "doctor":
            doctor(cfg)
            return
        if command == "report":
            generate(cfg)
            return
        seed_all(cfg.seed)
        if cfg.backend == "synthetic":
            import torch

            torch.set_num_threads(1)
        started = time.monotonic()
        global_deadline = Deadline(cfg.total_budget_hours)
        setup_deadline = Deadline(min(3, global_deadline.remaining / 3600))
        # Lazy load once per invocation. All commands use exactly the same model/config.
        backend = None
        simulator = None

        def invoke(phase):
            nonlocal simulator
            global_deadline.check()
            with Telemetry(cfg, phase):
                if phase == "prepare":
                    simulator = make_simulator(cfg)
                    return prepare(cfg, backend, simulator, setup_deadline)
                manifest = load_manifest(cfg, verify_files=phase in {"benchmark", "train", "evaluate"})
                if phase == "benchmark":
                    return benchmark(cfg, backend, manifest, setup_deadline)
                if phase == "features":
                    cache_features(cfg, backend, manifest, Deadline(min(2, global_deadline.remaining / 3600)))
                    fit_detectors(cfg, manifest)
                elif phase == "train":
                    budget = read_json(cfg.root / "benchmark.json")
                    train(
                        cfg,
                        backend,
                        manifest,
                        budget,
                        Deadline(min(cfg.train_budget_hours, global_deadline.remaining / 3600)),
                    )
                elif phase == "evaluate":
                    offline(cfg, backend, manifest, Deadline(min(2, global_deadline.remaining / 3600)))
                elif phase == "rollout":
                    if simulator is None:
                        simulator = make_simulator(cfg)
                    rollouts(
                        cfg,
                        backend,
                        simulator,
                        manifest,
                        Deadline(min(cfg.rollout_budget_hours, global_deadline.remaining / 3600)),
                    )
                else:
                    raise ValueError(phase)

        phases = (
            ["prepare", "benchmark", "features", "train", "evaluate", "rollout"]
            if command == "run"
            else [command]
        )
        status = {
            "command": command,
            "complete": False,
            "phases": [],
            "synthetic": cfg.backend == "synthetic",
        }
        phase = "model-load"
        try:
            with Telemetry(cfg, phase):
                backend = make_backend(cfg)
            for phase in phases:
                print(f"Starting {phase}", flush=True)
                then = time.monotonic()
                invoke(phase)
                status["phases"].append(
                    {"phase": phase, "status": "complete", "wall_seconds": time.monotonic() - then}
                )
                write_json(cfg.root / "status.json", status)
            status["complete"] = True
        except Exception as error:
            status.update(error=f"{type(error).__name__}: {error}", failed_phase=phase)
            raise
        finally:
            status["wall_seconds"] = time.monotonic() - started
            write_json(cfg.root / "status.json", status)
            # Partial results and missing rollout status are explicitly exposed, never relabeled success.
            try:
                generate(cfg)
            except Exception as report_error:
                status["report_error"] = f"{type(report_error).__name__}: {report_error}"
                if status["complete"]:
                    status.update(complete=False, failed_phase="report")
                    write_json(cfg.root / "status.json", status)
                    raise
                write_json(cfg.root / "status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--output")
    parser.add_argument("--data-root")
    parser.add_argument("--libero-root")
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "prepare",
            "benchmark",
            "features",
            "train",
            "evaluate",
            "rollout",
            "report",
            "run",
        ],
    )
    args = parser.parse_args(argv)
    cfg = load(args.config, output=args.output, data_root=args.data_root, libero_root=args.libero_root)
    execute(cfg, args.command)


if __name__ == "__main__":
    main()
