"""Small stage-oriented interface; every scientific artifact carries provenance."""
import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config
from .io import digest, read_json


def load_manifest(cfg):
    path = Path(cfg["output"]) / "manifest.json"
    manifest = read_json(path)
    if manifest["config"] != cfg:
        raise ValueError("Configuration differs from prepared manifest; restore it or choose a new output")
    if digest({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("Manifest integrity check failed")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test predictability of deletion-induced VLA weight changes")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--telemetry-dir", default=os.environ.get("VLA_TELEMETRY_DIR"),
                        help="Directory for live GPU JSONL samples and per-command summaries")
    parser.add_argument("--telemetry-interval", type=float, default=2.0)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "benchmark", "extract", "predict", "evaluate", "report", "run", "doctor"):
        sub.add_parser(command)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--task")
    train_parser.add_argument("--run-id")
    train_parser.add_argument("--max-updates", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.telemetry_interval <= 0:
        parser.error("--telemetry-interval must be positive")
    from .telemetry import GpuTelemetry
    with GpuTelemetry(args.telemetry_dir, cfg["training"]["device"],
                      args.telemetry_interval, scope=args.command):
        return execute(args, cfg, parser)


def execute(args, cfg, parser):
    if args.command == "doctor":
        import importlib.util
        import platform
        import torch
        print(json.dumps({"python": platform.python_version(), "python_executable": sys.executable,
                          "environment_prefix": sys.prefix, "code_path": str(Path(__file__).resolve()),
                          "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                          "lerobot_installed": importlib.util.find_spec("lerobot") is not None, "backend": cfg["backend"]}, indent=2))
        return
    if args.command in {"prepare", "run"}:
        from .prepare import prepare
        manifest = prepare(cfg)
    else:
        manifest = load_manifest(cfg)
    if args.command == "prepare":
        print(f"Prepared {len(manifest['tasks'])} tasks: {cfg['output']}/manifest.json")
        return
    from .benchmark import benchmark
    from .training import train
    from .extraction import extract
    from .prediction import predict
    from .evaluation import evaluate
    from .report import report
    if args.command == "run":
        benchmark(manifest)
        train(manifest)
        extract(manifest)
        predict(manifest)
        evaluate(manifest)
        print(f"Report: {report(manifest)}")
    elif args.command == "train":
        if args.max_updates is not None and args.max_updates < 1:
            parser.error("--max-updates must be positive")
        train(manifest, args.task, args.run_id, args.max_updates)
    else:
        function = {"benchmark": benchmark, "extract": extract, "predict": predict, "evaluate": evaluate, "report": report}[args.command]
        result = function(manifest)
        print(f"{args.command}: {result}" if args.command == "report" else f"{args.command}: complete ({cfg['output']})")


if __name__ == "__main__":
    main()
