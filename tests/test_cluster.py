"""Cluster-specific scientific safeguards and an executable Linux launcher check."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vla_weightspace.cluster import compare, scratch_config
from vla_weightspace.io import read_json, write_json
from vla_weightspace.telemetry import GpuTelemetry, statistics


ROOT = Path(__file__).parents[1]


def test_cpu_telemetry_missing_values_and_exception_preservation(tmp_path):
    with pytest.raises(ValueError, match="original failure"):
        with GpuTelemetry(tmp_path, device="cpu", interval=0.01) as monitor:
            raise ValueError("original failure")
    summary = read_json(monitor.summary_path)
    assert summary["exit"] == "error"
    assert summary["exception_type"] == "ValueError"
    assert summary["metrics"]["utilization_gpu_percent"]["mean"] is None
    assert summary["metrics"]["device_used_gb"]["samples"] == 0
    assert len(monitor.samples_path.read_text().splitlines()) >= 2
    # Repeated commands keep both sets of samples rather than overwriting them.
    with GpuTelemetry(tmp_path, device="cpu"):
        pass
    assert len(list(tmp_path.glob("*-summary.json"))) == 2


class Unsupported(Exception):
    pass


def fake_nvml(mig=True):
    def utilization(handle):
        if mig:
            raise Unsupported("MIG utilization unavailable")
        return SimpleNamespace(gpu=73, memory=40)
    return SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                           nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
                           nvmlDeviceGetUUID=lambda handle: handle,
                           nvmlDeviceGetMigMode=lambda handle: (1, 1), NVMLError_NotSupported=Unsupported,
                           nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=12 * 2**30, total=24 * 2**30),
                           nvmlDeviceGetUtilizationRates=utilization)


def test_mig_samples_correct_uuid_and_no_fake_utilization(tmp_path):
    nvml = fake_nvml()
    props = SimpleNamespace(name="MIG slice", total_memory=24 * 2**30, uuid="GPU-parent")
    with patch.dict(sys.modules, pynvml=nvml), patch.dict(os.environ, CUDA_VISIBLE_DEVICES="MIG-slice"), \
         patch.object(torch.cuda, "is_initialized", return_value=True), \
         patch.object(torch.cuda, "get_device_properties", return_value=props), \
         patch.object(torch.cuda, "mem_get_info", return_value=(12 * 2**30, 24 * 2**30)), \
         patch.object(torch.cuda, "memory_allocated", return_value=8 * 2**30), \
         patch.object(torch.cuda, "memory_reserved", return_value=10 * 2**30), \
         patch.object(torch.cuda, "max_memory_allocated", return_value=9 * 2**30), \
         patch.object(torch.cuda, "max_memory_reserved", return_value=11 * 2**30):
        with GpuTelemetry(tmp_path, device="cuda:0") as monitor:
            pass
    assert monitor.summary["device_uuid"] == "MIG-slice"
    assert monitor.summary["metrics"]["device_used_gb"]["max"] == 12
    assert monitor.summary["metrics"]["torch_peak_reserved_gb"]["max"] == 11
    assert monitor.summary["metrics"]["utilization_gpu_percent"]["mean"] is None
    assert "MIG utilization unavailable" in " ".join(monitor.summary["errors"])


def test_refuse_mig_parent_and_numeric_index_fallback():
    with patch.dict(sys.modules, pynvml=fake_nvml()), patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2"):
        monitor = GpuTelemetry(None, device="cuda:0")
        monitor._connect_nvml(SimpleNamespace(uuid="GPU-parent"))
        assert monitor.handle is None
        assert "MIG parent" in " ".join(monitor.errors)
        missing = GpuTelemetry(None, device="cuda:0")
        missing._connect_nvml(SimpleNamespace())
        assert missing.handle is None
        assert "physical-index fallback" in " ".join(missing.errors)


def test_scratch_config_is_immutable(tmp_path):
    destination = tmp_path / "config.json"
    output = tmp_path / "artifacts"
    source = ROOT / "configs/smoke.json"
    scratch_config(source, destination, output, microbatch=1)
    scratch_config(source, destination, output, microbatch=1)
    assert read_json(destination)["output"] == str(output.resolve())
    with pytest.raises(ValueError, match="Existing scratch config"):
        scratch_config(source, destination, output, microbatch=2)


def measured_benchmark():
    return {"synthetic": False, "workload_key": "matching-protocol", "microbatch": 2,
            "job": {"VLA_JOB_LEVEL": "small", "SLURM_JOB_ID": "123"},
            "hardware": {"total_gb": 24}, "timing": {"mean_seconds": 1, "p95_seconds": 1.1},
            "samples_per_second": 16, "training_hours_estimate": 13.3,
            "gpu_telemetry": {"metrics": {"torch_peak_reserved_gb": statistics([18]),
                                           "device_used_gb": statistics([20]),
                                           "utilization_gpu_percent": statistics([None])}}}


def test_selection_uses_device_peak_and_rejects_mixed_protocols(tmp_path):
    record = measured_benchmark()
    path = tmp_path / "benchmark.json"
    write_json(path, record)
    row = compare([path])[0]
    # Allocator peak alone fits a 20% margin; sampled total occupancy does not.
    assert row["vram_headroom_pass"] is False
    assert row["mean_gpu_util_percent"] is None
    other = copy.deepcopy(record)
    other["workload_key"] = "different-protocol"
    second = tmp_path / "other.json"
    write_json(second, other)
    with pytest.raises(ValueError, match="Workloads differ"):
        compare([path, second])
    record["synthetic"] = True
    write_json(path, record)
    with pytest.raises(ValueError, match="Synthetic"):
        compare([path])


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="Linux Slurm launcher")
def test_bash_launchers_and_cpu_job_integration(tmp_path):
    for script in (ROOT / "cluster").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {"srun": 'exec "$@"', "scontrol": 'echo "fixture allocation"',
                       "sacct": 'echo "fixture accounting"', "nvidia-smi": 'echo "fixture inventory"',
                       "sbatch": 'printf "%s\\n" "$@"'}.items():
        script = binaries / name
        script.write_text("#!/bin/bash\n" + body + "\n")
        script.chmod(0o755)
    cfg = read_json(ROOT / "configs/smoke.json")
    cfg["benchmark"].update(warmup=1, updates=2)
    source = tmp_path / "source.json"
    write_json(source, cfg)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               VLA_REPO_ROOT=str(ROOT), VLA_LOG_ROOT=str(tmp_path / "logs"),
               VLA_SCRATCH_ROOT=str(tmp_path / "scratch"), VLA_EXPERIMENT="test-medium",
               VLA_PYTHON=sys.executable, VLA_CONFIG=str(source), VLA_BENCHMARK_REPEATS="1",
               SLURM_JOB_ID="123", SLURM_JOB_PARTITION="gpu_medium", VLA_JOB_LEVEL="medium")
    submission = subprocess.run(["bash", str(ROOT / "cluster/submit.sh"), "medium", "train", "--max-updates", "1"],
                                env=env, check=True, text=True, capture_output=True)
    assert (tmp_path / "logs").is_dir()
    assert "--error=" in submission.stdout and "--output=" in submission.stdout
    assert "--max-updates\n1" in submission.stdout
    subprocess.run(["bash", str(ROOT / "cluster/job_gpu_medium.sh"), "benchmark"], env=env, check=True)
    output = tmp_path / "scratch/experiments/test-medium"
    assert read_json(output / "benchmark.json")["synthetic"] is True
    logs = tmp_path / "logs/jobs/123-0"
    assert (logs / "benchmark-1.json").exists()
    assert (logs / "exit-code.txt").read_text().strip() == "0"
    assert list((logs / "command-1").glob("*-summary.json"))
    # CLI errors must propagate to Slurm; diagnostics still get written.
    failure = subprocess.run(["bash", str(ROOT / "cluster/job_gpu_medium.sh"), "invalid-stage"], env=env)
    assert failure.returncode != 0
    assert int((logs / "exit-code.txt").read_text()) != 0
