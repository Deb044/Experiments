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
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with GpuTelemetry(tmp_path, device="cpu") as failed:
            raise torch.cuda.OutOfMemoryError("fixture OOM")
    assert failed.summary["oom"] is True


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


LINUX_LAUNCHER = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="Linux Slurm launcher")


def simulated_slurm(tmp_path, sbatch='printf "%s\\n" "$@"'):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {"srun": 'exec "$@"', "scontrol": 'echo "fixture allocation"',
                       "sacct": 'echo "fixture accounting"', "nvidia-smi": 'echo "fixture inventory"',
                       "sbatch": sbatch}.items():
        script = binaries / name
        script.write_text("#!/bin/bash\n" + body + "\n")
        script.chmod(0o755)
    cfg = read_json(ROOT / "configs/smoke.json")
    cfg["benchmark"].update(warmup=1, updates=2)
    source = tmp_path / "source.json"
    write_json(source, cfg)
    return dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                VLA_REPO_ROOT=str(ROOT), VLA_LOG_ROOT=str(tmp_path / "logs"),
                VLA_SCRATCH_ROOT=str(tmp_path / "scratch"), VLA_EXPERIMENT="test-medium",
                VLA_PYTHON=sys.executable, VLA_CONFIG=str(source), VLA_BENCHMARK_REPEATS="1",
                SLURM_JOB_ID="123", SLURM_JOB_PARTITION="gpu_medium", VLA_JOB_LEVEL="medium")


@LINUX_LAUNCHER
def test_bash_launchers_and_cpu_job_integration(tmp_path):
    for script in (ROOT / "cluster").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    env = simulated_slurm(tmp_path)
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


@LINUX_LAUNCHER
def test_training_jobs_for_separate_tasks_share_an_experiment(tmp_path):
    env = simulated_slurm(tmp_path)
    output = tmp_path / "scratch/experiments/test-medium"

    def job(*command):
        return subprocess.run(["bash", str(ROOT / "cluster/job_gpu_medium.sh"), *command], env=env, text=True, capture_output=True)

    # A slice of the experiment cannot be trained before the shared manifest exists.
    early = job("train", "--task", "goal-0", "--max-updates", "1")
    assert early.returncode == 2 and "Prepare the experiment once" in early.stderr
    assert job("prepare").returncode == 0
    # Stand in for a running goal-0 training job: experiment lock shared, task lock exclusive.
    holder = subprocess.Popen(["bash", "-c", 'exec 9> "$0/.job.lock"; flock -s 9; exec 8> "$0/.job-goal-0.lock"; flock 8; '
                               'echo held; exec sleep 120', str(output)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert job("train", "--task", "object-0", "--max-updates", "1").returncode == 0
        same_task = job("train", "--task", "goal-0", "--max-updates", "1")
        assert same_task.returncode == 2 and "already has a running training job" in same_task.stderr
        one_seed = job("train", "--task=goal-0", "--seed=0", "--max-updates", "1")
        assert one_seed.returncode == 2 and "covering every seed" in one_seed.stderr
        whole = job("benchmark")
        assert whole.returncode == 2 and "already has a running job" in whole.stderr
    finally:
        holder.kill()
        holder.wait()
    assert job("train", "--task", "goal-0", "--seed", "0", "--max-updates", "1").returncode == 0
    started = {path.relative_to(output / "runs").parts[:2] for path in (output / "runs").rglob("resume.pt")}
    assert ("goal-0", "seed-0") in started and ("goal-0", "seed-1") not in started
    assert {parts[0] for parts in started} == {"goal-0", "object-0"}


@LINUX_LAUNCHER
@pytest.mark.parametrize("split, training_jobs", [("task", 3), ("seed", 6)])
def test_parallel_submission_chains_prepare_training_and_the_remaining_stages(tmp_path, split, training_jobs):
    record = tmp_path / "submissions.txt"
    env = simulated_slurm(tmp_path, sbatch=f'printf "%s " "$@" >> "{record}"; echo >> "{record}"; '
                                           f'echo "Submitted batch job $(wc -l < "{record}")"')
    subprocess.run(["bash", str(ROOT / "cluster/submit_parallel.sh"), "small", split], env=env, check=True, capture_output=True)
    submissions = record.read_text().splitlines()
    assert len(submissions) == training_jobs + 2
    assert submissions[0].rstrip().endswith("job_gpu_small.sh prepare") and "--dependency" not in submissions[0]
    for line in submissions[1:-1]:
        assert "--dependency=afterok:1 " in line and "job_gpu_small.sh train --task " in line
        assert ("--seed" in line) == (split == "seed")
    assert {line.split("train ")[1].strip() for line in submissions[1:-1]} == (
        {f"--task {task}" for task in ("goal-0", "object-0", "spatial-0")} if split == "task" else
        {f"--task {task} --seed {seed}" for task in ("goal-0", "object-0", "spatial-0") for seed in (0, 1)})
    waits = ":".join(str(number) for number in range(2, training_jobs + 2))
    assert f"--dependency=afterany:{waits} " in submissions[-1] and submissions[-1].rstrip().endswith("job_gpu_small.sh run")
