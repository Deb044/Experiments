"""Scientific invariants: isolation, matched arms, calibration and honest artifacts."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from o3_vla.actions import ActionCodec, hold_action
from o3_vla.backend import Synthetic
from o3_vla.cli import execute
from o3_vla.config import Config
from o3_vla.data import build_manifest, split_trajectories
from o3_vla.detector import calibrate_threshold
from o3_vla.gating import Gate
from o3_vla.io import Deadline, load_torch, read_json, seed_all
from o3_vla.lora import Adapters
from o3_vla.simulator import ToySimulator
from o3_vla.training import benchmark, pool, sample_batch, train


def small_config(tmp_path):
    return Config(
        backend="synthetic",
        output=str(tmp_path),
        rank=2,
        alpha=2,
        trajectories_per_task=15,
        windows_per_trajectory=2,
        eval_windows_per_trajectory=1,
        updates_per_stage=2,
        effective_batch=2,
        checkpoint_every=1,
        benchmark_warmup=1,
        benchmark_updates=2,
        rollout_trials=2,
        rollout_min_trials=2,
        settle_steps=1,
        video_episodes_per_configuration=0,
    ).validate()


def test_trajectory_splits_are_disjoint_and_seeded():
    names = [f"demo_{i}" for i in range(40)]
    split = split_trajectories(names, 7, 30)
    assert [len(split[k]) for k in ["train", "validation", "test"]] == [20, 5, 5]
    assert len({name for names in split.values() for name in names}) == 30
    assert split == split_trajectories(list(reversed(names)), 7, 30)
    with pytest.raises(ValueError):
        split_trajectories(names[:14], 7, 30)


def test_gate_zero_restores_base_and_only_adapter_trains():
    model = nn.Sequential(nn.Linear(4, 5))
    original = model[0]
    x = torch.randn(3, 4)
    baseline = original(x).detach()
    adapters = Adapters(model, 2, 2, {"0"})
    with torch.no_grad():
        model[0].B.fill_(0.3)
    adapters.gate(0)
    assert torch.equal(model(x), baseline)
    adapters.gate(1)
    assert not torch.equal(model(x), baseline)
    model(x).sum().backward()
    assert all(p.grad is None for p in original.parameters())
    assert all(p.grad is not None for p in adapters.parameters())


def test_orthogonality_uses_output_factor_and_detaches_previous():
    model = nn.Sequential(nn.Linear(4, 5))
    adapters = Adapters(model, 2, 2, {"0"})
    previous = adapters.state()
    previous["0"]["B"].fill_(1)
    previous["0"]["B"].requires_grad_(True)
    with torch.no_grad():
        model[0].B.fill_(0.3)
    loss = adapters.orthogonal_loss(previous)
    expected = (previous["0"]["B"].detach().T @ model[0].B).square().sum()
    assert torch.equal(loss, expected)
    loss.backward()
    assert model[0].B.grad is not None
    assert model[0].A.grad is None
    assert previous["0"]["B"].grad is None


def test_hold_targets_roundtrip_in_simulator_coordinates():
    codec = ActionCodec(32000, {"q01": [-0.8] * 7, "q99": [0.9] * 7, "mask": [True] * 6 + [False]})
    for aperture, expected in [(np.array([0.03, -0.03]), -1), (np.zeros(2), 1)]:
        hold = hold_action(aperture, 0.04)
        assert hold[6] == expected
        decoded = codec.decode(codec.encode(hold))
        assert decoded[6] == expected
        assert np.all(np.abs(decoded[:6]) <= codec.half_bin_tolerance[:6] * 2)


def test_threshold_ties_do_not_exceed_allowed_fpr():
    probabilities = np.array([0.9, 0.9, 0.2, 0.8, 1.0])
    labels = np.array([0, 0, 0, 1, 1])
    threshold = calibrate_threshold(probabilities, labels, 0.05)
    assert np.mean(probabilities[labels == 0] >= threshold) <= 0.05
    assert np.mean(probabilities[labels == 1] >= threshold) == 0.5


def test_smoothing_resets_between_episodes():
    gate = Gate("smoothed", smoothing=0.8)
    assert gate.step(1) == 1
    assert gate.step(0) == pytest.approx(0.8)
    gate.reset()
    assert gate.step(0) == 0


def test_current_request_pool_does_not_replay_old_requests(tmp_path):
    cfg = small_config(tmp_path)
    manifest = build_manifest(cfg, ToySimulator(cfg).tasks(), [])
    forget, retain = pool(manifest, 2)
    assert {r["stage"] for r in forget} == {2}
    batch = sample_batch(np.random.default_rng(0), forget, retain, 8)
    assert sum(r["role"] == "forget" for r in batch) == 4
    assert all(r["split"] == "train" for r in batch)


def test_matched_stage_one_and_resume(tmp_path):
    cfg = small_config(tmp_path)
    seed_all(cfg.seed)
    backend = Synthetic(cfg)
    manifest = build_manifest(cfg, ToySimulator(cfg).tasks(), [])
    initial = backend.adapters.state()
    budget = benchmark(cfg, backend, manifest, Deadline(1))
    for name in initial:
        assert torch.equal(initial[name]["B"], backend.adapters.state()[name]["B"])
    train(cfg, backend, manifest, budget, Deadline(1))
    states = [load_torch(cfg.root / "adapters" / arm / "stage-1.pt") for arm in ["sequential", "orthogonal"]]
    for name in initial:
        for factor in ["A", "B"]:
            assert torch.equal(states[0]["adapters"][name][factor], states[1]["adapters"][name][factor])
    final_path = cfg.root / "adapters/orthogonal/stage-3.pt"
    before = final_path.read_bytes()
    train(cfg, backend, manifest, budget, Deadline(1))
    assert final_path.read_bytes() == before


def test_complete_smoke_pipeline_and_identity_guard(tmp_path):
    cfg = small_config(tmp_path)
    execute(cfg, "run")
    assert read_json(cfg.root / "status.json")["complete"]
    summary = read_json(cfg.root / "report/summary.json")
    assert summary["synthetic"] and summary["rollouts"]["complete"]
    assert summary["valid_episode_count"] == 60
    assert len(list((cfg.root / "adapters").glob("*/*.pt"))) == 6
    assert len(list((cfg.root / "report").glob("*.png"))) >= 3
    assert "NOT RESEARCH RESULTS" in (cfg.root / "report/results.md").read_text()
    execute(cfg, "run")
    assert read_json(cfg.root / "report/summary.json")["valid_episode_count"] == 60
    # A stale status file must not certify a missing paired episode.
    from o3_vla.report import generate

    next((cfg.root / "episodes").glob("*.json")).unlink()
    assert not generate(cfg)["rollouts"]["complete"]
    with pytest.raises(ValueError, match="configuration changed"):
        execute(replace(cfg, seed=8), "run")


def test_partial_optimizer_resume_matches_uninterrupted_training(tmp_path):
    cfg = small_config(tmp_path / "reference")
    seed_all(cfg.seed)
    backend = Synthetic(cfg)
    manifest = build_manifest(cfg, ToySimulator(cfg).tasks(), [])
    budget = {"selected_updates_per_stage": 2}
    train(cfg, backend, manifest, budget, Deadline(1))
    expected = load_torch(cfg.root / "adapters/orthogonal/stage-3.pt")["adapters"]

    resumed = replace(cfg, output=str(tmp_path / "resumed"))
    seed_all(cfg.seed)
    backend = Synthetic(resumed)
    manifest = build_manifest(resumed, ToySimulator(resumed).tasks(), [])

    class StopAfterOneUpdate:
        checks = 0

        def check(self):
            self.checks += 1
            if self.checks > 2:
                raise TimeoutError("simulated job cutoff")

    with pytest.raises(TimeoutError):
        train(resumed, backend, manifest, budget, StopAfterOneUpdate())
    state = load_torch(resumed.root / "adapters/sequential/stage-1.pt")
    assert state["completed_updates"] == 1 and not state["complete"]
    train(resumed, backend, manifest, budget, Deadline(1))
    actual = load_torch(resumed.root / "adapters/orthogonal/stage-3.pt")["adapters"]
    for name in expected:
        for factor in ["A", "B"]:
            assert torch.equal(expected[name][factor], actual[name][factor])


def test_native_setup_failure_produces_incomplete_engineering_report(tmp_path, monkeypatch):
    cfg = replace(small_config(tmp_path), backend="openvla")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        execute(cfg, "run")
    status = read_json(cfg.root / "status.json")
    assert not status["complete"] and status["failed_phase"] == "model-load"
    summary = read_json(cfg.root / "report/summary.json")
    assert not summary["synthetic"] and not summary["rollouts"]["complete"]
