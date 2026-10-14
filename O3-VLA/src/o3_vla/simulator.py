"""LIBERO setup, official initial states, bounded episodes and synthetic fixtures."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .actions import hold_action
from .io import git_revision, hash_file
from .render import robosuite_import


def quaternion_distance(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))


class Libero:
    def __init__(self, cfg):
        import yaml

        if not cfg.libero_root or not cfg.data_root:
            raise ValueError("set --libero-root and --data-root; see README.md")
        self.cfg = cfg
        root = Path(cfg.libero_root).expanduser().resolve()
        if git_revision(root) != cfg.libero_revision:
            raise ValueError("LIBERO source revision does not match configuration")
        config_dir = cfg.root / "libero-config"
        config_dir.mkdir(parents=True, exist_ok=True)
        benchmark_root = root / "libero" / "libero"
        paths = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "assets": str(benchmark_root / "assets"),
            "datasets": str(Path(cfg.data_root).expanduser().resolve()),
        }
        (config_dir / "config.yaml").write_text(yaml.safe_dump(paths), encoding="utf-8")
        # Must be set before first LIBERO import, avoiding interactive import-time prompts.
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from libero.libero import benchmark

        self.suite = benchmark.get_benchmark_dict()[cfg.suite](task_order_index=cfg.task_order_index)
        self.benchmark_root = benchmark_root

    def tasks(self):
        result = []
        for index in range(self.suite.n_tasks):
            task = self.suite.get_task(index)
            initial_path = self.benchmark_root / "init_files" / task.problem_folder / task.init_states_file
            result.append(
                {
                    "id": index,
                    "name": task.name,
                    "instruction": task.language,
                    "file": str(Path(self.cfg.data_root) / self.suite.get_task_demonstration(index)),
                    "bddl": str(self.suite.get_task_bddl_file_path(index)),
                    "initial_states_file": str(initial_path),
                    "initial_states_sha256": hash_file(initial_path),
                }
            )
        return result

    def environment(self, task, trial):
        with robosuite_import():
            from libero.libero.envs import OffScreenRenderEnv

        path = Path(task["initial_states_file"])
        if hash_file(path) != task["initial_states_sha256"]:
            raise ValueError("initial-state data changed")
        states = torch.load(path, map_location="cpu", weights_only=False)
        if trial >= len(states):
            raise ValueError("requested initial-state index exceeds suite states")
        env = OffScreenRenderEnv(bddl_file_name=task["bddl"], camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        obs = env.set_init_state(states[trial])
        return env, obs

    @property
    def horizon(self):
        return 280


class ToyEnvironment:
    def __init__(self):
        self.position = np.zeros(3)
        self.steps = 0

    def observation(self):
        return {
            "agentview_image": np.full((16, 16, 3), 100, dtype=np.uint8),
            "robot0_eef_pos": self.position.copy(),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([0.03, -0.03]),
        }

    def step(self, action):
        self.position += np.asarray(action)[:3] * 0.1
        self.steps += 1
        done = bool(self.position[0] > 0.07)
        return self.observation(), float(done), done, {}

    def close(self):
        pass


class ToySimulator:
    def __init__(self, cfg):
        self.cfg = cfg

    def tasks(self):
        return [
            {"id": index, "name": f"synthetic_{index}", "instruction": f"synthetic task {index}"}
            for index in range(6)
        ]

    def environment(self, task, trial):
        env = ToyEnvironment()
        return env, env.observation()

    @property
    def horizon(self):
        return 8


def make_simulator(cfg):
    return ToySimulator(cfg) if cfg.backend == "synthetic" else Libero(cfg)


def observation_window(obs, instruction):
    return {
        "image": Image.fromarray(obs["agentview_image"][::-1, ::-1].copy()),
        "gripper_qpos": obs["robot0_gripper_qpos"],
        "instruction": instruction,
    }


def hold_sanity(cfg, simulator, task, codec=None):
    env, obs = simulator.environment(task, 0)
    try:
        for _ in range(cfg.settle_steps):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        position, orientation = obs["robot0_eef_pos"].copy(), obs["robot0_eef_quat"].copy()
        gripper = obs["robot0_gripper_qpos"].copy()
        distance, angle, gripper_drift = 0.0, 0.0, 0.0
        for _ in range(10):
            action = hold_action(obs["robot0_gripper_qpos"], cfg.gripper_open_threshold)
            if codec is not None:
                action = codec.decode(codec.encode(action))
            obs, _, _, _ = env.step(action)
            distance = max(distance, float(np.linalg.norm(obs["robot0_eef_pos"] - position)))
            angle = max(angle, quaternion_distance(orientation, obs["robot0_eef_quat"]))
            gripper_drift = max(gripper_drift, float(np.linalg.norm(obs["robot0_gripper_qpos"] - gripper)))
        result = {
            "position_drift_m": distance,
            "rotation_drift_rad": angle,
            "gripper_drift": gripper_drift,
            "quantized_hold": codec is not None,
            "passed": distance <= cfg.hold_position_tolerance
            and angle <= cfg.hold_rotation_tolerance
            and gripper_drift <= cfg.hold_gripper_tolerance,
        }
        if not result["passed"]:
            raise ValueError(f"hold controller sanity check failed: {result}")
        return result
    finally:
        env.close()
