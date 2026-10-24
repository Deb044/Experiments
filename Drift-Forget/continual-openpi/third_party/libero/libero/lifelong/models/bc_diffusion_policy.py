"""
Diffusion Policy for continual learning.
Ported from robomimic with language conditioning from libero.
"""
from collections import deque
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as parse_version

# Requires diffusers>=0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import robomimic.utils.tensor_utils as TensorUtils

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.diffusion_policy_nets import ConditionalUnet1D
from libero.lifelong.models.modules.rgb_modules import SpatialSoftmax
from libero.lifelong.models.modules.language_modules import MLPEncoder


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with the output of func.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int = 16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    This is critical for EMA to work properly.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, 
            num_channels=x.num_features)
    )
    return root_module


class DiffusionImageEncoder(nn.Module):
    """
    ResNet18-based image encoder matching robomimic's configuration.
    Uses SpatialSoftmax pooling with 32 keypoints.
    """
    
    def __init__(self, input_shape, output_size=64, num_kp=32):
        super().__init__()
        import torchvision
        
        C, H, W = input_shape
        
        # Use ResNet18 backbone (matching robomimic's ResNet18Conv)
        resnet = torchvision.models.resnet18(pretrained=False)
        # Remove avgpool and fc layers
        layers = list(resnet.children())[:-2]
        
        # Modify first conv if needed for different input channels
        if C != 3:
            layers[0] = nn.Conv2d(
                C, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        self.backbone = nn.Sequential(*layers)
        
        # Compute backbone output shape
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            out = self.backbone(dummy)
            _, out_c, out_h, out_w = out.shape
        
        # SpatialSoftmax pooling
        self.spatial_softmax = SpatialSoftmax(out_c, out_h, out_w, num_kp=num_kp)
        
        # Output dimension is num_kp * 2 (x, y coordinates for each keypoint)
        self.output_dim = num_kp * 2
        
        # Project to desired output size
        self.fc = nn.Linear(self.output_dim, output_size)
        self.output_size = output_size
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            features: (B, output_size) feature vector
        """
        h = self.backbone(x)
        h = self.spatial_softmax(h)
        h = self.fc(h)
        return h


class BCDiffusionPolicy(BasePolicy):
    """
    Diffusion Policy for behavior cloning with language conditioning.
    
    Ported from robomimic's DiffusionPolicyUNet with libero's language embeddings.
    """
    
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        
        policy_cfg = cfg.policy
        
        # Horizon parameters (from robomimic config)
        self.observation_horizon = policy_cfg.get("observation_horizon", 2)
        self.action_horizon = policy_cfg.get("action_horizon", 8)
        self.prediction_horizon = policy_cfg.get("prediction_horizon", 16)
        
        # Image encoders
        embed_size = policy_cfg.get("embed_size", 64)
        self.image_encoders = nn.ModuleDict()
        
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                input_shape = shape_meta["all_shapes"][name]
                encoder = DiffusionImageEncoder(
                    input_shape=input_shape,
                    output_size=embed_size,
                    num_kp=policy_cfg.get("num_kp", 32)
                )
                # Replace BatchNorm with GroupNorm for EMA compatibility
                encoder = replace_bn_with_gn(encoder)
                self.image_encoders[name] = encoder
        
        # Proprioceptive encoder (joint states, gripper states)
        self.proprio_dim = 0
        proprio_keys = []
        if cfg.data.use_joint:
            self.proprio_dim += 7  # joint_states
            proprio_keys.append("joint_states")
        if cfg.data.use_gripper:
            self.proprio_dim += 2  # gripper_states
            proprio_keys.append("gripper_states")
        if cfg.data.use_ee:
            self.proprio_dim += 3  # ee_states
            proprio_keys.append("ee_states")
        self.proprio_keys = proprio_keys
        
        if self.proprio_dim > 0:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size)
            )
        else:
            self.proprio_encoder = None
            
        # Language encoder (from libero)
        lang_cfg = policy_cfg.language_encoder.network_kwargs
        self.language_encoder = MLPEncoder(
            input_size=lang_cfg.input_size,
            hidden_size=lang_cfg.hidden_size,
            output_size=embed_size,
            num_layers=lang_cfg.num_layers
        )
        
        # Calculate global conditioning dimension
        num_images = len(self.image_encoders)
        obs_per_step = num_images * embed_size
        if self.proprio_dim > 0:
            obs_per_step += embed_size
        # Language is added once (not per timestep)
        lang_dim = embed_size
        
        global_cond_dim = obs_per_step * self.observation_horizon + lang_dim
        
        # UNet for noise prediction
        ac_dim = shape_meta["ac_dim"]
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=ac_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=policy_cfg.get("diffusion_step_embed_dim", 256),
            down_dims=policy_cfg.get("down_dims", [256, 512, 1024]),
            kernel_size=policy_cfg.get("kernel_size", 5),
            n_groups=policy_cfg.get("n_groups", 8)
        )
        
        # Noise scheduler (DDPM by default)
        ddpm_cfg = policy_cfg.get("ddpm", {})
        if ddpm_cfg.get("enabled", True):
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=ddpm_cfg.get("num_train_timesteps", 100),
                beta_schedule=ddpm_cfg.get("beta_schedule", "squaredcos_cap_v2"),
                clip_sample=ddpm_cfg.get("clip_sample", True),
                prediction_type=ddpm_cfg.get("prediction_type", "epsilon")
            )
            self.num_inference_timesteps = ddpm_cfg.get("num_inference_timesteps", 100)
        else:
            ddim_cfg = policy_cfg.get("ddim", {})
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=ddim_cfg.get("num_train_timesteps", 100),
                beta_schedule=ddim_cfg.get("beta_schedule", "squaredcos_cap_v2"),
                clip_sample=ddim_cfg.get("clip_sample", True),
                set_alpha_to_one=ddim_cfg.get("set_alpha_to_one", True),
                steps_offset=ddim_cfg.get("steps_offset", 0),
                prediction_type=ddim_cfg.get("prediction_type", "epsilon")
            )
            self.num_inference_timesteps = ddim_cfg.get("num_inference_timesteps", 10)
        
        self.ema = None
            
        # Action dimension
        self.ac_dim = ac_dim
        
        # Inference state
        self.obs_queue = None
        self.action_queue = None
        
    def _get_img_tuple(self, data):
        """Override to use our image encoder keys."""
        img_tuple = tuple(
            [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        """Override to use our image encoder keys."""
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict
        
    def encode_obs(self, data):
        """
        Encode observations into a global conditioning vector.
        
        Args:
            data: dict with 'obs' containing image and proprio observations,
                  and 'task_emb' for language embedding
        Returns:
            global_cond: (B, global_cond_dim) conditioning vector
        """
        obs = data["obs"]
        B = None
        T = None
        
        features = []
        
        # Encode images
        for img_name, encoder in self.image_encoders.items():
            img = obs[img_name]  # (B, T, C, H, W)
            if B is None:
                B, T = img.shape[:2]
            # Only use observation_horizon timesteps
            img = img[:, :self.observation_horizon]
            # Reshape for encoding
            img_flat = img.reshape(B * self.observation_horizon, *img.shape[2:])
            feat = encoder(img_flat)  # (B*T, embed_size)
            feat = feat.reshape(B, self.observation_horizon, -1)  # (B, T, embed_size)
            features.append(feat)
        
        # Encode proprioceptive state
        if self.proprio_encoder is not None:
            proprio_list = []
            for key in self.proprio_keys:
                proprio_list.append(obs[key][:, :self.observation_horizon])
            proprio = torch.cat(proprio_list, dim=-1)  # (B, T, proprio_dim)
            proprio_feat = self.proprio_encoder(proprio)  # (B, T, embed_size)
            features.append(proprio_feat)
        
        # Concatenate all observation features
        obs_features = torch.cat(features, dim=-1)  # (B, T, total_obs_dim)
        obs_features = obs_features.flatten(start_dim=1)  # (B, T * total_obs_dim)
        
        # Encode language
        lang_feat = self.language_encoder(data)  # (B, embed_size)
        
        # Concatenate obs and language features
        global_cond = torch.cat([obs_features, lang_feat], dim=-1)
        
        return global_cond
    
    def forward(self, data):
        """
        Forward pass for training. Computes diffusion loss.
        
        Args:
            data: dict containing observations and actions
        Returns:
            loss: diffusion training loss
        """
        actions = data["actions"]
        B, T, D = actions.shape
        
        # Pad or truncate actions to prediction_horizon
        if T < self.prediction_horizon:
            # Pad with zeros (or repeat last action)
            pad_len = self.prediction_horizon - T
            padding = actions[:, -1:, :].repeat(1, pad_len, 1)
            actions = torch.cat([actions, padding], dim=1)
        else:
            actions = actions[:, :self.prediction_horizon]
        
        B = actions.shape[0]
        
        # Encode observations
        global_cond = self.encode_obs(data)
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Sample diffusion timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=actions.device
        ).long()
        
        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=global_cond)
        
        # L2 loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def compute_loss(self, data, reduction="mean"):
        """Compute training loss."""
        data = self.preprocess_input(data, train_mode=True)
        loss = self.forward(data)
        return loss
    
    def reset(self):
        """Reset inference state."""
        self.obs_queue = deque(maxlen=self.observation_horizon)
        self.action_queue = deque(maxlen=self.action_horizon)
    
    def get_action(self, data):
        """
        Get action for inference.
        
        Args:
            data: dict with current observation
        Returns:
            action: (B, ac_dim) action to execute
        """
        self.eval()
        
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            
            curr_obs = data["obs"]
            
            # Initialize queue if needed (e.g. if reset wasn't called or empty)
            if self.obs_queue is None:
                self.obs_queue = deque(maxlen=self.observation_horizon)
                
            # If queue is empty (start of episode), duplicate current frame
            if len(self.obs_queue) == 0:
                for _ in range(self.observation_horizon):
                    self.obs_queue.append(curr_obs)
            else:
                self.obs_queue.append(curr_obs)
            
            # Stack observations from queue along time dimension
            # Assumes keys are consistent and shapes are (B, C, H, W) or (B, D)
            stacked_obs = {}
            for key in curr_obs.keys():
                tensors = [x[key] for x in self.obs_queue]
                # Check if we need to stack (add T dim) or cat (T dim exists)
                # If 4D (image) or 2D (proprio), stack to get 5D/3D
                if tensors[0].ndim in [2, 4]:
                    stacked_obs[key] = torch.stack(tensors, dim=1)
                else:
                    # Assume T dim is at dim 1, so cat
                    stacked_obs[key] = torch.cat(tensors, dim=1)
            
            # Create data with stacked history
            data_history = {
                "obs": stacked_obs,
                "task_emb": data["task_emb"]
            }
            
            if len(self.action_queue) == 0:
                # Need to run denoising to get new action trajectory
                action_traj = self._denoise_actions(data_history)  # (B, prediction_horizon, ac_dim)
                
                # Extract actions using action_horizon
                start = self.observation_horizon - 1
                end = start + self.action_horizon
                actions = action_traj[:, start:end]  # (B, action_horizon, ac_dim)
                
                # Add to queue
                for t in range(actions.shape[1]):
                    self.action_queue.append(actions[:, t])
            
            # Pop action from queue
            action = self.action_queue.popleft()
            
        return action.cpu().numpy()
    
    def _denoise_actions(self, data):
        """
        Run denoising diffusion to generate action trajectory.
        """
        # Currently using main model (EMA disabled)
        
        # Encode observations
        global_cond = self.encode_obs(data)
        B = global_cond.shape[0]
        
        # Initialize from noise
        noisy_action = torch.randn(
            (B, self.prediction_horizon, self.ac_dim),
            device=self.device
        )
        
        # Set scheduler timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
        
        # Iterative denoising
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=noisy_action,
                timestep=k,
                global_cond=global_cond
            )
            
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample
        
        return noisy_action
