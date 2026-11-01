"""Dueling DQN with masked one-step TD targets and explicit checkpoint state."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AgentConfig:
    hidden_channels: int = 128
    hidden_layers: int = 3
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_capacity: int = 100000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 50000
    target_update_frequency: int = 1000
    learning_starts: int = 1000

    def __post_init__(self):
        for name in ("hidden_channels", "hidden_layers", "batch_size", "buffer_capacity",
                     "epsilon_decay", "target_update_frequency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"agent.{name} must be a positive integer")
        if self.hidden_channels < 2 or self.batch_size > self.buffer_capacity:
            raise ValueError("Invalid hidden width or batch size exceeds replay capacity")
        if not isinstance(self.learning_starts, int) or self.learning_starts < 0:
            raise ValueError("learning_starts must be a nonnegative integer")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


class DuelingDQN(nn.Module):
    """Default shared layers: 52 -> 128 -> 128 -> 128; V/A branches: 128 -> 64."""

    def __init__(self, state_dim: int, hidden_channels: int = 128, hidden_layers: int = 3):
        super().__init__()
        layers = []
        width = state_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_channels), nn.ReLU()])
            width = hidden_channels
        # Names intentionally match the original policy state_dict.
        self.feature_extractor = nn.Sequential(*layers)
        self.value_stream = nn.Sequential(nn.Linear(width, width // 2), nn.ReLU(),
                                          nn.Linear(width // 2, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(width, width // 2), nn.ReLU(),
                                              nn.Linear(width // 2, 5))

    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def read_policy_checkpoint(path: Path, device="cpu", allow_legacy_pickle=False):
    """Use restricted torch loading unless the caller explicitly trusts legacy pickle."""
    try:
        return torch.load(path, map_location=device, weights_only=not allow_legacy_pickle)
    except Exception as exc:
        raise ValueError(f"Cannot read policy checkpoint {path}: {exc}") from exc


class DQNAgent:
    def __init__(self, state_dim: int, config: AgentConfig, device="cpu", seed=42):
        self.state_dim = state_dim
        self.config = config
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.policy = DuelingDQN(state_dim, config.hidden_channels, config.hidden_layers).to(self.device)
        self.target = DuelingDQN(state_dim, config.hidden_channels, config.hidden_layers).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.replay = []
        self.replay_position = 0
        self.frames = self.updates = self.completed_episodes = 0

    @property
    def epsilon(self):
        c = self.config
        return c.epsilon_end + (c.epsilon_start - c.epsilon_end) * math.exp(-self.frames / c.epsilon_decay)

    def select(self, state, mask, *, training=False, temperature=0.0):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        mask = np.asarray(mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if mask.shape != (5,) or not len(valid):
            raise ValueError("A nonempty five-action mask is required")
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid.tolist()))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(tensor)[0].detach().cpu().numpy()[valid]
        if not np.isfinite(q).all():
            raise ValueError("Policy produced non-finite Q values")
        if temperature == 0:
            return int(valid[int(np.argmax(q))])
        logits = (q - q.max()) / temperature
        weights = np.exp(logits).tolist()
        return int(self.rng.choices(valid.tolist(), weights=weights, k=1)[0])

    def observe(self, state, action, reward, next_state, done, next_mask):
        transition = (np.asarray(state, dtype=np.float32).copy(), int(action), float(reward),
                      np.asarray(next_state, dtype=np.float32).copy(), bool(done),
                      np.asarray(next_mask, dtype=bool).copy())
        if len(self.replay) < self.config.buffer_capacity:
            self.replay.append(transition)
        else:
            self.replay[self.replay_position] = transition
        self.replay_position = (self.replay_position + 1) % self.config.buffer_capacity
        self.frames += 1

    def update(self):
        c = self.config
        if len(self.replay) < c.batch_size or self.frames < c.learning_starts:
            return None
        batch = self.rng.sample(self.replay, c.batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        states = torch.as_tensor(np.asarray(states), device=self.device)
        next_states = torch.as_tensor(np.asarray(next_states), device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        masks = torch.as_tensor(np.asarray(masks), device=self.device)
        predictions = self.policy(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).masked_fill(~masks, -torch.inf).max(dim=1).values
            # Terminal all-false masks must never create 0 * -inf NaNs.
            next_q = torch.where(dones | ~masks.any(dim=1), torch.zeros_like(next_q), next_q)
            targets = rewards + c.gamma * next_q
        loss = F.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite DQN loss")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % c.target_update_frequency == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def policy_state(self):
        return {"schema_version": 1, "kind": "graphene_dueling_policy", "state_dim": self.state_dim,
                "agent_config": asdict(self.config), "policy_net_state_dict": self.policy.state_dict(),
                "completed_episodes": self.completed_episodes, "frames": self.frames}

    def training_state(self):
        result = self.policy_state()
        result.update(kind="graphene_dueling_training", target_state_dict=self.target.state_dict(),
                      optimizer_state_dict=self.optimizer.state_dict(), updates=self.updates,
                      replay_position=self.replay_position,
                      random_state=self.rng.getstate(), torch_rng_state=torch.get_rng_state(),
                      cuda_rng_states=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])
        if self.replay:
            states, actions, rewards, next_states, dones, masks = zip(*self.replay)
            result["replay"] = {
                "states": torch.from_numpy(np.asarray(states)),
                "actions": torch.tensor(actions), "rewards": torch.tensor(rewards, dtype=torch.float64),
                "next_states": torch.from_numpy(np.asarray(next_states)),
                "dones": torch.tensor(dones), "masks": torch.from_numpy(np.asarray(masks)),
            }
        else:
            result["replay"] = None
        return result

    def load_policy(self, checkpoint):
        if "state_dim" in checkpoint and checkpoint["state_dim"] != self.state_dim:
            raise ValueError("Policy state dimension does not match environment")
        if "agent_config" in checkpoint:
            saved = checkpoint["agent_config"]
            for key in ("hidden_channels", "hidden_layers"):
                if saved[key] != getattr(self.config, key):
                    raise ValueError(f"Policy {key} differs from current configuration")
        # Legacy files lack full architecture metadata; strict loading still
        # checks the explicitly configured architecture rather than guessing.
        if "policy_net_state_dict" not in checkpoint:
            raise ValueError("Checkpoint has no policy_net_state_dict")
        self.policy.load_state_dict(checkpoint["policy_net_state_dict"], strict=True)
        self.policy.eval()

    def restore_training(self, checkpoint):
        if checkpoint.get("kind") != "graphene_dueling_training":
            raise ValueError("Resume requires a full training checkpoint from this package")
        if checkpoint.get("agent_config") != asdict(self.config):
            raise ValueError("Resume agent configuration does not match checkpoint")
        self.load_policy(checkpoint)
        self.policy.train()
        self.target.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.completed_episodes = int(checkpoint["completed_episodes"])
        self.frames = int(checkpoint["frames"])
        self.updates = int(checkpoint["updates"])
        self.rng.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        self.replay.clear()
        self.replay_position = int(checkpoint["replay_position"])
        replay = checkpoint.get("replay")
        if replay is not None:
            for values in zip(*(replay[key].cpu().numpy() for key in
                                ("states", "actions", "rewards", "next_states", "dones", "masks"))):
                state, action, reward, next_state, done, mask = values
                self.replay.append((state.copy(), int(action), float(reward), next_state.copy(),
                                    bool(done), mask.copy()))
