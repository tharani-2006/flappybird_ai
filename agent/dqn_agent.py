from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_sizes[0]),
			nn.ReLU(),
			nn.Linear(hidden_sizes[0], hidden_sizes[1]),
			nn.ReLU(),
			nn.Linear(hidden_sizes[1], output_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


@dataclass
class DQNConfig:
	state_dim: int = 5
	action_dim: int = 2
	gamma: float = 0.99
	lr: float = 1e-3
	batch_size: int = 64
	replay_size: int = 50_000
	start_learning_after: int = 5_000
	target_update_freq: int = 1_000
	eps_start: float = 1.0
	eps_end: float = 0.05
	eps_decay_steps: int = 50_000
	gradient_clip_norm: float = 5.0
	device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
	def __init__(self, capacity: int):
		self.capacity = capacity
		self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

	def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
		self.buffer.append((state, action, reward, next_state, done))

	def __len__(self) -> int:
		return len(self.buffer)

	def sample(self, batch_size: int):
		indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
		states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))
		return (
			np.stack(states, axis=0),
			np.array(actions, dtype=np.int64),
			np.array(rewards, dtype=np.float32),
			np.stack(next_states, axis=0),
			np.array(dones, dtype=np.float32),
		)