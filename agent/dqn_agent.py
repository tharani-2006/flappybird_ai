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


class DQNAgent:
	def __init__(self, config: Optional[DQNConfig] = None):
		self.cfg = config or DQNConfig()
		self.device = torch.device(self.cfg.device)

		self.q_net = QNetwork(self.cfg.state_dim, self.cfg.action_dim).to(self.device)
		self.target_net = QNetwork(self.cfg.state_dim, self.cfg.action_dim).to(self.device)
		self.target_net.load_state_dict(self.q_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
		self.replay = ReplayBuffer(self.cfg.replay_size)

		self.total_steps = 0
		self.epsilon = self.cfg.eps_start

	def select_action(self, state: np.ndarray) -> int:
		self.total_steps += 1
		# Epsilon decay
		decay_ratio = min(1.0, self.total_steps / float(self.cfg.eps_decay_steps))
		self.epsilon = self.cfg.eps_start + decay_ratio * (self.cfg.eps_end - self.cfg.eps_start)

		if np.random.rand() < self.epsilon:
			return np.random.randint(self.cfg.action_dim)

		state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # [1, state_dim]
		with torch.no_grad():
			q_values = self.q_net(state_t)
		action = int(torch.argmax(q_values, dim=1).item())
		return action

	def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
		self.replay.push(state, action, reward, next_state, done)

	def train_step(self) -> Optional[float]:
		# Only learn after we have a decent buffer
		if len(self.replay) < max(self.cfg.batch_size, self.cfg.start_learning_after):
			return None

		states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size)

		states_t = torch.from_numpy(states).float().to(self.device)               # [B, S]
		actions_t = torch.from_numpy(actions).long().to(self.device)             # [B]
		rewards_t = torch.from_numpy(rewards).float().to(self.device)            # [B]
		next_states_t = torch.from_numpy(next_states).float().to(self.device)    # [B, S]
		dones_t = torch.from_numpy(dones).float().to(self.device)                # [B]

		# Q(s,a)
		q_values = self.q_net(states_t)                                          # [B, A]
		q_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)             # [B]

		# Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
		with torch.no_grad():
			next_q_values = self.target_net(next_states_t)                       # [B, A]
			next_q_max = torch.max(next_q_values, dim=1)[0]                      # [B]
			target = rewards_t + (1.0 - dones_t) * self.cfg.gamma * next_q_max   # [B]

		loss = nn.functional.mse_loss(q_a, target)

		self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.gradient_clip_norm)
		self.optimizer.step()

		# Periodically update target network
		if self.total_steps % self.cfg.target_update_freq == 0:
			self.target_net.load_state_dict(self.q_net.state_dict())

		return float(loss.item())

	def save(self, path: str) -> None:
		torch.save(
			{
				"q_net": self.q_net.state_dict(),
				"target_net": self.target_net.state_dict(),
				"optimizer": self.optimizer.state_dict(),
				"total_steps": self.total_steps,
				"epsilon": self.epsilon,
				"config": self.cfg.__dict__,
			},
			path,
		)

	def load(self,  path: str) -> None:
		ckpt = torch.load(path, map_location=self.device)
		self.q_net.load_state_dict(ckpt["q_net"])
		self.target_net.load_state_dict(ckpt.get("target_net", ckpt["q_net"]))
		self.optimizer.load_state_dict(ckpt["optimizer"])
		self.total_steps = ckpt.get("total_steps", 0)
		self.epsilon = ckpt.get("epsilon", self.cfg.eps_start)