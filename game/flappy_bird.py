import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
	import pygame
	_PYGAME_AVAILABLE = True
except Exception:
	_PYGAME_AVAILABLE = False


class FlappyBirdEnv:
	"""
	Simple Flappy Bird environment for DQN.
	Observation (np.float32, shape [5], all normalized to [~0,1] range):
		[ bird_y_norm, bird_vel_norm, next_pipe_x_norm, next_pipe_top_norm, next_pipe_bottom_norm ]
	Actions:
		0 = do nothing, 1 = jump
	Rewards:
		+1 for each time-step survived, -100 on collision/death.
	"""

	def __init__(
		self,
		screen_width: int = 288,
		screen_height: int = 512,
		pipe_gap: int = 120,
		pipe_width: int = 52,
		pipe_speed: float = 3.0,
		gravity: float = 0.5,
		jump_velocity: float = -8.0,
		bird_size: int = 24,
		bird_x: int = 50,
		seed: Optional[int] = None,
	):
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.pipe_gap = pipe_gap
		self.pipe_width = pipe_width
		self.pipe_speed = pipe_speed
		self.gravity = gravity
		self.jump_velocity = jump_velocity
		self.bird_size = bird_size
		self.bird_x = bird_x

		self.random = random.Random(seed)
		self.np_random = np.random.RandomState(seed)

		self._init_pygame_done = False
		self.screen = None
		self.clock = None
		self.font = None

		self.bird_y = 0.0
		self.bird_vy = 0.0
		self.pipes: List[Dict[str, float]] = []
		self.score = 0
		self.ticks = 0
		self.done = False

	def seed(self, seed: Optional[int] = None) -> None:
		self.random = random.Random(seed)
		self.np_random = np.random.RandomState(seed)

	def reset(self) -> np.ndarray:
		self.bird_y = self.screen_height * 0.5
		self.bird_vy = 0.0
		self.score = 0
		self.ticks = 0
		self.done = False

		self.pipes = []
		# Spawn two pipes at the start to avoid empty screen
		first_x = self.screen_width + 60
		second_x = first_x + 180
		self.pipes.append(self._create_pipe(first_x))
		self.pipes.append(self._create_pipe(second_x))

		return self._get_state()

	def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
		if self.done:
			# If step is called after done, return same terminal state
			return self._get_state(), 0.0, True, {"score": self.score}

		# Action: 1 = jump
		if action == 1:
			self.bird_vy = self.jump_velocity

		# Physics update
		self.bird_vy += self.gravity
		self.bird_y += self.bird_vy
		self.ticks += 1

		# Move pipes
		for pipe in self.pipes:
			pipe["x"] -= self.pipe_speed

		# Remove off-screen pipes and spawn new ones
		self.pipes = [p for p in self.pipes if p["x"] + self.pipe_width > 0]
		if len(self.pipes) == 0 or (self.pipes[-1]["x"] < self.screen_width - 180):
			self.pipes.append(self._create_pipe(self.screen_width + 20))

		# Collision detection
		collision = self._check_collision()

		reward = 1.0  # survive bonus
		if collision:
			reward = -100.0
			self.done = True

		if not self.done:
			# Increment score when passing a pipe center
			for pipe in self.pipes:
				# Count score only once per pipe
				if not pipe.get("counted", False) and (pipe["x"] + self.pipe_width) < self.bird_x:
					self.score += 1
					pipe["counted"] = True

		return self._get_state(), reward, self.done, {"score": self.score}

	def render(self, fps: int = 30) -> None:
		if not _PYGAME_AVAILABLE:
			return
		if not self._init_pygame_done:
			pygame.init()
			self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
			pygame.display.set_caption("FlappyBirdEnv")
			self.clock = pygame.time.Clock()
			self.font = pygame.font.SysFont("Arial", 18)
			self._init_pygame_done = True

		# Handle events (let user close window)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.close()
				return
		# Draw background
		self.screen.fill((135, 206, 235))  # sky blue
		# Ground line (visual)
		pygame.draw.rect(
			self.screen,
			(222, 184, 135),
			pygame.Rect(0, self.screen_height - 40, self.screen_width, 40),
		)

		# Draw pipes
		for pipe in self.pipes:
			top_rect, bottom_rect = self._pipe_rects(pipe)
			pygame.draw.rect(self.screen, (34, 139, 34), top_rect)     # forest green
			pygame.draw.rect(self.screen, (34, 139, 34), bottom_rect)

		# Draw bird
		bird_rect = pygame.Rect(
			int(self.bird_x - self.bird_size // 2),
			int(self.bird_y - self.bird_size // 2),
			self.bird_size,
			self.bird_size,
		)
		pygame.draw.rect(self.screen, (255, 215, 0), bird_rect)  # gold

		# HUD
		if self.font:
			score_surf = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
			self.screen.blit(score_surf, (8, 8))

		pygame.display.flip()
		if self.clock:
			self.clock.tick(fps)

	def close(self) -> None:
		if _PYGAME_AVAILABLE and self._init_pygame_done:
			pygame.display.quit()
			pygame.quit()
		self._init_pygame_done = False
		self.screen = None
		self.clock = None
		self.font = None

	# -----------------------
	# Internal helpers
	# -----------------------
	def _create_pipe(self, x: float) -> Dict[str, float]:
		margin = 60
		gap_center = self.random.randint(margin + self.pipe_gap // 2, self.screen_height - 40 - margin - self.pipe_gap // 2)
		return {"x": float(x), "gap_center": float(gap_center)}

	def _pipe_rects(self, pipe: Dict[str, float]) -> Tuple["pygame.Rect", "pygame.Rect"]:
		top_height = int(pipe["gap_center"] - self.pipe_gap / 2)
		bottom_y = int(pipe["gap_center"] + self.pipe_gap / 2)
		top_rect = pygame.Rect(int(pipe["x"]), 0, self.pipe_width, top_height)
		bottom_rect = pygame.Rect(int(pipe["x"]), bottom_y, self.pipe_width, self.screen_height - bottom_y - 40)
		return top_rect, bottom_rect

	def _check_collision(self) -> bool:
		# Ground or ceiling
		if (self.bird_y - self.bird_size // 2) < 0:
			return True
		if (self.bird_y + self.bird_size // 2) > (self.screen_height - 40):
			return True

		# Pipes
		bird_left = self.bird_x - self.bird_size // 2
		bird_right = self.bird_x + self.bird_size // 2
		bird_top = self.bird_y - self.bird_size // 2
		bird_bottom = self.bird_y + self.bird_size // 2

		for pipe in self.pipes:
			top_height = pipe["gap_center"] - self.pipe_gap / 2
			bottom_y = pipe["gap_center"] + self.pipe_gap / 2

			pipe_left = pipe["x"]
			pipe_right = pipe["x"] + self.pipe_width

			# Overlap in x?
			if bird_right > pipe_left and bird_left < pipe_right:
				# If bird is above top gap or below bottom gap => collision
				if bird_top < top_height or bird_bottom > bottom_y:
					return True

		return False

	def _get_state(self) -> np.ndarray:
		# Find next pipe (first with x + width >= bird_x)
		next_pipe = None
		for pipe in self.pipes:
			if (pipe["x"] + self.pipe_width) >= self.bird_x:
				next_pipe = pipe
				break
		if next_pipe is None:
			next_pipe = self.pipes[0]

		# Normalizations
		bird_y_norm = float(self.bird_y) / float(self.screen_height)
		# Velocity normalization: map roughly from [-20, 20] -> [0, 1]
		vel_clip = max(-20.0, min(20.0, self.bird_vy))
		bird_vel_norm = (vel_clip + 20.0) / 40.0

		next_pipe_x_norm = (next_pipe["x"] - self.bird_x) / float(self.screen_width)
		next_pipe_top_norm = (next_pipe["gap_center"] - self.pipe_gap / 2) / float(self.screen_height)
		next_pipe_bottom_norm = (next_pipe["gap_center"] + self.pipe_gap / 2) / float(self.screen_height)

		state = np.array(
			[
				bird_y_norm,
				bird_vel_norm,
				next_pipe_x_norm,
				next_pipe_top_norm,
				next_pipe_bottom_norm,
			],
			dtype=np.float32,
		)
		return state