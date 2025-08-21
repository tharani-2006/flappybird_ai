import os
import time
import torch
import numpy as np
from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent, DQNConfig

CKPT_PATH = "agent/flappy_dqn_live.pth"

def maybe_reload(agent, last_mtime):
	try:
		mtime = os.path.getmtime(CKPT_PATH)
	except FileNotFoundError:
		return last_mtime
	if last_mtime is None or mtime != last_mtime:
		try:
			agent.load(CKPT_PATH)
			agent.q_net.eval()
			print(f"Loaded weights @ {time.strftime('%H:%M:%S')}")
			return mtime
		except Exception as e:
			print("Checkpoint load failed:", e)
	return last_mtime

def main():
	env = FlappyBirdEnv(seed=123)
	agent = DQNAgent(DQNConfig())
	print(f"Waiting for checkpoint: {CKPT_PATH}")
	while not os.path.exists(CKPT_PATH):
		time.sleep(1.0)

	last_mtime = None
	while True:
		last_mtime = maybe_reload(agent, last_mtime)
		state = env.reset()
		done = False
		while not done:
			with torch.no_grad():
				state_t = torch.from_numpy(state).float().unsqueeze(0).to(agent.cfg.device)
				action = int(torch.argmax(agent.q_net(state_t), dim=1).item())
			state, reward, done, info = env.step(action)
			env.render(fps=60)
		time.sleep(0.2)

if __name__ == "__main__":
	main()