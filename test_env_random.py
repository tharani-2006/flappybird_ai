from game.flappy_bird import FlappyBirdEnv
import random, time

env = FlappyBirdEnv(seed=42)
state = env.reset()
total_reward = 0.0

for t in range(5000):
	action = random.randint(0, 1)  # 0 = no-op, 1 = jump
	state, reward, done, info = env.step(action)
	env.render(fps=100)
	total_reward += reward
	if done:
		break

print("Episode finished. Reward:", total_reward, "Score:", info.get("score"))
env.close()