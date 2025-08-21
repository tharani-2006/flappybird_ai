# test_jump.py
from game.flappy_bird import FlappyBirdEnv
env = FlappyBirdEnv(seed=0)
s = env.reset()
for t in range(300):
    a = 1 if t % 8 == 0 else 0  # jump every 8 frames
    s, r, d, info = env.step(a)
    env.render(fps=60)
    if d: break
env.close()
print("OK if you saw periodic jumps.")