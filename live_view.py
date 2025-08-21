import os, time, torch, numpy as np
from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent, DQNConfig

CKPT_PATH = "agent/flappy_dqn_live.pth"
EPS_VIEW = 0.1  # small exploration so you see jumps early

def maybe_reload(agent, last_mtime):
    try:
        mtime = os.path.getmtime(CKPT_PATH)
    except FileNotFoundError:
        return last_mtime
    if last_mtime is None or mtime != last_mtime:
        agent.load(CKPT_PATH)
        agent.q_net.eval()
        print(f"Loaded weights | steps={agent.total_steps} | eps={agent.epsilon:.3f} @ {time.strftime('%H:%M:%S')}")
        return mtime
    return last_mtime

def main():
    env = FlappyBirdEnv(pipe_gap=160, pipe_speed=2.0, seed=123)
    agent = DQNAgent(DQNConfig())
    print(f"Waiting for checkpoint: {CKPT_PATH}")
    while not os.path.exists(CKPT_PATH):
        time.sleep(1.0)

    last_mtime = None
    while True:
        last_mtime = maybe_reload(agent, last_mtime)
        s = env.reset()
        done = False
        while not done:
            if np.random.rand() < EPS_VIEW:
                a = np.random.randint(2)
            else:
                with torch.no_grad():
                    st = torch.from_numpy(s).float().unsqueeze(0).to(agent.cfg.device)
                    a = int(torch.argmax(agent.q_net(st), dim=1).item())
            s, r, done, info = env.step(a)
            env.render(fps=60)
        time.sleep(0.2)

if __name__ == "__main__":
    main()