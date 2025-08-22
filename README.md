# Flappy Bird AI with Deep Q-Learning

A complete implementation of an AI agent that learns to play Flappy Bird using Deep Q-Network (DQN) reinforcement learning.

## 🎯 What This Project Does

This project creates an AI that learns to play Flappy Bird from scratch. The AI starts by making random moves, but through trial and error, it learns the best actions to survive longer and score higher points.

## 📁 Project Structure

```
flappybird_ai/
│── venv/                    # Python virtual environment
│── game/
│   ├── flappy_bird.py       # The Flappy Bird game environment
│── agent/
│   ├── dqn_agent.py         # The AI agent that learns to play
│── train_flappybird.ipynb   # Jupyter notebook for training
│── live_view.py             # Real-time viewer to watch AI play
│── requirements.txt         # Python dependencies
│── README.md               # This file
```

## 🧠 How It Works (Simple Explanation)

### 1. The Game Environment (`game/flappy_bird.py`)

**What it does:**
- Creates a Flappy Bird game that the AI can interact with
- Tracks the bird's position, pipes, and collisions
- Gives rewards: +1 for surviving each frame, -100 for dying

**Key parts:**
```python
# The AI sees 5 pieces of information:
state = [
    bird_y_norm,           # Bird's height (normalized)
    bird_vel_norm,         # Bird's vertical speed
    next_pipe_x_norm,      # Distance to next pipe
    next_pipe_top_norm,    # Top of pipe gap
    next_pipe_bottom_norm  # Bottom of pipe gap
]

# The AI can do 2 actions:
action = 0  # Do nothing (fall)
action = 1  # Jump (flap wings)
```

**Why this matters:** The AI needs to understand the game state to make good decisions.

### 2. The AI Agent (`agent/dqn_agent.py`)

**What it does:**
- Contains a neural network that learns to predict the best action
- Uses "experience replay" to learn from past games
- Balances exploration (trying new things) vs exploitation (using what it knows)

**Key components:**

#### Neural Network (QNetwork)
```python
class QNetwork(nn.Module):
    def __init__(self, input_dim=5, output_dim=2):
        self.net = nn.Sequential(
            nn.Linear(5, 128),    # Input: 5 game state values
            nn.ReLU(),            # Activation function
            nn.Linear(128, 128),  # Hidden layer
            nn.ReLU(),
            nn.Linear(128, 2),    # Output: Q-values for 2 actions
        )
```
**What this does:** Takes the 5 game state values and outputs 2 numbers (Q-values) - one for each action. Higher Q-value = better action.

#### Experience Replay (ReplayBuffer)
```python
class ReplayBuffer:
    def push(self, state, action, reward, next_state, done):
        # Store: "I was in state X, did action Y, got reward Z, ended up in state W"
        self.buffer.append((state, action, reward, next_state, done))
```
**What this does:** Remembers past experiences so the AI can learn from them later, not just the most recent action.

#### Epsilon-Greedy Strategy
```python
def select_action(self, state):
    if np.random.rand() < self.epsilon:
        return np.random.randint(2)  # Random action (exploration)
    else:
        return best_action_from_network  # Best known action (exploitation)
```
**What this does:** Early in training, the AI mostly tries random actions to explore. Later, it uses what it learned.

### 3. Training Process (`train_flappybird.ipynb`)

**What happens during training:**

1. **Episode Loop:** Play many games (episodes)
2. **Step Loop:** For each frame of the game:
   - Get current game state
   - AI chooses action (jump or don't jump)
   - Game updates (bird moves, pipes move)
   - Get reward and new state
   - Store experience in memory
   - Train the neural network
   - Save progress periodically

**Key training code:**
```python
for episode in range(500):  # Play 500 games
    state = env.reset()     # Start new game
    for step in range(10000):  # Max 10,000 frames per game
        action = agent.select_action(state)  # AI chooses action
        next_state, reward, done, info = env.step(action)  # Game responds
        agent.store(state, action, reward, next_state, done)  # Remember
        loss = agent.train_step()  # Learn from experience
        if done: break  # Game over
```

### 4. Live Viewer (`live_view.py`)

**What it does:**
- Shows a real-time Pygame window of the AI playing
- Automatically reloads the latest trained model
- Lets you watch the AI improve as training progresses

## 🔧 How to Run the Project

### Step 1: Setup Environment
```powershell
# Navigate to project folder
cd D:\2025\ML\flappybird_ai

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train the AI
```powershell
# Start Jupyter notebook
jupyter notebook train_flappybird.ipynb
```

In the notebook:
1. Run the setup cell (imports and initialization)
2. Run the training cell (starts learning)
3. Watch the console output for progress

### Step 3: Watch AI Play (Optional)
In a second terminal:
```powershell
cd D:\2025\ML\flappybird_ai
.\venv\Scripts\Activate.ps1
python live_view.py
```

This opens a Pygame window showing the AI playing with the latest learned strategy.

## 🎮 Understanding the Learning Process

### Phase 1: Random Exploration (Episodes 1-50)
- AI mostly makes random moves
- Bird crashes quickly
- Average score: 0-5 points

### Phase 2: Basic Learning (Episodes 50-200)
- AI starts to understand simple patterns
- Bird survives longer occasionally
- Average score: 5-20 points

### Phase 3: Strategy Development (Episodes 200-500)
- AI learns optimal timing for jumps
- Bird navigates through pipes more consistently
- Average score: 20-100+ points

### Phase 4: Fine-tuning (Episodes 500+)
- AI optimizes for maximum survival
- Very consistent performance
- Average score: 100+ points

## 🧮 The Math Behind It (Simplified)

### Q-Learning Formula
```
Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
```

**What this means:**
- `Q(s,a)`: How good action `a` is in state `s`
- `r`: Immediate reward
- `γ`: Future reward discount (0.99 = future rewards matter a lot)
- `max(Q(s',a'))`: Best possible future reward
- `α`: Learning rate (how much to update)

**In simple terms:** "If this action led to a good outcome, remember it. If it led to a bad outcome, avoid it."

### Loss Function
```python
loss = (predicted_q_value - target_q_value)²
```

**What this means:** The neural network tries to predict Q-values that match the actual rewards the AI receives.

## 🔍 Key Parameters and What They Do

### Environment Parameters
```python
pipe_gap = 120        # Space between top and bottom pipes
pipe_speed = 3.0      # How fast pipes move left
gravity = 0.5         # How fast bird falls
jump_velocity = -8.0  # How much bird jumps when flapping
```

### Training Parameters
```python
gamma = 0.99          # Future reward importance (0.99 = very important)
lr = 1e-3             # Learning rate (how fast AI learns)
batch_size = 64       # How many experiences to learn from at once
replay_size = 50_000  # How many past experiences to remember
eps_start = 1.0       # Start with 100% random actions
eps_end = 0.05        # End with 5% random actions
```

## 🎯 Expected Results

### Training Progress Indicators
- **Episode Return:** Total reward per game (higher = better)
- **50-episode Average:** Smoothed performance over 50 games
- **Epsilon:** Exploration rate (starts at 1.0, decreases to 0.05)

### Good Performance Metrics
- Episode return > 50 (survives ~50 frames)
- 50-episode average > 30
- Consistent survival through multiple pipes

## 🐛 Common Issues and Solutions

### Problem: AI only falls down
**Solution:** 
- Check if action=1 actually makes the bird jump
- Increase `pipe_gap` to make the game easier
- Reduce `pipe_speed` to give AI more time to react

### Problem: Training is too slow
**Solution:**
- Reduce `num_episodes` for testing
- Use CPU instead of GPU if available
- Disable rendering during training

### Problem: AI doesn't improve
**Solution:**
- Increase learning rate (`lr = 2e-3`)
- Reduce `start_learning_after` to learn earlier
- Add more exploration time (`eps_decay_steps = 100_000`)

## 🚀 Next Steps and Improvements

### Easy Improvements
1. **Frame Stacking:** Give AI multiple frames to see movement
2. **Prioritized Replay:** Learn more from important experiences
3. **Double DQN:** More stable learning algorithm

### Advanced Improvements
1. **Dueling DQN:** Separate value and advantage learning
2. **Rainbow DQN:** Combine multiple DQN improvements
3. **PPO/A3C:** Different RL algorithms for better performance

## 📚 Learning Resources

- **Reinforcement Learning Basics:** Sutton & Barto's "Reinforcement Learning: An Introduction"
- **Deep Q-Learning:** "Playing Atari with Deep Reinforcement Learning" (DeepMind)
- **PyTorch Tutorials:** Official PyTorch documentation
- **Pygame Basics:** Pygame documentation for game development

## 🤝 Contributing

Feel free to experiment with:
- Different neural network architectures
- Alternative reward functions
- New game mechanics
- Visualization improvements

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Happy Learning! 🐦✨**
