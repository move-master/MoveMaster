import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from JengaEnv import JengaEnv 

# ----- DQN -----
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_size),
        )
    def forward(self, x): return self.net(x)

def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    masked = q_values.copy()
    masked[mask == 0] = -1e9
    return int(masked.argmax())

def get_action(state, epsilon, mask, policy_net, device):
    legal = np.nonzero(mask)[0]
    if len(legal) == 0:
        return 0
    if np.random.rand() < epsilon:
        return int(np.random.choice(legal))
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy_net(s).cpu().numpy()[0]
    return masked_argmax(q, mask)

# ----- Hyperparams -----
gamma = 0.99
epsilon, epsilon_min, epsilon_decay = 1.0, 0.01, 0.995
lr = 1e-3
batch_size = 64
memory_size = 100_000
target_update_freq = 1000
warmup = 1000

# ----- Env -----
env = JengaEnv(start_height_layers=5, max_steps=200, seed=0)
state, info = env.reset()
mask = info["action_mask"]
state_size = len(state)                         # should be 109
action_size = env.action_space.n               # should be 162

# ----- Nets / Opt -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.SmoothL1Loss()

# ----- Replay -----
memory = deque(maxlen=memory_size)

# ----- Training loop -----
global_step = 0
episodes = 2000

for ep in range(episodes):
    state, info = env.reset()
    mask = info["action_mask"]
    ep_reward = 0.0

    while True:
        action = get_action(state, epsilon, mask, policy_net, device)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info["action_mask"]

        memory.append((state, action, reward, next_state, float(done)))
        state, mask = next_state, next_mask
        ep_reward += reward
        global_step += 1

        # Learn
        if len(memory) >= warmup and global_step % 4 == 0:
            batch = random.sample(memory, batch_size)
            s,a,r,s2,d = zip(*batch)
            s  = torch.as_tensor(np.array(s),  dtype=torch.float32, device=device)
            a  = torch.as_tensor(a,           dtype=torch.int64,   device=device).unsqueeze(1)
            r  = torch.as_tensor(r,           dtype=torch.float32, device=device).unsqueeze(1)
            s2 = torch.as_tensor(np.array(s2),dtype=torch.float32, device=device)
            d  = torch.as_tensor(d,           dtype=torch.float32, device=device).unsqueeze(1)

            # Double DQN target
            with torch.no_grad():
                next_q_online = policy_net(s2)                  # (B, A)
                a2 = torch.argmax(next_q_online, dim=1, keepdim=True)  # (B,1)
                next_q_target = target_net(s2).gather(1, a2)    # (B,1)
                target_q = r + gamma * (1 - d) * next_q_target

            q = policy_net(s).gather(1, a)
            loss = loss_fn(q, target_q)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

        if global_step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    # epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Ep {ep+1} | R={ep_reward:.2f} | eps={epsilon:.3f}")