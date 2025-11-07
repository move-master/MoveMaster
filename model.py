import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

from JengaEnv import JengaEnv 

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.adv = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_size),
        )
        self.val = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        h = self.feature(x)
        adv = self.adv(h)
        val = self.val(h)
        return val + adv - adv.mean(dim=1, keepdim=True)

def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    q = q_values.copy()
    q[mask == 0] = -1e9
    return int(q.argmax())

@torch.no_grad()
def get_action(state, epsilon, mask, policy, device):
    legal = np.nonzero(mask)[0]
    if len(legal) == 0:
        return 0
    if np.random.rand() < epsilon:
        return int(np.random.choice(legal))
    s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    q = policy(s).cpu().numpy()[0]
    return masked_argmax(q, mask)

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr

gamma = 0.99
eps, eps_min, eps_decay = 1.0, 0.05, 0.9995
anneal_start = 30_000
anneal_len = 20_000
eps_floor_late = 0.02
eps_delta = (0.05 - eps_floor_late) / max(anneal_len, 1)

lr = 1e-3
batch_size = 128
replay_cap = 100_000
warmup = 5000
tau = 0.01
episodes = 50_000
save_every = 5000
save_path = os.path.join("checkpoints", "policy.pth")
os.makedirs("checkpoints", exist_ok=True)

def make_env(seed=None):
    return JengaEnv(start_height_layers=5, max_steps=200, seed=seed, com_half_span=1.0)

env = make_env(seed=0)
state, info = env.reset()
mask = info["action_mask"]
state_size = len(state)
action_size = env.action_space.n 
print("state_size", state_size, "action_size", action_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = DQN(state_size, action_size).to(device)
target = DQN(state_size, action_size).to(device)
target.load_state_dict(policy.state_dict())
target.eval()

opt = optim.Adam(policy.parameters(), lr=lr)
loss_fn = nn.SmoothL1Loss()
buf = deque(maxlen=replay_cap)

@torch.no_grad()
def evaluate(policy, episodes=200):
    def run(mode="greedy"):
        total_R = total_moves = total_H = 0.0
        total_dcom = total_depth = total_two = total_collapse = 0.0
        for _ in range(episodes):
            e = make_env()
            s, info = e.reset()
            m = info.get("action_mask", np.ones(action_size, dtype=np.int32))
            done = False
            ep_R, moves = 0.0, 0
            com0 = e._com_x_norm() if hasattr(e, "_com_x_norm") else 0.0
            while not done:
                if mode == "random":
                    legal = np.nonzero(m)[0]
                    a = int(np.random.choice(legal)) if len(legal) else 0
                else:
                    a = get_action(s, 0.0, m, policy, device)
                s, r, term, trunc, info2 = e.step(a)
                done = bool(term) or bool(trunc)
                m = info2.get("action_mask", m)
                ep_R += float(r)
                moves += 1
            total_R += ep_R
            total_moves += moves
            total_H += getattr(e, "height", 0)
            com1 = e._com_x_norm() if hasattr(e, "_com_x_norm") else 0.0
            total_dcom += abs(com1 - com0)
            total_depth += 0.0
            total_two += 0.0
            total_collapse += 1.0
            e.close()
        return dict(
            R=total_R/episodes,
            moves=total_moves/episodes,
            maxH=total_H/episodes,
            dCOM=-(total_dcom/episodes),
            depth=total_depth/episodes,
            twoBlk=total_two/episodes,
            collapse=total_collapse/episodes,
        )
    g = run("greedy"); r = run("random")
    print(f"[EVAL {episodes}] greedy: R={g['R']:.2f}  moves={g['moves']:.1f}  maxH={g['maxH']:.2f}  "
          f"dCOM={g['dCOM']:.3f}  depth={g['depth']:.2f}  twoBlk={g['twoBlk']:.2f}  collapse={g['collapse']:.2f} | "
          f"random: R={r['R']:.2f}  moves={r['moves']:.1f}  maxH={r['maxH']:.2f}  "
          f"dCOM={r['dCOM']:.3f}  depth={r['depth']:.2f}  twoBlk={r['twoBlk']:.2f}  collapse={r['collapse']:.2f}")

global_step = 0
for ep in range(episodes):
    state, info = env.reset()
    mask = info["action_mask"]
    ep_return = 0.0
    done = False

    while not done:
        action = get_action(state, eps, mask, policy, device)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info["action_mask"]

        buf.append((state, action, reward, next_state, float(done)))
        state, mask = next_state, next_mask
        ep_return += reward
        global_step += 1

        if len(buf) >= warmup and global_step % 4 == 0:
            batch = random.sample(buf, batch_size)
            s, a, r, s2, d = zip(*batch)

            s  = torch.as_tensor(np.array(s),  dtype=torch.float32, device=device)
            a  = torch.as_tensor(a,           dtype=torch.int64,   device=device).unsqueeze(1)
            r  = torch.as_tensor(r,           dtype=torch.float32, device=device).unsqueeze(1)
            s2 = torch.as_tensor(np.array(s2),dtype=torch.float32, device=device)
            d  = torch.as_tensor(d,           dtype=torch.float32, device=device).unsqueeze(1)

            with torch.no_grad():
                a2 = policy(s2).argmax(dim=1, keepdim=True) 
                q_tgt_next = target(s2).gather(1, a2)
                y = r + gamma * (1 - d) * q_tgt_next

            q = policy(s).gather(1, a)
            loss = loss_fn(q, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()

            with torch.no_grad():
                for tp, p in zip(target.parameters(), policy.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)

    if eps > eps_min:
        eps = max(eps_min, eps * eps_decay)
    if (ep + 1) >= anneal_start and (ep + 1) < (anneal_start + anneal_len):
        eps = max(eps_floor_late, eps - eps_delta)
    if (ep + 1) == anneal_start:
        set_lr(opt, 5e-4)

    print(f"ep {ep+1:05d}  return {ep_return:7.2f}  eps {eps:.3f}")

    if (ep + 1) % save_every == 0:
        torch.save(policy.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")
        evaluate(policy, episodes=200)

env.close()
