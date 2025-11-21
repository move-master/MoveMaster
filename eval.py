import argparse
import numpy as np
import torch
import torch.nn as nn

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

def make_env(seed=None):
    return JengaEnv(start_height_layers=5, max_steps=200, seed=seed, com_half_span=1.0)

def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    q = q_values.copy()
    q[mask == 0] = -1e9
    return int(q.argmax())

@torch.no_grad()
def get_action(state, mask, policy, device):
    legal = np.nonzero(mask)[0]
    if len(legal) == 0:
        return 0
    s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    q = policy(s).cpu().numpy()[0]
    return masked_argmax(q, mask)

@torch.no_grad()
def evaluate(policy, action_size, episodes=200, device="cpu"):
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
                    a = get_action(s, m, policy, device)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to dueling-DQN checkpoint (policy.pth)")
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(seed=0)
    state, info = env.reset()
    state_size = len(state)
    action_size = env.action_space.n
    env.close()

    policy = DQN(state_size, action_size).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    policy.load_state_dict(sd, strict=True)
    policy.eval()

    evaluate(policy, action_size, episodes=args.episodes, device=device)

if __name__ == "__main__":
    main()