import argparse, numpy as np, torch
from JengaEnv import JengaEnv
from model import DQN, get_action

def make_env(seed=None):
    return JengaEnv(start_height_layers=5, max_steps=200, seed=seed, com_half_span=1.0)

@torch.no_grad()
def eval_once(policy, episodes=200, device="cpu"):
    def run(mode="greedy"):
        total = dict(R=0.0, moves=0.0, maxH=0.0, dCOM=0.0)
        for _ in range(episodes):
            e = make_env()
            s, info = e.reset()
            m = info.get("action_mask")
            done = False; ep_R=0.0; moves=0
            com0 = e._com_x_norm() if hasattr(e,"_com_x_norm") else 0.0
            while not done:
                if mode == "random":
                    legal = np.nonzero(m)[0]
                    a = int(np.random.choice(legal)) if len(legal) else 0
                else:
                    a = get_action(s, 0.0, m, policy, device)
                s, r, term, trunc, info2 = e.step(a)
                done = bool(term) or bool(trunc)
                m = info2.get("action_mask", m)
                ep_R += float(r); moves += 1
            total["R"] += ep_R
            total["moves"] += moves
            total["maxH"] += getattr(e, "height", 0)
            com1 = e._com_x_norm() if hasattr(e,"_com_x_norm") else 0.0
            total["dCOM"] += abs(com1 - com0) * (-1)
            e.close()
        for k in total: total[k] /= float(episodes)
        return total
    g = run("greedy"); r = run("random")
    return g, r

def mean_ci(x, alpha=0.05):
    x = np.array(x, dtype=np.float64)
    m = x.mean()
    se = x.std(ddof=1) / max(len(x)**0.5, 1.0)
    return m, 1.96*se

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes_per_seed", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tmp_env = make_env(seed=0)
    s0, info = tmp_env.reset()
    state_size = len(s0); action_size = tmp_env.action_space.n
    tmp_env.close()

    device = torch.device(args.device)
    policy = DQN(state_size, action_size).to(device)
    policy.load_state_dict(torch.load(args.ckpt, map_location=device))
    policy.eval()

    greedy_Rs = []; random_Rs = []
    for sd in range(args.seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        g, r = eval_once(policy, episodes=args.episodes_per_seed, device=device)
        greedy_Rs.append(g["R"]); random_Rs.append(r["R"])
        print(f"[seed {sd}] greedy R={g['R']:.2f} moves={g['moves']:.2f} maxH={g['maxH']:.2f} dCOM={g['dCOM']:.3f} | "
              f"random R={r['R']:.2f} moves={r['moves']:.2f} maxH={r['maxH']:.2f} dCOM={r['dCOM']:.3f}")

    gm, gci = mean_ci(greedy_Rs); rm, rci = mean_ci(random_Rs)
    print(f"\nGreedy R mean±95%CI: {gm:.2f} ± {gci:.2f}   (over {args.seeds} seeds x {args.episodes_per_seed} eps)")
    print(f"Random R mean±95%CI: {rm:.2f} ± {rci:.2f}")
    print(f"Greedy - Random: {(gm-rm):.2f}")

if __name__ == "__main__":
    main()
