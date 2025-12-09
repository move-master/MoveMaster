import argparse, json, sys
import numpy as np
import torch
import torch.nn as nn

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

def load_weights(policy: nn.Module, ckpt_path: str, device):
    blob = torch.load(ckpt_path, map_location=device)
    if isinstance(blob, dict):
        if all(k.startswith(("feature.","adv.","val.")) for k in blob.keys()):
            policy.load_state_dict(blob)
        elif "policy" in blob and isinstance(blob["policy"], dict):
            policy.load_state_dict(blob["policy"])
        else:
            raise RuntimeError(
                "Unrecognized checkpoint format. Expected a state_dict with "
                "feature/adv/val keys or a dict containing 'policy'."
            )
    else:
        raise RuntimeError("Checkpoint is not a dict.")

def masked_topk(q: np.ndarray, mask: np.ndarray, k: int):
    q = q.copy()
    if mask is not None:
        q[mask == 0] = -1e30
    idx = np.argsort(-q)[:k]
    return [(int(i), float(q[i])) for i in idx]

def parse_vec(s: str):
    s = s.strip()
    if s.startswith("["):
        return np.array(json.loads(s), dtype=np.float32)
    parts = [p for p in s.replace(",", " ").split() if p]
    return np.array([float(x) for x in parts], dtype=np.float32)

def load_mask(path: str, action_size: int):
    with open(path, "r") as f:
        data = json.load(f)
    arr = np.array(data, dtype=np.int32)
    if arr.size != action_size:
        raise ValueError(f"Mask length {arr.size} != action_size {action_size}")
    return arr

def run_probe_env(policy, device, action_size, topk):
    from JengaEnv import JengaEnv
    
    env = JengaEnv(start_height_layers=5, max_steps=200, seed=None, com_half_span=1.0)
    state, info = env.reset()
    mask = info.get("action_mask", np.ones(action_size, dtype=np.int32))
    state = np.array(state, dtype=np.float32)

    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy(s).cpu().numpy()[0]
    tops = masked_topk(q, mask, topk)

    print("\n[Probe] grabbed a real env state; you can paste this into interactive mode:")
    print(json.dumps([float(f) for f in state.tolist()]))
    if mask is not None:
        print(f"[Probe] mask has {int(mask.sum())}/{len(mask)} legal actions.")

    print("\nTop-5 actions (from probe):")
    for rank, (a, qv) in enumerate(tops, 1):
        print(f"  #{rank}: action={a:3d}  Q={qv:.6f}")
    env.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--topk", type=int, default=5, help="How many actions to show")
    p.add_argument("--state_size", type=int, default=56)
    p.add_argument("--action_size", type=int, default=162)
    p.add_argument("--mask", type=str, default=None, help="Optional JSON mask (0/1 x action_size)")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--probe-env", dest="probe_env", action="store_true",
                   help="Grab a real env state and show suggested moves")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    policy = DQN(args.state_size, args.action_size).to(device).eval()
    load_weights(policy, args.ckpt, device)
    print("\nModel loaded.", "Using CPU" if device.type == "cpu" else "Using CUDA")

    if args.probe_env:
        run_probe_env(policy, device, args.action_size, args.topk)
        return

    mask = None
    if args.mask:
        mask = load_mask(args.mask, args.action_size)

    print("\nPaste a state vector of length", args.state_size)
    print(" - Accepts comma- or space-separated floats, or a JSON list.")

    while True:
        try:
            line = input("state> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        try:
            vec = parse_vec(line)
        except Exception as e:
            print("Could not parse:", e)
            continue

        if vec.size != args.state_size:
            print(f"Got {vec.size} values, expected {args.state_size}. Try again.")
            continue

        with torch.no_grad():
            s = torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
            q = policy(s).cpu().numpy()[0]

        tops = masked_topk(q, mask, args.topk)
        print("Top-{} actions:".format(args.topk))
        for rank, (a, qv) in enumerate(tops, 1):
            print(f"  #{rank}: action={a:3d}  Q={qv:.6f}")

if __name__ == "__main__":
    main()
