import numpy as np
from JengaEnv import JengaEnv, decode_action
from com import com as compute_COM
print("COM demo:", compute_COM([1,1,1, 1,1,1, 1,1,1]))

def show_mask_stats(mask):
    legal = np.nonzero(mask)[0]
    print(f"legal actions: {len(legal)} (of {len(mask)})")
    # sanity: no top-layer removals should appear
    tops = []
    for a in legal:
        layer, pos, place = decode_action(a)
        tops.append(layer)
    print(f"min removal layer: {min(tops)}, max removal layer: {max(tops)}")

env = JengaEnv(start_height_layers=5, max_steps=50, seed=0, com_half_span=1.0)

obs, info = env.reset()
mask = info["action_mask"]

print("obs len:", len(obs))                         # expect 56
print("height_norm:", obs[0], "com_x_norm:", obs[1])
show_mask_stats(mask)

# Roll a few random-legal steps; print COM and height changes
total_r = 0.0
for t in range(10):
    legal = np.nonzero(mask)[0]
    assert len(legal) > 0, "No legal actions — mask bug"
    a = int(np.random.choice(legal))
    layer, pos, place = decode_action(a)
    obs2, r, term, trunc, info = env.step(a)
    print(f"t={t:02d}  act=(layer={layer}, pos={pos}, place={place})  "
          f"r={r:+.3f}  H={env.height}  COMx_norm={(obs2[1]*2-1):+.3f}")
    total_r += r
    mask = info["action_mask"]
    k2_layers = [l for l in range(env.height) if int(env.occ[l].sum())==2]
    allows_k2 = any(decode_action(a)[0] in k2_layers for a in np.nonzero(info["action_mask"])[0])
    print("2-block layers:", k2_layers, "mask-allows-2blk?", allows_k2)
    print(f"depth={env.height-1-layer} k={int(env.occ[layer].sum())} comx={(obs[1]*2-1):+.3f}")
    if term or trunc: 
        print("done:", "terminated" if term else "truncated")
        break

print("total reward (rollout):", round(total_r, 3))
env.close()
