# smoke_env.py
import numpy as np
from JengaEnv import JengaEnv
env = JengaEnv(start_height_layers=5, max_steps=200, seed=0)
obs, info = env.reset()
mask = info["action_mask"]
assert obs.shape[0] == 109, f"obs size {obs.shape[0]} != 109"
assert mask.shape[0] == 162, f"mask size {mask.shape[0]} != 162"

total, steps = 0.0, 0
while True:
    legal = np.nonzero(mask)[0]
    assert len(legal) > 0, "No legal actions available!"
    a = int(np.random.choice(legal))
    obs, r, term, trunc, info = env.step(a)
    mask = info["action_mask"]
    total += r; steps += 1
    if term or trunc:
        break

print(f"Smoke OK — steps={steps}, return={total:.2f}")