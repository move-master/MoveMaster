import gymnasium as gym
from gymnasium import spaces
import numpy as np

from com import com as compute_COM

LEFT, MIDDLE, RIGHT = 0, 1, 2

def encode_action(block_id: int, place_slot: int) -> int:
    return block_id * 3 + place_slot

def decode_action(a: int) -> tuple[int, int, int]:
    block_id = a // 3
    place_slot = a % 3
    layer, pos = divmod(block_id, 3)
    return layer, pos, place_slot

class JengaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        start_height_layers: int = 5,
        max_steps: int = 200,
        seed: int | None = None,
        com_half_span: float = 1.0):
        super().__init__()
        assert 1 <= start_height_layers <= 18
        self.start_height_layers = start_height_layers
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.com_half_span = float(com_half_span)  
        self.gamma_rl = 0.99
        self.alpha_phi = 1.0

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(56,), dtype=np.float32)
        self.action_space = spaces.Discrete(54 * 3)

        self.occ: np.ndarray | None = None
        self.height: int = 0
        self.steps: int = 0

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.height = self.start_height_layers
        self.occ = np.zeros((18, 3), dtype=np.uint8)
        for l in range(self.height):
            self.occ[l, :] = 1
        return self._get_obs(), {"action_mask": self._action_mask()}

    def step(self, action: int):
        assert self.occ is not None
        self.steps += 1

        reward = -0.01
        done = False
        truncated = False

        mask = self._action_mask()
        if mask.sum() == 0:
            reward += -1.0
            done = True
            return self._get_obs(), reward, done, truncated, {"action_mask": mask}
        illegal_by_mask = not (0 <= action < self.action_space.n and mask[action] == 1)

        layer, pos, place_slot = decode_action(action)
        top = self.height - 1 if self.height > 0 else -1
        k_layer = int(self.occ[layer].sum()) if (0 <= layer < self.height) else 0
        d = top - layer

        guard_illegal = (
            (self.height <= 0) or
            (layer < 0 or layer >= self.height) or
            (d == 0) or
            (d == 1 and k_layer < 3) or
            (k_layer == 2) or
            (self.occ[layer, pos] == 0)
        )
        if illegal_by_mask or guard_illegal:
            reward += -10.0
            if self.steps >= self.max_steps:
                truncated = True
            return self._get_obs(), reward, done, truncated, {"action_mask": mask}

        phi_before = -abs(self._com_x_norm())
        fell, height_gained = self._apply_move(layer, pos, place_slot)

        if fell:
            reward += -1.0
            done = True
        else:
            reward += 1.0
            if height_gained:
                reward += 0.1

            phi_after = -abs(self._com_x_norm())
            shaping = self.gamma_rl * phi_after - phi_before
            reward += self.alpha_phi * shaping

        if self.steps >= self.max_steps and not done:
            truncated = True

        return self._get_obs(), reward, done, truncated, {"action_mask": self._action_mask()}

    def _get_obs(self) -> np.ndarray:
        height_norm = np.float32(self.height / 18.0)
        com_x_norm = np.float32((self._com_x_norm() + 1.0) / 2.0)  # [-1,1] -> [0,1]
        occ_flat = self.occ.astype(np.float32).reshape(-1)
        return np.concatenate(([height_norm, com_x_norm], occ_flat), dtype=np.float32)

    def _placement_layer_index(self) -> int:
        if self.height <= 0:
            return 0
        top = self.height - 1
        return top if self.occ[top].sum() < 3 else self.height

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self.height <= 0:
            return mask

        top_layer = self.height - 1
        place_layer = self._placement_layer_index()

        if place_layer < self.height:
            open_placements = tuple(int(i) for i in np.where(self.occ[place_layer] == 0)[0])
        else:
            open_placements = (LEFT, MIDDLE, RIGHT)

        for l in range(self.height):
            d = top_layer - l
            k = int(self.occ[l].sum())

            if d == 0: continue                     
            if d == 1 and k < 3: continue

            for pos in (0, 1, 2):
                if self.occ[l, pos] == 0:
                    continue
                block_id = l * 3 + pos
                for place in open_placements:
                    a = encode_action(block_id, place)
                    mask[a] = 1
        return mask

    def _apply_move(self, layer: int, pos: int, place_slot: int) -> tuple[bool, bool]:
        top = self.height - 1
        d = top - layer
        denom = max(self.height - 1, 1)
        depth_frac = d / denom
        risk = 0.02 + 0.30 * (self.height / 18.0) + 0.25 * depth_frac + 0.60 * abs(self._com_x_norm())
        k_layer = int(self.occ[layer].sum())
        if k_layer == 2:
            side = -1 if pos == 0 else (0 if pos == 1 else 1)
            com = self._com_x_norm()
            heavy_side = 1 if com > 0 else (-1 if com < 0 else 0)
            align = 1.0 if (side != 0 and side == heavy_side) else 0.0
            two_mid_bonus = 0.25 if side == 0 else 0.0
            two_edge_bonus = 0.15 * align
            depth_amp = 0.5 + 0.5 * depth_frac
            risk += depth_amp * (two_mid_bonus + two_edge_bonus)
        risk = float(np.clip(risk, 0.0, 0.95))
        if self.rng.random() < risk:
            return True, False

        self.occ[layer, pos] = 0

        top = self.height - 1
        if self.occ[top].sum() == 3 and self.height < 18:
            self.height += 1
            self.occ[self.height - 1, :] = 0
            top = self.height - 1

        self.occ[top, place_slot] = 1

        height_gained = False
        if self.occ[top].sum() == 3 and self.height < 18:
            self.height += 1
            self.occ[self.height - 1, :] = 0
            height_gained = True

        return False, height_gained

    def _com_vec(self) -> np.ndarray:
        if self.height <= 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bits = self.occ[: self.height].reshape(-1).astype(int).tolist()
        vec = compute_COM(bits)
        v = np.asarray(vec, dtype=np.float32)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v

    def _com_x_norm(self) -> float:
        x = float(self._com_vec()[0])
        span = max(self.com_half_span, 1e-6)
        return float(np.clip(x / span, -1.0, 1.0))

    def render(self): ...
    def close(self): ...