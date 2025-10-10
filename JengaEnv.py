import gymnasium as gym
from gymnasium import spaces
import numpy as np

LEFT, MIDDLE, RIGHT = 0, 1, 2

def encode_action(block_id: int, place_slot: int) -> int:
    return block_id * 3 + place_slot

def decode_action(a: int) -> tuple[int, int]:
    return a // 3, a % 3

class JengaEnv(gym.Env):
    """
    Jenga environment with fixed-size observation and action spaces.
    Action masking is provided via info['action_mask'] on reset() and step().
    
    Observation: [ height_norm, (edge_center_i, dist_norm_i) for i in 0..53 ]
    Action: Discrete(162) -> (block_id ∈ [0,53], place_slot ∈ {0,1,2})
    A move = choose block, remove, place on top (one step).
    """
    metadata = {"render_modes": []}

    def __init__(self, start_height_layers: int = 5, max_steps: int = 200, seed: int | None = None):
        super().__init__()
        self.start_height_layers = start_height_layers
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # --- Spaces ---
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1 + 54*2,), dtype=np.float32)
        self.action_space = spaces.Discrete(54 * 3)

        # --- Properties that define the "board" ---
        # Edge/center label is static. Example pattern for a classic 3-per-layer stack:
        # We'll mark indices within each layer [3k, 3k+1, 3k+2] with [edge=1, center=0, edge=1]
        self.edge_center = np.zeros(54, dtype=np.int32)
        for layer in range(18):
            i0, i1, i2 = 3*layer, 3*layer + 1, 3*layer + 2
            self.edge_center[i0] = 1
            self.edge_center[i1] = 0
            self.edge_center[i2] = 1

        # Dynamic state
        self.height = None                 # integer layers currently built (0..54/3) but we’ll treat as blocks/3; you can choose
        self.distance = None               # int distance-from-top per block (0 means on top)
        self.steps = 0

        # Bookkeeping for reward shaping
        self.prev_height = None

    # ---------- Core Gym API ----------
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Initialize distances based on starting tower height
        # Let the top layer blocks have distance 0, next layer 1, etc.
        H_layers = self.start_height_layers
        self.height = H_layers
        self.prev_height = self.height
        self.steps = 0

        # Fill distance:
        # For the first H_layers*3 blocks, assign distances by layer-from-top
        self.distance = np.full(54, 99, dtype=np.int32)  # 99 means not yet in tower (below base)
        # Build from bottom to top: bottom layer has largest distance, top has distance 0.
        # We'll store distance-from-top = (top_index - layer_index)
        # More simply: for each layer ℓ=0..H_layers-1 from bottom, its distance = (H_layers-1 - ℓ)
        for layer in range(H_layers):
            d = H_layers - 1 - layer
            i0, i1, i2 = 3*layer, 3*layer+1, 3*layer+2
            self.distance[i0] = d
            self.distance[i1] = d
            self.distance[i2] = d

        obs = self._get_obs()
        info = {"action_mask": self._action_mask()}
        return obs, info

    def step(self, action: int):
        self.steps += 1
        block_id, place_slot = decode_action(action)

        illegal = self.distance[block_id] < 2
        fell = False
        height_gain = 0
        reward = -0.01  # step cost

        if illegal:
            reward += -10.0
        else:
            # Apply the move: remove block, place on top
            fell = self._apply_move(block_id, place_slot)

            if not fell:
                reward += 1.0
                # Height increases when we complete a new top layer of 3 blocks.
                # In this simple model, every successful placement increases height only
                # when we just finished a layer. We’ll approximate: if the placed block’s
                # distance becomes 0 and completes a set of three zeros → +0.1 and height += 1
                completed_now = self._update_height_if_completed()
                if completed_now:
                    height_gain = 1
                    reward += 0.1

        done = fell or (self.steps >= self.max_steps)
        if fell:
            reward += -1.0

        obs = self._get_obs()
        info = {"action_mask": self._action_mask()}

        return obs, reward, done, False, info

    # ---------- Helpers ----------
    def _get_obs(self):
        height_norm = np.array([self.height / 54.0], dtype=np.float32)
        dist_norm = np.clip(self.distance, 0, 54) / 54.0
        ec = self.edge_center.astype(np.float32)
        # Interleave (edge, dist) per block
        per_block = np.stack([ec, dist_norm], axis=1).reshape(-1).astype(np.float32)
        return np.concatenate([height_norm, per_block], dtype=np.float32)

    def _action_mask(self):
        # legal if distance >= 2
        legal_blocks = (self.distance >= 2)
        # Build mask of length 162 = 54*3
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for b in range(54):
            if legal_blocks[b]:
                for slot in (LEFT, MIDDLE, RIGHT):
                    mask[encode_action(b, slot)] = 1
        return mask

    def _apply_move(self, block_id: int, place_slot: int) -> bool:
        """
        Placeholder physics -> replace with your Unity bridge.
        Logic:
        - Removing a block near the top is safer than deep ones.
        - Edge blocks slightly riskier than center.
        - Higher towers are riskier.
        We compute a "collapse probability" and sample.
        If not collapsed:
          - Set removed block distance to 0 (placed on new top)
          - Increment all other distances by +1 (tower “shifted up”)
        """
        d = int(self.distance[block_id])
        is_edge = int(self.edge_center[block_id])

        # --- Collapse probability heuristic (replace later) ---
        # Base risk grows with normalized height and “depth” of the removed block.
        h_norm = min(self.height / 54.0, 1.0)
        depth_factor = np.tanh(d / 8.0)   # deeper pulls riskier, saturates
        edge_bonus = 0.05 if is_edge else 0.0

        # Tighten this as you like:
        p_collapse = 0.02 + 0.25*h_norm + 0.20*depth_factor + edge_bonus
        p_collapse = float(np.clip(p_collapse, 0.0, 0.95))

        if self.rng.random() < p_collapse:
            return True  # fell

        # --- Update distances after successful pull+place ---
        # Everyone moves "one deeper" because we build up
        self.distance[self.distance < 99] += 1

        # Placed block goes on very top
        self.distance[block_id] = 0

        # (Optional) slight effect of place_slot on stability could be modeled later

        return False

    def _update_height_if_completed(self) -> bool:
        """
        If we now have three blocks with distance==0 (a full new top layer), count it as a height gain.
        Then "relabel" them as the top layer and shift distances accordingly:
        - increment height
        - increase every block's distance by +1 so new placements start from 0 next step
        """
        zeros = np.where(self.distance == 0)[0]
        if len(zeros) >= 3:
            # Count a new layer. We won't enforce exact triplets by layer index in this simple placeholder.
            self.height += 1
            # Push all distances down by +1 to make room for next top
            self.distance[self.distance < 99] += 1
            # Keep the just-placed three as distance 0 to represent the new top
            # We’ll pick any three zeros (already at zero). Ensure 3 remain at zero:
            # First set all zeros to 1, then set three back to 0.
            z = np.where(self.distance == 1)[0]  # those that were zero before the +1
            if len(z) > 0:
                self.distance[z] = 1
            keep = zeros[:3]
            self.distance[keep] = 0
            return True
        return False