"""Permutation-invariant feature extractor for the EV Charging Station Recommendation environment.

Implements a DeepSets-inspired architecture (Zaheer et al., 2017) that:
- Processes each vehicle observation independently with a shared encoder (permutation invariance)
- Handles variable-length fleets by masking padded observation slots
- Optionally incorporates global context features (simulation time, station assignment counts)
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Feature layout per vehicle observation vector (see environment.py as source of truth):
#   [0]    battery SoC              normalized to [0, 1]; -1 for unspawned/unavailable
#   [1]    distance to destination  normalized to [0, 1]; -1 for unspawned/unavailable
#   [2:6]  distances to cs_1..4    normalized to [0, 1]; -1 for unreachable
#   [6:11] last_action one-hot      {0,1} × 5  (0=do_nothing, 1–4=charging station)
#   [11]   arrived at destination   {0, 1}
FEATURES_PER_VEHICLE = 12


class VehicleSetExtractor(BaseFeaturesExtractor):
    """DeepSets-inspired feature extractor for the EV charging recommendation environment.

    The environment exposes a gymnasium Dict observation with three required keys and two
    optional keys:
        "active_vehicle":             Box(12,)              — vehicle filing the current request
        "other_vehicles":             Box(max_vehicles, 12) — all other vehicles, zero-padded
        "vehicle_mask":               Box(max_vehicles,)    — 1.0 for live rows, 0.0 for padding
        "simulation_time"*:           Box(1,)               — simulation time in [0, 1]
        "station_assignment_counts"*: Box(4,)               — per-station congestion in [0, 1]
    (* optional; present only when the corresponding key is in obs_features in CustomEnv)

    Architecture
    ------------
    1. **Shared encoder φ**: a small MLP with shared weights is applied identically to
       every vehicle row (active and other), producing embeddings in R^vehicle_embed_dim.
       Shared weights are what makes the representation permutation-invariant.
    2. **Active embedding**: the active vehicle row is encoded directly → R^vehicle_embed_dim.
    3. **Context embedding**: the other_vehicles rows are encoded, then masked mean-pooled
       over live slots (vehicle_mask == 1.0) → R^vehicle_embed_dim.
    4. **Global features**: if present, simulation_time and/or station_assignment_counts
       are concatenated directly to the output (already normalized to [0, 1]).
    5. **Output**: [active_emb ‖ context_emb (‖ simulation_time) (‖ station_assignment_counts)]
       — size 2 * vehicle_embed_dim (+1 if simulation_time, +4 if station_assignment_counts).

    Properties
    ----------
    - **Permutation-invariant**: vehicle row order in other_vehicles does not affect the
      output, because context vehicles are aggregated by mean pooling.
    - **Variable-fleet-capable**: padded slots are excluded from aggregation via vehicle_mask,
      so the same model can be evaluated on fleets larger than those seen during training
      (as long as max_vehicles in the observation space stays constant).

    Args:
        observation_space (spaces.Dict): The observation space of the environment.
        vehicle_embed_dim (int): Dimension of the per-vehicle embedding. The base output
            size is 2 * vehicle_embed_dim, plus extra dimensions for any optional keys.
            Defaults to 64.
        encoder_hidden_dim (int): Width of the hidden layer in the shared encoder MLP.
            Defaults to 64.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        vehicle_embed_dim: int = 64,
        encoder_hidden_dim: int = 64,
    ):
        # Detect optional global-context keys and account for their dimensions so SB3
        # sizes the policy head correctly.
        extra_dim = sum(
            observation_space.spaces[k].shape[0]
            for k in ("simulation_time", "station_assignment_counts")
            if k in observation_space.spaces
        )
        super().__init__(observation_space, features_dim=2 * vehicle_embed_dim + extra_dim)
        self.vehicle_embed_dim = vehicle_embed_dim
        self._has_simulation_time = "simulation_time" in observation_space.spaces
        self._has_station_counts = "station_assignment_counts" in observation_space.spaces

        # Shared encoder φ: R^12 → R^vehicle_embed_dim.
        # Identical weights are applied to every vehicle row (active and other),
        # which is what makes the representation permutation-invariant.
        self.encoder = nn.Sequential(
            nn.Linear(FEATURES_PER_VEHICLE, encoder_hidden_dim),
            nn.Tanh(), # Chose Tanh over ReLU because the input contains -1 for unspawned vehicles/unreachable stations. ReLU would map these to zero, collapsing the distinction between "unavailable" and "zero" after the first layer.
            nn.Linear(encoder_hidden_dim, vehicle_embed_dim),
            nn.Tanh(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """Extract a fixed-size embedding from the fleet observation.

        Args:
            observations: Dict of tensors as provided by SB3 to a custom feature extractor.
                Required keys: "active_vehicle" [batch, 12], "other_vehicles" [batch, max_vehicles, 12],
                "vehicle_mask" [batch, max_vehicles]. Optional: "simulation_time", "station_assignment_counts".

        Returns:
            Tensor of shape [batch, features_dim].
        """
        active = observations["active_vehicle"]          # (batch, 12)
        others = observations["other_vehicles"]          # (batch, max_vehicles, 12)
        batch_size, n_slots, _ = others.shape

        # --- Active vehicle embedding ---
        active_emb = self.encoder(active)                # (batch, vehicle_embed_dim)

        # --- Context embedding ---
        # Encode all other-vehicle slots with the same shared encoder (flatten to apply
        # the linear layers, then reshape back), then mean-pool over live slots only.
        flat = others.view(batch_size * n_slots, FEATURES_PER_VEHICLE)
        encoded_others = self.encoder(flat).view(batch_size, n_slots, self.vehicle_embed_dim)

        # Clamp denominator to 1.0 to handle the edge case where no context vehicles
        # are live (e.g., single-vehicle scenario or all others have already arrived).
        mask = observations["vehicle_mask"].unsqueeze(-1)   # (batch, n_slots, 1)
        context_emb = (encoded_others * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # --- Assemble output ---
        # Base: [active_emb ‖ context_emb]. Append any optional global-context features,
        # which are already normalized to [0, 1] and concatenated directly (no extra MLP).
        parts = [active_emb, context_emb]
        if self._has_simulation_time:
            parts.append(observations["simulation_time"])
        if self._has_station_counts:
            parts.append(observations["station_assignment_counts"])
        return torch.cat(parts, dim=-1)
