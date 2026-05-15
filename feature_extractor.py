import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EVChargingFeatureExtractor(BaseFeaturesExtractor):
    """DeepSets-inspired feature extractor for the EV charging recommendation environment.

    Applies a shared encoder identically to every vehicle row, then aggregates:
    - active_emb:  embedding of the vehicle requesting a recommendation (from "active_vehicle" key)
    - context_emb: mean-pooled embeddings of other spawned vehicles (from "other_vehicles" key,
                   masked by "vehicle_mask")

    Output: [active_emb || context_emb (|| simulation_time) (|| station_assignment_counts)]
    Size:    2 * vehicle_embed_dim  (+1 if simulation_time present, +4 if station_assignment_counts present)

    Permutation invariant by construction: vehicle row order in other_vehicles does not affect output.
    Padding rows (vehicle_mask == 0) are excluded from context aggregation.

    Required observation space keys:
        "active_vehicle": Box(12,)              — the vehicle that filed the charging request
        "other_vehicles": Box(max_vehicles, 12) — all other spawned vehicles, zero-padded
        "vehicle_mask":   Box(max_vehicles,)    — 1 for active rows, 0 for padding

    Optional observation space keys (detected automatically, concatenated directly to output if present):
        "simulation_time":           Box(1,) — current simulation time normalized to [0, 1]
        "station_assignment_counts": Box(4,) — per-station congestion counts normalized to [0, 1]

    Feature layout per vehicle row (12 features):
        0:    battery_soc             [0, 1]; -1 if unavailable
        1:    distance_to_destination [0, 1]; -1 if unavailable
        2–5:  distance_to_cs_1..4     [0, 1]; -1 if unreachable
        6–10: last_action one-hot     {0,1} x 5 (0=do_nothing, 1–4=charging station)
        11:   arrived_at_destination  {0, 1}
    """

    FEATURES_PER_VEHICLE = 12

    def __init__(
        self,
        observation_space: spaces.Dict,
        vehicle_embed_dim: int = 64,
        encoder_hidden_dim: int = 64,
    ):
        extra_dim = sum(
            observation_space.spaces[k].shape[0]
            for k in ("simulation_time", "station_assignment_counts")
            if k in observation_space.spaces
        )
        super().__init__(observation_space, features_dim=2 * vehicle_embed_dim + extra_dim)
        self.vehicle_embed_dim = vehicle_embed_dim
        self._has_simulation_time = "simulation_time" in observation_space.spaces
        self._has_station_counts = "station_assignment_counts" in observation_space.spaces

        self.encoder = nn.Sequential(
            nn.Linear(self.FEATURES_PER_VEHICLE, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, vehicle_embed_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        active = observations["active_vehicle"]          # (batch, 12)
        others = observations["other_vehicles"]          # (batch, max_vehicles, 12)
        batch_size, n_slots, _ = others.shape

        active_emb = self.encoder(active)                # (batch, vehicle_embed_dim)

        flat = others.view(batch_size * n_slots, self.FEATURES_PER_VEHICLE)
        encoded_others = self.encoder(flat).view(batch_size, n_slots, self.vehicle_embed_dim)

        mask = observations["vehicle_mask"].unsqueeze(-1)   # (batch, n_slots, 1)
        context_emb = (encoded_others * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        parts = [active_emb, context_emb]
        if self._has_simulation_time:
            parts.append(observations["simulation_time"])
        if self._has_station_counts:
            parts.append(observations["station_assignment_counts"])
        return torch.cat(parts, dim=-1)
