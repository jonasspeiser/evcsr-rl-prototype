# Permutation-Invariant Feature Extraction for Fleet Observations

## Motivation

When scaling from 5 to 50 vehicles, two structural problems emerge with a standard flat MLP policy.

**Positional encoding of vehicles.**
A flat MLP assigns a distinct set of weights to each position in the input vector.
If the observation concatenates vehicle states in the order they were spawned, the network treats the state at position 0 as fundamentally different from an identical state at position 3, even though the agent's recommendation should be the same in both cases.
When vehicles are spawned in different orders across episodes, the policy cannot generalise: it has learned associations tied to input positions rather than to vehicle properties.
This is the primary cause of policy collapse at 50 vehicles, where the ordering of vehicles across episodes is essentially random.

**Fixed observation space dimensionality.**
With named per-vehicle keys (`observable_ev_0`, `observable_ev_1`, …), the observation space shape is fixed at training time by the number of keys.
A model trained on 5 vehicles cannot be loaded into an environment with 50, because the input dimensionality changes.
Any experiment that varies fleet size therefore requires training a separate model from scratch.

Both problems motivate a representation that processes every vehicle with the same function and aggregates the results — an approach that is both permutation-invariant by construction and independent of the exact fleet size.

---

## Observation Space Design

To enable permutation-invariant processing while keeping the observation space shape fixed, the fleet observation is structured as a **grouped tensor format** rather than a flat per-vehicle dictionary.

The observation at each decision step contains three required keys and two optional keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `active_vehicle` | `(12,)` | Features of the vehicle filing the current charging request |
| `other_vehicles` | `(N_max, 12)` | Features of all other vehicles, zero-padded to fixed capacity `N_max` |
| `vehicle_mask` | `(N_max,)` | `1.0` for live rows in `other_vehicles`, `0.0` for padding |
| `simulation_time`* | `(1,)` | Current simulation time normalised to [0, 1] |
| `station_assignment_counts`* | `(4,)` | Per-station count of concurrently arriving vehicles, normalised to [0, 1] |

*Optional; included only when the corresponding key is passed in `obs_features` to `CustomEnv`.

`N_max` is a fixed capacity chosen at training time (typically equal to the fleet size).
Rows in `other_vehicles` beyond the number of currently live vehicles are filled with zeros, and `vehicle_mask` identifies which rows contain valid data.
As vehicles arrive at their destination or run out of battery, their rows are masked out (`vehicle_mask[i] = 0`), preventing the network from treating them as active context.

**Separating the active vehicle** into its own key rather than encoding its identity via a flag feature inside the shared vehicle vector has two advantages.
First, it removes one input dimension from the shared feature vector (no `is_active` flag needed).
Second, and more importantly, it ensures the extractor can handle the active vehicle through a dedicated code path without having to scan the observation for a flag value at runtime.

### Per-Vehicle Feature Vector

Each vehicle — whether active or not — is represented by 12 scalar features:

| Index | Feature | Range |
|-------|---------|-------|
| 0 | State of charge (SoC), normalised by 100 kWh maximum | [0, 1]; −1 if unavailable |
| 1 | Distance to destination, normalised by network diameter | [0, 1]; −1 if unavailable |
| 2–5 | Distance to each of the four charging stations, normalised | [0, 1]; −1 if unreachable |
| 6–10 | Last recommended action, one-hot encoded (5 classes) | {0, 1}^5 |
| 11 | Arrived at destination | {0, 1} |

The last action is encoded as a one-hot vector rather than as a single integer to avoid implying a false numeric ordering between charging stations.
Unavailable values (vehicle not yet spawned, station unreachable) are represented by −1, which lies outside the normal [0, 1] range and is therefore unambiguously distinguishable from valid observations.

---

## Feature Extractor Architecture

The feature extractor follows the **DeepSets** framework (Zaheer et al., 2017), which defines a family of permutation-invariant set functions of the form

$$f(X) = \rho\!\left(\sum_{x \in X} \phi(x)\right)$$

where $\phi$ is a per-element encoder applied identically to each set element and $\rho$ aggregates the resulting embeddings.
Any function that is invariant to the permutation of its inputs can be represented in this form.

The architecture consists of four steps:

**Step 1 — Shared encoder φ.**
A two-layer MLP is applied identically to every vehicle row (both the active vehicle and every row of `other_vehicles`):

$$\phi : \mathbb{R}^{12} \to \mathbb{R}^{h}, \quad \phi(x) = \mathrm{ReLU}\!\left(W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2\right)$$

where $h$ is the embedding dimension (default: 64).
Because the same weights $W_1, W_2$ are used for every vehicle, the encoder learns a vehicle-agnostic representation: the same state observed for any vehicle produces the same embedding.

**Step 2 — Active embedding.**
The active vehicle row is encoded directly, producing a single embedding vector:

$$z_\mathrm{active} = \phi(\texttt{active\_vehicle}) \in \mathbb{R}^h$$

**Step 3 — Context embedding.**
The `other_vehicles` rows are encoded with the same shared encoder, then mean-pooled over live slots only.
The `vehicle_mask` excludes padding rows and rows of vehicles that have already arrived or run out of battery:

$$z_\mathrm{context} = \frac{\displaystyle\sum_{i=1}^{N_\mathrm{max}} \mathrm{mask}_i \cdot \phi(\texttt{other\_vehicles}_i)}{\displaystyle\max\!\left(1,\;\sum_{i=1}^{N_\mathrm{max}} \mathrm{mask}_i\right)} \in \mathbb{R}^h$$

The denominator is clamped to at least 1 to handle the edge case where no context vehicles are live (e.g., a single-vehicle scenario or all other vehicles have already arrived).

**Step 4 — Output.**
The active and context embeddings are concatenated.
If optional observation keys are present, their values are appended directly (they are already normalised to [0, 1] and do not require further transformation):

$$z = [\,z_\mathrm{active} \;\|\; z_\mathrm{context} \;(\|\; \texttt{simulation\_time}) \;(\|\; \texttt{station\_assignment\_counts})\,] \in \mathbb{R}^{2h + \delta}$$

where $\delta \in \{0, 1, 4, 5\}$ depending on which optional keys are present.
This vector $z$ is passed to the PPO policy and value networks in place of the raw observation.

---

## Properties

**Permutation invariance.**
The output $z$ does not depend on the order in which vehicles appear in `other_vehicles`, because context vehicles are aggregated by mean pooling — an operation that is insensitive to ordering.
The active vehicle is handled through its own key rather than by searching for a flag value, so its representation is also independent of any vehicle ordering.

**Variable fleet size.**
Padding rows in `other_vehicles` are excluded from the context embedding via `vehicle_mask`.
The weight matrices $W_1$ and $W_2$ of the shared encoder are independent of $N_\mathrm{max}$, so the same trained model can be evaluated on any fleet size as long as $N_\mathrm{max}$ (the observation space capacity) remains constant across training and evaluation.

**Encoding separation.**
The output $z$ explicitly encodes *who is requesting a recommendation* ($z_\mathrm{active}$) separately from *what the fleet context looks like* ($z_\mathrm{context}$).
This structural split gives the policy two distinct, interpretable signals rather than a single undifferentiated aggregate.
Because the active vehicle always has its own dedicated encoding path, the quality of its representation is unaffected by the number of context vehicles.

---

## Limitations and Alternatives

**Mean-pooling dilution.**
As the number of context vehicles grows, each individual vehicle's embedding contributes only $1/N$ of the aggregate.
At very large fleet sizes, the context embedding converges toward a fleet average and may lose the ability to represent that a specific station is heavily loaded.
For the current task — where the primary decision signal is the active vehicle's own SoC and proximity to stations, with fleet context as a secondary signal — this is acceptable.
At fleet sizes above 50, an attention-based aggregation (Transformer encoder) could be considered as an alternative that weights each vehicle's contribution by its relevance to the current decision.

**Encoder cold-start.**
Training the system involves two coupled optimisation problems: learning a useful vehicle embedding and learning a policy on top of those embeddings.
Early in training, the randomly initialised encoder maps different observations to similar embeddings, producing a weak policy gradient.
In practice, the DeepSets approach requires more training steps to reach the same performance as a flat MLP, which can learn position-specific patterns from the first gradient update.

---

## Integration

The extractor is implemented as `EVChargingFeatureExtractor` in [`feature_extractor.py`](../feature_extractor.py), subclassing `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`.

It is activated in `train_model` and `train_and_evaluate` via the `use_custom_extractor=True` flag, which passes `policy_kwargs={"features_extractor_class": EVChargingFeatureExtractor}` to the SB3 algorithm constructor.
The extractor dimensions (`vehicle_embed_dim`, `encoder_hidden_dim`, both default 64) can be tuned by passing `features_extractor_kwargs` inside `policy_kwargs` directly.

---

## References

Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). *Deep Sets*. Advances in Neural Information Processing Systems, 30.
