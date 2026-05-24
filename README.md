# Mountain Car Continuous — SAC with Reward Shaping

University coursework for *Data & AI 6: Reinforcement Learning* (2025-2026,
Thomas More). The project trains a Soft Actor-Critic (SAC) agent on
`MountainCarContinuous-v0`, compares a default-reward baseline against a
hand-crafted dense reward function, and sweeps three learning rates to study
hyperparameter sensitivity. The whole pipeline uses Stable Baselines 3.

---

## The Problem

`MountainCarContinuous-v0` places a car at the bottom of a valley with a flag
at position ≥ 0.45. The engine is too weak to drive straight up; the agent
must build momentum by swinging back and forth. The default reward is +100 on
success and −0.1 × action² otherwise — sparse enough that random exploration
rarely stumbles onto the goal, making it a useful testbed for reward shaping.

State: `(position, velocity)`. Action: continuous force in `[−1, 1]`.

---

## Algorithm

A single algorithm is used throughout: **SAC (Soft Actor-Critic)**, an
off-policy actor-critic method that maximises a weighted sum of return and
policy entropy. SAC handles continuous action spaces cleanly and is a standard
choice for this environment. All experiments use the `MlpPolicy` from SB3 with
default network architecture.

No alternative algorithms were compared; the extension is hyperparameter
tuning rather than an algorithm swap.

---

## Experiments

### 1. Baseline
SAC with the default Gymnasium reward. Three independent trials, 200 000
timesteps each. Checkpoints saved every 10 000 steps. Training curves logged
to TensorBoard under `logs/baseline/`.

### 2. Custom Reward
The default reward is augmented with dense shaping signals so the agent
receives useful feedback before it ever reaches the flag:

| Component | Scale | Purpose |
|---|---|---|
| Position reward | `norm_pos × 1.5` | Encourages rightward progress |
| Velocity reward | `max(v, 0) × 0.5` | Rewards forward momentum |
| Height reward | `max(pos, 0) × 1.0` | Rewards climbing the slope |
| Near-goal bonus | `5.0 × (pos − 0.35)` when `pos > 0.35` | Extra gradient near the flag |
| Goal bonus | +20.0 when `pos ≥ 0.45` | Explicit terminal success signal |
| Energy penalty | `−action² × 0.02` | Discourages unnecessary force |
| Time penalty | −0.01 per step | Mild pressure to solve quickly |

Scales are configured in `config/config_custom.yaml`. Three trials, same
setup as baseline.

### 3. Extension — Learning Rate Sweep
With the custom reward fixed, SAC is trained at three learning rates:
`0.0001`, `0.0003`, `0.001`. One run per value, 200 000 timesteps each.
Results logged under `logs/extension/`.

---

## Project Structure

```
.
├── config/
│   ├── config_baseline.yaml      # Baseline SAC settings
│   ├── config_custom.yaml        # Custom reward + shaping scales
│   └── config_extension.yaml     # Learning rate sweep settings
├── src/
│   ├── train_baseline.py         # Train SAC with default reward (3 trials)
│   ├── train_custom_reward.py    # Train SAC with custom reward (3 trials)
│   ├── train_extension.py        # Learning rate sweep
│   ├── evaluate.py               # Evaluate a single saved model
│   ├── comparison.py             # Baseline vs. custom + extension comparison
│   ├── custom_env_wrapper.py     # Gymnasium RewardWrapper implementation
│   └── utils.py                  # YAML config loader
├── requirements.txt
└── RL_project.pdf                # Original assignment specification
```

`logs/` and `results/` are excluded from the repo via `.gitignore`. Run the
training scripts to regenerate them locally.

---

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Dependencies: `gymnasium==1.2.2`, `stable-baselines3==2.7.0`,
`tensorboard==2.19.0`, `numpy`, `PyYAML`.

---

## How to Run

All commands from the repo root.

```bash
# 1. Baseline — 3 trials with default reward
python src/train_baseline.py

# 2. Custom reward — 3 trials with reward shaping
python src/train_custom_reward.py

# 3. Extension — learning rate sweep (lr ∈ {0.0001, 0.0003, 0.001})
python src/train_extension.py

# Compare baseline vs. custom, then extension sweep
python src/comparison.py

# Evaluate a specific checkpoint (edit the path inside the script first)
python src/evaluate.py

# Monitor training curves
tensorboard --logdir logs/
```

---

## Results

Metrics reported by `comparison.py`:

- **Mean max position** reached per episode (averaged over 20 eval episodes)
- **Distance to goal** (0.45 − max pos, clipped to 0)
- **Success rate** (% of eval episodes where max pos ≥ 0.45)

The custom reward is expected to converge faster than the baseline because
the dense shaping signals provide a gradient even in failed episodes. The
learning rate sweep compares `0.0001`, `0.0003`, and `0.001`; SB3's SAC
default is `0.0003`.

Training artifacts are not committed to this repo. To reproduce results, run
the three training scripts and then `comparison.py`.

---

## Context

This is a coursework project for a university RL module. The code follows the
folder structure required by the assignment and is not intended as a
general-purpose library. Reward scales were tuned manually, there is no
cross-trial seeding, and no statistical significance tests were run.
