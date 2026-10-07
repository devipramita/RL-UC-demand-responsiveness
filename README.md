# RL-UC-demand-responsiveness

Cost and feasibility conceal demand-blind scheduling in reinforcement learning for unit commitment

Tabular Q-Learning, DQN and PPO are trained on the 51-generator EPRI AI-ccelerating Unit Commitment benchmark. The paper shows that optimality ratio and demand fulfilment cannot tell a scheduling policy from a constant fleet, and proposes *demand responsiveness* (within-episode Pearson correlation between committed generator count and net demand) as a diagnostic.

## Layout

```
notebooks/
  00_prepare_dataset_and_eda.ipynb            data split + EDA
  01_train_agents_and_window_ablation.ipynb   trains all agents, window ablation, 2x2 (runs top to bottom, ~40 min on CPU)
  02_evaluation_bootstrap_static_fleet.ipynb  evaluation, bootstrap, static-fleet control
  03_seed_replication_2x2.ipynb               3-seed replication of the 2x2
  archive/                                    original exploratory notebook 01 (not runnable top to bottom)
  saved_models_{dqn,ppo,qlearning}/           checkpoints
scripts/                                      Q-Learning retrain, Table 2, dispatch figures, checkpoint re-evaluation
results/                                      CSVs behind the paper's tables
figures/                                      dispatch and EDA figures
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # or requirements-lock.txt for exact versions (Python 3.13)
```

**Data.** Download the EPRI AI-ccelerating Unit Commitment starting kit *(add dataset link)* and place `Train_Data` at `starting_kit_ai-uc_v2_warm-up/Train_Data`. Run notebook 00 once; it writes `rl_ready_competition_data/`. Experiments use only its `train/` folder. Paths can be set with `UC_RAW_DATA` and `UC_DATA_ROOT`; scripts run from the repository root.
