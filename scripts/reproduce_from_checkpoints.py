"""Re-evaluate the checkpoint-based results of the paper (Table 4 single-seed values and Section 5.6)
from the shipped checkpoints. Evaluation is deterministic, so these numbers reproduce exactly.
Run from the repository root:  python scripts/reproduce_from_checkpoints.py"""
import os, time, json, gzip, re, pickle, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, vstack as sp_vstack
from scipy.stats import pearsonr

t_script_start = time.time()
ED_LEVELS = [round(x * 0.05, 2) for x in range(1, 21)]

def _first_existing(folder, names, required=True):
    folder = Path(folder)
    for n in names:
        if (folder / n).exists(): return folder / n
    if required: raise FileNotFoundError(f"None of {names} in {folder}")
    return None

def _read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f: return json.load(f)

def _norm_col(df, name):
    df = df.copy()
    if len(df.columns) > 0 and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.rename(columns={df.columns[0]: name})
    return df

def _load_profiles_and_ic(folder):
    folder = Path(folder)
    if (folder / "profiles.csv").exists():
        return (_norm_col(pd.read_csv(folder/"profiles.csv"), "hour"),
                _norm_col(pd.read_csv(folder/"initial_conditions.csv"), "generator"))
    xls = pd.ExcelFile(folder/"explanatory_variables.xlsx")
    def find(kws):
        for k in kws:
            for s in xls.sheet_names:
                if all(x.lower() in s.lower() for x in k): return s
        return xls.sheet_names[0]
    return (_norm_col(pd.read_excel(folder/"explanatory_variables.xlsx",
                                    sheet_name=find([("profile",)])), "hour"),
            _norm_col(pd.read_excel(folder/"explanatory_variables.xlsx",
                                    sheet_name=find([("initial",)])), "generator"))

def get_profile_series(df):
    def fc(kws):
        for c in df.columns:
            if all(k in str(c).lower() for k in kws): return c
        return None
    d = pd.to_numeric(df[fc(["demand"])], errors="coerce").fillna(0).to_numpy(float)
    w = pd.to_numeric(df[fc(["wind"])], errors="coerce").fillna(0).to_numpy(float) if fc(["wind"]) else np.zeros_like(d)
    s = pd.to_numeric(df[fc(["solar"])], errors="coerce").fillna(0).to_numpy(float) if fc(["solar"]) else np.zeros_like(d)
    return d, w, s, d - w - s

def list_instance_dirs(root):
    dirs = sorted(p for p in Path(root).rglob("instance_*") if p.is_dir())
    if not dirs: raise FileNotFoundError(f"No instance_* under {root}")
    return dirs

def split_instance_dirs(dirs, val_quarters=None):
    val_quarters = val_quarters or set()
    train, valid = [], []
    for d in dirs:
        m = re.match(r"instance_(\d{4})_(Q[1-4])_(\d+)", d.name)
        if m and (int(m.group(1)), m.group(2)) in val_quarters: valid.append(d)
        else: train.append(d)
    return train, valid

def load_single_bundle(d):
    d = Path(d)
    profiles, ic = _load_profiles_and_ic(d)
    inp = _read_json(_first_existing(d, ["InputData.json", "InputData.json.gz"]))
    outp_path = _first_existing(d, ["OutputData.json", "OutputData.json.gz"], required=False)
    outp = _read_json(outp_path) if outp_path else None
    return {"instance_id": d.name, "profiles": profiles, "initial_conditions": ic,
            "input_json": inp, "output_json": outp}

def _as_float_array(v):
    if v is None: return np.array([], float)
    return pd.to_numeric(pd.Series(v if isinstance(v, list) else [v]),
                          errors="coerce").dropna().to_numpy(float)

def approx_marginal_cost(mw, cost):
    mw, cost = _as_float_array(mw), _as_float_array(cost)
    if len(mw) < 2: return 0.0
    return float((cost[1] - cost[0]) / max(mw[1] - mw[0], 1e-9))

def parse_generators(inp):
    gen = pd.DataFrame.from_dict(inp["Generators"], orient="index").reset_index().rename(columns={"index": "generator"})
    gen["Type_clean"] = gen["Type"].str.strip().str.lower()
    th = gen[gen["Type_clean"] == "thermal"].copy()
    if th.empty: raise ValueError("No thermal generators")
    th["cost_curve_mw"] = th["Production cost curve (MW)"].apply(_as_float_array)
    th["cost_curve_usd"] = th["Production cost curve ($)"].apply(_as_float_array)
    th["pmin"] = th["cost_curve_mw"].apply(lambda x: float(x[0]) if len(x) else 0.0)
    th["pmax"] = th["cost_curve_mw"].apply(lambda x: float(x[-1]) if len(x) else 0.0)
    th["marginal_cost"] = th.apply(lambda r: approx_marginal_cost(r["cost_curve_mw"], r["cost_curve_usd"]), axis=1)
    th["startup_cost"] = th["Startup costs ($)"].apply(
        lambda x: float(_as_float_array(x)[0]) if len(_as_float_array(x)) else 0.0)
    for src, tgt in [("Initial power (MW)", "init_power"), ("Initial status (h)", "init_status"),
                      ("Minimum uptime (h)", "min_up"), ("Minimum downtime (h)", "min_down"),
                      ("Ramp up limit (MW)", "ramp_up"), ("Ramp down limit (MW)", "ramp_down")]:
        th[tgt] = pd.to_numeric(th.get(src, 0), errors="coerce").fillna(0.0)
    th = th.sort_values(["marginal_cost", "startup_cost", "pmax"]).reset_index(drop=True)
    return th[["generator", "cost_curve_mw", "cost_curve_usd", "pmin", "pmax",
               "marginal_cost", "startup_cost", "init_power", "init_status",
               "min_up", "min_down", "ramp_up", "ramp_down"]]

def get_balance_penalty(inp, default=12533.0):
    params = inp.get("Parameters", {})
    for k in ["Power balance penalty ($/MW)", "Power balance penalty"]:
        if k in params:
            v = pd.to_numeric(pd.Series([params[k]]), errors="coerce").iloc[0]
            if pd.notna(v): return float(v)
    return default

def bundle_to_episode(b):
    th = parse_generators(b["input_json"])
    d, w, s, nd = get_profile_series(b["profiles"])
    return {"instance_id": b["instance_id"], "thermal_df": th,
            "demand": d, "wind": w, "solar": s, "net_demand": nd,
            "output_json": b["output_json"],
            "balance_penalty": get_balance_penalty(b["input_json"])}

def validate_compat(eps):
    ref = eps[0]["thermal_df"]["generator"].tolist()
    for e in eps[1:]:
        if e["thermal_df"]["generator"].tolist() != ref:
            raise ValueError(f"Mismatch: {e['instance_id']}")
    return True

def load_episodes(dirs, max_n=None, seed=42, verbose=True):
    dirs = list(dirs)
    np.random.default_rng(seed).shuffle(dirs)
    if max_n: dirs = dirs[:max_n]
    eps, fails = [], []
    for d in dirs:
        try: eps.append(bundle_to_episode(load_single_bundle(d)))
        except Exception as e: fails.append((str(d), repr(e)))
    if not eps: raise RuntimeError(f"No episodes. Failures: {fails[:3]}")
    validate_compat(eps)
    if verbose:
        print(f"Loaded {len(eps)} | Failed {len(fails)} | "
              f"Generators: {len(eps[0]['thermal_df'])} | Horizon: {len(eps[0]['demand'])}h")
    return eps

print("Loading data...")
DATA_ROOT = Path(os.environ.get("UC_DATA_ROOT", "rl_ready_competition_data"))
TRAIN_ROOT = DATA_ROOT / "train"
N_TRAIN, N_VALID, N_TEST = 300, 25, 100

all_dirs = list_instance_dirs(TRAIN_ROOT)
train_dirs, valid_dirs = split_instance_dirs(all_dirs, {(2026, "Q4"), (2027, "Q4"), (2028, "Q4"), (2029, "Q4")})
train_episodes = load_episodes(train_dirs, max_n=N_TRAIN, seed=42)
valid_episodes = load_episodes(valid_dirs, max_n=N_VALID, seed=43)
used_ids = {e["instance_id"] for e in train_episodes} | {e["instance_id"] for e in valid_episodes}
test_dirs = [d for d in all_dirs if d.name not in used_ids]
test_episodes = load_episodes(test_dirs, max_n=N_TEST, seed=99)
assert len(test_episodes) == 100

class UCOnlyEnv:
    def __init__(self, episodes, spill_penalty=8000.0, reward_scale=1e5,
                 random_reset=True, seed=42, lookahead=8,
                 state_version="v2", obs_window=72):
        validate_compat(episodes)
        self.episodes = list(episodes)
        self.spill_penalty = spill_penalty
        self.reward_scale = reward_scale
        self.random_reset = random_reset
        self.rng = np.random.default_rng(seed)
        self.lookahead = lookahead
        self.state_version = state_version
        self.obs_window = int(obs_window)

        th = episodes[0]["thermal_df"].reset_index(drop=True)
        self.generator_names = th["generator"].tolist()
        self.n_gen = len(th)
        self.horizon = len(episodes[0]["demand"])
        self.pmin = th["pmin"].to_numpy(float)
        self.pmax = th["pmax"].to_numpy(float)
        self.marginal_cost = th["marginal_cost"].to_numpy(float)
        self.startup_cost_arr = th["startup_cost"].to_numpy(float)
        self.min_up = th["min_up"].to_numpy(float)
        self.min_down = th["min_down"].to_numpy(float)
        self.ramp_up = th["ramp_up"].to_numpy(float)
        self.ramp_down = th["ramp_down"].to_numpy(float)
        self.cost_curve_mw = th["cost_curve_mw"].tolist()
        self.cost_curve_usd = th["cost_curve_usd"].tolist()
        self.thermal_cap = float(self.pmax.sum())

        self._mu_scale = max(float(self.min_up.max()), 1.0)
        self._md_scale = max(float(self.min_down.max()), 1.0)

        if self.state_version == "v1":
            self.obs_dim = 1 + self.n_gen + self.horizon + self.n_gen + 4
        else:
            self.obs_dim = 3 + self.obs_window + 4 + 3 * self.n_gen

        self.reset(options={"episode_index": 0})

    @property
    def obs_layout(self):
        n, W = self.n_gen, self.obs_window
        if self.state_version == "v1":
            H = self.horizon
            return {"time": (0, 1), "prev_on": (1, 1 + n),
                    "net_demand": (1 + n, 1 + n + H),
                    "status": (1 + n + H, 1 + n + H + n),
                    "lookahead": (1 + n + H + n, 1 + n + H + n + 4)}
        o = 0
        lay = {}
        lay["time"] = (o, o + 3); o += 3
        lay["window"] = (o, o + W); o += W
        lay["lookahead"] = (o, o + 4); o += 4
        lay["on_off"] = (o, o + n); o += n
        lay["can_start"] = (o, o + n); o += n
        lay["can_stop"] = (o, o + n); o += n
        return lay

    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        options = options or {}
        idx = int(options.get("episode_index",
                  self.rng.integers(len(self.episodes)) if self.random_reset else 0))
        self.ep_idx = idx
        ep = self.episodes[idx]
        self.demand = np.asarray(ep["demand"], float)
        self.wind = np.asarray(ep["wind"], float)
        self.solar = np.asarray(ep["solar"], float)
        self.net_demand = self.demand - self.wind - self.solar
        self.balance_penalty = float(ep.get("balance_penalty", 12533.0))

        th = ep["thermal_df"]
        self.current_dispatch = th["init_power"].to_numpy(float).copy()
        self.status_duration = th["init_status"].to_numpy(float).copy()
        self.prev_on = (self.current_dispatch > 1e-6).astype(int)
        self.t = 0
        return self._obs(), {"instance_id": ep["instance_id"]}

    def _constraint_clocks(self):
        sd = self.status_duration
        is_on = (self.current_dispatch > 1e-6) | (sd > 0)
        hours_off = np.where(sd < 0, -sd, 0.0)
        can_start = np.where(is_on, 0.0, np.maximum(0.0, self.min_down - hours_off))
        hours_on = np.where(sd > 0, sd, 0.0)
        can_stop = np.where(~is_on, 0.0, np.maximum(0.0, self.min_up - hours_on))
        return (can_start / self._md_scale, can_stop / self._mu_scale, is_on.astype(float))

    def _demand_window(self):
        cap = max(self.thermal_cap, 1.0)
        W = self.obs_window
        seg = self.net_demand[self.t:self.t + W] / cap
        if len(seg) < W:
            pad = np.full(W - len(seg), seg[-1] if len(seg) else 0.0)
            seg = np.concatenate([seg, pad])
        return seg.astype(np.float32)

    def _lookahead_stats(self):
        cap = max(self.thermal_cap, 1.0)
        L = self.lookahead
        window = self.net_demand[self.t:min(self.t + L, self.horizon)]
        if len(window) < 2:
            return np.zeros(4, dtype=np.float32)
        return np.array([
            float(window.min()) / cap, float(window.max()) / cap,
            float(window.mean()) / cap,
            float(window[-1] - window[0]) / (cap * L),
        ], dtype=np.float32)

    def _obs(self):
        cap = max(self.thermal_cap, 1.0)
        if self.state_version == "v1":
            status_norm = np.clip(self.status_duration / 10.0, -1.0, 1.0)
            return np.concatenate([
                [self.t / max(self.horizon - 1, 1)],
                self.prev_on.astype(float),
                self.net_demand / cap,
                status_norm,
                self._lookahead_stats(),
            ]).astype(np.float32)
        h = self.t % 24
        ang = 2.0 * np.pi * h / 24.0
        remaining = (self.horizon - self.t) / max(self.horizon, 1)
        can_start, can_stop, on_off = self._constraint_clocks()
        return np.concatenate([
            [np.sin(ang), np.cos(ang), remaining],
            self._demand_window(),
            self._lookahead_stats(),
            on_off, can_start, can_stop,
        ]).astype(np.float32)

    def _legalise(self, desired):
        on = np.asarray(desired, int).copy()
        cur_on = (self.current_dispatch > 1e-6) | (self.status_duration > 0)
        on[cur_on & (self.status_duration > 0) & (self.status_duration < self.min_up)] = 1
        on[(~cur_on) & (self.status_duration < 0) & (-self.status_duration < self.min_down)] = 0
        on[cur_on & (on == 0) & (self.current_dispatch > self.ramp_down + 1e-6)] = 1
        return on

    def _update_durations(self, on):
        nxt = np.zeros(self.n_gen, float)
        for i in range(self.n_gen):
            if on[i] == 1:
                nxt[i] = max(1.0, self.status_duration[i] + 1.0) if self.prev_on[i] == 1 else 1.0
            else:
                nxt[i] = min(-1.0, self.status_duration[i] - 1.0) if self.prev_on[i] == 0 else -1.0
        return nxt

    def step(self, action):
        raw = np.asarray(action, int).ravel()[:self.n_gen]
        on = self._legalise(raw)
        n_overrides = int(np.sum(on != raw))
        nd_t = float(self.net_demand[self.t])
        required = max(0.0, nd_t)
        total_pmin_on = float(self.pmin[on == 1].sum())
        total_pmax_on = float(self.pmax[on == 1].sum())
        cur_on = (self.current_dispatch > 1e-6) | (self.status_duration > 0)
        startup_cost = float(np.sum(((on == 1) & (~cur_on)) * self.startup_cost_arr))
        fuel_proxy = float(np.sum(on * self.pmin * self.marginal_cost))
        unserved = max(0.0, required - total_pmax_on)
        over_commit = max(0.0, total_pmin_on - required)
        spill_cost = self.spill_penalty * over_commit
        if required > 0:
            on_indices = np.where(on == 1)[0]
            sorted_by_pmin = on_indices[np.argsort(-self.pmax[on_indices])]
            cumsum = 0.0; needed = set()
            for idx in sorted_by_pmin:
                if cumsum < required * 1.15:
                    cumsum += self.pmax[idx]; needed.add(idx)
            n_excess = len(on_indices) - len(needed)
            excess_unit_penalty = n_excess * 100.0
        else:
            excess_unit_penalty = float(on.sum()) * 500.0
        lookahead_penalty = 0.0
        L = self.lookahead
        t_end = min(self.t + L, self.horizon)
        future_min_demand = float(self.net_demand[self.t:t_end].min())
        if future_min_demand < required * 0.5 and over_commit > 0:
            lookahead_penalty = over_commit * 500.0
        cost = (fuel_proxy + startup_cost + self.balance_penalty * unserved
                + spill_cost + excess_unit_penalty + lookahead_penalty)
        reward = -cost / self.reward_scale
        dur_next = self._update_durations(on)
        self.current_dispatch = on.astype(float) * self.pmin
        info = {
            "instance_id": self.episodes[self.ep_idx]["instance_id"],
            "hour": self.t, "net_demand": nd_t,
            "on": on.copy(), "raw_action": raw.copy(),
            "n_overrides": n_overrides,
            "startup_cost": startup_cost, "fuel_proxy": fuel_proxy,
            "cost": cost, "reward": reward,
            "unserved_proxy": unserved, "over_commit": over_commit,
            "n_startups": int(((on == 1) & (~cur_on)).sum()),
            "n_online": int(on.sum()),
        }
        self.prev_on = on.copy()
        self.status_duration = dur_next
        self.t += 1
        done = self.t >= self.horizon
        obs = np.zeros(self.obs_dim, np.float32) if done else self._obs()
        return obs, reward, done, False, info


def _reset(env_obj, idx=None):
    out = env_obj.reset(options={"episode_index": idx}) if idx is not None else env_obj.reset()
    return out[0]

def _step(env_obj, action):
    obs, rew, term, trunc, info = env_obj.step(action)
    return obs, rew, bool(term or trunc), info


env = UCOnlyEnv(train_episodes, spill_penalty=1000, reward_scale=1e5,
                random_reset=True, seed=42, lookahead=8,
                state_version="v2", obs_window=72)
test_env = UCOnlyEnv(test_episodes, spill_penalty=1000, reward_scale=1e5,
                      random_reset=False, seed=99, lookahead=8,
                      state_version="v2", obs_window=72)
print(f"env ready -- obs_dim={env.obs_dim} | n_gen={env.n_gen}")

def _ed_bounds(on, pmin, pmax, current_dispatch, ramp_up, ramp_down, status_duration):
    lower = np.zeros(len(on), float); upper = np.zeros(len(on), float)
    cur_on = (current_dispatch > 1e-6) | (status_duration > 0)
    for i in range(len(on)):
        if on[i] == 0: continue
        if cur_on[i]:
            lower[i] = max(pmin[i], current_dispatch[i] - ramp_down[i])
            upper[i] = min(pmax[i], current_dispatch[i] + ramp_up[i])
        else:
            lower[i] = pmin[i]; upper[i] = min(pmax[i], ramp_up[i])
        if upper[i] < lower[i]: upper[i] = lower[i]
    return lower, upper

def dispatch_proportional(required_mw, on, pmin, pmax, marginal_cost,
                           current_dispatch, ramp_up, ramp_down, status_duration, levels=ED_LEVELS):
    lower, upper = _ed_bounds(on, pmin, pmax, current_dispatch, ramp_up, ramp_down, status_duration)
    dispatch = lower.copy(); remaining = max(0.0, float(required_mw) - float(dispatch.sum()))
    headroom = np.maximum(upper - lower, 0.0) * on; total_h = headroom.sum()
    if total_h > 1e-9:
        for i in range(len(on)):
            if on[i] == 0: continue
            frac = headroom[i] / total_h
            dispatch[i] += min(frac * remaining, headroom[i])
    dispatch_snapped = np.zeros(len(on), float)
    for i in range(len(on)):
        if on[i] == 0: continue
        lo, hi = lower[i], upper[i]; target = dispatch[i]
        best_level, best_dist = lo, float("inf")
        for lv in levels:
            candidate = lv * pmax[i]
            if candidate < lo - 1e-6 or candidate > hi + 1e-6: continue
            dist = abs(candidate - target)
            if dist < best_dist: best_dist = dist; best_level = candidate
        if best_level > 1e-6: best_level = max(best_level, pmin[i])
        dispatch_snapped[i] = best_level
    return dispatch_snapped

ED_METHODS = {"proportional": dispatch_proportional}

def apply_ed_to_uc(uc_env, policy_fn, ed_fn, episodes, instance_index=0, levels=ED_LEVELS):
    obs = _reset(uc_env, instance_index)
    ep = episodes[instance_index]
    n_gen = uc_env.n_gen
    current_dispatch = ep["thermal_df"]["init_power"].to_numpy(float).copy()
    status_duration = ep["thermal_df"]["init_status"].to_numpy(float).copy()
    done, trace = False, []
    total_overrides = 0
    n_steps = 0
    while not done:
        a = policy_fn(obs)
        obs, rew_uc, done, info_uc = _step(uc_env, a)
        total_overrides += info_uc["n_overrides"]
        n_steps += 1
        on = info_uc["on"]
        nd_t = float(ep["net_demand"][info_uc["hour"]])
        required = max(0.0, nd_t)
        dispatch = ed_fn(required_mw=required, on=on, pmin=uc_env.pmin, pmax=uc_env.pmax,
                          marginal_cost=uc_env.marginal_cost, current_dispatch=current_dispatch,
                          ramp_up=uc_env.ramp_up, ramp_down=uc_env.ramp_down,
                          status_duration=status_duration, levels=levels)
        fuel_cost = sum(float(np.interp(dispatch[i], np.asarray(uc_env.cost_curve_mw[i], float),
                       np.asarray(uc_env.cost_curve_usd[i], float)))
                       if dispatch[i] > 1e-9 else 0.0 for i in range(n_gen))
        prev_on_flag = (current_dispatch > 1e-6) | (status_duration > 0)
        startup_cost = float(np.sum(((on == 1) & (~prev_on_flag)) * uc_env.startup_cost_arr))
        thermal_supply = float(dispatch.sum())
        unserved = max(0.0, required - thermal_supply)
        spill = max(0.0, thermal_supply - required) + max(0.0, -nd_t)
        cost = fuel_cost + startup_cost + uc_env.balance_penalty * unserved + 500.0 * spill
        current_dispatch = dispatch.copy()
        new_status = np.zeros(n_gen, float)
        for i in range(n_gen):
            cur = prev_on_flag[i]
            if on[i] == 1:
                new_status[i] = max(1.0, status_duration[i] + 1.0) if cur else 1.0
            else:
                new_status[i] = min(-1.0, status_duration[i] - 1.0) if not cur else -1.0
        status_duration = new_status
        trace.append({"instance_id": ep["instance_id"], "hour": info_uc["hour"],
                       "net_demand": nd_t, "required_mw": required, "on": on.copy(),
                       "dispatch": dispatch.copy(), "fuel_cost": fuel_cost,
                       "startup_cost": startup_cost, "cost": cost, "unserved": unserved,
                       "spill": spill, "n_startups": info_uc["n_startups"],
                       "n_online": int(on.sum()), "feasible": unserved < 1.0,
                       "n_overrides": info_uc["n_overrides"]})
    return pd.DataFrame(trace)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO as SB3_PPO
import gymnasium as gym

class UCOnlyEnvV3(UCOnlyEnv):
    def __init__(self, *args, **kwargs):
        kwargs["state_version"] = "v2"
        self._v3_ready = False
        self.starts_so_far = None
        self.prev_spill = 0.0
        self.prev_unserved = 0.0
        super().__init__(*args, **kwargs)
        self._max_starts = np.maximum(self.horizon / np.maximum(self.min_up + self.min_down, 1.0), 1.0)
        self._v3_ready = True
        self.state_version = "v3"
        self.obs_dim = 3 + self.obs_window + 4 + 3 * self.n_gen + 4 + self.n_gen
        self.reset(options={"episode_index": 0})

    @property
    def obs_layout(self):
        n, W, o = self.n_gen, self.obs_window, 0
        lay = {}
        for key, size in [("time", 3), ("window", W), ("lookahead", 4), ("on_off", n),
                          ("can_start", n), ("can_stop", n), ("balance", 4), ("starts", n)]:
            lay[key] = (o, o + size); o += size
        return lay

    def reset(self, seed=None, options=None):
        self.starts_so_far = np.zeros(self.n_gen, float)
        self.prev_spill = 0.0
        self.prev_unserved = 0.0
        self.state_version = "v2"
        obs, info = super().reset(seed=seed, options=options)
        if not self._v3_ready:
            return obs, info
        self.state_version = "v3"
        return self._obs(), info

    def _balance_features(self):
        denom = max(float(self.net_demand[self.t]), 1.0)
        on_m = (self.current_dispatch > 1e-6) | (self.status_duration > 0)
        return np.array([
            np.clip(float(self.pmin[on_m].sum()) / denom, 0.0, 3.0),
            np.clip(float(self.pmax[on_m].sum()) / denom, 0.0, 3.0),
            np.clip(self.prev_spill / denom, 0.0, 3.0),
            np.clip(self.prev_unserved / denom, 0.0, 3.0),
        ], dtype=np.float32)

    def _obs(self):
        if not self._v3_ready:
            return super()._obs()
        self.state_version = "v2"
        base = super()._obs()
        self.state_version = "v3"
        return np.concatenate([base, self._balance_features(),
                               np.clip(self.starts_so_far / self._max_starts, 0.0, 1.0)]).astype(np.float32)

    def step(self, action):
        cur_on = (self.current_dispatch > 1e-6) | (self.status_duration > 0)
        self.state_version = "v2"
        obs, rew, done, trunc, info = super().step(action)
        self.state_version = "v3"
        self.starts_so_far = self.starts_so_far + ((info["on"] == 1) & (~cur_on)).astype(float)
        self.prev_spill = info["over_commit"]
        self.prev_unserved = info["unserved_proxy"]
        obs = np.zeros(self.obs_dim, np.float32) if done else self._obs()
        return obs, rew, done, trunc, info


import torch
from stable_baselines3 import PPO as SB3_PPO

MD = Path(os.environ.get("UC_MODELS_DIR", "notebooks"))
OUT = Path("outputs_repro"); OUT.mkdir(exist_ok=True)


def policy_of(model):
    return lambda o: np.asarray(model.predict(o, deterministic=True)[0]).ravel()


def collect(env_obj, pol, i):
    obs = _reset(env_obj, i); done = False; rows = []
    while not done:
        obs, rew, done, info = _step(env_obj, pol(obs))
        rows.append(info)
    return pd.DataFrame(rows)


def evaluate(model, env_obj):
    pol = policy_of(model)
    corr, amp, uns, spl, cost, onl = [], [], [], [], [], []
    for i in range(len(test_episodes)):
        tr = collect(env_obj, pol, i)
        n_on = tr["n_online"].to_numpy(float); nd = tr["net_demand"].to_numpy(float)
        if n_on.std() > 1e-9 and nd.std() > 1e-9:
            corr.append(np.corrcoef(n_on, nd)[0, 1])
        amp.append(n_on.std()); uns.append(tr["unserved_proxy"].sum()); spl.append(tr["over_commit"].sum())
        cost.append(tr["cost"].sum()); onl.append(n_on.mean())
    return {"responsiveness": np.mean(corr), "amplitude": np.mean(amp), "mean_online": np.mean(onl),
            "unserved_MWh": np.mean(uns), "spill_MWh": np.mean(spl), "cost_$M": np.mean(cost) / 1e6}


def legality(model, state_version, n=20):
    e = UCOnlyEnv(test_episodes, spill_penalty=1000, random_reset=False, state_version=state_version, obs_window=72)
    pol = policy_of(model)
    ov, ill, cost, onl = [], [], [], []
    for i in range(n):
        tr = collect(e, pol, i)
        ov.append(tr["n_overrides"].sum()); ill.append(tr["n_overrides"].sum() / (72 * e.n_gen))
        cost.append(tr["cost"].sum()); onl.append(tr["n_online"].mean())
    return {"overrides_per_episode": np.mean(ov), "illegal_rate": np.mean(ill),
            "mean_cost_M": np.mean(cost) / 1e6, "mean_online": np.mean(onl)}


print("\n== Table 4 (single seed): re-evaluated from the four shipped PPO checkpoints ==")
P = MD / "saved_models_ppo"
spec = [("v2, spill 1,000", "ppo_a_v2_sp1000", "v2"), ("v3, spill 1,000", "ppo_c_v3_sp1000", "v3"),
        ("v2, spill 3,000", "ppo_b_v2_sp3000", "v2"), ("v3, spill 3,000", "ppo_d_v3_sp3000", "v3")]
rows = []
for label, name, sv in spec:
    model = SB3_PPO.load(str(P / name), device="cpu")
    env_e = (UCOnlyEnvV3(test_episodes, spill_penalty=1000, random_reset=False, obs_window=72) if sv == "v3"
             else UCOnlyEnv(test_episodes, spill_penalty=1000, random_reset=False, state_version="v2", obs_window=72))
    rows.append({"configuration": label, **evaluate(model, env_e)})
fact = pd.DataFrame(rows)
fact.to_csv(OUT / "factorial_single_seed.csv", index=False)
print(fact.round(3).to_string(index=False))

print("\n== Section 5.6: v1 against v2 observation, re-evaluated from the shipped checkpoints ==")
v1 = legality(SB3_PPO.load(str(P / "ppo_v1_prev_on"), device="cpu"), "v1")
v2 = legality(SB3_PPO.load(str(P / "ppo_v2_section56"), device="cpu"), "v2")
v2a = legality(SB3_PPO.load(str(P / "ppo_a_v2_sp1000"), device="cpu"), "v2")
v = pd.DataFrame({"v1 (prev_on only)": v1,
                  "v2 (constraint clocks), checkpoint used in Section 5.6": v2,
                  "v2 (constraint clocks), baseline checkpoint ppo_a": v2a}).T
v.to_csv(OUT / "v1_vs_v2.csv")
print(v.round(3).to_string())
