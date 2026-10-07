import os, time, json, gzip, re, pickle, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
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
            "n_overrides": int(np.sum(on != raw)),
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

def dispatch_merit_order(required_mw, on, pmin, pmax, marginal_cost,
                          current_dispatch, ramp_up, ramp_down, status_duration, levels=ED_LEVELS):
    lower, upper = _ed_bounds(on, pmin, pmax, current_dispatch, ramp_up, ramp_down, status_duration)
    dispatch = np.zeros(len(on), float); remaining = float(required_mw)
    for i in range(len(on)):
        if on[i] == 0: continue
        lo, hi = lower[i], upper[i]; best = lo
        for lv in sorted(levels):
            candidate = lv * pmax[i]
            if candidate < lo - 1e-6 or candidate > hi + 1e-6: continue
            if candidate <= remaining + 1e-6: best = candidate
        if best > 1e-6: best = max(best, pmin[i])
        dispatch[i] = best; remaining -= best
    return dispatch

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

def dispatch_qp_milp(required_mw, on, pmin, pmax, marginal_cost,
                      current_dispatch, ramp_up, ramp_down, status_duration, levels=ED_LEVELS):
    lower, upper = _ed_bounds(on, pmin, pmax, current_dispatch, ramp_up, ramp_down, status_duration)
    on_idx = np.where(on == 1)[0]; n_on = len(on_idx)
    if n_on == 0: return np.zeros(len(on))
    feasible_mw = []
    for i in on_idx:
        lo, hi = lower[i], upper[i]
        gen_levels = [lv * pmax[i] for lv in levels if lo - 1e-6 <= lv * pmax[i] <= hi + 1e-6]
        if not gen_levels: gen_levels = [lo]
        feasible_mw.append(gen_levels)
    var_list = []
    for i_local, (i_global, mw_options) in enumerate(zip(on_idx, feasible_mw)):
        for mw in mw_options:
            var_list.append((i_local, i_global, mw, marginal_cost[i_global]))
    n_vars = len(var_list)
    c_obj = np.array([v[2] * v[3] for v in var_list])
    rows_pick, cols_pick, data_pick = [], [], []
    for vi, (i_local, i_global, mw, mc) in enumerate(var_list):
        rows_pick.append(i_local); cols_pick.append(vi); data_pick.append(1.0)
    A_pick = csr_matrix((data_pick, (rows_pick, cols_pick)), shape=(n_on, n_vars))
    b_lo_pick = np.ones(n_on); b_hi_pick = np.ones(n_on)
    mw_vals = np.array([v[2] for v in var_list])
    A_demand = csr_matrix(mw_vals.reshape(1, -1))
    b_lo_dem = np.array([max(0.0, float(required_mw))])
    b_hi_dem = np.array([float(pmax[on_idx].sum())])
    A_all = sp_vstack([A_pick, A_demand])
    b_lo_all = np.concatenate([b_lo_pick, b_lo_dem]); b_hi_all = np.concatenate([b_hi_pick, b_hi_dem])
    constraints = LinearConstraint(A_all, b_lo_all, b_hi_all)
    integrality = np.ones(n_vars)
    bounds_milp = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    result = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds_milp)
    dispatch = np.zeros(len(on), float)
    if result.success:
        for vi, (i_local, i_global, mw, mc) in enumerate(var_list):
            if result.x[vi] > 0.5: dispatch[i_global] = mw
    else:
        dispatch = dispatch_merit_order(required_mw, on, pmin, pmax, marginal_cost,
                                         current_dispatch, ramp_up, ramp_down, status_duration, levels)
    return dispatch

ED_METHODS = {"merit_order": dispatch_merit_order, "proportional": dispatch_proportional,
              "qp_milp": dispatch_qp_milp}

def apply_ed_to_uc(uc_env, policy_fn, ed_fn, episodes, instance_index=0, levels=ED_LEVELS):
    obs = _reset(uc_env, instance_index)
    ep = episodes[instance_index]
    n_gen = uc_env.n_gen
    current_dispatch = ep["thermal_df"]["init_power"].to_numpy(float).copy()
    status_duration = ep["thermal_df"]["init_status"].to_numpy(float).copy()
    done, trace = False, []
    while not done:
        a = policy_fn(obs)
        obs, rew_uc, done, info_uc = _step(uc_env, a)
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
                       "n_online": int(on.sum()), "feasible": unserved < 1.0})
    return pd.DataFrame(trace)

N_BINS = 6

def compact_state_for_gen(obs, gen_idx, n_gen, horizon=72, env_obj=None):
    e = env_obj if env_obj is not None else env
    lay = e.obs_layout
    sin_h, cos_h, remaining = obs[slice(*lay["time"])]
    win = obs[slice(*lay["window"])]
    on_off = obs[slice(*lay["on_off"])]
    can_start = obs[slice(*lay["can_start"])]
    can_stop = obs[slice(*lay["can_stop"])]
    nd_now = float(win[0]); fmin = float(win.min())
    return np.array([float(sin_h), float(cos_h), float(remaining), nd_now,
                      float(win.max()), float(win.mean()),
                      float(on_off.sum()) / max(n_gen, 1),
                      float(on_off[gen_idx]), float(can_start[gen_idx]), float(can_stop[gen_idx]),
                      float(fmin - nd_now)], dtype=float)

def estimate_bounds(env_obj, n_samples=500):
    obs = _reset(env_obj)
    feats = []
    for _ in range(n_samples):
        a = np.random.randint(0, 2, env_obj.n_gen)
        obs, _, done, _ = _step(env_obj, a)
        feats.append(compact_state_for_gen(obs, 0, env_obj.n_gen, env_obj=env_obj))
        if done: obs = _reset(env_obj)
    arr = np.vstack(feats)
    lo = np.quantile(arr, 0.02, axis=0); hi = np.quantile(arr, 0.98, axis=0)
    return lo, np.where(hi <= lo + 1e-9, lo + 1.0, hi)

def build_edges(lo, hi, n_bins=N_BINS):
    return [np.linspace(l, h, n_bins + 1)[1:-1] for l, h in zip(lo, hi)]

def discretize_all_gens(obs, env_obj=None, edges=None):
    e = env_obj if env_obj is not None else env
    ed = edges if edges is not None else EDGES
    lay = e.obs_layout; n = e.n_gen
    sin_h, cos_h, remaining = obs[slice(*lay["time"])]
    win = obs[slice(*lay["window"])]
    on_off = obs[slice(*lay["on_off"])]
    can_start = obs[slice(*lay["can_start"])]
    can_stop = obs[slice(*lay["can_stop"])]
    nd_now, win_max, win_mean = float(win[0]), float(win.max()), float(win.mean())
    fmin = float(win.min()); frac_on = float(on_off.sum()) / max(n, 1)
    g_vals = [float(sin_h), float(cos_h), float(remaining), nd_now, win_max, win_mean, frac_on]
    g_bins = [int(np.clip(np.digitize(v, ed[k]), 0, N_BINS - 1)) for k, v in enumerate(g_vals)]
    delta = float(fmin - nd_now)
    d_bin = int(np.clip(np.digitize(delta, ed[10]), 0, N_BINS - 1))
    b_on = np.clip(np.digitize(on_off, ed[7]), 0, N_BINS - 1)
    b_start = np.clip(np.digitize(can_start, ed[8]), 0, N_BINS - 1)
    b_stop = np.clip(np.digitize(can_stop, ed[9]), 0, N_BINS - 1)
    g0, g1, g2, g3, g4, g5, g6 = g_bins
    return [(g0, g1, g2, g3, g4, g5, g6, int(b_on[i]), int(b_start[i]), int(b_stop[i]), d_bin)
            for i in range(n)]

QLEARNING_FINAL_SEED = 42

def train_qlearning(seed=QLEARNING_FINAL_SEED, n_epochs=5, n_q=None,
                     alpha=0.10, gamma=0.99, eps0=0.30, eps_min=0.05, eps_decay=0.998,
                     verbose_every=1):
    if n_q is None: n_q = min(300, len(train_episodes))
    np.random.seed(seed); random.seed(seed)
    train_env = UCOnlyEnv(train_episodes, spill_penalty=1000, reward_scale=1e5,
                           random_reset=True, seed=seed, lookahead=8,
                           state_version="v2", obs_window=72)
    Q_tables_local = [defaultdict(lambda: np.zeros(2)) for _ in range(train_env.n_gen)]
    eps = eps0; t0 = time.time()
    for epoch in range(n_epochs):
        for ep in range(n_q):
            obs = _reset(train_env)
            states = discretize_all_gens(obs, env_obj=train_env)
            done = False
            while not done:
                n = len(states); action = np.empty(n, int)
                explore = np.random.rand(n) < eps; rand_a = np.random.randint(0, 2, n)
                for i, s in enumerate(states):
                    if explore[i]: action[i] = rand_a[i]
                    else:
                        q = Q_tables_local[i][s]; action[i] = 0 if q[0] >= q[1] else 1
                nobs, rew, done, info = _step(train_env, action)
                if done:
                    for i in range(n):
                        q = Q_tables_local[i][states[i]]; ai = int(action[i])
                        q[ai] += alpha * (rew - q[ai])
                    next_states = None
                else:
                    next_states = discretize_all_gens(nobs, env_obj=train_env)
                    for i in range(n):
                        q = Q_tables_local[i][states[i]]; ai = int(action[i])
                        q[ai] += alpha * (rew + gamma * Q_tables_local[i][next_states[i]].max() - q[ai])
                states = next_states
            eps = max(eps_min, eps * eps_decay)
        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"  epoch {epoch+1}/{n_epochs} done ({time.time()-t0:.0f}s)")
    print(f"Q-Learning retrain done in {time.time()-t0:.0f}s")
    return Q_tables_local

def make_q_policy(Q_tables_local):
    def policy(obs, epsilon=0.0):
        states = discretize_all_gens(obs, env_obj=env)
        n = len(states); action = np.empty(n, int)
        for i, s in enumerate(states):
            q = Q_tables_local[i][s]; action[i] = 0 if q[0] >= q[1] else 1
        return action
    return policy

print("\nEstimating fresh OBS_LO/OBS_HI/EDGES on the fixed v2 environment...")
np.random.seed(QLEARNING_FINAL_SEED)
OBS_LO, OBS_HI = estimate_bounds(env)
EDGES = build_edges(OBS_LO, OBS_HI)
print(f"EDGES has {len(EDGES)} entries (expect 11).")

print(f"\nTraining FINAL Q-Learning (seed={QLEARNING_FINAL_SEED})...")
Q_tables = train_qlearning(seed=QLEARNING_FINAL_SEED)
q_policy_fn = make_q_policy(Q_tables)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qtable_io import save_q_tables

SAVE_DIR_FINAL = Path("notebooks/saved_models_qlearning")
SAVE_DIR_FINAL.mkdir(parents=True, exist_ok=True)
Q_PATH = SAVE_DIR_FINAL / "q_tables_final_seed42.npz"
save_q_tables(Q_PATH, Q_tables, OBS_LO, OBS_HI, QLEARNING_FINAL_SEED)
print(f"Saved to {Q_PATH}")

def responsiveness_for_instance(commitment, net_demand):
    committed_count = commitment.sum(axis=1)
    if np.std(committed_count) <= 1e-9 or np.std(net_demand) <= 1e-9:
        return np.nan
    r, _ = pearsonr(committed_count, net_demand)
    return float(r)

def scheduling_amplitude(commitment):
    return float(np.std(commitment.sum(axis=1)))

def collect_uc_only_trace_test(policy_fn, episodes, instance_index):
    obs = _reset(test_env, instance_index)
    ep = episodes[instance_index]; on_rows = []; done = False
    while not done:
        a = policy_fn(obs)
        obs, rew, done, info = _step(test_env, a)
        on_rows.append(info["on"].copy())
    return np.vstack(on_rows), np.asarray(ep["net_demand"], float)

def benchmark_cost_for_episode(ep, generator_names, cost_curve_mw, cost_curve_usd, startup_cost_arr):
    gen_to_idx = {g: i for i, g in enumerate(generator_names)}
    oj = ep.get("output_json"); bench_cost = 0.0
    if oj:
        bt = pd.DataFrame(oj.get("Thermal production (MW)", {}))
        bo = pd.DataFrame(oj.get("Is on", {}))
        gc = [c for c in bt.columns if c.lower() != "hour"]
        if gc:
            bdisp = bt[gc].to_numpy(float); bon = bo[gc].to_numpy(float)
            for t in range(len(bdisp)):
                for j, col in enumerate(gc):
                    if col in gen_to_idx:
                        idx = gen_to_idx[col]; mw = bdisp[t, j]
                        if mw > 1e-9:
                            mw_c = np.asarray(cost_curve_mw[idx], float)
                            c_c = np.asarray(cost_curve_usd[idx], float)
                            bench_cost += float(np.interp(mw, mw_c, c_c))
                        if t > 0 and bon[t, j] > 0.5 and bon[t-1, j] < 0.5:
                            bench_cost += startup_cost_arr[idx]
    return bench_cost

def evaluate_policy_raw(policy_fn, n_instances):
    commitment_rows = []
    for vi in range(n_instances):
        commitment, net_demand = collect_uc_only_trace_test(policy_fn, test_episodes, vi)
        commitment_rows.append({"instance": vi,
            "responsiveness": responsiveness_for_instance(commitment, net_demand),
            "amplitude": scheduling_amplitude(commitment)})
    dispatch_rows = []
    for ed_name, ed_fn in ED_METHODS.items():
        for vi in range(n_instances):
            ep = test_episodes[vi]
            tr = apply_ed_to_uc(test_env, policy_fn, ed_fn, test_episodes, vi)
            rl_cost = tr["fuel_cost"].sum() + tr["startup_cost"].sum()
            bench_cost = benchmark_cost_for_episode(ep, env.generator_names, env.cost_curve_mw,
                                                     env.cost_curve_usd, env.startup_cost_arr)
            OR = rl_cost / bench_cost if bench_cost > 1 else np.nan
            total_demand = np.maximum(0, tr["net_demand"].to_numpy(float)).sum()
            unserved = tr["unserved"].sum()
            fulfilment_pct = max(0, 1 - unserved / max(total_demand, 1e-6)) * 100
            dispatch_rows.append({"instance": vi, "ed_method": ed_name, "OR": OR,
                                   "fulfilment_pct": fulfilment_pct,
                                   "unserved_mwh": unserved, "spill_mwh": tr["spill"].sum()})
    return pd.DataFrame(commitment_rows), pd.DataFrame(dispatch_rows)

print("\nEvaluating final Q-Learning on 100 test instances x 3 ED methods...")
t0 = time.time()
commit_df, disp_df = evaluate_policy_raw(q_policy_fn, 100)
print(f"Done in {time.time()-t0:.0f}s")

RAW_OUT = Path("outputs/results_raw")
RAW_OUT.mkdir(parents=True, exist_ok=True)
commit_df.to_csv(RAW_OUT / "Q-Learning_FINAL_commitment.csv", index=False)
disp_df.to_csv(RAW_OUT / "Q-Learning_FINAL_dispatch.csv", index=False)

pooled_or = disp_df["OR"].median()
pooled_fulfil = disp_df["fulfilment_pct"].mean()
prop = disp_df[disp_df.ed_method == "proportional"]
prop_or = prop["OR"].median()
prop_fulfil = prop["fulfilment_pct"].mean()

print("\n===== FINAL Q-LEARNING RESULTS (seed=42, canonical) =====")
print(f"Pooled (all 3 ED methods):  OR_median={pooled_or:.3f} | fulfilment_mean={pooled_fulfil:.3f}%")
print(f"Proportional only:          OR_median={prop_or:.3f} | fulfilment_mean={prop_fulfil:.3f}%")
print(f"Responsiveness (mean, skipna): {commit_df['responsiveness'].mean(skipna=True):.3f}")
print(f"Amplitude (mean): {commit_df['amplitude'].mean():.3f}")
print(f"\nTotal script time: {(time.time()-t_script_start)/60:.1f} min")
