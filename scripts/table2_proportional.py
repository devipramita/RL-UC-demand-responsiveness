import numpy as np
import pandas as pd

R = "results"
N_BOOT, SEED = 1000, 0
AGENTS = {"Tabular Q-Learning": "qlearning_final", "Deep Q-Network": "dqn", "PPO": "ppo"}


def bootstrap(disp, commit):
    prop = disp[disp.ed_method == "proportional"].sort_values("instance").reset_index(drop=True)
    resp = commit.set_index("instance").reindex(prop["instance"])["responsiveness"].to_numpy()
    orr, ful = prop["OR"].to_numpy(), prop["fulfilment_pct"].to_numpy()
    n = len(prop)
    assert n == 100
    rng = np.random.default_rng(SEED)
    b_or, b_ful, b_resp = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        b_or.append(np.median(orr[idx])); b_ful.append(np.mean(ful[idx])); b_resp.append(np.nanmean(resp[idx]))
    q = lambda a, p: float(np.percentile(a, p))
    return {
        "or_p25": q(b_or, 25), "or_median": float(np.median(b_or)), "or_p75": q(b_or, 75),
        "fulfilment_pct": float(np.mean(ful)), "responsiveness": float(np.nanmean(resp)),
        "amplitude": float(commit["amplitude"].mean()),
        "or_point": float(np.median(orr)),
    }


rows = []
for name, stem in AGENTS.items():
    d = pd.read_csv(f"{R}/{stem}_dispatch.csv"); c = pd.read_csv(f"{R}/{stem}_commitment.csv")
    rows.append({"agent": name, **bootstrap(d, c)})

static = pd.read_csv(f"{R}/static_fleet_sweep_k10_45.csv")
sp = static[static.ed_method == "proportional"].sort_values("k")
k45 = sp[sp.k == 45].iloc[0]
rows.append({"agent": "Static fleet, k = 45", "or_p25": k45.or_p25, "or_median": k45.or_p50, "or_p75": k45.or_p75,
             "fulfilment_pct": k45.fulfilment_pct, "responsiveness": 0.0, "amplitude": 0.0, "or_point": k45.or_p50})
table2 = pd.DataFrame(rows)
table2.to_csv(f"{R}/table2_proportional.csv", index=False)
print("Table 2 (proportional loading, bootstrap over 100 test instances)")
print(table2.round(3).to_string(index=False))

fr_f, fr_o = sp.fulfilment_pct.to_numpy(), sp.or_p50.to_numpy()
assert np.all(np.diff(fr_f) > 0)
cmp_rows = []
for r in rows[:3]:
    frontier = float(np.interp(r["fulfilment_pct"], fr_f, fr_o))
    cmp_rows.append({"agent": r["agent"], "fulfilment_pct": r["fulfilment_pct"], "agent_or": r["or_median"],
                     "static_frontier_or": frontier, "agent_vs_frontier_pct": 100 * (r["or_median"] / frontier - 1)})
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(f"{R}/matched_fulfilment_comparison.csv", index=False)
print("\nMatched-fulfilment comparison (negative = agent cheaper than the static frontier)")
print(cmp.round(3).to_string(index=False))
