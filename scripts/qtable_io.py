from collections import defaultdict

import numpy as np


def save_q_tables(path, Q_tables, OBS_LO, OBS_HI, seed=None):
    keys, vals, owner = [], [], []
    for g, table in enumerate(Q_tables):
        for state, q in table.items():
            keys.append(state)
            vals.append(q)
            owner.append(g)
    np.savez_compressed(
        path,
        keys=np.asarray(keys, dtype=np.int8),
        vals=np.asarray(vals, dtype=np.float64),
        owner=np.asarray(owner, dtype=np.int16),
        n_generators=np.int16(len(Q_tables)),
        OBS_LO=np.asarray(OBS_LO, dtype=np.float64),
        OBS_HI=np.asarray(OBS_HI, dtype=np.float64),
        seed=np.int64(-1 if seed is None else seed),
    )


def load_q_tables(path):
    d = np.load(path)
    n_gen = int(d["n_generators"])
    Q_tables = [defaultdict(lambda: np.zeros(2)) for _ in range(n_gen)]
    keys, vals, owner = d["keys"], d["vals"], d["owner"]
    for k, v, g in zip(keys.tolist(), vals, owner.tolist()):
        Q_tables[g][tuple(k)] = v.copy()
    seed = int(d["seed"])
    return Q_tables, d["OBS_LO"], d["OBS_HI"], (None if seed < 0 else seed)
