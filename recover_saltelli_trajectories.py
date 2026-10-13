"""
recover_saltelli_trajectories.py
--------------------------------
Rebuild ./results/song_control_trajectories_saltelli.npz from any trajectory
file in ./results/, by matching trajectories to the rows of
./results/song_control_audit_saltelli.csv on the control-path metrics
(each trajectory's metrics are a fingerprint of that run).

Why: runs made before the mode-suffixed output names wrote their
trajectories to the unsuffixed song_control_trajectories.npz, which later
plain-design runs appended to or overwrote. If the Saltelli trajectories
are still in that file, this script recovers them; if not, it says so, and
the design has to be re-solved (SALTELLI_N=512 RESUME=0).
"""
import glob, numpy as np, pandas as pd

csv = pd.read_csv('./results/song_control_audit_saltelli.csv')
n = len(csv)
print(f"Saltelli CSV: {n} runs")

def metrics(beta, pop, yrs):
    i_min = int(beta.argmin())
    after = np.where((yrs > yrs[i_min]) & (beta >= 2.10))[0]
    return (round(float(beta.min()), 9), int(yrs[i_min]), int((beta < 2.0).sum()),
            round(float(np.clip(2.16 - beta, 0, None).sum()), 6),
            int(yrs[after[0]]) if after.size else int(yrs[-1]))

from collections import defaultdict
# several design rows can share a fingerprint (rows differing only in an
# inert input give identical trajectories), so keep a list per fingerprint
want = defaultdict(list)
for i, r in csv.iterrows():
    want[(round(float(r.tfr_min), 9), int(r.year_tfr_min), int(r.years_below_2),
          round(float(r.cum_deficit), 6), int(r.year_return_repl))].append(i)

for path in sorted(glob.glob('./results/*trajector*.npz')):
    z = np.load(path); beta, pop, yrs = z['beta'], z['pop'], z['years']
    pending = {k: list(v) for k, v in want.items()}
    hits = {}
    for k in range(len(beta)):
        key = metrics(beta[k], pop[k], yrs)
        if pending.get(key):
            hits[pending[key].pop(0)] = k
    print(f"{path}: {len(beta)} trajectories, {len(hits)}/{n} CSV runs matched")
    if len(hits) == n:
        order = [hits[i] for i in range(n)]
        np.savez_compressed('./results/song_control_trajectories_saltelli.npz',
                            beta=beta[order], pop=pop[order], years=yrs)
        print("recovered -> ./results/song_control_trajectories_saltelli.npz "
              "(rerun: SALTELLI_N=512 RESUME=1 python3 Song_control_audit.py)")
        break
else:
    print("no file contains all Saltelli trajectories: re-solve the design with "
          "SALTELLI_N=512 RESUME=0 python3 Song_control_audit.py (~3 h at 2 s/solve)")
