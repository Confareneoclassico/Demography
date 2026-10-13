"""
momp_robustness.py
------------------
Bootstrap confidence intervals and sample-size convergence for the Sobol'
indices of the global projection experiment, computed entirely from the
saved evaluations in ./results/simulation_results_v2.csv (no model reruns).

Why prefixes are valid designs: SALib's Saltelli sampler emits rows in
blocks of (D+2) per base sample (A_i, AB_i^(1..D), B_i) when
calc_second_order=False, as in the paper; the block size is detected from
the file. sobol.analyze indexes rows by that stride, hence the first N'*block rows
of an N-base design constitute the same scrambled-sequence design at base
size N'. A sanity assertion below verifies the stride assumption by
checking that the full-prefix analysis reproduces the headline indices.

Env knobs:
  B_FULL   bootstrap resamples at full N for the headline CIs (default 1000)
  B_CONV   resamples per convergence point (default 200)
  N_LIST   comma-separated base sizes (default "32,64,128,256,512,1024")

Outputs:
  ./results/momp_robustness.json
  ./figures/sobol_convergence.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SALib.analyze import sobol as sobol_an

D = 10
NAMES = ['paradigm', 'fertility_model', 'mortality_model', 'arima_order',
         'fertility_scale', 'tfr_floor', 'tfr_trend_scale', 'e0_gain',
         'mort_realization', 'arima_realization']
PROBLEM = {'num_vars': D, 'names': NAMES,
           'bounds': [[0, 1.9999], [0, 2.9999], [0, 1.9999], [0, 3.9999],
                      [0.90, 1.10], [1.45, 1.95], [0.50, 1.50], [0.10, 0.25],
                      [0.001, 0.999], [0.001, 0.999]]}
BLOCK = None   # detected from the file: N(D+2) for first-order designs, N(2D+2) with a second-order block

B_FULL = int(os.environ.get('B_FULL', 1000))
B_CONV = int(os.environ.get('B_CONV', 200))
N_LIST = [int(x) for x in os.environ.get('N_LIST', '32,64,128,256,512,1024,2048').split(',')]

df = pd.read_csv('./results/simulation_results_v2.csv')
if len(df) % (D + 2) == 0:
    BLOCK, SECOND = D + 2, False
elif len(df) % (2 * D + 2) == 0:
    BLOCK, SECOND = 2 * D + 2, True
else:
    raise RuntimeError(f"{len(df)} rows: not a multiple of D+2 or 2D+2")
N_full = len(df) // BLOCK
print(f"loaded {len(df)} rows = {N_full} base samples x {BLOCK} (calc_second_order={SECOND})")

out = {'meta': dict(B_full=B_FULL, B_conv=B_CONV, N_list=N_LIST)}
for tag in ('2050', '2075'):
    Y = df[f'population_{tag}_millions'].values.astype(float)

    conv = {n: {'N': [], 'S1': [], 'ST': [], 'S1_conf': [], 'ST_conf': []}
            for n in NAMES}
    for N in N_LIST:
        Yp = Y[:N * BLOCK]
        Si = sobol_an.analyze(PROBLEM, Yp, calc_second_order=SECOND,
                              num_resamples=(B_FULL if N == N_full else B_CONV),
                              conf_level=0.95, print_to_console=False, seed=11)
        for j, n in enumerate(NAMES):
            conv[n]['N'].append(N)
            conv[n]['S1'].append(float(Si['S1'][j]))
            conv[n]['ST'].append(float(Si['ST'][j]))
            conv[n]['S1_conf'].append(float(Si['S1_conf'][j]))
            conv[n]['ST_conf'].append(float(Si['ST_conf'][j]))
        if N == N_full:
            # sanity: reproduce headline indices (stride assumption check).
            # Skipped with a warning if MoMP_world.py has not written the
            # summary file (older runs); the analysis itself is unaffected.
            try:
                prev = json.load(open('./results/momp_v2_summary.json'))['sobol'][tag]
                for j, n in enumerate(NAMES):
                    assert abs(Si['S1'][j] - prev[n][0]) < 1e-9, (tag, n)
                print(f"  [{tag}] stride sanity OK; full-N CIs from {B_FULL} resamples")
            except FileNotFoundError:
                print(f"  [{tag}] note: momp_v2_summary.json not found — "
                      f"stride sanity check skipped (rerun MoMP_world.py to enable)")
    out[tag] = conv

    top = ['paradigm', 'arima_order', 'arima_realization', 'fertility_model']
    print(f"  [{tag}] N={N_full} 95% CIs:")
    for n in top:
        s1, c1 = conv[n]['S1'][-1], conv[n]['S1_conf'][-1]
        st, ct = conv[n]['ST'][-1], conv[n]['ST_conf'][-1]
        print(f"    {n:20s} S1 {s1:5.2f} ± {c1:4.2f}   ST {st:5.2f} ± {ct:4.2f}")

# ---- optimal-transport indices with bootstrap CIs (B_OT resamples of rows) ----
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demographic_models import ot_indices_given_data
B_OT = int(os.environ.get('B_OT', 1000))
Xo = df[NAMES].values.astype(float).copy()
for j, n in enumerate(NAMES):
    if n in ('paradigm', 'fertility_model', 'mortality_model', 'arima_order'):
        Xo[:, j] = np.floor(Xo[:, j]) + 1
rng = np.random.default_rng(7)
out['ot'] = {}
for tag in ('2050', '2075'):
    Y = df[f'population_{tag}_millions'].values.astype(float)
    pt = np.asarray(ot_indices_given_data(Xo, Y, M=25)); bo = np.empty((B_OT, D))
    for b in range(B_OT):
        ii = rng.integers(0, len(Y), len(Y)); bo[b] = ot_indices_given_data(Xo[ii], Y[ii], M=25)
    # intervals centred on the estimate (pt +/- 1.96 SE), as for the Sobol' indices;
    # percentile bounds kept for transparency (biased upward for weak inputs)
    se = bo.std(axis=0, ddof=1); plo, phi = np.percentile(bo, [2.5, 97.5], axis=0)
    out['ot'][tag] = {n: dict(OT=float(pt[j]), SE=float(se[j]), lo=float(pt[j] - 1.959964*se[j]),
                              hi=float(pt[j] + 1.959964*se[j]), pct_lo=float(plo[j]), pct_hi=float(phi[j]))
                      for j, n in enumerate(NAMES)}
    print(f"  [{tag}] OT (B_OT={B_OT}): " + ", ".join(f"{n}={pt[j]:.2f}[{lo[j]:.2f},{hi[j]:.2f}]" for j, n in enumerate(NAMES) if pt[j] > 0.05))
json.dump(out, open('./results/momp_robustness.json', 'w'), indent=1)

# ---------------------------------------------------------------- figure ----
os.makedirs('./figures', exist_ok=True)
import matplotlib.gridspec as gridspec
cmap = plt.get_cmap('tab10')
COLS = {n: cmap(i) for i, n in enumerate(NAMES)}
EMPH  = ['paradigm', 'arima_order', 'arima_realization', 'fertility_model']
WHISK = set(EMPH)
LAB = {'paradigm': 'paradigm', 'fertility_model': 'fertility model',
       'mortality_model': 'mortality model', 'arima_order': 'ARIMA order',
       'fertility_scale': 'fertility scale', 'tfr_floor': 'TFR floor',
       'tfr_trend_scale': 'TFR trend scale', 'e0_gain': '$e_0$ gain',
       'mort_realization': 'mort. realisation',
       'arima_realization': 'ARIMA realisation'}

fig = plt.figure(figsize=(13, 10.4))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.18,
                       bottom=0.155, top=0.96)

for col, tag in ((0, '2050'), (1, '2075')):
    ax = fig.add_subplot(gs[0, col])
    conv = out[tag]
    for n in NAMES:
        c = conv[n]
        emph = n in EMPH
        ax.plot(c['N'], c['ST'], '-', marker='o', color=COLS[n],
                lw=2.0 if emph else 1.0, ms=4 if emph else 2.6,
                alpha=1.0 if emph else 0.45)
        ax.plot(c['N'], c['S1'], '--', marker='s', color=COLS[n],
                lw=1.5 if emph else 0.8, ms=3.4 if emph else 2.2,
                alpha=0.85 if emph else 0.35)
        if n in WHISK:
            ax.errorbar(c['N'], c['ST'], yerr=c['ST_conf'], fmt='none',
                        color=COLS[n], capsize=3, lw=1.1)
            ax.errorbar(c['N'], c['S1'], yerr=c['S1_conf'], fmt='none',
                        color=COLS[n], capsize=2, lw=0.8, alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_LIST); ax.set_xticklabels(N_LIST, fontsize=10)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_title(f'Convergence — {tag}  (solid $S_T$, dashed $S_1$)', fontsize=12)
    ax.set_xlabel('Base sample size $N$ (log$_2$)', fontsize=11)
    if col == 0:
        ax.set_ylabel("Sobol' index", fontsize=12)
    ax.tick_params(labelsize=10)

ax = fig.add_subplot(gs[1, :])
x = np.arange(len(NAMES))
for tag, off, colr in (('2050', -0.17, '#7f7f7f'), ('2075', 0.17, '#1b6ca8')):
    c = out[tag]
    st = [c[n]['ST'][-1] for n in NAMES]; ct = [c[n]['ST_conf'][-1] for n in NAMES]
    s1 = [c[n]['S1'][-1] for n in NAMES]; c1 = [c[n]['S1_conf'][-1] for n in NAMES]
    ax.errorbar(x + off, st, yerr=ct, fmt='o', ms=5, capsize=3,
                color=colr, label=f'$S_T$ {tag}')
    ax.errorbar(x + off, s1, yerr=c1, fmt='s', ms=4, capsize=3,
                color=colr, alpha=0.55, label=f'$S_1$ {tag}')
ax.set_xticks(x)
TICKLAB = dict(LAB); TICKLAB['mort_realization'] = 'mort. realis.'
TICKLAB['arima_realization'] = 'ARIMA realis.'
ax.set_xticklabels([TICKLAB[n] for n in NAMES], rotation=18, ha='right',
                   fontsize=9.5)
ax.set_ylabel(f'Index at $N={N_full}$', fontsize=12)
ax.axhline(0, color='0.6', lw=0.8)
ax.set_title(f'Full-sample indices with 95% CIs ({B_FULL} resamples)', fontsize=12)
ax.legend(fontsize=10, framealpha=0.9, ncol=4)

handles = [plt.Line2D([], [], color=COLS[n],
                      lw=2.0 if n in EMPH else 1.0,
                      alpha=1.0 if n in EMPH else 0.5,
                      marker='o', ms=4, label=LAB[n]) for n in NAMES]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, 0.005), framealpha=0.9,
           title='Inputs (top panels; whiskers on emphasised inputs only)',
           title_fontsize=10)
plt.savefig('./figures/sobol_convergence.png', dpi=300, bbox_inches='tight')
print("figure: ./figures/sobol_convergence.png")
