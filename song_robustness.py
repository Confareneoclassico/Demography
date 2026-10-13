"""
song_robustness.py
------------------
Bootstrap confidence intervals and sample-size convergence for the Song
control-path sensitivity measures, computed from ONE design.

Source: ./results/song_control_audit_saltelli.csv when present (the
Saltelli design, SALTELLI_N mode of Song_control_audit.py; the paper's
design), otherwise ./results/song_control_audit.csv (plain QMC, fallback).
No re-solves are performed.

Convergence figure: one row per control-path metric, two columns.
  1. Sobol' S_T (solid) and S_1 (dashed) on nested prefixes of the Saltelli
     design (base sizes 8, 16, ..., N), with SALib's bootstrap intervals.
     On the plain fallback: given-data S_1 on power-of-two prefixes.
  2. Optimal-transport index on the same prefixes (given-data, one class
     per level for discrete inputs), with bootstrap intervals.
Whiskers are drawn for the two dominant inputs only (pyramid, source);
all bounds are stored in the JSON.

Env:  B        bootstrap resamples, full-sample CI section   (default 1000)
      B_CONV   resamples per convergence point               (default 100)
      NS       comma-separated prefix sizes (base N for Saltelli; rows for plain)
      SAVE_INDIVIDUAL=1  also write each panel as a separate PNG
Outputs: results/song_robustness.json, figures/song_convergence_grid.png
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demographic_models import ot_indices_given_data, first_order_given_data
from SALib.analyze import delta as delta_an

B      = int(os.environ.get('B', 1000))
B_CONV = int(os.environ.get('B_CONV', 100))
SAVE_IND = os.environ.get('SAVE_INDIVIDUAL', '0') == '1'
rng = np.random.default_rng(3)

NAMES = ['initial_pyramid_type','fertility_rate_source','mortality_evolution_factor',
         'newborn_female_proportion','optimizer_tolerance','optimizer_max_iterations',
         'penalty_factor','fertility_age_shift_factor']
SHORT = ['pyramid','fert. source','mortality evo.','female prop.',
         'opt. tolerance','opt. max iter.','penalty','age shift']
METRICS = ['tfr_min','year_tfr_min','years_below_2','cum_deficit','year_return_repl']
MLAB = {'tfr_min':'Minimum TFR','year_tfr_min':'Trough year','years_below_2':'Years below 2.0',
        'cum_deficit':'Cumulative deficit','year_return_repl':'Return year'}
K = len(NAMES); BLK = K + 2

SPATH, PPATH = './results/song_control_audit_saltelli.csv', './results/song_control_audit.csv'
SALT = os.path.exists(SPATH)
df = pd.read_csv(SPATH if SALT else PPATH)
X = df[[f'param_{n}' for n in NAMES]].values.astype(float)
print(f"source: {'Saltelli design' if SALT else 'plain design (fallback)'}, {len(df)} rows")

if SALT:
    assert len(df) % BLK == 0, f"{len(df)} rows is not a multiple of {BLK}: incomplete run? (RESUME=1)"
    Nb = len(df) // BLK
    if os.environ.get('NS'):
        NS = [int(x) for x in os.environ['NS'].split(',')]
    else:
        NS, n = [], 8
        while n <= Nb: NS.append(n); n *= 2
        if not NS or NS[-1] != Nb: NS.append(Nb)
    rows = lambda n: n * BLK
    PROB = {'num_vars': K, 'names': NAMES,
            'bounds': [[0,2.9999],[0,1.9999],[0,0.1],[0.47,0.5],[-7,-5],[1,3],[8,12],[0,0.2]]}
    from SALib.analyze import sobol as sobol_an
    print(f"base N={Nb}, prefix base sizes {NS}")
else:
    NS = [int(x) for x in os.environ['NS'].split(',')] if os.environ.get('NS') else \
         [n for n in (16, 32, 64, 128, 256, 512, 1024, 2048) if n <= len(df)]
    rows = lambda n: n
    PROB = {'num_vars': K, 'names': NAMES,
            'bounds': [[0,2],[0,1],[0,0.1],[0.47,0.5],[1e-7,1e-5],[10,1000],[1e8,1e12],[0,0.2]]}

def boot(fn, Xp, yp, Bn):
    """Point estimate and 95% interval centred on it, pt +/- 1.96 SE, with SE
    from Bn row resamples (the convention SALib uses for Sobol' indices).
    Percentile intervals are NOT used: resampling with replacement duplicates
    rows, which inflates the transport index of weak inputs, so percentile
    intervals sit above the estimate itself for such inputs."""
    pt = np.asarray(fn(Xp, yp, M=25)); bo = np.empty((Bn, K))
    for b in range(Bn):
        ii = rng.integers(0, len(yp), len(yp)); bo[b] = fn(Xp[ii], yp[ii], M=25)
    se = bo.std(axis=0, ddof=1)
    return pt, pt - 1.959964 * se, pt + 1.959964 * se

# ---------------- convergence, both estimator families, CIs at every point ----
conv = {m: {'OT': {}, 'SOBOL': {}} for m in METRICS}
for m in METRICS:
    y = df[m].values.astype(float)
    for n in NS:
        Xp, yp = X[:rows(n)], y[:rows(n)]
        pt, lo, hi = boot(ot_indices_given_data, Xp, yp, B_CONV)
        conv[m]['OT'][n] = dict(val=pt.tolist(), lo=lo.tolist(), hi=hi.tolist())
        if SALT:
            Si = sobol_an.analyze(PROB, yp, calc_second_order=False, num_resamples=B_CONV,
                                  conf_level=0.95, print_to_console=False, seed=5)
            conv[m]['SOBOL'][n] = dict(S1=[float(v) for v in Si['S1']], S1c=[float(v) for v in Si['S1_conf']],
                                       ST=[float(v) for v in Si['ST']], STc=[float(v) for v in Si['ST_conf']])
        else:
            pt, lo, hi = boot(first_order_given_data, Xp, yp, B_CONV)
            conv[m]['SOBOL'][n] = dict(S1=pt.tolist(), S1lo=lo.tolist(), S1hi=hi.tolist())

# ---------------- full-sample CIs (headline) ---------------------------------
out = {'meta': {'source': 'saltelli' if SALT else 'plain', 'rows': int(len(df)),
                'base_N': (Nb if SALT else None), 'B': B, 'B_conv': B_CONV, 'NS': NS},
       'conv': conv, 'full': {}}
print(f"{'metric':18s} {'OT pyramid [95% CI]':>22s} {'OT source [95% CI]':>22s}" + ("   Sobol' S1/ST pyr" if SALT else ""))
for m in METRICS:
    y = df[m].values.astype(float)
    pt, lo, hi = boot(ot_indices_given_data, X, y, B)
    out['full'][m] = dict(OT=pt.tolist(), OT_lo=lo.tolist(), OT_hi=hi.tolist(),
                          S1_partition=[float(v) for v in first_order_given_data(X, y, M=25)])
    line = f"{m:18s}  {pt[0]:.2f} [{lo[0]:.2f},{hi[0]:.2f}]      {pt[1]:.2f} [{lo[1]:.2f},{hi[1]:.2f}]"
    if SALT:
        Si = sobol_an.analyze(PROB, y, calc_second_order=False, num_resamples=B,
                              conf_level=0.95, print_to_console=False, seed=5)
        out['full'][m].update(S1=[float(v) for v in Si['S1']], S1_conf=[float(v) for v in Si['S1_conf']],
                              ST=[float(v) for v in Si['ST']], ST_conf=[float(v) for v in Si['ST_conf']])
        line += f"   {Si['S1'][0]:.2f}/{Si['ST'][0]:.2f}"
    print(line)

# delta: for the record only (excluded from the analysis: KDE fails on the censored trough)
out['delta_reference'] = {'_note': 'excluded from the analysis (KDE fails on censored tfr_min); '
                                   'S1_binned_salib uses equal-count rank bins (biased low for discrete inputs)'}
for m in METRICS:
    try:
        r = delta_an.analyze(PROB, X, df[m].values.astype(float), num_resamples=min(B, 200),
                             conf_level=0.95, print_to_console=False, seed=5)
        dk = 'delta' if 'delta' in r else 'delta_raw'
        out['delta_reference'][m] = {n: dict(delta=float(r[dk][j]), delta_conf=float(r[dk+'_conf'][j]),
                                             S1_binned_salib=float(r['S1'][j]))
                                     for j, n in enumerate(NAMES)}
    except Exception as e:
        out['delta_reference'][m] = f'failed: {type(e).__name__}'
json.dump(out, open('./results/song_robustness.json', 'w'), indent=1)

# ---------------- grid: metrics x (Sobol' | OT) ------------------------------
cmap = plt.get_cmap('tab10'); COLS = [cmap(i) for i in range(K)]; EMPH = {0, 1}
fig, axes = plt.subplots(len(METRICS), 2, figsize=(10.0, 3.4 * len(METRICS)))
col1 = (f"Sobol' $S_T$ (solid), $S_1$ (dashed), base $N \\leq {Nb}$" if SALT
        else f"Given-data $S_1$, plain design, $n \\leq {NS[-1]}$")
col2 = f"OT index $\\iota^{{W}}$, {'base $N$' if SALT else '$n$'} $\\leq$ {NS[-1]}"

def draw(ax, m, est):
    xs = NS
    for j in range(K):
        em = j in EMPH; lw, ms, al = (2.2, 4, 1.0) if em else (1.1, 2.8, 0.55)
        if est == 'OT':
            d = conv[m]['OT']; v = [d[n]['val'][j] for n in xs]
            ax.plot(xs, v, '-', marker='o', color=COLS[j], lw=lw, ms=ms, alpha=al)
            if em:
                ax.errorbar(xs, v, yerr=[[max(0, d[n]['val'][j]-d[n]['lo'][j]) for n in xs],
                                        [max(0, d[n]['hi'][j]-d[n]['val'][j]) for n in xs]],
                            fmt='none', color=COLS[j], capsize=3, lw=1.2)
        elif SALT:
            d = conv[m]['SOBOL']
            st = [d[n]['ST'][j] for n in xs]; s1 = [d[n]['S1'][j] for n in xs]
            ax.plot(xs, st, '-', marker='o', color=COLS[j], lw=lw, ms=ms, alpha=al)
            ax.plot(xs, s1, '--', marker='s', color=COLS[j], lw=lw*0.75, ms=ms*0.85, alpha=al*0.9)
            if em:
                ax.errorbar(xs, st, yerr=[d[n]['STc'][j] for n in xs], fmt='none', color=COLS[j], capsize=3, lw=1.2)
                ax.errorbar(xs, s1, yerr=[d[n]['S1c'][j] for n in xs], fmt='none', color=COLS[j], capsize=2, lw=0.9, alpha=0.7)
        else:
            d = conv[m]['SOBOL']; v = [d[n]['S1'][j] for n in xs]
            ax.plot(xs, v, '--', marker='s', color=COLS[j], lw=lw, ms=ms, alpha=al)
            if em:
                ax.errorbar(xs, v, yerr=[[max(0, d[n]['S1'][j]-d[n]['S1lo'][j]) for n in xs],
                                        [max(0, d[n]['S1hi'][j]-d[n]['S1'][j]) for n in xs]],
                            fmt='none', color=COLS[j], capsize=3, lw=1.2)
    ax.set_xscale('log', base=2); ax.set_xticks(xs); ax.set_xticklabels(xs, fontsize=10)
    ax.axhline(0, color='0.6', lw=0.8); ax.tick_params(labelsize=10)

for i, m in enumerate(METRICS):
    for k, (est, title) in enumerate((('SOBOL', col1), ('OT', col2))):
        ax = axes[i, k]; draw(ax, m, est)
        if i == 0: ax.set_title(title, fontsize=12)
        if k == 0: ax.set_ylabel(MLAB[m], fontsize=12)
        if i == len(METRICS) - 1:
            ax.set_xlabel('Base sample size $N$ (log$_2$)' if SALT else 'Sample size $n$ (log$_2$)', fontsize=12)
handles = [plt.Line2D([], [], color=COLS[j], lw=2.2 if j in EMPH else 1.2, alpha=1.0 if j in EMPH else 0.6,
                      marker='o', ms=4, label=SHORT[j]) for j in range(K)]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.005),
           framealpha=0.9, title='Inputs (whiskers: 95% CIs, emphasised inputs only)', title_fontsize=11)
plt.tight_layout(rect=(0, 0.045, 1, 1))
plt.savefig('./figures/song_convergence_grid.png', dpi=300, bbox_inches='tight')
print("figure: ./figures/song_convergence_grid.png")
if SAVE_IND:
    for m in METRICS:
        for est, title in (('SOBOL', col1), ('OT', col2)):
            f1, a1 = plt.subplots(figsize=(5.2, 4.0)); draw(a1, m, est)
            a1.set_title(f'{MLAB[m]}: {title}', fontsize=11)
            f1.legend(handles=handles, loc='center left', fontsize=8, bbox_to_anchor=(1.0, 0.5))
            f1.tight_layout(); f1.savefig(f'./figures/song_conv_{m}_{est}.png', dpi=300, bbox_inches='tight'); plt.close(f1)
    print("individual panels written")
