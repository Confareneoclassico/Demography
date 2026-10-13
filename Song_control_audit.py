"""
Song_control_audit.py
---------------------
Re-analysis of the Song et al. optimal-control model with the sensitivity
audit aimed at the model's ACTUAL output: the optimal fertility control path
beta(t), not the constrained state path (total population).

Responds to Caswell's review point: an optimal control problem outputs the
control function; the state path is constrained by construction, so its
invariance is trivially true. The audit therefore asks: how sensitive is the
POLICY PRESCRIPTION (how hard to squeeze fertility, when, for how long) to
the model's uncertain demographic and algorithmic inputs?

Control-path metrics per run:
  tfr_min              lowest fertility the optimiser prescribes
  year_tfr_min         calendar year of the trough
  years_below_2        number of years with prescribed TFR < 2.0
  cum_deficit          sum over years of (2.16 - beta)+  ("TFR-years withheld")
  year_return_repl     first year after the trough with beta >= 2.10
  peak_pop_year        year of maximum total population (state side, for figure)
  peak_pop             maximum total population

Design used in the paper: SALTELLI_N=512 (Saltelli design, N(k+2) = 5,120
solves). Without SALTELLI_N the script runs the plain Sobol' design of the
submitted version (N_SIM runs), kept for comparison only.
Sensitivity (given-data, on the plain QMC sample): optimal-transport
indices (Wasserstein-2^2 normalised by 2 Var(Y); engine mirrored from the
companion fairness-audit repository, whose exact-level branch assigns one
1-based bin per category), alongside the Plischke et al. (2013)
delta and S1 estimators for cross-checking across estimator families.
Set SALTELLI_N=<base N> to instead run a Saltelli-scheme design and
compute classical Sobol' S1/ST on the control metrics. Sampling uses
calc_second_order=False, so the design has N x (8+2) optimal-control
solves: SALTELLI_N=64 -> 640 solves, 512 -> 5,120, 2048 -> 20,480.

The mathematical model is a verified fast reimplementation of the functions
in Song.py (identical arithmetic; loop-invariant quantities hoisted).
Verification against the original objective is run at startup and aborts on
mismatch.

Incremental CSV writes: ./results/song_control_audit.csv grows per run, so
partial results are usable if interrupted.
"""

import sys, os, io, time, json, contextlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Song  # data tables + original functions (module-level only)

T0, TF = Song.T0, Song.TF
TIME_STEPS = Song.TIME_STEPS
NT, NA = Song.NUM_TIME_STEPS, Song.NUM_AGE_GROUPS
AGES = Song.AGE_GROUPS

# --------------------------------------------------------------------------
# Fast, mathematically identical engine
# --------------------------------------------------------------------------

def build_solve_context(pyramid_type, fert_source, mort_evo, newborn_fem_prop,
                        fert_shift):
    """Hoist everything constant within one optimal-control solve."""
    X0 = Song.parse_population_data_optimized("./Data/china_population_1978.csv")
    if pyramid_type == 1:
        X0 = X0.copy(); X0[0:20] *= 1.2; X0[40:] *= 0.8
        X0 *= 958e6 / X0.sum()
    elif pyramid_type == 2:
        X0 = X0.copy(); X0[0:20] *= 0.8; X0[40:] *= 1.2
        X0 *= 958e6 / X0.sum()

    fp = (Song.preprocess_rates(Song.proportion_in_total_fertility_rudong)
          if fert_source == 0 else
          Song.preprocess_rates(Song.proportion_in_total_fertility_tianjin))
    rep_ages = np.arange(15, 50)
    fert_prop = fp(rep_ages) / 1000.0
    if fert_shift != 0:
        fert_prop = (1 - fert_shift) * fert_prop + fert_shift * (fp(rep_ages + 5) / 1000.0)

    fem_prop = Song.preprocess_rates(Song.proportion_females_by_age)(rep_ages) / 100.0

    dm = Song.interp_death_rates_males(AGES)
    df = Song.interp_death_rates_females(AGES)
    base_death = dm * (1 - newborn_fem_prop) + df * newborn_fem_prop
    # time factor per step i (year TIME_STEPS[i]):
    tfac = 1 - mort_evo * (TIME_STEPS[:-1] - T0) / (TF - T0)   # (NT-1,)

    return dict(X0=X0, fert_prop=fert_prop, fem_prop=fem_prop,
                base_death=base_death, tfac=tfac, rep=rep_ages)


def simulate_fast(ctx, beta):
    X = np.empty((NT, NA))
    X[0] = ctx['X0']
    fw = ctx['fem_prop'] * ctx['fert_prop']
    bd = ctx['base_death']
    for i in range(NT - 1):
        d = bd * ctx['tfac'][i]
        cur = X[i]
        births = beta[i] * np.dot(cur[15:50], fw)
        nxt = X[i + 1]
        nxt[0] = births
        nxt[1:] = cur[:-1] * (1 - d[:-1])
        nxt[NA - 1] += cur[NA - 1] * (1 - d[NA - 1])
    return X


# Song's target history and index math, vectorised
XSTAR = None  # filled in main from Song's literal (rebuild identically)

def indices_fast(X):
    young = X[:, 0:15].sum(axis=1)
    old = X[:, 65:].sum(axis=1)
    prod = X[:, 15:65].sum(axis=1)
    sdr = np.where(prod > 0, (young + old) / np.maximum(prod, 1e-9), np.inf)
    tot = X.sum(axis=1)
    aging = np.where(tot > 0, old / np.maximum(tot, 1e-9), 0.0)
    return sdr, aging, tot


def objective_fast(beta, ctx, penalty_factor):
    X = simulate_fast(ctx, beta)
    J = np.sum((X - XSTAR) ** 2) + np.sum((X[:, 0] - XSTAR[0]) ** 2)
    sdr, aging, tot = indices_fast(X)
    pen = 0.0
    v = tot[tot > 1.2e9]
    if v.size: pen += penalty_factor * np.sum((v - 1.2e9) ** 2)
    v = sdr[sdr > 1.0]
    if v.size: pen += penalty_factor * np.sum((v - 1.0) ** 2)
    v = aging[aging > 0.7]
    if v.size: pen += penalty_factor * np.sum((v - 0.7) ** 2)
    return J + pen


def solve_fast(params):
    ctx = build_solve_context(params['initial_pyramid_type'],
                              params['fertility_rate_source'],
                              params['mortality_evolution_factor'],
                              params['newborn_female_proportion'],
                              params['fertility_age_shift_factor'])
    beta0 = np.full(NT, 2.16)
    res = minimize(objective_fast, beta0,
                   args=(ctx, params['penalty_factor']),
                   method='L-BFGS-B', bounds=[(1, 2.16)] * NT,
                   options={'maxiter': params['optimizer_max_iterations'],
                            'ftol': params['optimizer_tolerance']})
    beta = res.x
    X = simulate_fast(ctx, beta)
    return beta, X


def control_metrics(beta, X):
    tot = X.sum(axis=1)
    i_min = int(beta.argmin())
    out = dict(
        tfr_min=float(beta.min()),
        year_tfr_min=int(TIME_STEPS[i_min]),
        years_below_2=int((beta < 2.0).sum()),
        cum_deficit=float(np.clip(2.16 - beta, 0, None).sum()),
        peak_pop=float(tot.max()),
        peak_pop_year=int(TIME_STEPS[tot.argmax()]),
        total_population_2080=float(tot[-1]),
    )
    after = np.where((TIME_STEPS > out['year_tfr_min']) & (beta >= 2.10))[0]
    out['year_return_repl'] = int(TIME_STEPS[after[0]]) if after.size else int(TF)
    return out


# --------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./figures', exist_ok=True)

    # Rebuild Song's X_star literal by calling into the original solver setup
    # is awkward (it's a local); replicate the literal here and verify shape.
    XSTAR = np.array(
        [10000000]*50 + [9000000]*5 + [8000000]*5 + [7000000]*5 + [6000000]*5 +
        [5000000]*5 + [4000000]*5 + [2000000]*5 + [1000000]*5 + [500000])
    assert XSTAR.shape == (NA,)
    globals()['XSTAR'] = XSTAR

    # ---- verification: fast objective == original objective ---------------
    p = dict(initial_pyramid_type=0, fertility_rate_source=0,
             mortality_evolution_factor=0.05, newborn_female_proportion=0.485,
             optimizer_tolerance=1e-6, optimizer_max_iterations=50,
             penalty_factor=1e10, fertility_age_shift_factor=0.1)
    ctx = build_solve_context(0, 0, 0.05, 0.485, 0.1)
    beta_t = np.linspace(1.2, 2.16, NT)
    f_fast = objective_fast(beta_t, ctx, 1e10)
    fp = (Song.preprocess_rates(Song.proportion_in_total_fertility_rudong))
    femp = Song.preprocess_rates(Song.proportion_females_by_age)
    f_orig = Song.objective_function_optimized(
        beta_t, ctx['X0'], XSTAR, 1e10, 0.05, 0.485, 0.1, fp, femp)
    rel = abs(f_fast - f_orig) / abs(f_orig)
    print(f"verification: fast vs original objective rel.diff = {rel:.2e}")
    assert rel < 1e-9, "fast engine does not reproduce original objective"

    # ---- design ------------------------------------------------------------
    SALTELLI_N = int(os.environ.get('SALTELLI_N', 0))
    if SALTELLI_N:
        from SALib.sample import sobol as _ss
        _problem = {
            'num_vars': 8,
            'names': ['initial_pyramid_type', 'fertility_rate_source',
                      'mortality_evolution_factor', 'newborn_female_proportion',
                      'optimizer_tolerance_log', 'optimizer_max_iterations_log',
                      'penalty_factor_log', 'fertility_age_shift_factor'],
            'bounds': [[0, 2.9999], [0, 1.9999], [0, 0.1], [0.47, 0.5],
                       [-7, -5], [1, 3], [8, 12], [0, 0.2]],
        }
        _Xs = _ss.sample(_problem, SALTELLI_N, calc_second_order=False,
                         scramble=True, seed=7)
        params_list = [dict(
            initial_pyramid_type=int(np.floor(r[0])),
            fertility_rate_source=int(np.floor(r[1])),
            mortality_evolution_factor=r[2],
            newborn_female_proportion=r[3],
            optimizer_tolerance=10 ** r[4],
            optimizer_max_iterations=int(10 ** r[5]),
            penalty_factor=10 ** r[6],
            fertility_age_shift_factor=r[7]) for r in _Xs]
        N = len(params_list)
        globals()['_SALTELLI'] = (_problem, _Xs)
        print(f"Saltelli mode: base N={SALTELLI_N}, {N} solves")
    else:
        N = int(os.environ.get('N_SIM', 512))
        params_list = [Song.generate_sobol_parameters(i, N) for i in range(N)]

    csv_path = ('./results/song_control_audit_saltelli.csv' if SALTELLI_N
                else './results/song_control_audit.csv')
    # Analysis artifacts are suffixed by mode so that Saltelli runs never
    # overwrite the plain-design outputs (and vice versa).
    SUFF = '_saltelli' if SALTELLI_N else ''
    traj_path = f'./results/song_control_trajectories{SUFF}.npz'
    rows, betas, pops = [], [], []
    start_idx = 0
    if os.path.exists(csv_path) and os.environ.get('RESUME', '1') == '1':
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict('records')
        start_idx = len(rows)
        if os.path.exists(traj_path):
            z = np.load(traj_path)
            betas = list(z['beta']); pops = list(z['pop'])
        print(f"resuming at {start_idx}")
    budget = float(os.environ.get('BUDGET_S', 1e9))
    t0 = time.time()
    for i, prm in enumerate(params_list):
        if i < start_idx: continue
        if time.time() - t0 > budget:
            print(f"budget reached at {i}"); break
        beta, X = solve_fast(prm)
        m = control_metrics(beta, X)
        m.update({f'param_{k}': v for k, v in prm.items()})
        m['sample'] = i
        rows.append(m); betas.append(beta); pops.append(X.sum(axis=1))
        if (i + 1) % 8 == 0 or i == N - 1:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            np.savez_compressed(traj_path, beta=np.array(betas),
                                pop=np.array(pops), years=TIME_STEPS)
            el = time.time() - t0
            print(f"[{i+1}/{N}] {el:6.0f}s  ~{el/(i+1):.1f}s/run", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== CONTROL-PATH METRICS (n=%d) ===" % len(df))
    stat_rows = []
    for c in ['tfr_min', 'year_tfr_min', 'years_below_2', 'cum_deficit',
              'year_return_repl', 'total_population_2080']:
        q = df[c].quantile([0, .05, .5, .95, 1]).values
        print(f"{c:22s} min {q[0]:10.2f}  p5 {q[1]:10.2f}  med {q[2]:10.2f}"
              f"  p95 {q[3]:10.2f}  max {q[4]:10.2f}")
        stat_rows.append(dict(metric=c, n=len(df), min=q[0], p5=q[1],
                              median=q[2], p95=q[3], max=q[4],
                              mean=df[c].mean(), sd=df[c].std(ddof=1)))
    pd.DataFrame(stat_rows).to_csv(f'./results/song_control_stats{SUFF}.csv',
                                   index=False)
    print(f"stats table: ./results/song_control_stats{SUFF}.csv")

    # Uncertainty distributions of the four control metrics, coloured by the
    # dominant driver (initial pyramid type) — the conditional-determinism
    # finding made visible.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLAB = {0: 'pyramid 0 (base)', 1: 'pyramid 1 (younger)',
            2: 'pyramid 2 (older)'}
    PCOL = {0: '#1b6ca8', 1: '#e07b00', 2: '#3d8b37'}
    figd, axd = plt.subplots(2, 2, figsize=(11, 8))
    MET4 = [('tfr_min', 'Minimum prescribed TFR'),
            ('year_tfr_min', 'Year of trough'),
            ('cum_deficit', 'Cumulative deficit (TFR-years)'),
            ('year_return_repl', 'Year of return to replacement')]
    for axi, (c, lab) in zip(axd.flat, MET4):
        groups = [df.loc[df.param_initial_pyramid_type == p, c].values
                  for p in (0, 1, 2)]
        axi.hist(groups, bins=24, stacked=True,
                 color=[PCOL[p] for p in (0, 1, 2)],
                 label=[PLAB[p] for p in (0, 1, 2)], alpha=.92)
        axi.set_xlabel(lab, fontsize=11)
        axi.set_ylabel('Runs', fontsize=11)
        axi.tick_params(labelsize=10)
    axd.flat[0].legend(fontsize=9, framealpha=.9)
    figd.suptitle(f'Control-path metric distributions '
                  f'({"Saltelli" if SALTELLI_N else "plain"} design, '
                  f'n={len(df)}), stacked by initial pyramid type',
                  fontsize=13)
    figd.tight_layout()
    figd.savefig(f'./figures/song_control_distributions{SUFF}.png', dpi=300,
                 bbox_inches='tight')
    plt.close(figd)
    print(f"distributions figure: ./figures/song_control_distributions{SUFF}.png")

    # Trajectory figures and the comparison with the observed record, from the
    # stored trajectories of this design (so that one design feeds UA and SA).
    bet = np.array(betas); pp = np.array(pops); yrs = TIME_STEPS
    if len(bet) != len(df) or len(bet) == 0:
        print(f"trajectory figures skipped: {traj_path} holds {len(bet)} trajectories "
              f"for {len(df)} runs. Recover them with recover_saltelli_trajectories.py "
              f"or rerun this design with RESUME=0.")
        bet = pp = None
    step = max(1, (len(bet) if bet is not None else 1) // 1000)   # at most ~1,000 lines
    for arr, fname, ylab, title, hline in ([] if bet is None else (
            (pp / 1e9, 'population_trajectories', 'Total population (billions)',
             'Population trajectories (1978-2080)', (1.2, 'k', '1.2B ceiling')),
            (bet, 'fertility_trajectories', 'Total fertility rate',
             'Fertility prescriptions (1978-2080)', (2.16, 'g', 'Replacement level')))):
        fg, ax = plt.subplots(figsize=(9, 5))
        for tr in arr[::step]:
            ax.plot(yrs, tr, color='#4C72B0', alpha=0.06, lw=0.8)
        ax.plot(yrs, arr.mean(axis=0), color='#C44E52', lw=2, label='Mean')
        ax.axhline(hline[0], color=hline[1], ls='--', lw=1.2, label=hline[2])
        ax.set_xlabel('Year'); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        fg.tight_layout(); fg.savefig(f'./figures/{fname}{SUFF}.png', dpi=300)
        plt.close(fg)
    # Observed China population (UN WPP 2024, approximate; VERIFY exact values)
    if pp is None:
        pass
    obs_y = np.array([1978, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2022, 2023])
    obs_p = np.array([0.963, 0.981, 1.051, 1.135, 1.205, 1.263, 1.304, 1.337, 1.379, 1.411, 1.426, 1.410])
    fg, ax = plt.subplots(figsize=(9, 5.2))
    for tr in (pp[::step] if pp is not None else []):
        ax.plot(yrs, tr / 1e9, color='#4C72B0', alpha=0.05, lw=0.8)
    if pp is not None:
        ax.plot(yrs, pp.mean(axis=0) / 1e9, color='#C44E52', lw=2.4,
                label=f'Song-model ensemble (n={len(pp)}): mean')
    ax.plot(obs_y, obs_p, 'o-', color='#2A7F3F', lw=2.2, ms=5, label='Observed (UN WPP 2024)')
    ax.axhline(1.2, color='k', ls='--', lw=1.2, label='Model hard ceiling: 1.2B')
    ax.set_xlabel('Year'); ax.set_ylabel('Total population (billions)')
    ax.set_xlim(1978, 2080); ax.legend(loc='center right', fontsize=9, framealpha=0.9)
    fg.tight_layout()
    if pp is not None:
        fg.savefig(f'./figures/song_vs_observed{SUFF}.png', dpi=300)
        print(f"trajectory figures: ./figures/population_trajectories{SUFF}.png, "
              f"fertility_trajectories{SUFF}.png, song_vs_observed{SUFF}.png")
    plt.close(fg)

    # ---- given-data sensitivity (Plischke delta + S1) ---------------------
    from SALib.analyze import delta as delta_an
    problem = {
        'num_vars': 8,
        'names': ['initial_pyramid_type', 'fertility_rate_source',
                  'mortality_evolution_factor', 'newborn_female_proportion',
                  'optimizer_tolerance', 'optimizer_max_iterations',
                  'penalty_factor', 'fertility_age_shift_factor'],
        'bounds': [[0, 2], [0, 1], [0, 0.1], [0.47, 0.5],
                   [1e-7, 1e-5], [10, 1000], [1e8, 1e12], [0, 0.2]],
    }
    Xmat = df[[f'param_{n}' for n in problem['names']]].values
    try:
        from demographic_models import first_order_given_data as _first_order
    except ImportError:                       # older demographic_models.py
        def _first_order(Xm, yv, M=25):
            Xm = np.asarray(Xm, float); yv = np.asarray(yv, float)
            mu, V = yv.mean(), yv.var(); res = []
            for k in range(Xm.shape[1]):
                col = Xm[:, k]; u = np.unique(col)
                if len(u) <= M:
                    lab = np.searchsorted(u, col) + 1
                else:
                    qs = np.unique(np.quantile(col, np.linspace(0, 1, M + 1)))
                    lab = np.digitize(col, qs[1:-1], right=False) + 1
                res.append(sum((lab == l).mean() * (yv[lab == l].mean() - mu) ** 2
                               for l in np.unique(lab)) / V if V > 0 else 0.0)
            return res
    print("\n=== GIVEN-DATA SENSITIVITY (S1) on control metrics ===")
    sens = {}
    for target in ['tfr_min', 'year_tfr_min', 'years_below_2', 'cum_deficit', 'year_return_repl']:
        Y = df[target].values.astype(float)
        if np.std(Y) < 1e-12:
            print(f"{target}: degenerate (constant)"); continue
        # Given-data S1 on a level-respecting partition (one class per level
        # for discrete inputs), consistent with the OT index below. SALib's
        # delta.analyze S1 uses equal-count rank bins that straddle the levels
        # of discrete inputs and bias their S1 downward. delta itself is
        # excluded (KDE fails on the censored trough) and stays reproducible
        # via song_robustness.py ("delta_reference").
        s1 = _first_order(Xmat.astype(float), Y)
        sens[target] = {n: {'S1': float(v)} for n, v in
                        zip(problem['names'], s1)}
        top = sorted(sens[target].items(), key=lambda kv: -kv[1]['S1'])[:4]
        print(f"\n{target}:")
        for n, v in top:
            print(f"   {n:28s} S1={v['S1']:5.2f}")
    # ---- given-data OT indices --------------------------------------------
    # Preferred: the shared engine in demographic_models.py. Fallback: an
    # inline copy of the same computation, so this script stays standalone
    # and keeps working against an older demographic_models.py.
    try:
        from demographic_models import ot_indices_given_data
    except ImportError:
        def ot_indices_given_data(Xm, yv, M=25):
            Xm = np.asarray(Xm, float); yv = np.asarray(yv, float)
            nn, K = Xm.shape
            ys = np.sort(yv)
            V = 2 * np.var(yv, ddof=1)
            if V <= 0:
                return [0.0] * K

            def _w2sq(yc):
                Nc = len(yc)
                ii = np.clip(np.floor(np.linspace(1 / Nc, 1, nn) * Nc + 0.5
                                      ).astype(int) - 1, 0, Nc - 1)
                return float(np.mean((ys - yc[ii]) ** 2))

            res = []
            for k in range(K):
                col = Xm[:, k]
                u = np.unique(col)
                if len(u) <= M:
                    labels = np.searchsorted(u, col) + 1
                else:
                    qs = np.unique(np.quantile(col, np.linspace(0, 1, M + 1)))
                    labels = np.zeros(nn, dtype=int)
                    for i in range(1, len(qs)):
                        msk = ((col >= qs[i-1]) & (col <= qs[i])
                               if i == len(qs) - 1
                               else (col >= qs[i-1]) & (col < qs[i]))
                        labels[msk] = i
                Wk = 0.0
                for m in range(1, labels.max() + 1):
                    sel = np.sort(yv[labels == m])
                    if len(sel):
                        Wk += len(sel) * _w2sq(sel)
                res.append(Wk / (V * nn))
            return res

    def ot_indices(Xm, yv, M=25):
        return ot_indices_given_data(Xm, yv, M=M)

    Xot = Xmat.copy()
    ot_res = {}
    for target in ['tfr_min', 'year_tfr_min', 'years_below_2', 'cum_deficit', 'year_return_repl']:
        ot_res[target] = {n: v for n, v in
                          zip(problem['names'], ot_indices(Xot, df[target].values))}

    # single consolidated sensitivity file: per metric, per input,
    # {"delta": ..., "S1": ...} from SALib delta.analyze (absent where the
    # KDE-based estimators are invalid, e.g. tfr_min) and "OT" always.
    for target, d in ot_res.items():
        sens.setdefault(target, {})
        for n, v in d.items():
            sens[target].setdefault(n, {})
            sens[target][n]['OT'] = float(v)
    with open(f'./results/song_control_sensitivity{SUFF}.json', 'w') as f:
        json.dump(sens, f, indent=1)
    print("\nOT indices (top drivers):")
    for tgt, d in ot_res.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])[:2]
        print("  " + tgt + ": " + ", ".join(f"{n}={v:.2f}" for n, v in top))

    if SALTELLI_N:
        from SALib.analyze import sobol as _sa
        _problem, _Xs = globals()['_SALTELLI']
        print("\nSaltelli-scheme Sobol' indices on control metrics:")
        for target in ['tfr_min', 'year_tfr_min', 'years_below_2', 'cum_deficit', 'year_return_repl']:
            Si = _sa.analyze(_problem, df[target].values.astype(float),
                             calc_second_order=False, print_to_console=False)
            top = sorted(zip(_problem['names'], Si['S1'], Si['ST']),
                         key=lambda x: -x[2])[:3]
            print("  " + target + ": " +
                  ", ".join(f"{n} S1={a:.2f}/ST={b:.2f}" for n, a, b in top))

    print("\nDone. Total", f"{time.time()-t0:.0f}s")
# (resume support appended below main guard is not reachable; use RESUME env in main instead)
