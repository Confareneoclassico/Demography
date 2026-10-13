"""
MoMP_world.py  (v2 — component-level redesign, Sept 2026)
---------------------------------------------------------
Global population MoMP experiment, restructured so the structural forks sit
where demographic practice actually argues (R1's point): at the level of the
COMPONENTS of change within the cohort-component paradigm, with the aggregate
ARIMA paradigm as the structural outsider. Lotka removed (Caswell: it is the
CC asymptote, not a paradigm; v1 additionally contained a maternity-function
error — see demographic_models.lotka_intrinsic_r).

Ten uncertain inputs, D = 10, Saltelli N x (D+2) evaluations:

  #  name                 range          active in            role
  1  paradigm             [0, 1.9999]    all                  L1: 0 CC | 1 ARIMA
  2  fertility_model      [0, 2.9999]    CC                   L2: const|trend|converge
  3  mortality_model      [0, 1.9999]    CC                   L2: det. e0 | stoch. index
  4  arima_order          [0, 3.9999]    ARIMA                L2: 4 (p,d,q) specs
  5  fertility_scale      [0.90, 1.10]   CC                   level of ASFR   (WPP-PI*)
  6  tfr_floor            [1.45, 1.95]   CC & f=2             long-run TFR    (WPP-PI*)
  7  tfr_trend_scale      [0.50, 1.50]   CC & f=1             x fitted decline rate
  8  e0_gain              [0.10, 0.25]   CC & m=0             yr of e0 per yr (WPP-PI*)
  9  mort_realization     [0.001,0.999]  CC & m=1             quantile of e0 RWD path
 10  arima_realization    [0.001,0.999]  ARIMA                quantile of forecast s.e.

(*) Ranges marked WPP-PI are placeholders anchored on my reading of the WPP
2024 probabilistic projections; SYNC THESE against the downloaded WPP PI
files before the paper-grade run and cite the exact interval used.

Migration: identically zero at planetary scale; not an input (stated design
motivation — removes the component most error-prone at national scale).

Effective configurations (10):
  CC-const-e0trend  CC-const-stoch   CC-trend-e0trend  CC-trend-stoch
  CC-conv-e0trend   CC-conv-stoch    ARIMA(0,1,1) (1,2,0) (0,2,1) (2,1,0)

Run:      python MoMP_world.py             (requires ./Data files as before)
Selftest: python MoMP_world.py --selftest  (synthetic data, N=16, no files)
"""

import os, sys, time
import numpy as np
import pandas as pd

from demographic_models import (
    cohort_component_step, life_expectancy, build_e0_scale_map,
    survival_for_e0, fit_e0_random_walk, fit_recent_tfr_trend, tfr_path,
    fit_all_arima_orders, arima_growth_path, ARIMA_ORDERS,
)
from scipy.stats import norm

N_YEARS = 52                      # 2024..2075
YEAR0 = 2023

problem = {
    'num_vars': 10,
    'names': ['paradigm', 'fertility_model', 'mortality_model', 'arima_order',
              'fertility_scale', 'tfr_floor', 'tfr_trend_scale', 'e0_gain',
              'mort_realization', 'arima_realization'],
    'bounds': [[0, 1.9999], [0, 2.9999], [0, 1.9999], [0, 3.9999],
               [0.90, 1.10], [1.45, 1.95], [0.50, 1.50], [0.10, 0.25],
               [0.001, 0.999], [0.001, 0.999]],
}

FERT_LABEL = {0: 'const', 1: 'trend', 2: 'conv'}
MORT_LABEL = {0: 'e0trend', 1: 'stoch'}


def effective_label(row):
    if int(row[0]) == 0:
        return f"CC-{FERT_LABEL[int(row[1])]}-{MORT_LABEL[int(row[2])]}"
    return f"ARIMA{ARIMA_ORDERS[int(row[3])]}"


# --------------------------------------------------------------------------
def run_simulation(x, data):
    paradigm      = int(np.floor(x[0]))
    fert_model    = int(np.floor(x[1]))
    mort_model    = int(np.floor(x[2]))
    arima_order   = int(np.floor(x[3]))
    fert_scale    = x[4]
    tfr_floor     = x[5]
    trend_scale   = x[6]
    e0_gain       = x[7]
    mort_u        = x[8]
    arima_u       = x[9]

    if paradigm == 1:                                  # ---- ARIMA ----------
        g = arima_growth_path(data['arima_fits'][arima_order], arima_u)
        pop = data['P0'] * np.cumprod(1.0 + g)
        return pop[N_YEARS - (2075 - 2050) - 1], pop[-1]   # 2050, 2075

    # ---- Cohort-Component ------------------------------------------------
    tfr = tfr_path(fert_model, N_YEARS, data['tfr0'], fert_scale,
                   trend_g=data['tfr_trend_g'], trend_scale=trend_scale,
                   floor=tfr_floor)
    if mort_model == 0:
        e0_traj = data['e0_0'] + e0_gain * np.arange(1, N_YEARS + 1)
    else:
        z = norm.ppf(mort_u)
        t = np.arange(1, N_YEARS + 1)
        e0_traj = data['e0_0'] + data['e0_drift'] * t + data['e0_sigma'] * np.sqrt(t) * z

    pop = data['pop_age_2023'].copy()
    shape = data['asfr_shape']                          # sums to 1 over 15-49
    p2050 = p2075 = None
    for i in range(N_YEARS):
        S = survival_for_e0(e0_traj[i], data['e0_to_s'], data['mu'])
        asfr = shape * tfr[i]
        pop = cohort_component_step(pop, S, asfr)
        year = YEAR0 + 1 + i
        if year == 2050: p2050 = pop.sum()
        if year == 2075: p2075 = pop.sum()
    return p2050, p2075


# --------------------------------------------------------------------------
def load_real_data():
    """Load WPP-derived inputs; identical files as v1 (see repo README)."""
    wpp = pd.read_excel("./Data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx",
                        sheet_name="Estimates", skiprows=16)
    world = wpp[wpp['Region, subregion, country or area *'] == 'World']
    w = world[world['Year'] == 2023].iloc[0]
    e0_series = world[(world['Year'] >= 1950) & (world['Year'] <= 2023)][
        'Life Expectancy at Birth, both sexes (years)'].values
    tfr_series = world[(world['Year'] >= 1950) & (world['Year'] <= 2023)][
        'Total Fertility Rate (live births per woman)'].values
    hist = pd.read_csv("./Data/world_population_combined_1950_2100.csv")
    hist_pop = hist[hist["Year"] <= 2023][
        "Total Population, as of 1 July (thousands)"].values * 1000.0

    return build_data(
        P0=w['Total Population, as of 1 July (thousands)'] * 1000.0,
        tfr0=float(w['Total Fertility Rate (live births per woman)']),
        e0_0=float(w['Life Expectancy at Birth, both sexes (years)']),
        e0_series=e0_series, tfr_series=tfr_series, hist_pop=hist_pop,
        base_survival=build_base_survival_from_indicators(w))


def build_base_survival_from_indicators(w):
    """Single-year survival schedule from the WPP summary indicators
    (unchanged logic from v1; the schedule is then anchored to e0 via the
    proportional-hazard map, so residual shape error affects levels little)."""
    S = np.ones(101)
    imr = w['Infant Mortality Rate (infant deaths per 1,000 live births)'] / 1000
    S[0] = 1 - imr
    u5 = w['Under-Five Mortality (deaths under age 5 per 1,000 live births)'] / 1000
    S[1:5] = 1 - (u5 - imr) / 4
    m40 = w['Mortality before Age 40, both sexes (deaths under age 40 per 1,000 live births)'] / 1000
    S[5:40] = 1 - (1 - (1 - m40) ** (1 / 35))
    m60 = w['Mortality before Age 60, both sexes (deaths under age 60 per 1,000 live births)'] / 1000
    S[40:60] = 1 - (1 - (1 - (m60 - m40)) ** (1 / 20))
    ages = np.arange(60, 101)
    S[60:] = 1 - np.minimum(0.01 * np.exp(0.07 * (ages - 60)), 0.99)
    return S


def build_data(P0, tfr0, e0_0, e0_series, tfr_series, hist_pop, base_survival,
               age_shares=None):
    if age_shares is None:                      # smooth plausible world shares
        raw = np.exp(-0.5 * ((np.arange(101) - 0) / 55.0) ** 2) + 0.15
        raw[85:] *= np.linspace(1, 0.15, 16)
        age_shares = raw / raw.sum()
    # anchor the baseline schedule so its e0 equals the observed e0_0
    e0_to_s, mu = build_e0_scale_map(base_survival)
    base_anchored = survival_for_e0(e0_0, e0_to_s, mu)
    e0_to_s, mu = build_e0_scale_map(base_anchored)

    sh = np.zeros(101)
    g = np.exp(-0.5 * ((np.arange(15, 50) - 27) / 7.0) ** 2)
    sh[15:50] = g / g.sum()

    E0_FIT_YEARS = 30   # fit window for the e0 random walk (last 30 annual
                        # increments, 1993-2023, COVID included); full-series
                        # drift overstates the current pace of improvement.
    drift, sigma = fit_e0_random_walk(e0_series[-(E0_FIT_YEARS + 1):])
    d_full, s_full = fit_e0_random_walk(e0_series)
    print(f"  e0 RWD (full series, for reference): drift {d_full:+.3f}, sigma {s_full:.3f}")
    print(f"  e0 RWD fit: drift {drift:+.3f} yr/yr, sigma {sigma:.3f}")
    trend_g = fit_recent_tfr_trend(tfr_series, years=20)
    print(f"  TFR recent log-trend: {trend_g:+.4f}/yr "
          f"(~{100*(np.exp(trend_g)-1):+.2f}%/yr)")
    print("  fitting ARIMA candidates...")
    arima_fits, labels = fit_all_arima_orders(hist_pop, n_steps=N_YEARS)

    return dict(P0=P0, tfr0=tfr0, e0_0=e0_0,
                pop_age_2023=P0 * age_shares, asfr_shape=sh,
                e0_to_s=e0_to_s, mu=mu, e0_drift=drift, e0_sigma=sigma,
                tfr_trend_g=trend_g, arima_fits=arima_fits)


def make_synthetic_data():
    """Selftest fixture: plausible world-like series, no files needed."""
    yrs = np.arange(1950, 2024)
    e0_series = 47 + 26.5 * (1 - np.exp(-(yrs - 1950) / 30.0))     # 47 -> ~73.4
    tfr_series = 1.55 + 3.45 * np.exp(-(yrs - 1950) / 33.0)        # ~5 -> ~2.25
    hist_pop = 2.5e9 * np.exp(np.cumsum(
        np.concatenate([[0], 0.021 * np.exp(-(yrs[1:] - 1950) / 55.0)])))
    A = 101; ages = np.arange(A)
    q = np.full(A, 0.0012); q[0] = 0.027; q[1:5] = 0.0023
    q[40:60] = 0.004; q[60:] = np.minimum(0.011 * np.exp(0.075 * (ages[60:] - 60)), .6)
    return build_data(P0=8.09e9, tfr0=2.25, e0_0=73.4,
                      e0_series=e0_series, tfr_series=tfr_series,
                      hist_pop=hist_pop, base_survival=1 - q)


# --------------------------------------------------------------------------
if __name__ == '__main__':
    selftest = '--selftest' in sys.argv
    os.makedirs('./results', exist_ok=True)

    print("Loading data..." + (" (synthetic selftest)" if selftest else ""))
    data = make_synthetic_data() if selftest else load_real_data()

    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_an

    import os as _os
    N = int(_os.environ.get('N_BASE', 16 if selftest else 2048))
    # Same design as the Song audit: N(k+2) runs, first- and total-order
    # indices only (no second-order block).
    X = sobol_sample.sample(problem, N, calc_second_order=False,
                            scramble=True, seed=42)
    print(f"{len(X)} evaluations (N={N}, D={problem['num_vars']})")

    t0 = time.time()
    Y50 = np.empty(len(X)); Y75 = np.empty(len(X))
    for i, row in enumerate(X):
        a, b = run_simulation(row, data)
        Y50[i], Y75[i] = a / 1e6, b / 1e6            # millions
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(X)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"simulation loop: {time.time()-t0:.0f}s")

    lab = np.array([effective_label(r) for r in X])
    df = pd.DataFrame(X, columns=problem['names'])
    df['effective_run'] = lab
    df['population_2050_millions'] = Y50
    df['population_2075_millions'] = Y75
    df.to_csv('./results/simulation_results_v2.csv', index=False)

    print("\nPer-configuration medians, 2075 (billions):")
    med = df.groupby('effective_run')['population_2075_millions'].median() / 1000
    for k, v in med.sort_values().items():
        print(f"   {k:22s} {v:6.2f}B")
    print(f"\nOverall 2075: {Y75.min()/1e3:.2f}–{Y75.max()/1e3:.2f}B, "
          f"mean {Y75.mean()/1e3:.2f}B; 2050 mean {Y50.mean()/1e3:.2f}B")

    import json as _json
    from demographic_models import ot_indices_given_data as _ot

    summary = {'stats': {}, 'sobol': {}, 'ot': {},
               'fits': dict(e0_drift=float(data['e0_drift']),
                            e0_sigma=float(data['e0_sigma']),
                            tfr_trend=float(data['tfr_trend_g']))}
    try:
        _hist = pd.read_csv('./Data/world_population_combined_1950_2100.csv')
        summary['un_medium'] = {str(y): float(
            _hist[_hist.Year == y]['Total Population, as of 1 July (thousands)'
                                   ].values[0] / 1e6)
            for y in (2050, 2075)}
    except Exception:
        summary['un_medium'] = {}

    X_ot = param_values.copy() if 'param_values' in dir() else X.copy()
    for _j, _n in enumerate(problem['names']):
        if _n in ('paradigm', 'fertility_model', 'mortality_model', 'arima_order'):
            X_ot[:, _j] = np.floor(X_ot[:, _j]) + 1

    for tag, Y in (('2050', Y50), ('2075', Y75)):
        Si = sobol_an.analyze(problem, Y, calc_second_order=False,
                              print_to_console=False)
        print(f"\nSobol' indices — {tag}:")
        print(f"{'input':22s} {'S1':>7s} {'ST':>7s}")
        for j, n in enumerate(problem['names']):
            print(f"  {n:20s} {Si['S1'][j]:7.3f} {Si['ST'][j]:7.3f}")
        with open(f'./results/sobol_v2_{tag}.txt', 'w') as f:
            for j, n in enumerate(problem['names']):
                f.write(f"{n}\t{Si['S1'][j]:.4f}\t{Si['ST'][j]:.4f}\n")

        summary['sobol'][tag] = {n: (float(Si['S1'][j]), float(Si['ST'][j]))
                                 for j, n in enumerate(problem['names'])}
        _oti = _ot(X_ot, Y, M=25)
        summary['ot'][tag] = {n: float(_oti[j])
                              for j, n in enumerate(problem['names'])}
        g = df.groupby('effective_run')[f'population_{tag}_millions']
        summary['stats'][tag] = {k: dict(med=float(v.median()/1e3),
                                         p5=float(v.quantile(.05)/1e3),
                                         p95=float(v.quantile(.95)/1e3),
                                         n=int(len(v)))
                                 for k, v in g}
        summary['stats'][tag]['_overall'] = dict(
            mean=float(Y.mean()/1e3), lo=float(Y.min()/1e3),
            hi=float(Y.max()/1e3),
            p5=float(np.quantile(Y, .05)/1e3), p95=float(np.quantile(Y, .95)/1e3))

    with open('./results/momp_v2_summary.json', 'w') as f:
        _json.dump(summary, f, indent=1)
    print("\nSummary written to ./results/momp_v2_summary.json")

    # Uncertainty distributions stacked by configuration (Figure 4 of the paper)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs('./figures', exist_ok=True)
    order = sorted([k for k in summary['stats']['2075'] if k != '_overall'],
                   key=lambda k: summary['stats']['2075'][k]['med'])
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, tag, Yv in ((axes[0], '2050', Y50), (axes[1], '2075', Y75)):
        arrs = [df.loc[df.effective_run == k, f'population_{tag}_millions'].values / 1e3 for k in order]
        ax.hist(arrs, bins=48, stacked=True, color=[cmap(i % 10) for i in range(len(order))],
                label=order, alpha=.92)
        if summary['un_medium'].get(tag):
            ax.axvline(summary['un_medium'][tag], color='k', ls=':', lw=1.4, label='UN medium')
        ax.set_xlabel(f'Projected population {tag} (billions)'); ax.set_ylabel('Runs'); ax.set_title(tag)
    axes[1].legend(fontsize=7, framealpha=.9)
    plt.tight_layout(); plt.savefig('./figures/uncertainty_distributions_world.png', dpi=300)
    print("figure: ./figures/uncertainty_distributions_world.png")
