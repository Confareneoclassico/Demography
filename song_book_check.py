"""
song_book_check.py
------------------
Checks the reimplementation in Song.py against what Song, Tuan and Yu (1985),
Population Control in China: Theory and Applications, report:

0. the compiled engine against the book's matrix form of the population
   equation, X(t+1) = [H(t) + beta(t) B(t)] X(t) (2.2-13), built here
   independently from the death rates and female shares of each year;
1. the transcribed tables reproduce the book's own totals (Table 2.5);
2. the target X* (eqs. 8.2-1 to 8.2-3, N* = 700 million) and its dependency
   and aging indices, against the values the book gives for the stationary
   population of Program 1 (p. 264: dependency ratio 0.6, aging index 0.55);
3. the critical (replacement) fertility implied by the published death
   rates (Table 3.3), against the book's 2.16;
4. the optimal control of Program 1 (Section 8.5) solved with the base-case
   inputs, against Table 8.14 (beta(t) and N(t) for 1978-2080);
5. Song's own published control path projected through the model engine,
   which separates the projection from the optimisation;
6. items 4 and 5 again with all death rates scaled by 0.6 (life expectancy
   about 74 instead of 65), the survival level at which the book's figures
   for the stationary target are reproduced (the book sets the critical
   fertility at 2.16 "after allowing for improvements in survival", Cohen
   1990, but does not tabulate the improved rates).

Writes results/song_book_check.json and figures/song_book_check.png.
"""
import os, sys, json
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import Song
import Song_control_audit as sca

YEARS = [1978, 2000, 2020, 2040, 2060, 2080]
# Table 8.14 (p. 264), Program 1. Population printed in units of 10^8.
BOOK_BETA = [1.22, 1.09, 1.98, 2.16, 2.16, 2.16]
BOOK_N = [958.0, 1056.0, 1046.0, 913.9, 764.4, 738.8]          # millions
# Program 1 control path: Table 8.14 values, with the shape of Figure 8.8
# (p. 262) read off between them (approximate).
SONG_PATH_YEARS = [1978, 1980, 1984, 1990, 2000, 2003, 2006, 2012, 2016, 2020, 2026, 2030, 2080]
SONG_PATH_BETA = [1.22, 1.23, 1.05, 1.00, 1.09, 1.50, 1.85, 1.98, 1.89, 1.98, 2.13, 2.16, 2.16]
P_INFANT = Song.FEMALE_INFANT_SHARE_1978
IDX = [y - Song.T0 for y in YEARS]


def critical_tfr(h, mu_f):
    """TFR at which each woman is replaced by one daughter (net reproduction
    rate 1) under the model's timing: daughters per woman aged i are
    beta h_i times the female share of births, survivorship to age i."""
    l = np.ones(Song.NUM_AGE_GROUPS)
    l[1:] = np.cumprod(1 - mu_f[:-1])
    return 1.0 / (P_INFANT * float(np.dot(h, l[Song.REPRODUCTIVE_AGES])))


def matrix_form_check():
    """Largest relative difference between the engine and the matrix form
    (2.2-13), with b_i = (1 - mu_00)(1 - mu_0(t)) k_i(t) h_i in the first row
    of B(t), 1 - mu_i(t) on the sub-diagonal of H(t), and the infants
    x_0(t) = beta(t) C(t)'X(t), c_i = (1 - mu_00) k_i(t) h_i (8.4-6); k_i(t)
    and mu_i(t) are the female share and the death rate of both sexes."""
    rng = np.random.default_rng(5)
    worst = 0.0
    m = Song.NUM_AGE_GROUPS - 1
    df, dm = Song.death_rates_females, Song.death_rates_males
    for pyr, src, mef, p, shift in ((0, 0, 0.0, P_INFANT, 0.0), (1, 1, 0.1, 0.47, 0.2), (2, 0, 0.05, 0.5, 0.1)):
        ctx = sca.build_solve_context(pyr, src, mef, p, shift)
        beta = rng.uniform(*Song.BETA_BOUNDS, Song.NUM_TIME_STEPS)
        X = sca.simulate_fast(ctx, beta)
        X0 = Song.initial_population(pyr)
        F = X0 * Song.female_share_1978
        M = X0 - F
        F[0] = M[0] = 0.0
        h = np.zeros(m + 1)
        h[Song.REPRODUCTIVE_AGES] = Song.fertility_pattern(src, shift)
        for t in range(Song.NUM_TIME_STEPS - 1):
            tf = 1 - mef * t / (Song.TF - Song.T0)
            x = F + M
            k = np.divide(F, x, out=np.zeros_like(x), where=x > 0)
            x0 = beta[t] * np.sum((1 - Song.MU00) * k[1:] * h[1:] * x[1:])
            worst = max(worst, abs(x0 / X[t, 0] - 1))
            F[0], M[0] = p * x0, (1 - p) * x0
            k[0] = p
            mu = (k * df + (1 - k) * dm) * tf
            H = np.zeros((m, m))
            for i in range(1, m):
                H[i, i - 1] = 1 - mu[i]
            H[m - 1, m - 1] += 1 - mu[m]                 # open 90+ group
            B = np.zeros((m, m))
            B[0, :] = (1 - Song.MU00) * (1 - mu[0]) * k[1:] * h[1:]
            worst = max(worst, float(np.max(np.abs((H + beta[t] * B) @ x[1:] / X[t + 1, 1:] - 1))))
            Fn, Mn = np.zeros_like(F), np.zeros_like(M)
            Fn[1:] = F[:-1] * (1 - df[:-1] * tf)
            Mn[1:] = M[:-1] * (1 - dm[:-1] * tf)
            Fn[-1] += F[-1] * (1 - df[-1] * tf)
            Mn[-1] += M[-1] * (1 - dm[-1] * tf)
            F, M = Fn, Mn
    return worst


def stationary_indices(scale):
    w = Song.stationary_weights(Song.death_rates_females * scale,
                                Song.death_rates_males * scale, P_INFANT)
    X = Song.TARGET_POPULATION * w / w.sum()
    return dict(life_expectancy=float(w.sum()), infants_millions=float(X[0] / 1e6),
                dependency_ratio=float(Song.dependency_ratio(X)),
                aging_index=float(Song.aging_index(X, w.sum())[0]))


def solve_program_1(src, scale):
    """Program 1 with the base-case inputs and all death rates scaled by
    `scale` (the target X* and the life expectancy follow the scaled rates)."""
    from scipy.optimize import minimize
    ctx = sca.build_solve_context(0, src, 0.0, P_INFANT, 0.0)
    df, dm = Song.death_rates_females * scale, Song.death_rates_males * scale
    w = Song.stationary_weights(df, dm, P_INFANT)
    xstar = Song.TARGET_POPULATION * w / w.sum()
    e0 = np.full(Song.NUM_TIME_STEPS, w.sum())
    lo, hi = Song.BETA_BOUNDS
    beta = np.full(Song.NUM_TIME_STEPS, hi)
    for pf in Song.penalty_schedule(1e10):
        args = (ctx['F0'], ctx['M0'], ctx['h'], df, dm, ctx['tfac'], P_INFANT, Song.MU00, xstar, e0,
                sca.MIDAGES, pf, Song.SOCIAL_DEPENDENCY_MAX, Song.AGING_INDEX_MAX, Song.POPULATION_MAX, lo, hi)
        res = minimize(lambda b: sca._objective_and_gradient(np.asarray(b, float), *args), beta, jac=True,
                       method='L-BFGS-B', bounds=[Song.BETA_BOUNDS] * Song.NUM_TIME_STEPS,
                       options={'maxiter': 5000, 'ftol': 1e-12})
        beta = np.clip(res.x, lo, hi)
    X = sca._simulate(ctx['F0'], ctx['M0'], beta, ctx['h'], df, dm, ctx['tfac'], P_INFANT, Song.MU00)
    return beta, X, e0


def project(beta, src, scale):
    ctx = sca.build_solve_context(0, src, 0.0, P_INFANT, 0.0)
    X = sca._simulate(ctx['F0'], ctx['M0'], np.asarray(beta, float), ctx['h'],
                      Song.death_rates_females * scale, Song.death_rates_males * scale,
                      ctx['tfac'], P_INFANT, Song.MU00)
    return X.sum(axis=1)


def main():
    out = {}
    # 0. engine against the matrix form (2.2-13)
    out['equation_2_2_13_max_rel_diff'] = matrix_form_check()
    # 1. tables
    out['table_2_5_totals'] = dict(
        rudong=sum(v for k, v in Song.fertility_rates_rudong_1973.items() if k < 40) + 10 * Song.fertility_rates_rudong_1973[40],
        rudong_book=1731.6,
        tianjin=sum(Song.fertility_rates_tianjin_1978.values()), tianjin_book=1199.0)
    # 2. target and stationary indices
    out['target'] = dict(total_millions=float(Song.X_STAR.sum() / 1e6), **stationary_indices(1.0),
                         book_dependency_ratio=0.6, book_aging_index=0.55)
    # 3. critical fertility
    out['critical_tfr'] = {name: critical_tfr(Song.fertility_pattern(src), Song.death_rates_females)
                           for src, name in ((0, 'rudong'), (1, 'tianjin'))}
    out['critical_tfr']['book'] = 2.16
    scale216 = brentq(lambda s: critical_tfr(Song.fertility_pattern(0), Song.death_rates_females * s) - 2.16, 0.05, 1.0)
    out['hazard_scale_for_critical_2_16'] = dict(scale=scale216, **stationary_indices(scale216))
    out['improved_survival_0_6'] = stationary_indices(0.6)
    # 4. Program 1 solved with the base-case inputs
    out['program_1'] = {'years': YEARS, 'book_beta': BOOK_BETA, 'book_population_millions': BOOK_N}
    sols = {}
    for scale in (1.0, 0.6):
        for src, name in ((0, 'rudong'), (1, 'tianjin')):
            beta, X, e0 = solve_program_1(src, scale)
            tot = X.sum(axis=1)
            m = sca.control_metrics(beta, X)
            omega = Song.aging_index(X, e0)
            sols[(name, scale)] = (beta, tot)
            out['program_1'][f'{name}_hazard_x{scale}'] = dict(
                beta=[round(float(beta[i]), 3) for i in IDX],
                population_millions=[round(float(tot[i]) / 1e6, 1) for i in IDX],
                peak_population_millions=round(float(tot.max()) / 1e6, 1), peak_year=int(Song.TIME_STEPS[tot.argmax()]),
                max_aging_index=round(float(omega.max()), 3),
                max_dependency_ratio=round(float(Song.dependency_ratio(X).max()), 3),
                **{k: m[k] for k in ('tfr_min', 'year_tfr_min', 'years_below_2', 'cum_deficit', 'year_return_repl')})
    # 5. Song's published path through the engine
    song_beta = np.interp(Song.TIME_STEPS, SONG_PATH_YEARS, SONG_PATH_BETA)
    out['song_path_projected'] = {}
    proj = {}
    for src, name in ((0, 'rudong'), (1, 'tianjin')):
        for scale in (1.0, 0.6):
            tot = project(song_beta, src, scale)
            key = f'{name}_hazard_x{scale}'
            proj[key] = tot
            out['song_path_projected'][key] = [round(float(tot[i]) / 1e6, 1) for i in IDX]

    os.makedirs('results', exist_ok=True)
    with open('results/song_book_check.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    yrs = Song.TIME_STEPS
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(yrs, song_beta, color='0.55', lw=1.4, ls='--', label='Song et al., Program 1 (Fig. 8.8, approx.)')
    ax[0].plot(YEARS, BOOK_BETA, 'ko', ms=6, label='Song et al., Table 8.14')
    ax[1].plot(YEARS, BOOK_N, 'ko', ms=6, label='Song et al., Table 8.14')
    for (name, scale), (beta, tot) in sols.items():
        col = '#1b6ca8' if name == 'rudong' else '#e07b00'
        ls = '-' if scale == 1.0 else '--'
        lab = f"{name.capitalize()} pattern, {'Table 3.3 death rates' if scale == 1.0 else 'death rates x 0.6'}"
        ax[0].plot(yrs, beta, color=col, lw=1.8, ls=ls, label=lab)
        ax[1].plot(yrs, tot / 1e6, color=col, lw=1.8, ls=ls, label=lab)
    ax[0].set_ylabel('Total fertility rate'); ax[1].set_ylabel('Total population (millions)')
    for a in ax:
        a.set_xlabel('Year'); a.set_xlim(1978, 2080); a.legend(fontsize=8, framealpha=.9)
    ax[0].set_ylim(0.9, 2.3)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/song_book_check.png', dpi=300)
    print('figure: figures/song_book_check.png')


if __name__ == '__main__':
    main()
