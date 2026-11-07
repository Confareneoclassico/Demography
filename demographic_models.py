"""
demographic_models.py  (v2 — component-level redesign, Sept 2026)
-----------------------------------------------------------------
Model library for "What demographic prediction can and cannot achieve",
revised in response to the Risk Analysis reviews (R1: the structural forks
that matter to practitioners are at the level of the COMPONENTS of change;
Caswell: Lotka's stable-population r is the CC model's own asymptote, not a
separate paradigm, and the previous implementation additionally fed the
both-sex maternity schedule into the Euler-Lotka equation).

Design implemented here:

  Level 1 — projection paradigm (`paradigm`)
    0 → Cohort-Component: age-structured Leslie-matrix projection
        (Leslie 1945; Preston, Heuveline & Guillot 2001)
    1 → ARIMA: aggregate statistical extrapolation of total population,
        no demographic structure

  Level 2 — component sub-models WITHIN each paradigm
    CC fertility_model:
        0 → constant rates      TFR(t) = TFR0 · φ_f
        1 → trend extrapolation TFR(t) = max(TFR_MIN, TFR0·φ_f·exp(ĝ·λ·t))
                                 ĝ = fitted recent annual log-TFR change
        2 → convergence         TFR(t) = F + (TFR0·φ_f − F)·exp(−t/τ)
                                 (UN-style convergence toward long-run level F;
                                  Alkema et al. 2011; Raftery et al. 2012)
    CC mortality_model:
        0 → deterministic trend e0(t) = e0₀ + γ·t          (γ uncertain)
        1 → stochastic index    e0(t) = e0₀ + d̂·t + σ̂·√t·Φ⁻¹(u)
                                 (random-walk-with-drift fitted to the
                                  1993–2023 world e0 series — the Lee-Carter
                                  k_t logic applied to the life-expectancy
                                  index)
        Either path is turned into age-specific mortality by scaling the WPP
        2024 single-age death rates of 2023 (both sexes, and each sex) by a
        common proportional-hazard factor chosen so that the both-sex life
        table has the target e0, bounded to E0_CAP.

  Cohort-component projection (revised Nov 2026, see CHANGELOG.md): two
  sexes, single years of age 0-99 and 100+, base population and age pattern
  of fertility from WPP 2024 (world, 1 July 2023), survivorship ratios from
  single-year life tables (Preston, Heuveline & Guillot 2001, ch. 3 and 6),
  births from the average female exposure over the year.
    ARIMA arima_order: four (p,d,q) specifications (unchanged from v1)
    ARIMA statistical uncertainty: quantile path from the forecast s.e.

Migration is identically zero at the planetary scale — this is stated as a
motivation for the global-scale design (it removes the component R1
identifies as most error-prone at national scale) and it is NOT a sampled
input.

The Euler-Lotka function is retained below, corrected, ONLY to document and
reproduce the v1 error for the response letter: it now requires an explicit
daughters-only maternity schedule and raises if handed a schedule whose
gross sum looks like a both-sex TFR.
"""

import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.stats import norm

FEMALE_BIRTH_SHARE = 0.4854   # female share of births (sex ratio at birth ~106 males per 100 females)
TFR_MIN = 0.8                 # floor for trend extrapolation (≈ lowest observed)
CONV_TAU = 30.0               # convergence time constant, years (fixed; see text)
E0_CAP = (60.0, 92.0)         # bounds on projected life expectancy at birth
S_GRID = np.geomspace(0.02, 4.0, 600)   # proportional-hazard factors for the e0 map


# ==========================================================================
# Life tables (single years, open interval at the last age)
# ==========================================================================

def a0_coale_demeny(m0, sex='B'):
    """Person-years lived by infants who die in their first year (a0),
    Coale-Demeny West formulas as given by Preston et al. (2001, Table 3.3);
    both sexes as the mean of the two."""
    f = 0.35 if m0 >= 0.107 else 0.053 + 2.800 * m0
    m = 0.33 if m0 >= 0.107 else 0.045 + 2.684 * m0
    return {'F': f, 'M': m}.get(sex, 0.5 * (f + m))


def life_table(mx, sex='B'):
    """Single-year period life table from central death rates m_x, ages
    0..A-1, the last age open-ended. Returns l_x, L_x and T_x (radix 1), so
    that e0 = T_0."""
    mx = np.asarray(mx, dtype=float)
    a = np.full(mx.size, 0.5)
    a[0] = a0_coale_demeny(mx[0], sex)
    q = mx / (1 + (1 - a) * mx)
    q[-1] = 1.0
    l = np.concatenate([[1.0], np.cumprod(1 - q[:-1])])
    d = l * q
    L = l - (1 - a) * d
    L[-1] = l[-1] / mx[-1]
    T = np.cumsum(L[::-1])[::-1]
    return l, L, T


def life_expectancy_from_mx(mx, sex='B'):
    return float(life_table(mx, sex)[2][0])


def build_e0_scale_map(mx_both, s_grid=S_GRID):
    """Map a life-expectancy target to the proportional-hazard factor s that
    gives the both-sex schedule s * mx_both that life expectancy.

    The grid must cover E0_CAP: targets are clipped to E0_CAP and looked up
    with bounds_error=True, so no extrapolation or fill value is ever used.
    (Before Nov 2026 the map was built on an approximate e0 and its fill
    values were inverted: targets above about 91.3 years were given the
    largest hazard factor, and life expectancy collapsed to about 51 years.)"""
    e0s = np.array([life_expectancy_from_mx(s * mx_both) for s in s_grid])
    assert np.all(np.diff(e0s) < 0), "e0 must fall as hazards rise"
    assert e0s[-1] < E0_CAP[0] and e0s[0] > E0_CAP[1], \
        f"hazard grid covers e0 {e0s[-1]:.1f}-{e0s[0]:.1f}, not E0_CAP {E0_CAP}"
    return interp1d(e0s[::-1], s_grid[::-1], bounds_error=True)


def hazard_factor_for_e0(e0_target, e0_to_s):
    return float(e0_to_s(np.clip(e0_target, *E0_CAP)))


def survivorship(mx_female, mx_male, s):
    """Life tables of both sexes with hazards scaled by s: (L, T) per sex."""
    _, LF, TF = life_table(s * mx_female, 'F')
    _, LM, TM = life_table(s * mx_male, 'M')
    return LF, TF, LM, TM


# ==========================================================================
# Cohort-component projection, one year, two sexes
# ==========================================================================

def project_two_sex(F, M, asfr, LF, TF, LM, TM, female_birth_share=FEMALE_BIRTH_SHARE):
    """Project women F and men M (single years 0..A-2, A-1 open) one year.

    Survivors: P_{x+1}(t+1) = P_x(t) L_{x+1}/L_x, and the open interval
    receives (P_{A-2} + P_{A-1}) T_{A-1}/T_{A-2}. Births over the year are
    sum_x f_x (W_x(t) + W_x(t+1)) / 2, with W the women; the female share of
    births is female_birth_share and newborns enter at age 0 as births
    times L_0 (radix 1). Net migration is zero at the planetary scale."""
    A = F.size
    Fn = np.empty(A)
    Mn = np.empty(A)
    Fn[1:A - 1] = F[0:A - 2] * LF[1:A - 1] / LF[0:A - 2]
    Mn[1:A - 1] = M[0:A - 2] * LM[1:A - 1] / LM[0:A - 2]
    Fn[A - 1] = (F[A - 2] + F[A - 1]) * TF[A - 1] / TF[A - 2]
    Mn[A - 1] = (M[A - 2] + M[A - 1]) * TM[A - 1] / TM[A - 2]
    births = 0.5 * np.dot(asfr[1:], F[1:] + Fn[1:])   # no fertility at age 0
    Fn[0] = female_birth_share * births * LF[0]
    Mn[0] = (1 - female_birth_share) * births * LM[0]
    return Fn, Mn


def fit_e0_random_walk(e0_series):
    """Fit RWD to the historical world e0 series: returns (drift, sigma)."""
    d = np.diff(np.asarray(e0_series, dtype=float))
    return float(d.mean()), float(d.std(ddof=1))


# ==========================================================================
# Fertility sub-models: TFR path builders
# ==========================================================================

def fit_recent_tfr_trend(tfr_series, years=20):
    """Mean annual log-TFR change over the last `years` observations."""
    x = np.asarray(tfr_series, dtype=float)[-(years + 1):]
    return float(np.mean(np.diff(np.log(x))))


def tfr_path(model, n_years, tfr0, fertility_scale,
             trend_g=None, trend_scale=1.0, floor=1.8, tau=CONV_TAU):
    """TFR trajectory for the chosen fertility sub-model (t = 1..n_years)."""
    t = np.arange(1, n_years + 1, dtype=float)
    start = tfr0 * fertility_scale
    if model == 0:                                    # constant rates
        return np.full(n_years, start)
    if model == 1:                                    # trend extrapolation
        return np.maximum(TFR_MIN, start * np.exp(trend_g * trend_scale * t))
    if model == 2:                                    # convergence
        return floor + (start - floor) * np.exp(-t / tau)
    raise ValueError(f"unknown fertility_model {model}")


# ==========================================================================
# ARIMA paradigm
# ==========================================================================

ARIMA_ORDERS = [
    (0, 1, 1),   # exponential-smoothing equivalent; d=1
    (1, 2, 0),   # AR(1) on second differences; d=2
    (0, 2, 1),   # MA(1) on second differences; d=2
    (2, 1, 0),   # AR(2) on first differences; d=1
]


def fit_all_arima_orders(historical_population, n_steps=52):
    """Fit each candidate order to log world population; return, per order,
    the forecast mean log-level path and its standard-error path.

    Per-run trajectories are then mean + z*se with z = PPF(u): a coherent
    quantile path — one statistical-uncertainty realisation per run."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    import warnings as _w

    log_pop = np.log(np.asarray(historical_population, dtype=float))
    seed = log_pop[-1]
    out, labels = {}, []
    for idx, order in enumerate(ARIMA_ORDERS):
        labels.append(f"ARIMA{order}")
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = _ARIMA(log_pop, order=order).fit()
                fc = res.get_forecast(steps=n_steps)
            mean = np.asarray(fc.predicted_mean, dtype=float)
            se = np.asarray(fc.se_mean, dtype=float)
        except Exception as exc:                       # pragma: no cover
            print(f"    ARIMA{order}: fit failed ({exc}); linear fallback")
            g = np.log(historical_population[-1] / historical_population[-11]) / 10
            mean = seed + g * np.arange(1, n_steps + 1)
            se = np.zeros(n_steps)
        out[idx] = dict(mean=mean, se=se, seed=seed)
    return out, labels


def arima_growth_path(fit, u):
    """Annual growth rates for statistical-uncertainty quantile u in (0,1)."""
    z = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
    log_path = np.concatenate([[fit['seed']], fit['mean'] + z * fit['se']])
    return np.exp(np.diff(log_path)) - 1.0


# ==========================================================================
# Given-data optimal-transport sensitivity (shared by MoMP_world.py and
# Song_control_audit.py; Wasserstein-2^2 normalised by 2 Var(Y), one 1-based
# bin per level for low-cardinality inputs, quantile bins otherwise)
# ==========================================================================

def ot_indices_given_data(X, y, M=25):
    """Return the list of OT indices, one per column of X."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, K = X.shape
    ys = np.sort(y)
    V = 2 * np.var(y, ddof=1)
    if V <= 0:
        return [0.0] * K

    def _w2sq(yc):
        Nc = len(yc)
        idx = np.clip(np.floor(np.linspace(1 / Nc, 1, n) * Nc + 0.5).astype(int) - 1,
                      0, Nc - 1)
        return float(np.mean((ys - yc[idx]) ** 2))

    out = []
    for k in range(K):
        col = X[:, k]
        u = np.unique(col)
        if len(u) <= M:
            labels = np.searchsorted(u, col) + 1          # one bin per level
        else:
            qs = np.unique(np.quantile(col, np.linspace(0, 1, M + 1)))
            labels = np.zeros(n, dtype=int)
            for i in range(1, len(qs)):
                m = ((col >= qs[i-1]) & (col <= qs[i]) if i == len(qs) - 1
                     else (col >= qs[i-1]) & (col < qs[i]))
                labels[m] = i
        Wk = 0.0
        for m in range(1, labels.max() + 1):
            sel = np.sort(y[labels == m])
            if len(sel):
                Wk += len(sel) * _w2sq(sel)
        out.append(Wk / (V * n))
    return out


def first_order_given_data(X, y, M=25):
    """Given-data first-order index on the SAME partition as
    ot_indices_given_data: one class per level for inputs with <= M distinct
    values, quantile classes otherwise. Returns one index per column of X.

    Why not SALib's delta.analyze S1: its equal-count rank bins are designed
    for continuous inputs and straddle the levels of a discrete input, which
    biases that input's S1 downward (0.71 instead of 0.91 for the Song
    pyramid on the cumulative deficit). On a shared partition the transport
    index decomposes exactly as  iota = S1/2 + R,  where R >= 0 measures the
    input's effect on the spread and shape of the conditional distributions
    (R = (1 - sqrt(1 - S1))**2 / 2 for a pure shift of a Gaussian output)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, K = X.shape
    mu, V = y.mean(), y.var()
    if V <= 0:
        return [0.0] * K
    out = []
    for k in range(K):
        col = X[:, k]
        u = np.unique(col)
        if len(u) <= M:
            labels = np.searchsorted(u, col) + 1
        else:
            qs = np.unique(np.quantile(col, np.linspace(0, 1, M + 1)))
            labels = np.zeros(n, dtype=int)
            for i in range(1, len(qs)):
                m = ((col >= qs[i-1]) & (col <= qs[i]) if i == len(qs) - 1
                     else (col >= qs[i-1]) & (col < qs[i]))
                labels[m] = i
        Vi = 0.0
        for lab in np.unique(labels):
            sel = labels == lab
            Vi += sel.sum() / n * (y[sel].mean() - mu) ** 2
        out.append(Vi / V)
    return out


# ==========================================================================
# Euler-Lotka (retained, corrected — NOT part of the v2 design)
# ==========================================================================

def lotka_intrinsic_r(l_x, maternity_female, r0=0.005, max_iter=1000,
                      tol=1e-8, _allow_suspicious=False):
    """Intrinsic rate of natural increase from the Euler-Lotka renewal
    equation (Lotka 1907; Sharpe & Lotka 1911; Dublin & Lotka 1925) —
    developed for human populations; the stable-population asymptote of the
    cohort-component model under time-invariant rates (Caswell 2019).

    `maternity_female` must be the DAUGHTERS-ONLY schedule (both-sex ASFR x
    female birth share). v1 of this code passed the both-sex schedule,
    roughly doubling the implied NRR and inflating r from ~0.1%/yr to
    ~2.8%/yr at 2023 world rates; the guard below refuses schedules whose
    gross sum looks like a both-sex TFR unless explicitly overridden."""
    gross = float(np.sum(maternity_female))
    if gross > 1.6 and not _allow_suspicious:
        raise ValueError(
            f"maternity schedule sums to {gross:.2f} — looks like a both-sex "
            "TFR; multiply by the female birth share (~0.485) first, or pass "
            "_allow_suspicious=True to reproduce the v1 (erroneous) result.")
    ages = np.arange(len(l_x))

    def f(r):
        return np.sum(np.exp(-r * ages) * l_x * maternity_female) - 1.0

    try:
        return root_scalar(f, x0=r0, x1=r0 + 0.01, method='secant',
                           maxiter=max_iter, xtol=tol).root
    except Exception:
        r = r0
        for _ in range(max_iter):
            fr = f(r); h = 1e-7
            fp = (f(r + h) - fr) / h
            if abs(fp) < 1e-12: break
            rn = r - fr / fp
            if abs(rn - r) < tol: return rn
            r = rn
        return r


# ==========================================================================
# Self-test
# ==========================================================================

if __name__ == '__main__':
    import os
    import pandas as pd
    w = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data', 'wpp2024_world_2023_single_age.csv'))
    print("e0 from WPP 2024 single-age mx, world 2023: both sexes",
          round(life_expectancy_from_mx(w.mxB.values), 2), "| women",
          round(life_expectancy_from_mx(w.mxF.values, 'F'), 2), "| men",
          round(life_expectancy_from_mx(w.mxM.values, 'M'), 2), "(WPP 2024: 73.17, 75.89, 70.55)")
    e0_to_s = build_e0_scale_map(w.mxB.values)
    for tgt in (60, 73.17, 85, 91.7, 92, 95):
        sfac = hazard_factor_for_e0(tgt, e0_to_s)
        print(f"  e0 target {tgt:6.2f} -> factor {sfac:.3f} -> achieved "
              f"{life_expectancy_from_mx(sfac * w.mxB.values):.2f} (cap {E0_CAP})")
    p = tfr_path(2, 52, 2.25, 1.0, floor=1.6)
    print("convergence TFR: t=1", round(p[0], 3), " t=52", round(p[-1], 3))
    l_x = life_table(w.mxF.values, 'F')[0]
    asfr = 2.25 * w.pasfr.values / 100
    try:
        lotka_intrinsic_r(l_x, asfr)
    except ValueError as e:
        print("guard OK:", str(e)[:58], "...")
    print("corrected r =", f"{lotka_intrinsic_r(l_x, asfr * FEMALE_BIRTH_SHARE):+.5f}")
