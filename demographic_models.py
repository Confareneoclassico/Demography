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
                                  1950–2023 world e0 series — the Lee-Carter
                                  k_t logic applied to the life-expectancy
                                  index, without pretending to a full
                                  age-pattern fit we lack life tables for)
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

FEMALE_BIRTH_SHARE = 0.4854   # sex ratio at birth ~106 males/100 females (WPP)
TFR_MIN = 0.8                 # floor for trend extrapolation (≈ lowest observed)
CONV_TAU = 30.0               # convergence time constant, years (fixed; see text)
E0_CAP = (60.0, 92.0)         # sanity bounds on projected e0


# ==========================================================================
# Cohort-Component paradigm — Leslie projection step
# ==========================================================================

def cohort_component_step(pop_age, survival, asfr,
                          female_birth_share=FEMALE_BIRTH_SHARE):
    """One-year Leslie-matrix projection step (Leslie 1945).

    Uses age-specific fertility and survival rates to project the age
    structure of the population forward one year (Caswell 2001, 2019).
    Net migration is zero at the planetary scale and is not an argument.
    """
    A = len(pop_age)
    nxt = np.zeros(A)
    rep = np.arange(15, min(50, A))
    births = np.sum(pop_age[rep] * female_birth_share * asfr[rep])
    nxt[0] = births * survival[0]
    nxt[1:] = pop_age[:-1] * survival[:-1]
    nxt[A - 1] += pop_age[A - 1] * survival[A - 1]   # open-ended top group
    return nxt


# ==========================================================================
# Mortality machinery: baseline schedule, e0 <-> hazard-scale mapping
# ==========================================================================

def life_expectancy(survival):
    """e0 from a single-year survival schedule: e0 ≈ Σ l_x (person-years)."""
    l_x = np.cumprod(np.insert(survival[:-1], 0, 1.0))
    return float(l_x.sum())


def build_e0_scale_map(base_survival, s_grid=None):
    """Map life-expectancy targets to a proportional-hazard scale factor.

    Survival under scale s: S_x(s) = exp(-s·mu_x), mu_x = -ln S_x(base).
    Returns interp1d e0 -> s over a monotone grid, replacing per-year
    root-finding with table lookup.
    """
    mu = -np.log(np.clip(base_survival, 1e-9, 1 - 1e-12))
    if s_grid is None:
        s_grid = np.linspace(0.25, 3.0, 120)
    e0s = np.array([life_expectancy(np.exp(-s * mu)) for s in s_grid])
    order = np.argsort(e0s)
    return interp1d(e0s[order], s_grid[order], bounds_error=False,
                    fill_value=(s_grid[order][-1], s_grid[order][0])), mu


def survival_for_e0(e0_target, e0_to_s, mu):
    s = float(e0_to_s(np.clip(e0_target, *E0_CAP)))
    return np.exp(-s * mu)


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
    A = 101; ages = np.arange(A)
    q = np.full(A, 0.0012); q[0] = 0.027; q[1:5] = 0.0023
    q[40:60] = 0.004; q[60:] = np.minimum(0.011*np.exp(0.075*(ages[60:]-60)), .6)
    S = 1 - q
    print("e0(base) =", round(life_expectancy(S), 2))
    e0_to_s, mu = build_e0_scale_map(S)
    for tgt in (70, 77, 85):
        print(f"  e0 target {tgt} -> achieved "
              f"{life_expectancy(survival_for_e0(tgt, e0_to_s, mu)):.2f}")
    p = tfr_path(2, 52, 2.25, 1.0, floor=1.6)
    print("convergence TFR: t=1", round(p[0], 3), " t=52", round(p[-1], 3))
    l_x = np.cumprod(np.insert(S[:-1], 0, 1.0))
    asfr = np.zeros(A); sh = np.exp(-0.5*((np.arange(15,50)-27)/7)**2)
    asfr[15:50] = sh/sh.sum()*2.25
    try:
        lotka_intrinsic_r(l_x, asfr)
    except ValueError as e:
        print("guard OK:", str(e)[:58], "...")
    print("corrected r =", f"{lotka_intrinsic_r(l_x, asfr*FEMALE_BIRTH_SHARE):+.5f}")
