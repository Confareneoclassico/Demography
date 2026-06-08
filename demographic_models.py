"""
demographic_models.py
---------------------
Model library for "What demographic prediction can and cannot achieve".

Implements the three-paradigm, two-level experimental design:

  Level 1 – Projection paradigm (discrete variable: `paradigm`)
    paradigm = 0  →  Cohort-Component family (age-structured, discrete)
    paradigm = 1  →  Lotka intrinsic growth rate (continuous, equilibrium)
    paradigm = 2  →  ARIMA (purely statistical, no demographic structure)

  Level 2 – Mortality sub-model within the CC paradigm (discrete: `mortality_model`)
    mortality_model = 0  →  Exogenous mortality scaling (bespoke calibration)
    mortality_model = 1  →  Lee-Carter log-bilinear extrapolation

  Parametric uncertainty (continuous, active across all paradigms):
    fertility_scale, mortality_scale, migration_scale, trend_strength,
    and Lee-Carter-specific parameters when mortality_model = 1.

Note: net migration is zero at the planetary scale used in the paper; the
migration argument in cohort_component_model is a dummy for interface
completeness.
"""

import numpy as np
from scipy.optimize import root_scalar


# ---------------------------------------------------------------------------
# Level 1 / Level 2 — Cohort-Component model (shared across CC sub-models)
# ---------------------------------------------------------------------------

def cohort_component_model(population_age_t, survival_prob, fertility_rates,
                           net_migration_age, female_ratio=0.5):
    """
    Cohort-Component Model — age-structured population accounting.

    Shared by both CC sub-models (exogenous mortality scaling and
    Lee-Carter-augmented). The mortality driver differs between sub-models;
    this function is the accounting identity that applies whatever survival
    probabilities are supplied.

    Parameters
    ----------
    population_age_t : np.ndarray, shape (n_ages,)
        Population at each age x at time t.
    survival_prob : np.ndarray, shape (n_ages,)
        One-year survival probability S_x (from age x to x+1).
    fertility_rates : np.ndarray, shape (n_ages,)
        Age-specific fertility rates (births per woman per year).
    net_migration_age : np.ndarray or scalar
        Net migration at each age. Pass scalar 0 for the planetary case.
        If array, must match n_ages.
    female_ratio : float
        Proportion of population that is female (default 0.5).

    Returns
    -------
    np.ndarray, shape (n_ages,)
        Population at each age at time t+1.
    """
    num_ages = len(population_age_t)

    # Broadcast scalar migration to array; validate array length.
    if np.isscalar(net_migration_age):
        migration = np.full(num_ages, float(net_migration_age))
    else:
        migration = np.asarray(net_migration_age, dtype=float)
        if len(migration) != num_ages:
            raise ValueError(
                f"net_migration_age length ({len(migration)}) must match "
                f"population length ({num_ages})."
            )

    population_age_t_plus_1 = np.zeros(num_ages)

    # Births from female reproductive-age population (ages 15–49).
    reproductive_ages = np.arange(15, min(50, num_ages))
    female_pop = population_age_t[reproductive_ages] * female_ratio
    total_births = np.sum(female_pop * fertility_rates[reproductive_ages])

    # Age 0: newborn survivors + migration.
    population_age_t_plus_1[0] = total_births * survival_prob[0] + migration[0]

    # Ages 1 … n_ages-1.
    for age in range(1, num_ages):
        if age < num_ages - 1:
            survived = population_age_t[age - 1] * survival_prob[age - 1]
        else:
            # Open-ended oldest age group: survivors from age-1 and age itself.
            survived = (population_age_t[age - 1] +
                        population_age_t[age]) * survival_prob[age - 1]
        population_age_t_plus_1[age] = survived + migration[age]

    return population_age_t_plus_1


# ---------------------------------------------------------------------------
# Level 2 — Lee-Carter mortality sub-model (embedded in CC shell)
# ---------------------------------------------------------------------------

def lee_carter_model(a_x, b_x, k_t, e_x_t):
    """
    Lee-Carter log-bilinear mortality sub-model.

    Predicts log age-specific mortality rates ln(m_x(t)) = a_x + b_x*k_t + e_x_t.
    Used as a *mortality sub-model* inside the CC shell: the returned
    log-mortality rates are converted to survival probabilities and passed
    to cohort_component_model or the aggregate accounting step.

    Parameters
    ----------
    a_x : np.ndarray     — average log mortality by age
    b_x : np.ndarray     — age-specific sensitivity to k_t
    k_t : float          — overall time-varying mortality index
    e_x_t : np.ndarray   — age-specific error term

    Returns
    -------
    np.ndarray : predicted ln(m_x(t))
    """
    return a_x + b_x * k_t + e_x_t


# ---------------------------------------------------------------------------
# Level 1 — Lotka intrinsic growth rate paradigm
# ---------------------------------------------------------------------------

def lotka_intrinsic_growth_rate_model(l_x, m_x, r_initial_guess=0.01,
                                      max_iter=1000, tol=1e-6):
    """
    Lotka's intrinsic growth rate model (Euler-Lotka equation, Lotka 1925).

    Solves  ∫₀^∞ exp(-r·x)·l_x·m_x dx = 1  for the intrinsic rate of
    natural increase r. Represents a fundamentally different paradigm from
    the CC family: characterises long-run dynamics through an equilibrium
    growth rate rather than tracking cohort flows.

    Note: assumes stable age structure converging to equilibrium under
    current vital rates; application to declining-fertility human populations
    should be treated with care.

    Parameters
    ----------
    l_x : np.ndarray   — survival probability to age x from birth
    m_x : np.ndarray   — age-specific fertility rates
    r_initial_guess, max_iter, tol : solver controls

    Returns
    -------
    float : intrinsic rate of natural increase r
    """
    ages = np.arange(len(l_x))

    def euler_lotka(r):
        return np.sum(np.exp(-r * ages) * l_x * m_x) - 1

    try:
        result = root_scalar(euler_lotka, x0=r_initial_guess,
                             x1=r_initial_guess + 0.01,
                             method='secant', maxiter=max_iter, xtol=tol)
        return result.root
    except Exception:
        r = r_initial_guess
        for _ in range(max_iter):
            f_r = euler_lotka(r)
            h = 1e-7
            f_prime = (euler_lotka(r + h) - f_r) / h
            if abs(f_prime) < 1e-10:
                break
            r_new = r - f_r / f_prime
            if abs(r_new - r) < tol:
                return r_new
            r = r_new
        return r


# ---------------------------------------------------------------------------
# Level 1 — ARIMA paradigm (purely statistical, no demographic structure)
# ---------------------------------------------------------------------------

def fit_arima_baseline(historical_population, order=(1, 2, 0)):
    """
    Fit an ARIMA model to the historical world population series and return
    the fitted model result for later forecasting.

    Uses ARIMA(1,2,0) by default — a common specification for a second-order
    integrated series with moderate autocorrelation in increments, which
    describes the decelerating-growth trajectory of world population well.
    The model is fitted on log-transformed population for numerical stability.

    Parameters
    ----------
    historical_population : np.ndarray
        Annual total world population (persons), ordered chronologically.
    order : tuple (p, d, q)
        ARIMA order. Default (1, 2, 0).

    Returns
    -------
    ARIMAResultsWrapper
        Fitted statsmodels ARIMA result object.
    float
        Log-population value at the final historical year (used as seed
        for forecast reconstruction).
    """
    from statsmodels.tsa.arima.model import ARIMA

    log_pop = np.log(historical_population)
    model = ARIMA(log_pop, order=order)
    fitted = model.fit()
    return fitted, log_pop[-1]


def arima_step_growth_rates(fitted_arima, log_pop_seed, n_steps):
    """
    Extract the ARIMA-implied annual growth rates for the next n_steps years.

    Parameters
    ----------
    fitted_arima : ARIMAResultsWrapper
        Fitted ARIMA model.
    log_pop_seed : float
        Log-population at the last historical year.
    n_steps : int
        Number of projection years.

    Returns
    -------
    np.ndarray, shape (n_steps,)
        Annual growth rates r_t = exp(Δ ln P_t) - 1 implied by the ARIMA
        point forecast.
    """
    forecast_log = fitted_arima.forecast(steps=n_steps)
    # Prepend seed to get differences
    log_series = np.concatenate([[log_pop_seed], forecast_log])
    growth_rates = np.exp(np.diff(log_series)) - 1
    return growth_rates


# ---------------------------------------------------------------------------
# Aggregate accounting step (shared by UN WPP-style and Lee-Carter CC runs)
# ---------------------------------------------------------------------------

def aggregate_accounting_step(population_t, deaths_t, births_t, migration_t):
    """
    Aggregate cohort-component accounting identity:
        P(t+1) = P(t) - D(t) + B(t) + M(t).

    Used by:
    - The CC run when applying aggregate crude rates (avoids updating the
      full age distribution every step when only the total is needed).
    - The Lee-Carter CC run, where mortality comes from the LC sub-model
      and births from scaled crude birth rates.

    Parameters
    ----------
    population_t : float   — total population at time t
    deaths_t     : float   — total deaths
    births_t     : float   — total births
    migration_t  : float   — net migration (zero at planetary scale)

    Returns
    -------
    float : total population at time t+1
    """
    return population_t - deaths_t + births_t + migration_t


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Cohort-Component (scalar migration — planetary case)
    pop_t = np.array([1000, 900, 800, 700, 600])
    surv   = np.array([0.99, 0.98, 0.97, 0.96, 0.95])
    asfr   = np.zeros(5)
    print("CC P(t+1):", cohort_component_model(pop_t, surv, asfr, 0))

    # Lee-Carter sub-model
    a_x = np.array([-5.1, -4.6, -4.1])
    b_x = np.array([ 0.10,  0.09,  0.08])
    e_x = np.array([ 0.01,  0.005, 0.002])
    print("LC ln(m_x):", lee_carter_model(a_x, b_x, k_t=-1.0, e_x_t=e_x))

    # Lotka
    l_x = np.array([1.0, 0.98, 0.95, 0.90, 0.80, 0.65, 0.45, 0.25, 0.10, 0.01])
    m_x = np.array([0.0, 0.00, 0.05, 0.15, 0.20, 0.18, 0.10, 0.03, 0.00, 0.00])
    print("Lotka r:", lotka_intrinsic_growth_rate_model(l_x, m_x))

    # Aggregate accounting
    print("Aggregate P(t+1):", aggregate_accounting_step(8e9, 60e6, 140e6, 0))


# ---------------------------------------------------------------------------
# ARIMA sub-model: multi-order pre-fitting and selection
# ---------------------------------------------------------------------------

# Candidate ARIMA specifications — four genuinely distinct assumptions about
# the data-generating process for world population:
#
#   0: ARIMA(0,1,1) — equivalent to exponential smoothing (ETS); d=1, assumes
#                     the growth rate is stationary with an MA(1) correction.
#                     The simplest non-trivial extrapolation of recent trends.
#
#   1: ARIMA(1,2,0) — AR(1) on second differences; d=2, assumes the change in
#                     growth rate is autoregressive. Our previous fixed choice.
#
#   2: ARIMA(0,2,1) — IMA(2,1); d=2 with an MA(1) correction on second
#                     differences. Different autocorrelation structure from (1,2,0).
#
#   3: ARIMA(2,1,0) — AR(2) on first differences; d=1, allows for oscillatory
#                     components in the annual growth rate.
#
# d=1 specifications assume growth rates are stationary (convergence to a
# fixed long-run rate). d=2 specifications allow the growth rate itself to
# trend, capturing decelerating growth patterns more naturally.

ARIMA_ORDERS = [
    (0, 1, 1),   # index 0: ETS-equivalent
    (1, 2, 0),   # index 1: AR on 2nd differences (previous default)
    (0, 2, 1),   # index 2: MA on 2nd differences
    (2, 1, 0),   # index 3: AR(2) on 1st differences
]


def fit_all_arima_orders(historical_population):
    """
    Pre-fit all candidate ARIMA specifications to the historical population
    series and return their implied annual growth rates for 2024–2075.

    Called once at startup; results are stored in the data dict and indexed
    by `arima_order` in the simulation loop.

    Parameters
    ----------
    historical_population : np.ndarray
        Annual world population in persons, 1950–2023.

    Returns
    -------
    dict mapping int → np.ndarray of shape (52,)
        Annual growth rates g_t = exp(Δ ln P_t) - 1 for each specification.
    list of str
        Human-readable label for each specification.
    """
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    import warnings as _warnings

    log_pop = np.log(historical_population)
    log_seed = log_pop[-1]
    growth_rates_by_order = {}
    labels = []

    for idx, order in enumerate(ARIMA_ORDERS):
        label = f"ARIMA{order}"
        labels.append(label)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                result = _ARIMA(log_pop, order=order).fit()
            forecast_log = result.forecast(steps=52)
            log_series = np.concatenate([[log_seed], forecast_log])
            gr = np.exp(np.diff(log_series)) - 1
            growth_rates_by_order[idx] = gr
            print(f"    ARIMA{order}: fitted OK. "
                  f"Mean 52-year growth rate = {gr.mean():.4f}")
        except Exception as exc:
            # Fallback: linear extrapolation of the last 10-year growth rate
            print(f"    ARIMA{order}: fit failed ({exc}). Using linear fallback.")
            recent_gr = (historical_population[-1] /
                         historical_population[-10]) ** (1/10) - 1
            growth_rates_by_order[idx] = np.full(52, recent_gr)

    return growth_rates_by_order, labels
