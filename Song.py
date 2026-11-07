import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import qmc
import seaborn as sns

# Reference implementation of the optimal-control model of Song, Tuan and Yu
# (1985), Population Control in China: Theory and Applications (Praeger),
# chapter 8 "Target population and optimum control", with the data tables of
# chapters 2, 3 and 6. Equation, table and page numbers refer to that book.
# The program reproduced is Program 1 of Section 8.5 (pp. 262-264): controls in
# [1, 2.16], peak population at most 1.2 billion, dependency index at most 1,
# aging index at most 0.7, target a stationary population of 700 million.

# Constants and initial conditions
T0 = 1978  # Initial year (Section 8.5)
TF = 2080  # Final year (Table 8.14 reports beta(t) and N(t) up to 2080)
TARGET_POPULATION = 700e6  # N*, Section 8.2

AGE_GROUPS = np.arange(0, 91) # single years 0..89 and an open 90+ group
NUM_AGE_GROUPS = len(AGE_GROUPS)
TIME_STEPS = np.arange(T0, TF + 1) # From 1978 to 2080
NUM_TIME_STEPS = len(TIME_STEPS)

# Age-indexed inputs are built from the tables below as step functions: the value
# given for age x applies from x up to the next tabulated age (the last one is
# open-ended). This is the convention of abridged life tables and of open-ended
# fertility groups.
def step_rates(rates_dict):
    keys = np.array(sorted(rates_dict))
    vals = np.array([rates_dict[k] for k in keys], dtype=float)
    def f(ages):
        idx = np.searchsorted(keys, np.asarray(ages), side='right') - 1
        return vals[np.clip(idx, 0, len(keys) - 1)]
    return f

# ---------------------------------------------------------------------------
# Data, transcribed from the book
# ---------------------------------------------------------------------------
# Table 3.3 (p. 76): age-specific death rates in China, 1973-1975, "forward
# mortality" mu_i, for ages 0, 1-4, 5-9, ..., 85-89 and 90+.
age_specific_death_rates_males = {
    0: 0.04748,
    1: 0.00898,
    5: 0.00231,
    10: 0.00101,
    15: 0.00110,
    20: 0.00148,
    25: 0.00157,
    30: 0.00200,
    35: 0.00286,
    40: 0.00410,
    45: 0.00620,
    50: 0.00980,
    55: 0.01527,
    60: 0.02500,
    65: 0.03694,
    70: 0.05886,
    75: 0.09051,
    80: 0.13585,
    85: 0.20000,
    90: 0.26597
}

age_specific_death_rates_females = {
    0: 0.04321,
    1: 0.00919,
    5: 0.00211,
    10: 0.00085,
    15: 0.00096,
    20: 0.00146,
    25: 0.00171,
    30: 0.00208,
    35: 0.00280,
    40: 0.00373,
    45: 0.00511,
    50: 0.00773,
    55: 0.01146,
    60: 0.01898,
    65: 0.02845,
    70: 0.04523,
    75: 0.07004,
    80: 0.10673,
    85: 0.15772,
    90: 0.23256
}

# Table 2.5 (p. 39): births per 1,000 women by age. Rudong (1973) has no entry
# at age 15 and an open-ended 40+ entry; the column total, 1731.6, counts the
# 40+ rate once for each age 40-49 (1691.6 + 10 x 4.0). Tianjin (1978) reports
# ages 23-37 (total 1199.0). Before Nov 2026 both columns were read one row
# too young (Rudong 15-38, Tianjin 22-36) and a Rudong entry at 39 was invented.
fertility_rates_rudong_1973 = {
    16: 0.8,
    17: 5.8,
    18: 11.0,
    19: 28.8,
    20: 73.2,
    21: 111.2,
    22: 188.9,
    23: 209.6,
    24: 208.7,
    25: 196.9,
    26: 139.5,
    27: 115.3,
    28: 88.7,
    29: 84.6,
    30: 49.1,
    31: 30.4,
    32: 30.6,
    33: 21.7,
    34: 20.9,
    35: 13.1,
    36: 25.1,
    37: 15.2,
    38: 10.5,
    39: 12.0,
    40: 4.0  # 40+, applies to each age 40-49
}

fertility_rates_tianjin_1978 = {
    23: 0.6,
    24: 7.5,
    25: 39.1,
    26: 116.0,
    27: 143.2,
    28: 175.9,
    29: 212.3,
    30: 205.8,
    31: 101.4,
    32: 85.2,
    33: 57.3,
    34: 27.0,
    35: 17.6,
    36: 8.1,
    37: 2.0
}

# Table 2.5, columns (3) and (4): the same schedules as proportions h_i of the
# total, per 1,000 (no 40+ share is printed for Rudong; 4.0/1731.6 gives 2.3
# per age). Kept for reference: the model builds h_i from the rates above,
# which reproduces these columns to their rounding.
proportion_in_total_fertility_rudong = {
    16: 0.5,
    17: 3.4,
    18: 6.4,
    19: 16.6,
    20: 42.3,
    21: 64.3,
    22: 109.2,
    23: 121.2,
    24: 120.6,
    25: 113.8,
    26: 80.6,
    27: 66.6,
    28: 51.3,
    29: 48.9,
    30: 28.4,
    31: 17.6,
    32: 17.7,
    33: 12.5,
    34: 12.1,
    35: 7.6,
    36: 14.5,
    37: 8.8,
    38: 6.1,
    39: 6.9
}

proportion_in_total_fertility_tianjin = {
    23: 0.5,
    24: 6.3,
    25: 32.6,
    26: 96.7,
    27: 119.3,
    28: 146.6,
    29: 176.9,
    30: 171.4,
    31: 84.5,
    32: 71.0,
    33: 47.7,
    34: 22.5,
    35: 14.7,
    36: 6.7,
    37: 1.7
}

# Table 6.1 (chapter 6): percentage of females by age, China 1978, k(r). The last
# entry (50) is used for all older ages. (Before Nov 2026 the transcription had
# skipped the second 48.92 at age 6 and the 46.64 at age 38, shifting ages
# 6-14 and 38-49 by one year.)
proportion_females_by_age = {
    0: 48.43,
    1: 48.92,
    2: 48.51,
    3: 48.81,
    4: 48.89,
    5: 48.92,
    6: 48.92,
    7: 49.49,
    8: 48.83,
    9: 48.75,
    10: 48.71,
    11: 48.69,
    12: 48.72,
    13: 48.79,
    14: 48.70,
    15: 49.42,
    16: 49.44,
    17: 48.71,
    18: 48.95,
    19: 48.86,
    20: 49.22,
    21: 48.79,
    22: 48.60,
    23: 49.17,
    24: 48.83,
    25: 48.97,
    26: 48.52,
    27: 48.83,
    28: 48.12,
    29: 47.52,
    30: 47.65,
    31: 47.10,
    32: 47.68,
    33: 47.90,
    34: 47.61,
    35: 47.37,
    36: 47.11,
    37: 46.93,
    38: 46.64,
    39: 47.24,
    40: 46.41,
    41: 46.24,
    42: 46.59,
    43: 46.78,
    44: 46.93,
    45: 46.34,
    46: 46.77,
    47: 46.57,
    48: 46.80,
    49: 46.85,
    50: 47.23
}

# Table 2.10 (chapter 2): sex ratio of the Chinese, 1975-1978
sex_ratio = {
    1975: {"women": 0.4876, "men": 0.5124},
    1976: {"women": 0.4873, "men": 0.5127},
    1977: {"women": 0.4877, "men": 0.5123},
    1978: {"women": 0.4872, "men": 0.5128}
}

# Table 2.12 (p. 48): proportion of female infants in zero ages
proportion_female_infants_zero_age = {
    1953: 0.4881,
    1964: 0.4906,
    1975: 0.4835,
    1978: 0.4843
}

# Function to parse population data from CSV
def parse_population_data_optimized(file_path):
    df = pd.read_csv(file_path)
    initial_population_distribution = np.zeros(91)

    for index, row in df.iterrows():
        age_group_str = str(row["Age"])
        male_pop = row["M"]
        female_pop = row["F"]

        if age_group_str == "100+":
            initial_population_distribution[90] += male_pop + female_pop
        else:
            if "-" in age_group_str:
                start_age, end_age = map(int, age_group_str.split("-"))
            else:
                start_age = int(age_group_str)
                end_age = int(age_group_str)

            total_pop_in_group = male_pop + female_pop
            num_years_in_group = end_age - start_age + 1
            pop_per_year = total_pop_in_group / num_years_in_group

            ages_to_assign = np.arange(start_age, end_age + 1)
            ages_to_assign[ages_to_assign >= 90] = 90 # Map all 90+ to index 90
            
            for age_idx in ages_to_assign:
                initial_population_distribution[age_idx] += pop_per_year

    return initial_population_distribution

def get_female_infant_proportion(year):
    return proportion_female_infants_zero_age.get(year, proportion_female_infants_zero_age[1978])

# Age-indexed inputs on the model's single-year grid (0..89, 90+)
REPRODUCTIVE_AGES = np.arange(15, 50)   # a1..a2 (zero rates outside the tables)
death_rates_males = step_rates(age_specific_death_rates_males)(AGE_GROUPS)
death_rates_females = step_rates(age_specific_death_rates_females)(AGE_GROUPS)
# Female share k_i of the 1978 population by age (Table 6.1, 50 = 50+)
female_share_1978 = step_rates(proportion_females_by_age)(AGE_GROUPS) / 100.0
FEMALE_INFANT_SHARE_1978 = proportion_female_infants_zero_age[1978]

# Infant loss in the year of birth, mu_00 in c_i = (1 - mu_00) k_i h_i (p. 26,
# eq. 8.4-6). The book gives no value; the first-year loss is carried by the
# age-0 rate of Table 3.3, so mu_00 is set to zero.
MU00 = 0.0


def stationary_weights(mu_f, mu_m, female_share_at_birth):
    """Persons in each age group of the stationary population per infant counted
    at age 0, under the model's own dynamics x_{i+1} = (1 - mu_i) x_i with the
    open 90+ group, both sexes. This is the discrete form of eqs. (8.2-1)-(8.2-3):
    with forward mortality mu_i = 1 - L_{i+1}/L_i the products are L_i/L_0.
    The sum is the life expectancy at birth S_0."""
    def one(mu):
        l = np.ones(NUM_AGE_GROUPS)
        l[1:] = np.cumprod(1 - mu[:-1])
        l[-1] = l[-1] / mu[-1]          # 90+: l_90 (1 + (1-mu) + (1-mu)^2 + ...)
        return l
    return female_share_at_birth * one(mu_f) + (1 - female_share_at_birth) * one(mu_m)


def mortality_factor(mortality_evolution_factor):
    """Multiplier of the 1973-75 death rates in each year 1978..2080 (linear
    decline by the given fraction over the horizon)."""
    return 1 - mortality_evolution_factor * (TIME_STEPS - T0) / (TF - T0)


def period_life_expectancy(mortality_evolution_factor, newborn_female_proportion):
    """Life expectancy at birth S_0(t) in each year, from the death rates of
    that year and the newborn female proportion."""
    return np.array([stationary_weights(death_rates_females * f, death_rates_males * f,
                                        newborn_female_proportion).sum()
                     for f in mortality_factor(mortality_evolution_factor)])


# Target state X* (eqs. 8.2-1 to 8.2-3 and 8.4-2): the stationary population of
# 700 million implied by the model's mortality (Table 3.3), in which births
# equal deaths. (Before Nov 2026 X* was a hand-drawn step profile summing to
# 710.5 million.)
_W_STAR = stationary_weights(death_rates_females, death_rates_males, FEMALE_INFANT_SHARE_1978)
X_STAR = TARGET_POPULATION * _W_STAR / _W_STAR.sum()

# Program 1 (Section 8.5, p. 262; Figure 8.8): U = [beta_0, beta_1] = [1, 2.16]
# (8.2-8), peak population N_0 = 1.2 billion (8.2-9), dependency index
# rho_0 = 1 (8.2-10), aging index omega_0 = 0.7 (8.2-11). "For our purpose, the
# three important indices listed above suffice" (p. 252). Stationarity is the
# target X*, not a constraint.
POPULATION_MAX = 1.2e9
SOCIAL_DEPENDENCY_MAX = 1.0
AGING_INDEX_MAX = 0.7
BETA_BOUNDS = (1.0, 2.16)

# The book's gradient program (8.4-12) handles the control set U by projection;
# the state constraints are imposed here as exterior penalties on violations
# measured in persons, which makes them effectively hard. A cold start of the
# penalised problem can stall, so it is solved by continuation: penalty factors
# 1e4, 1e6, ... up to the sampled factor, each stage warm-started from the
# previous solution.
PENALTY_START = 1e4


def penalty_schedule(penalty_factor):
    out, pf = [], PENALTY_START
    while pf < penalty_factor:
        out.append(pf)
        pf *= 100
    out.append(penalty_factor)
    return out


def fertility_pattern(fertility_rate_source, fertility_age_shift_factor=0.0):
    """Age pattern of fertility h_i over ages 15-49, summing to one, so that the
    control beta(t) is exactly the total fertility rate (Table 2.5 columns 3-4).

    Built from Song's age-specific fertility rates: Rudong 1973 (the 40+ rate
    applies to each age 40-49) or Tianjin 1978. The age shift moves a share s of
    the schedule five years OLDER, h_s(x) = (1-s) h(x) + s h(x-5), renormalised
    so that the shift changes the timing of childbearing and not its level."""
    table = fertility_rates_rudong_1973 if fertility_rate_source == 0 else fertility_rates_tianjin_1978
    asfr = np.zeros(NUM_AGE_GROUPS)
    for age, rate in table.items():
        asfr[age] = rate
    if fertility_rate_source == 0:
        asfr[40:50] = table[40]
    h = asfr[REPRODUCTIVE_AGES] / asfr[REPRODUCTIVE_AGES].sum()
    if fertility_age_shift_factor != 0:
        older = np.zeros_like(h)
        older[5:] = h[:-5]
        h = (1 - fertility_age_shift_factor) * h + fertility_age_shift_factor * older
        h = h / h.sum()
    return h


def initial_population(initial_pyramid_type, file_path="./Data/china_population_1978.csv"):
    X0 = parse_population_data_optimized(file_path)
    if initial_pyramid_type == 1:          # younger pyramid
        X0[0:20] *= 1.2
        X0[40:] *= 0.8
        X0 = X0 * (958e6 / np.sum(X0))
    elif initial_pyramid_type == 2:        # older pyramid
        X0[0:20] *= 0.8
        X0[40:] *= 1.2
        X0 = X0 * (958e6 / np.sum(X0))
    return X0


# Equations (2.2-13), (8.3-1) and (8.4-1): X(t+1) = [H(t) + beta(t) B(t)] X(t),
# X(t0) = X0, with X the population aged 1..m. The infants of year t are
# x_0(t) = beta(t) C(t)'X(t), c_i = (1 - mu_00) k_i h_i (8.4-6), and reach age 1
# at t+1 with survival 1 - mu_0 (the first row of B(t), p. 26). Women and men
# are projected with their own death rates (Table 3.3), so the female shares
# k_i(t) of cohorts born after 1978 follow from the newborn female proportion;
# the 1978 shares are those of Table 6.1. The 1978 infants are generated by
# beta(1978), as in the book, whose first control is beta(1978).
def simulate_population_optimized(initial_X, beta_trajectory, mortality_evolution_factor,
                                  newborn_female_proportion, fertility_h, female_share=female_share_1978):
    F = initial_X * female_share
    M = initial_X - F
    F[0] = 0.0
    M[0] = 0.0
    p = newborn_female_proportion
    tfac = mortality_factor(mortality_evolution_factor)
    population_history = np.zeros((NUM_TIME_STEPS, NUM_AGE_GROUPS))
    A = NUM_AGE_GROUPS
    for i in range(NUM_TIME_STEPS):
        births = (1 - MU00) * beta_trajectory[i] * np.dot(F[REPRODUCTIVE_AGES], fertility_h)
        F[0] = p * births
        M[0] = (1 - p) * births
        population_history[i] = F + M
        if i == NUM_TIME_STEPS - 1:
            break
        sf = 1 - death_rates_females * tfac[i]
        sm = 1 - death_rates_males * tfac[i]
        F_next = np.zeros(A)
        M_next = np.zeros(A)
        F_next[1:] = F[:-1] * sf[:-1]
        M_next[1:] = M[:-1] * sm[:-1]
        F_next[A - 1] += F[A - 1] * sf[A - 1]          # open-ended 90+ group
        M_next[A - 1] += M[A - 1] * sm[A - 1]
        F, M = F_next, M_next
    return population_history


# Performance index (8.4-2): J = sum_t [X(t) - X*]'[X(t) - X*] + (x_0(t) - x_0*)^2,
# X over ages 1..m, so that every age 0..90 is counted once. The controls run
# over 1978-2080 and the sum over the same years (T = 2081 in the book's
# notation, whose Table 8.14 reports beta(2080)).
def performance_index_optimized(X_history, X_star_history):
    return float(np.sum((X_history - X_star_history) ** 2))


def dependency_ratio(population_distribution):
    """Social dependency index rho(t): dependants (ages <15 and 65+) per person
    aged 15-64 ("the number of elderly and children who depend on each worker",
    Cohen 1990). Works on one year or on a history."""
    P = np.atleast_2d(population_distribution)
    dependants = P[:, 0:15].sum(axis=1) + P[:, 65:].sum(axis=1)
    working = P[:, 15:65].sum(axis=1)
    out = dependants / working
    return out if np.ndim(population_distribution) > 1 else float(out[0])


def aging_index(population_history, life_expectancy):
    """Aging index omega(t): mean age of the whole population over life
    expectancy at birth (Song and Yu 1988; Cohen 1990). Ages at mid-interval."""
    P = np.atleast_2d(population_history)
    mean_age = (P @ (AGE_GROUPS + 0.5)) / P.sum(axis=1)
    return mean_age / np.asarray(life_expectancy)


def constraint_violations(population_history, life_expectancy):
    """Violations of the three state constraints of Program 1, in persons, so
    that one penalty factor weighs them alike: dependants above rho_0 times the
    working-age population (8.2-10), total population times the excess of the
    aging index over omega_0 (8.2-11), and population above 1.2 billion (8.2-9).
    (Before Nov 2026 the code penalised squared ratios, which could never bind,
    and imposed 65+/total <= 0.7 in place of the aging index; the version of
    early Nov 2026 imposed the paper's description instead: dependency ratio
    <= 0.7 and a stabilisation band in 2080.)"""
    P = population_history
    dependants = P[:, 0:15].sum(axis=1) + P[:, 65:].sum(axis=1)
    working = P[:, 15:65].sum(axis=1)
    total = P.sum(axis=1)
    years_lived = P @ (AGE_GROUPS + 0.5)
    social = np.clip(dependants - SOCIAL_DEPENDENCY_MAX * working, 0, None)
    aging = np.clip(years_lived / life_expectancy - AGING_INDEX_MAX * total, 0, None)
    ceiling = np.clip(total - POPULATION_MAX, 0, None)
    return social, aging, ceiling


# Objective function for scipy.optimize.minimize
def objective_function_optimized(beta_array, initial_X, X_star_history, penalty_factor, mortality_evolution_factor,
                                 newborn_female_proportion, fertility_h, female_share=female_share_1978):
    beta_array = np.clip(beta_array, *BETA_BOUNDS)
    population_history = simulate_population_optimized(initial_X, beta_array, mortality_evolution_factor,
                                                        newborn_female_proportion, fertility_h, female_share)
    J = performance_index_optimized(population_history, X_star_history)
    e0 = period_life_expectancy(mortality_evolution_factor, newborn_female_proportion)
    social, aging, ceiling = constraint_violations(population_history, e0)
    penalty = np.sum(social ** 2) + np.sum(aging ** 2) + np.sum(ceiling ** 2)
    return J + penalty_factor * penalty


def objective_and_gradient(beta_array, *args):
    """Objective with a forward-difference gradient (steps as in scipy's
    2-point scheme, backward at the upper bound). Supplying the gradient keeps
    scipy >= 1.18 from aborting when L-BFGS-B returns an iterate a rounding
    error outside its bounds."""
    beta = np.clip(np.asarray(beta_array, dtype=float), *BETA_BOUNDS)
    f0 = objective_function_optimized(beta, *args)
    g = np.empty_like(beta)
    for j in range(beta.size):
        h = 1.4901161193847656e-08 * max(1.0, abs(beta[j]))
        if beta[j] + h > BETA_BOUNDS[1]:
            h = -h
        b = beta.copy()
        b[j] = beta[j] + h
        g[j] = (objective_function_optimized(b, *args) - f0) / (b[j] - beta[j])
    return f0, g


def solve_population_trajectory_with_constraints_scipy_optimized(initial_pyramid_type, fertility_rate_source, mortality_evolution_factor, newborn_female_proportion, optimizer_tolerance, optimizer_max_iterations, penalty_factor, fertility_age_shift_factor):
    beta_initial_array = np.full(NUM_TIME_STEPS, 2.16)
    initial_population_distribution = initial_population(initial_pyramid_type)
    fertility_h = fertility_pattern(fertility_rate_source, fertility_age_shift_factor)
    X_star_history = X_STAR

    # Bounds for beta (TFR)
    bounds = [BETA_BOUNDS for _ in TIME_STEPS]

    # Optimization using L-BFGS-B, continuation in the penalty factor
    beta = beta_initial_array
    for pf in penalty_schedule(penalty_factor):
        result = minimize(
            objective_and_gradient,
            beta,
            jac=True,
            args=(initial_population_distribution, X_star_history, pf, mortality_evolution_factor,
                  newborn_female_proportion, fertility_h),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': optimizer_max_iterations, 'ftol': optimizer_tolerance}
        )
        beta = np.clip(result.x, *BETA_BOUNDS)

    optimized_beta_array = beta
    population_history = simulate_population_optimized(initial_population_distribution, optimized_beta_array,
                                                        mortality_evolution_factor, newborn_female_proportion, fertility_h)

    # Calculate output variables
    total_population_2080 = np.sum(population_history[-1])
    e0 = period_life_expectancy(mortality_evolution_factor, newborn_female_proportion)
    years_aging_index_exceeded = int(np.sum(aging_index(population_history, e0) > AGING_INDEX_MAX + 1e-6))
    years_social_dependency_exceeded = int(np.sum(dependency_ratio(population_history) > SOCIAL_DEPENDENCY_MAX + 1e-6))
    years_upper_threshold_exceeded = int(np.sum(np.sum(population_history, axis=1) > POPULATION_MAX))

    return optimized_beta_array, population_history, X_star_history, total_population_2080, years_aging_index_exceeded, years_social_dependency_exceeded, years_upper_threshold_exceeded

def generate_sobol_parameters(sample_index, num_samples):
    """
    Generate parameters using Sobol sequence for Monte Carlo sampling.
    
    Parameters:
    - sample_index: Index of the current sample
    - num_samples: Total number of samples to generate
    
    Returns:
    - Dictionary of parameters for the optimization
    """
    # Define parameter ranges
    ranges = {
        'initial_pyramid_type': [0, 2],  # Discrete: 0, 1, 2
        'fertility_rate_source': [0, 1], # Discrete: 0, 1
        'mortality_evolution_factor': [0, 0.1],
        'newborn_female_proportion': [0.47, 0.5],
        'optimizer_tolerance_log': [-7, -5], # For 10**x
        'optimizer_max_iterations_log': [1, 3], # For 10**x
        'penalty_factor_log': [8, 12],
        'fertility_age_shift_factor': [0, 0.2]
    }

    # Number of dimensions for Sobol sequence
    num_dimensions = len(ranges)

    # Generate a scrambled Sobol' sequence (fixed seed). The unscrambled sequence
    # starts at the origin, which made sample 0 the all-minimum corner.
    sampler = qmc.Sobol(d=num_dimensions, scramble=True, seed=1978)
    sample = sampler.random_base2(m=int(np.log2(num_samples)))[sample_index]

    # Map Sobol sample to parameter ranges
    parameters = {}
    i = 0

    # initial_pyramid_type
    # discrete inputs: equiprobable levels (rounding gave the middle level half the weight)
    parameters['initial_pyramid_type'] = min(int(np.floor(sample[i] * 3)), 2)
    i += 1

    # fertility_rate_source
    parameters['fertility_rate_source'] = min(int(np.floor(sample[i] * 2)), 1)
    i += 1

    # mortality_evolution_factor
    parameters['mortality_evolution_factor'] = sample[i] * (ranges['mortality_evolution_factor'][1] - ranges['mortality_evolution_factor'][0]) + ranges['mortality_evolution_factor'][0]
    i += 1

    # newborn_female_proportion
    parameters['newborn_female_proportion'] = sample[i] * (ranges['newborn_female_proportion'][1] - ranges['newborn_female_proportion'][0]) + ranges['newborn_female_proportion'][0]
    i += 1

    # optimizer_tolerance
    parameters['optimizer_tolerance'] = 10** (sample[i] * (ranges['optimizer_tolerance_log'][1] - ranges['optimizer_tolerance_log'][0]) + ranges['optimizer_tolerance_log'][0])
    i += 1

    # optimizer_max_iterations
    parameters['optimizer_max_iterations'] = int(10** (sample[i] * (ranges['optimizer_max_iterations_log'][1] - ranges['optimizer_max_iterations_log'][0]) + ranges['optimizer_max_iterations_log'][0]))
    i += 1

    # penalty_factor
    parameters['penalty_factor'] = 10** (sample[i] * (ranges['penalty_factor_log'][1] - ranges['penalty_factor_log'][0]) + ranges['penalty_factor_log'][0])
    i += 1

    # fertility_age_shift_factor
    parameters['fertility_age_shift_factor'] = sample[i] * (ranges['fertility_age_shift_factor'][1] - ranges['fertility_age_shift_factor'][0]) + ranges['fertility_age_shift_factor'][0]
    i += 1

    return parameters


if __name__ == '__main__':
    import time
    start_time = time.time()
    # Define parameters for analysis
    N_SIMULATIONS = 256 
    
    OUTPUT_VARIABLES = [
        'total_population_2080',
        'dependency_ratio_2080',
        'years_aging_index_above_0.7',
        'years_above_1.2B'
    ]
    
    # Initialize results storage
    results = {
        'parameters': [],
        'outputs': {var: [] for var in OUTPUT_VARIABLES},
        'population_trajectories': [],
        'fertility_trajectories': []
    }

    # Generate parameter samples using Sobol sequence
    print(f"\nGenerating {N_SIMULATIONS} parameter samples...")
    for sample_idx in range(N_SIMULATIONS):
        params = generate_sobol_parameters(sample_idx, N_SIMULATIONS)
        results['parameters'].append(params)

    # Run simulations
    print(f"\nRunning {N_SIMULATIONS} simulations...")
    for i, params in enumerate(results['parameters']):
        if i % 10 == 0:
            print(f"  Simulation {i+1}/{N_SIMULATIONS}")
            
        optimized_beta, pop_history, _, total_pop, aging_years, social_dep_years, threshold_years = solve_population_trajectory_with_constraints_scipy_optimized(
            params['initial_pyramid_type'],
            params['fertility_rate_source'],
            params['mortality_evolution_factor'],
            params['newborn_female_proportion'],
            params['optimizer_tolerance'],
            params['optimizer_max_iterations'],
            params['penalty_factor'],
            params['fertility_age_shift_factor']
        )
        
        # Store results
        results['outputs']['total_population_2080'].append(total_pop)
        results['outputs']['dependency_ratio_2080'].append(dependency_ratio(pop_history[-1]))
        results['outputs']['years_aging_index_above_0.7'].append(aging_years)
        results['outputs']['years_above_1.2B'].append(threshold_years)
        results['population_trajectories'].append(np.sum(pop_history, axis=1))
        results['fertility_trajectories'].append(optimized_beta)

    # Create DataFrames
    results_df = pd.DataFrame({
        'sample': range(N_SIMULATIONS),
        **results['outputs'],
        **{f'param_{k}': [p[k] for p in results['parameters']] for k in results['parameters'][0].keys()}
    })

    # Calculate fertility distances
    base_fertility = results['fertility_trajectories'][0]
    results_df['fertility_distance'] = [
        np.sqrt(np.sum((traj - base_fertility)**2)) for traj in results['fertility_trajectories']
    ]

    import os
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    # UNCERTAINTY VISUALIZATIONS
    print("\nGenerating uncertainty visualizations...")
    
    # 1. Population distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(results_df['total_population_2080'], kde=True, bins=10, ax=axes[0,0])
    sns.histplot(results_df['dependency_ratio_2080'], kde=True, bins=10, ax=axes[0,1])
    sns.histplot(results_df['years_aging_index_above_0.7'], kde=True, bins=10, ax=axes[1,0])
    sns.histplot(results_df['years_above_1.2B'], kde=True, bins=10, ax=axes[1,1])
    plt.tight_layout()
    plt.savefig('./figures/uncertainty_distributions.png')
    plt.close()

    # 2. Trajectory plots
    plt.figure(figsize=(12,6))
    for traj in results['population_trajectories']:
        plt.plot(TIME_STEPS, traj, alpha=0.1, color='blue')
    plt.plot(TIME_STEPS, np.mean(results['population_trajectories'], axis=0), 
            color='red', linewidth=2, label='Mean')
    plt.axhline(1.2e9, color='black', linestyle='--', label='1.2B Threshold')
    plt.title('Population Trajectories')
    plt.legend()
    plt.savefig('./figures/population_trajectories.png')
    plt.close()

    # 3. Fertility trajectories
    plt.figure(figsize=(12,6))
    for traj in results['fertility_trajectories']:
        plt.plot(TIME_STEPS, traj, alpha=0.1, color='blue')
    plt.plot(TIME_STEPS, np.mean(results['fertility_trajectories'], axis=0), 
            color='red', linewidth=2, label='Mean')
    plt.axhline(2.16, color='green', linestyle='--', label='Replacement Level')
    plt.title('Fertility Trajectories')
    plt.legend()
    plt.savefig('./figures/fertility_trajectories.png')
    plt.close()

    # SUMMARY STATISTICS
    print("\nFinal Results Summary:")
    print(f"Mean 2080 Population: {np.mean(results_df['total_population_2080']/1e6):.2f} million")
    print(f"Mean dependency ratio in 2080: {np.mean(results_df['dependency_ratio_2080']):.3f}")
    print(f"Mean Fertility Distance: {np.mean(results_df['fertility_distance']):.2f}")

    # Save results
    results_df.to_csv('./results/simulation_results.csv', index=False)
    print(f"\nTotal runtime: {time.time()-start_time:.2f} seconds")
