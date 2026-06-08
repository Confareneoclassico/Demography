"""
MoMP_world.py
-------------
Global population projection experiment for
"What demographic prediction can and cannot achieve".

Experimental design — two discrete structural variables + continuous parametric:

  `paradigm`       (Level 1)  0 = Cohort-Component family
                              1 = Lotka intrinsic growth rate
                              2 = ARIMA (statistical, no demographic structure)

  `mortality_model` (Level 2)  0 = exogenous mortality scaling  (CC only)
                               1 = Lee-Carter sub-model          (CC only)

  Continuous parametric (all paradigms):
    fertility_scale, mortality_scale, migration_scale, trend_strength

  Lee-Carter specific (paradigm=0, mortality_model=1 only):
    lee_carter_kt_std, lee_carter_error_scale, kt_realization

Total: 9 uncertain inputs, sampled via Sobol' quasi-random sequences.
N = 1024 base samples → 1024*(2*10+2) = 22,528 model evaluations.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

from demographic_models_revised import (
    cohort_component_model,
    lee_carter_model,
    lotka_intrinsic_growth_rate_model,
    aggregate_accounting_step,
    fit_all_arima_orders,
    ARIMA_ORDERS,
)

sns.set_theme(style="whitegrid")
mpl.rcParams.update({
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 14, "axes.titlesize": 16,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "axes.edgecolor": "black",
    "axes.linewidth": 0.8, "grid.color": "0.9",
    "grid.linewidth": 0.8, "savefig.dpi": 300, "figure.dpi": 300,
})

# ---------------------------------------------------------------------------
# Sobol' problem definition — symmetric two-level structural design
# ---------------------------------------------------------------------------
#
# Level 1 — `paradigm` (3 options, ~33% each):
#   0 = Cohort-Component (CC) family
#   1 = Lotka intrinsic growth rate
#   2 = ARIMA statistical paradigm
#
# Level 2 — sub-model choice within each structured paradigm:
#   `mortality_model` (CC only):  0=exogenous scaling, 1=Lee-Carter (~17% each)
#   `arima_order`     (ARIMA only): 0–3 → four ARIMA specifications (~8% each)
#
# Both Level-2 variables are sampled independently across ALL simulations
# but are active only within their respective paradigm. Their S1 indices
# are attenuated by inactivity; the ST−S1 gap reveals their interaction
# with `paradigm`. See methods section for full discussion.
#
# Effective run breakdown (~% of 22,528 evaluations):
#   CC + exogenous  (p=0, mm=0): ~16.7%
#   CC + LeeCarter  (p=0, mm=1): ~16.7%
#   Lotka           (p=1):       ~33.3%
#   ARIMA order 0   (p=2, ao=0): ~ 8.3%
#   ARIMA order 1   (p=2, ao=1): ~ 8.3%
#   ARIMA order 2   (p=2, ao=2): ~ 8.3%
#   ARIMA order 3   (p=2, ao=3): ~ 8.3%

problem = {
    'num_vars': 10,
    'names': [
        'fertility_scale',        # parametric — all paradigms
        'mortality_scale',        # parametric — all paradigms
        'migration_scale',        # parametric — dummy (zero at planetary scale)
        'trend_strength',         # parametric — all paradigms
        'paradigm',               # Level 1: 0=CC, 1=Lotka, 2=ARIMA
        'mortality_model',        # Level 2 (CC only): 0=exogenous, 1=Lee-Carter
        'arima_order',            # Level 2 (ARIMA only): 0–3 → 4 specifications
        'lee_carter_kt_std',      # Lee-Carter specific (p=0, mm=1 only)
        'lee_carter_error_scale', # Lee-Carter specific
        'kt_realization',         # Lee-Carter specific (uniform for inv-CDF)
    ],
    'bounds': [
        [0.85,   1.15  ],  # fertility_scale
        [0.97,   1.03  ],  # mortality_scale
        [0.8,    1.2   ],  # migration_scale
        [0.9,    1.1   ],  # trend_strength
        [0,      2.9999],  # paradigm
        [0,      1.9999],  # mortality_model  (CC only)
        [0,      3.9999],  # arima_order      (ARIMA only)
        [0.005,  0.08  ],  # lee_carter_kt_std
        [0.0,    0.1   ],  # lee_carter_error_scale
        [0.0,    1.0   ],  # kt_realization
    ]
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sobol_to_truncated_normal(sobol_value, mean=0, std=1, lower=-1, upper=1):
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm(a, b, loc=mean, scale=std).ppf(sobol_value)


def get_lee_carter_error_pattern(error_scale, n_ages):
    return np.full(n_ages, error_scale)


def transform_survival_probabilities(base_survival, mortality_scale):
    base_mortality = 1.0 - base_survival
    log_odds = np.log(base_mortality / (1 - base_mortality + 1e-9))
    scaled_log_odds = log_odds * (2.0 - mortality_scale)
    return 1.0 - 1.0 / (1.0 + np.exp(-scaled_log_odds))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def estimate_survival_probabilities_1yr(world_data):
    survival_probs = np.ones(101)
    imr  = world_data['Infant Mortality Rate (infant deaths per 1,000 live births)'] / 1000
    survival_probs[0] = 1 - imr
    u5   = world_data['Under-Five Mortality (deaths under age 5 per 1,000 live births)'] / 1000
    child_1_4 = (u5 - imr) / 4
    for age in range(1, 5):
        survival_probs[age] = 1 - child_1_4
    m40 = world_data['Mortality before Age 40, both sexes (deaths under age 40 per 1,000 live births)'] / 1000
    ann_5_39 = 1 - (1 - m40) ** (1/35)
    for age in range(5, 40):
        survival_probs[age] = 1 - ann_5_39
    m60 = world_data['Mortality before Age 60, both sexes (deaths under age 60 per 1,000 live births)'] / 1000
    ann_40_59 = 1 - (1 - (m60 - m40)) ** (1/20)
    for age in range(40, 60):
        survival_probs[age] = 1 - ann_40_59
    for age in range(60, 101):
        mortality = 0.01 * np.exp(0.07 * (age - 60))
        survival_probs[age] = 1 - min(mortality, 0.99)
    return survival_probs


def estimate_lee_carter_parameters():
    wpp = pd.read_excel(
        "./Data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx",
        sheet_name="Estimates", skiprows=16)
    world = wpp[(wpp['Region, subregion, country or area *'] == 'World') &
                (wpp['Year'] >= 1950) & (wpp['Year'] <= 2023)]
    cdr = world['Crude Death Rate (deaths per 1,000 population)'].values
    n_ages = 101
    ages = np.arange(n_ages)
    infant_m   = 0.02  * np.exp(-0.5 * ages[:5])
    young_m    = 0.001 * np.ones(21)
    adult_m    = 0.002 * np.exp(0.05 * (np.arange(26, 61) - 26))
    elderly_m  = 0.01  * np.exp(0.08 * (np.arange(61, 101) - 61))
    pattern    = np.concatenate([infant_m, young_m, adult_m, elderly_m])
    lc_a_x = np.log(pattern + 1e-9)
    lc_b_x = 0.15 * np.exp(-0.03 * ages)
    lc_e_x_t = np.random.normal(0, 0.05, n_ages)
    return lc_a_x, lc_b_x, lc_e_x_t


def extract_actual_demographic_parameters():
    wpp = pd.read_excel(
        "./Data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx",
        sheet_name="Estimates", skiprows=16)
    w23 = wpp[(wpp['Region, subregion, country or area *'] == 'World') &
              (wpp['Year'] == 2023)].iloc[0]
    total_pop = w23['Total Population, as of 1 July (thousands)'] * 1000
    age_props = np.array([
        *[0.0130]*5, *[0.0128]*5, *[0.0126]*5, *[0.0124]*5, *[0.0122]*5,
        *[0.0120]*5, *[0.0118]*5, *[0.0114]*5, *[0.0112]*5, *[0.0110]*5,
        *[0.0108]*5, *[0.0106]*5, *[0.0104]*5, *[0.0102]*5, *[0.0100]*5,
        *[0.0098]*5, *[0.0096]*5, *[0.0094]*5, *[0.0092]*5, *[0.0090]*5,
        0.001
    ])
    tfr = w23['Total Fertility Rate (live births per woman)']
    asfr_ages = np.arange(15, 50)
    asfr_pat  = np.exp(-0.5 * ((asfr_ages - 27) / 7)**2)
    asfr_pat  = asfr_pat / asfr_pat.sum() * tfr
    asfr_base = np.zeros(101)
    asfr_base[15:50] = asfr_pat
    lc_a_x, lc_b_x, lc_e_x_t = estimate_lee_carter_parameters()
    return {
        'total_population_2023': total_pop,
        'pop_age_dist': total_pop * age_props,
        'survival_prob_base': estimate_survival_probabilities_1yr(w23),
        'asfr_base': asfr_base,
        'net_migration_base': 0,
        'net_migration_rate': 0,
        'lc_a_x': lc_a_x, 'lc_b_x': lc_b_x, 'lc_e_x_t': lc_e_x_t,
        'crude_birth_rate': w23['Crude Birth Rate (births per 1,000 population)'],
        'crude_death_rate': w23['Crude Death Rate (deaths per 1,000 population)'],
        'total_fertility_rate': tfr,
        'total_births': w23['Births (thousands)'] * 1000,
        'total_deaths': w23['Total Deaths (thousands)'] * 1000,
        'life_expectancy': w23['Life Expectancy at Birth, both sexes (years)'],
    }


def load_actual_demographic_data():
    """Load and pre-compute all data needed by run_simulation."""
    params = extract_actual_demographic_parameters()
    hist = pd.read_csv("./Data/world_population_combined_1950_2100.csv")
    historical_pop = (hist[hist["Year"] <= 2023]
                      ["Total Population, as of 1 July (thousands)"].values * 1000)

    # Pre-fit all candidate ARIMA specifications once at startup.
    print("  Fitting all ARIMA candidate specifications...")
    arima_growth_by_order, arima_labels = fit_all_arima_orders(historical_pop)
    print(f"  ARIMA fitting complete. {len(arima_growth_by_order)} specifications ready.")

    # Apply a minimum survival probability floor to prevent degenerate
    # Lee-Carter projections at extreme parameter combinations.
    MIN_SURVIVAL = 0.5   # conservative floor; still allows very high mortality
    survival_base = np.maximum(params['survival_prob_base'], MIN_SURVIVAL)

    return {
        'initial_population':    params['total_population_2023'],
        'historical_population': historical_pop,
        'pop_age_dist':          params['pop_age_dist'].copy(),
        'survival_prob_base':    survival_base,
        'asfr_base':             params['asfr_base'],
        'net_migration_base':    params['net_migration_base'],
        'net_migration_rate':    params['net_migration_rate'],
        'lc_a_x':                params['lc_a_x'],
        'lc_b_x':                params['lc_b_x'],
        'lc_e_x_t':              params['lc_e_x_t'],
        'crude_birth_rate':      params['crude_birth_rate'],
        'crude_death_rate':      params['crude_death_rate'],
        'total_fertility_rate':  params['total_fertility_rate'],
        'arima_growth_by_order': arima_growth_by_order,  # dict: int → (52,) array
        'arima_labels':          arima_labels,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(params_row, data):
    """
    Run one projection from 2024 to 2075 for the given parameter vector.

    Parameters
    ----------
    params_row : array-like, length 9
        [fertility_scale, mortality_scale, migration_scale, trend_strength,
         paradigm, mortality_model,
         lee_carter_kt_std, lee_carter_error_scale, kt_realization]
    data : dict
        Pre-loaded demographic data from load_actual_demographic_data().

    Returns
    -------
    (float, float) : (population_2050_millions, population_2075_millions)
    """
    # -- Unpack parameters --------------------------------------------------
    fertility_scale        = params_row[0]
    mortality_scale        = params_row[1]
    migration_scale        = params_row[2]
    trend_strength         = params_row[3]
    paradigm               = int(np.floor(params_row[4]))   # 0=CC, 1=Lotka, 2=ARIMA
    mortality_model        = int(np.floor(params_row[5]))   # 0=exogenous, 1=LeeCarter (CC only)
    arima_order            = int(np.floor(params_row[6]))   # 0–3 (ARIMA only)
    lee_carter_kt_std      = params_row[7]
    lee_carter_error_scale = params_row[8]
    kt_realization         = params_row[9]

    # -- State variables ----------------------------------------------------
    current_total_population = data['initial_population']
    current_pop_age_dist     = data['pop_age_dist'].copy()

    # Pre-compute Lee-Carter k_t (used every year if paradigm=0, mm=1).
    if paradigm == 0 and mortality_model == 1:
        k_t_simulated = sobol_to_truncated_normal(
            kt_realization,
            mean  = -0.5 * lee_carter_kt_std,
            std   =  lee_carter_kt_std,
            lower = -3 * lee_carter_kt_std,
            upper =  3 * lee_carter_kt_std,
        )
        lc_e_x_t = get_lee_carter_error_pattern(
            lee_carter_error_scale, len(data['lc_a_x'])
        )
        predicted_log_mortality = lee_carter_model(
            data['lc_a_x'], data['lc_b_x'], k_t_simulated, lc_e_x_t
        )
        lc_survival = 1 - np.exp(predicted_log_mortality)
        if np.any(lc_survival < 0) or np.any(lc_survival > 1):
            lc_survival = np.clip(lc_survival, 0.0, 1.0)

    pop_2050 = pop_2075 = 0.0

    for year_idx in range(1, 53):   # 2024 → 2075 (52 steps)
        current_year = 2023 + year_idx
        years_from_start = year_idx - 1

        # Time-varying parametric scalings
        current_fertility_scale = fertility_scale * (1 - 0.01 * trend_strength) ** years_from_start
        current_mortality_scale = mortality_scale * (1 - 0.005 * trend_strength) ** years_from_start
        current_migration_scale = migration_scale * (1 + 0.001 * trend_strength) ** years_from_start

        current_asfr      = data['asfr_base'] * current_fertility_scale
        current_survival  = transform_survival_probabilities(
            data['survival_prob_base'], current_mortality_scale
        )
        current_migration = data['net_migration_base'] * current_migration_scale

        # ==================================================================
        # Paradigm = 0: Cohort-Component family
        # ==================================================================
        if paradigm == 0:

            if mortality_model == 0:
                # -- Level 2 sub-model 0: exogenous mortality scaling ------
                current_pop_age_dist = cohort_component_model(
                    current_pop_age_dist,
                    current_survival,
                    current_asfr,
                    current_migration,
                )
                current_total_population = np.sum(current_pop_age_dist)

            else:
                # -- Level 2 sub-model 1: Lee-Carter mortality embedded ----
                # Lee-Carter survival was pre-computed above; apply
                # parametric mortality scaling on top.
                lc_surv_scaled = transform_survival_probabilities(
                    lc_survival, current_mortality_scale
                )
                lc_crude_death = (
                    np.sum((1 - lc_surv_scaled) * current_pop_age_dist)
                    / current_total_population * 1000
                )
                total_deaths    = current_total_population * lc_crude_death / 1000
                total_births    = (current_total_population
                                   * data['crude_birth_rate']
                                   * current_fertility_scale / 1000)
                total_migration = (current_total_population
                                   * data['net_migration_rate']
                                   * current_migration_scale / 1000)
                current_total_population = aggregate_accounting_step(
                    current_total_population,
                    total_deaths, total_births, total_migration,
                )

        # ==================================================================
        # Paradigm = 1: Lotka intrinsic growth rate
        # ==================================================================
        elif paradigm == 1:
            l_x   = np.cumprod(np.insert(current_survival[:-1], 0, 1.0))
            r_val = lotka_intrinsic_growth_rate_model(l_x, current_asfr)
            current_total_population *= np.exp(r_val)

        # ==================================================================
        # Paradigm = 2: ARIMA — statistical paradigm, no demographic structure.
        # Sub-model choice: arima_order selects among four pre-fitted ARIMA
        # specifications (see ARIMA_ORDERS in demographic_models.py).
        # Parametric perturbations are applied as additive adjustments to the
        # ARIMA-implied growth rate, using global crude rates as conversion.
        # ==================================================================
        elif paradigm == 2:
            g_arima = data['arima_growth_by_order'][arima_order][year_idx - 1]
            CBR = data['crude_birth_rate'] / 1000
            CDR = data['crude_death_rate'] / 1000
            g_adjusted = (g_arima * trend_strength
                          + (current_fertility_scale - 1) * CBR
                          - (current_mortality_scale - 1) * CDR)
            current_total_population *= (1 + g_adjusted)

        # Record outputs
        if current_year == 2050:
            pop_2050 = current_total_population
        if current_year == 2075:
            pop_2075 = current_total_population

    return pop_2050 / 1e6, pop_2075 / 1e6   # return in millions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    print("Loading demographic data...")
    demographic_data = load_actual_demographic_data()
    print("Data loaded.\n")

    # Sobol' quasi-random sample
    N_BASE = 1024
    param_values = sobol_sample.sample(
        problem, N_BASE, calc_second_order=True, scramble=True, seed=42
    )
    print(f"Generated {len(param_values)} parameter vectors "
          f"({N_BASE} base samples, D={problem['num_vars']}).\n")

    Y_2050, Y_2075 = [], []
    for i, row in enumerate(param_values):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Simulation {i+1}/{len(param_values)} ...")
        p2050, p2075 = run_simulation(row, demographic_data)
        Y_2050.append(p2050)
        Y_2075.append(p2075)

    Y_2050 = np.array(Y_2050)
    Y_2075 = np.array(Y_2075)

    # -- Save results -------------------------------------------------------
    from demographic_models_revised import ARIMA_ORDERS as _ARIMA_ORDERS
    pars  = np.floor(param_values[:, 4]).astype(int)
    mm    = np.floor(param_values[:, 5]).astype(int)
    ao    = np.floor(param_values[:, 6]).astype(int)

    def _run_label(p, m, a):
        if p == 0:
            return 'CC-exogenous' if m == 0 else 'CC-LeeCarter'
        elif p == 1:
            return 'Lotka'
        else:
            order = _ARIMA_ORDERS[a] if a < len(_ARIMA_ORDERS) else _ARIMA_ORDERS[-1]
            return f'ARIMA{order}'

    effective_run = np.array([_run_label(pars[i], mm[i], ao[i])
                               for i in range(len(pars))])

    results_df = pd.DataFrame(param_values, columns=problem['names'])
    results_df['effective_run']          = effective_run
    results_df['population_2050_millions'] = Y_2050
    results_df['population_2075_millions'] = Y_2075
    results_df.to_csv('./results/simulation_results_actual.csv', index=False)
    print("\nSimulation results saved to ./results/simulation_results_actual.csv")

    # -- Sobol' sensitivity analysis ----------------------------------------
    Si_2050 = sobol.analyze(problem, Y_2050, calc_second_order=True,
                             print_to_console=False)
    Si_2075 = sobol.analyze(problem, Y_2075, calc_second_order=True,
                             print_to_console=False)

    print("\nSobol' Sensitivity Indices — 2050:")
    print(f"{'Parameter':<28} {'S1':>8} {'ST':>8}")
    for j, name in enumerate(problem['names']):
        print(f"  {name:<26} {Si_2050['S1'][j]:>8.3f} {Si_2050['ST'][j]:>8.3f}")

    print("\nSobol' Sensitivity Indices — 2075:")
    print(f"{'Parameter':<28} {'S1':>8} {'ST':>8}")
    for j, name in enumerate(problem['names']):
        print(f"  {name:<26} {Si_2075['S1'][j]:>8.3f} {Si_2075['ST'][j]:>8.3f}")

    with open('./results/sobol_results.txt', 'w') as f:
        f.write("Sobol' Analysis Results\n\n")
        for label, Si in [('2050', Si_2050), ('2075', Si_2075)]:
            f.write(f"--- {label} Projections ---\n")
            f.write(f"{'Parameter':<28} S1        ST\n")
            for j, name in enumerate(problem['names']):
                f.write(f"  {name:<26} {Si['S1'][j]:.4f}    {Si['ST'][j]:.4f}\n")
            f.write("\n")
    print("Sobol' results saved to ./results/sobol_results.txt")

    # -- Summary statistics -------------------------------------------------
    hist_csv = pd.read_csv("./Data/world_population_combined_1950_2100.csv")
    # Column is in thousands; divide by 1e6 to convert to billions.
    un_2050  = hist_csv[hist_csv["Year"] == 2050][
        "Total Population, as of 1 July (thousands)"].values[0] / 1e6
    un_2075  = hist_csv[hist_csv["Year"] == 2075][
        "Total Population, as of 1 July (thousands)"].values[0] / 1e6
    print(f"\nUN medium projection: 2050 = {un_2050:.2f}B,  2075 = {un_2075:.2f}B")
    print(f"Simulation range:     2050 = {Y_2050.min()/1000:.2f}–{Y_2050.max()/1000:.2f}B ({Y_2050.min():.0f}–{Y_2050.max():.0f}M)")
    print(f"                      2075 = {Y_2075.min()/1000:.2f}–{Y_2075.max()/1000:.2f}B ({Y_2075.min():.0f}–{Y_2075.max():.0f}M)")
    print(f"Simulation mean:      2050 = {Y_2050.mean()/1000:.2f}B,  2075 = {Y_2075.mean()/1000:.2f}B")

    # -- Scatter plots -------------------------------------------------------
    from demographic_models_revised import ARIMA_ORDERS as _AO
    run_colors = {
        'CC-exogenous':  '#2166ac',
        'CC-LeeCarter':  '#4dac26',
        'Lotka':         '#d7191c',
        f'ARIMA{_AO[0]}': '#f4a582',
        f'ARIMA{_AO[1]}': '#fdae61',
        f'ARIMA{_AO[2]}': '#e6ab02',
        f'ARIMA{_AO[3]}': '#a6761d',
    }

    # Discrete parameters whose raw sampled values cluster at integer positions.
    # Horizontal jitter is applied to spread overlapping dots.
    DISCRETE_PARAMS = {'paradigm', 'mortality_model', 'arima_order'}
    JITTER = {'paradigm': 0.08, 'mortality_model': 0.06, 'arima_order': 0.08}

    for year, Y, label in [('2050', Y_2050, 'population_2050_millions'),
                            ('2075', Y_2075, 'population_2075_millions')]:
        plot_df = pd.DataFrame(param_values, columns=problem['names'])
        plot_df[label]           = Y
        plot_df['effective_run'] = effective_run
        plot_df_sub = plot_df.iloc[::16].copy().reset_index(drop=True)

        for param_name in problem['names']:
            # Wider figure for discrete params to accommodate the outside legend
            is_discrete = param_name in DISCRETE_PARAMS
            fig, ax = plt.subplots(figsize=(7, 4))

            rng = np.random.default_rng(seed=42)

            for run, color in run_colors.items():
                mask = plot_df_sub['effective_run'] == run
                sub  = plot_df_sub.loc[mask]
                if sub.empty:
                    continue

                x_vals = sub[param_name].values.copy()
                if is_discrete:
                    x_vals = x_vals + rng.uniform(
                        -JITTER[param_name], JITTER[param_name], size=len(x_vals)
                    )

                ax.scatter(x_vals, sub[label].values,
                           color=color, alpha=0.45, s=28,
                           edgecolors='none', label=run)

            ax.set_xlabel(param_name.replace('_', ' ').capitalize(), fontsize=12)
            ax.set_ylabel('Population (millions)', fontsize=12)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            sns.despine(ax=ax)

            if is_discrete:
                # Legend placed outside the right edge — colours are the same
                # across all individual scatter plots so this acts as a shared key.
                ax.legend(fontsize=8, framealpha=0.8,
                          loc='upper left',
                          bbox_to_anchor=(1.01, 1.0),
                          borderaxespad=0)
                plt.tight_layout()
                plt.savefig(f'./figures/scatter_{year}_{param_name}.png',
                            dpi=300, bbox_inches='tight')
            else:
                plt.tight_layout()
                plt.savefig(f'./figures/scatter_{year}_{param_name}.png', dpi=300)
            plt.close()

    print("\nScatter plots saved to ./figures/")
