import numpy as np
import pandas as pd
from scipy.stats import norm
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from demographic_models import cohort_component_model, un_world_population_prospects_model, lee_carter_model, lotka_intrinsic_growth_rate_model, arima_model

from scipy.stats import truncnorm

def sobol_to_truncated_normal(sobol_value, mean=0, std=1, lower=-1, upper=1):
    """
    Convert Sobol sample in [0,1] to truncated normal using inverse CDF
    sobol_value: Sobol sample in range [0,1]
    returns: value from truncated normal distribution
    """
    # Calculate bounds in standard normal space
    a, b = (lower - mean) / std, (upper - mean) / std
    
    # Create truncated normal distribution
    trunc_dist = truncnorm(a, b, loc=mean, scale=std)
    
    # Use Sobol value as probability for inverse CDF
    return trunc_dist.ppf(sobol_value)

# Problem definition with realistic bounds based on UN uncertainty
problem = {
    'num_vars': 8,  # Added parameters for noise realizations
    'names': [
        'fertility_scale', 
        'mortality_scale', 
        'migration_scale', 
        'lee_carter_k_t_std',
        'model_choice',
        'future_trend_strength',
        'lee_carter_error_scale',
        'k_t_realization',  # For Lee-Carter k_t sampling
    ],
    'bounds': [
        [0.85, 1.15],      # fertility_scale
        [0.97, 1.03],      # mortality_scale
        [0.8, 1.2],        # migration_scale
        [0.005, 0.08],     # lee_carter_k_t_std
        [0, 3],            # model_choice #exclude ARIMA for now
        [0.9, 1.1],        # future_trend_strength
        [0.0, 0.1],        # lee_carter_error_scale
        [0.0, 1.0],         # k_t_realization (uniform for inverse transform)
    ]
}

def extract_actual_demographic_parameters():
    """
    Extract actual demographic parameters from WPP2024 Excel file with proper survival probability estimation.
    """
    # Load WPP2024 data
    wpp_data = pd.read_excel("./Data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx", 
                           sheet_name="Estimates",
                           skiprows=16)
    
    # Filter for World data and year 2023
    world_2023 = wpp_data[(wpp_data['Region, subregion, country or area *'] == 'World') & 
                         (wpp_data['Year'] == 2023)].iloc[0]
    
    # Extract core population data
    total_population_2023 = world_2023['Total Population, as of 1 July (thousands)'] * 1000
    
    # Create age distribution (approximation)
    # Using 1-year age groups (0, 1, 2, ..., 99, 100+) for compatibility with cohort_component_model
    age_proportions_1yr = np.array([
        # 0-4 years (distributed)
        0.013, 0.013, 0.013, 0.013, 0.013,
        # 5-9 years
        0.0128, 0.0128, 0.0128, 0.0128, 0.0128,
        # 10-14 years
        0.0126, 0.0126, 0.0126, 0.0126, 0.0126,
        # 15-19 years
        0.0124, 0.0124, 0.0124, 0.0124, 0.0124,
        # 20-24 years
        0.0122, 0.0122, 0.0122, 0.0122, 0.0122,
        # 25-29 years
        0.0120, 0.0120, 0.0120, 0.0120, 0.0120,
        # 30-34 years
        0.0118, 0.0118, 0.0118, 0.0118, 0.0118,
        # 35-39 years
        0.0114, 0.0114, 0.0114, 0.0114, 0.0114,
        # 40-44 years
        0.0112, 0.0112, 0.0112, 0.0112, 0.0112,
        # 45-49 years
        0.0110, 0.0110, 0.0110, 0.0110, 0.0110,
        # 50-54 years
        0.0108, 0.0108, 0.0108, 0.0108, 0.0108,
        # 55-59 years
        0.0106, 0.0106, 0.0106, 0.0106, 0.0106,
        # 60-64 years
        0.0104, 0.0104, 0.0104, 0.0104, 0.0104,
        # 65-69 years
        0.0102, 0.0102, 0.0102, 0.0102, 0.0102,
        # 70-74 years
        0.0100, 0.0100, 0.0100, 0.0100, 0.0100,
        # 75-79 years
        0.0098, 0.0098, 0.0098, 0.0098, 0.0098,
        # 80-84 years
        0.0096, 0.0096, 0.0096, 0.0096, 0.0096,
        # 85-89 years
        0.0094, 0.0094, 0.0094, 0.0094, 0.0094,
        # 90-94 years
        0.0092, 0.0092, 0.0092, 0.0092, 0.0092,
        # 95-99 years
        0.0090, 0.0090, 0.0090, 0.0090, 0.0090,
        # 100+ years
        0.001
    ])
    
    pop_age_dist = total_population_2023 * age_proportions_1yr
    
    # Extract fertility data
    tfr = world_2023['Total Fertility Rate (live births per woman)']
    crude_birth_rate = world_2023['Crude Birth Rate (births per 1,000 population)']
    total_births = world_2023['Births (thousands)'] * 1000
    
    # Create ASFR pattern for 1-year age groups (ages 15-49)
    # Typical ASFR pattern by single year of age
    asfr_ages = np.arange(15, 50)  # Ages 15 to 49
    # Bell-shaped curve for ASFR
    asfr_pattern = np.exp(-0.5 * ((asfr_ages - 27) / 7) ** 2)  # Normal distribution centered at age 27
    asfr_pattern = asfr_pattern / asfr_pattern.sum() * tfr
    asfr_base = np.zeros(101)  # 0 to 100 years
    asfr_base[15:50] = asfr_pattern
    
    # Extract migration data
    net_migration = 0 #world_2023.get('Net Number of Migrants (thousands)', 0) * 1000
    net_migration_rate = 0 #world_2023.get('Net Migration Rate (per 1,000 population)', 0)
    
    # ESTIMATE SURVIVAL PROBABILITIES FROM AVAILABLE MORTALITY DATA (1-year)
    survival_prob_base = estimate_survival_probabilities_1yr(world_2023)
    
    # LEE-CARTER PARAMETER ESTIMATION USING HISTORICAL DATA
    lc_a_x, lc_b_x, lc_e_x_t = estimate_lee_carter_parameters()
    
    return {
        'total_population_2023': total_population_2023,
        'pop_age_dist': pop_age_dist,  # Now 1-year age groups
        'survival_prob_base': survival_prob_base,  # Now 1-year survival probabilities
        'asfr_base': asfr_base,  # Now 1-year ASFR
        'net_migration_base': net_migration,
        'net_migration_rate': net_migration_rate,
        'lc_a_x': lc_a_x,
        'lc_b_x': lc_b_x,
        'lc_e_x_t': lc_e_x_t,
        'crude_birth_rate': crude_birth_rate,
        'crude_death_rate': world_2023['Crude Death Rate (deaths per 1,000 population)'],
        'total_fertility_rate': tfr,
        'total_births': total_births,
        'total_deaths': world_2023['Total Deaths (thousands)'] * 1000,
        'life_expectancy': world_2023['Life Expectancy at Birth, both sexes (years)']
    }

def estimate_survival_probabilities_1yr(world_data):
    """
    Estimate 1-year survival probabilities from available mortality data.
    """
    # 101 age groups (0 to 100+)
    survival_probs = np.ones(101)
    
    # 1. Infant mortality (age 0)
    infant_mortality_rate = world_data['Infant Mortality Rate (infant deaths per 1,000 live births)'] / 1000
    survival_probs[0] = 1 - infant_mortality_rate
    
    # 2. Child mortality (ages 1-4)
    under5_mortality = world_data['Under-Five Mortality (deaths under age 5 per 1,000 live births)'] / 1000
    child_mortality_1_4 = (under5_mortality - infant_mortality_rate) / 4
    
    for age in range(1, 5):
        survival_probs[age] = 1 - child_mortality_1_4
    
    # 3. Mortality before age 40
    mortality_before_40 = world_data['Mortality before Age 40, both sexes (deaths under age 40 per 1,000 live births)'] / 1000
    annual_mortality_5_39 = 1 - (1 - mortality_before_40) ** (1/35)
    
    for age in range(5, 40):
        survival_probs[age] = 1 - annual_mortality_5_39
    
    # 4. Mortality between 40-59
    mortality_before_60 = world_data['Mortality before Age 60, both sexes (deaths under age 60 per 1,000 live births)'] / 1000
    mortality_40_59 = mortality_before_60 - mortality_before_40
    annual_mortality_40_59 = 1 - (1 - mortality_40_59) ** (1/20)
    
    for age in range(40, 60):
        survival_probs[age] = 1 - annual_mortality_40_59
    
    # 5. Mortality for older ages
    # Use gradual increase in mortality for ages 60+
    for age in range(60, 101):
        # Exponential increase in mortality with age
        mortality = 0.01 * np.exp(0.07 * (age - 60))
        survival_probs[age] = 1 - min(mortality, 0.99)  # Cap at 99% mortality
    
    return survival_probs

def transform_survival_probabilities(base_survival, mortality_scale):
    """
    Directly transform survival probabilities using valid operations
    """
    # Convert survival to mortality space
    base_mortality = 1.0 - base_survival
    
    # Apply mortality scale in log-odds space to maintain bounds
    log_odds_mortality = np.log(base_mortality / (1 - base_mortality + 1e-9))
    
    # Scale effect diminishes near boundaries (logistic transformation property)
    scaled_log_odds = log_odds_mortality * (2.0 - mortality_scale)
    
    # Convert back to probability space
    scaled_mortality = 1.0 / (1.0 + np.exp(-scaled_log_odds))
    
    return 1.0 - scaled_mortality

def get_lee_carter_error_pattern(error_scale, n_ages):
    """
    Simple uniform error pattern - same magnitude for all ages
    """
    return np.full(n_ages, error_scale)

def estimate_lee_carter_parameters():
    """
    Estimate Lee-Carter parameters using historical mortality data 1950-2023.
    Now returns parameters for 101 age groups to match the 1-year cohort structure.
    """
    # Load historical data
    wpp_data = pd.read_excel("./Data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx", 
                           sheet_name="Estimates",
                           skiprows=16)
    
    # Filter for World data 1950-2023
    world_historical = wpp_data[(wpp_data['Region, subregion, country or area *'] == 'World') & 
                               (wpp_data['Year'] >= 1950) & (wpp_data['Year'] <= 2023)]
    
    # Extract crude death rates over time
    crude_death_rates = world_historical['Crude Death Rate (deaths per 1,000 population)'].values
    years = world_historical['Year'].values
    
    # Number of age groups - now 101 to match 1-year cohort structure
    n_ages = 101
    
    # a_x: average log mortality rate by age
    # Use typical age pattern of mortality for 1-year age groups
    # Create a smooth mortality curve across all ages
    ages = np.arange(n_ages)
    
    # Infant and child mortality (ages 0-4)
    infant_mortality = 0.02 * np.exp(-0.5 * ages[:5])
    
    # Young adult mortality (ages 5-25) - very low
    young_adult_mortality = 0.001 * np.ones(21)
    
    # Adult mortality (ages 26-60) - gradual increase
    adult_ages = np.arange(26, 61)
    adult_mortality = 0.002 * np.exp(0.05 * (adult_ages - 26))
    
    # Elderly mortality (ages 61-100) - rapid increase
    elderly_ages = np.arange(61, 101)
    elderly_mortality = 0.01 * np.exp(0.08 * (elderly_ages - 61))
    
    # Combine all mortality patterns
    typical_age_pattern = np.concatenate([
        infant_mortality,
        young_adult_mortality,
        adult_mortality,
        elderly_mortality
    ])
    
    lc_a_x = np.log(typical_age_pattern + 1e-9)
    
    # b_x: age-specific pattern of mortality change
    # Higher improvement for younger ages, less for older ages
    lc_b_x = 0.15 * np.exp(-0.03 * ages)
    
    # k_t: time trend of mortality (estimated from crude death rate trend)
    k_t_series = (crude_death_rates - crude_death_rates.mean()) / crude_death_rates.std()
    
    # e_x_t: error terms (random normal)
    lc_e_x_t = np.random.normal(0, 0.05, n_ages)
    
    return lc_a_x, lc_b_x, lc_e_x_t

def load_actual_demographic_data():
    """
    Load actual demographic data based on UN WPP 2024
    """
    # Extract parameters from UN data
    params = extract_actual_demographic_parameters()
    
    # Load historical population data
    historical_data = pd.read_csv("./Data/world_population_combined_1950_2100.csv")
    historical_population = historical_data[historical_data["Year"] <= 2023]["Total Population, as of 1 July (thousands)"].values * 1000
    
    return {
        'initial_population': params['total_population_2023'],
        'historical_population': historical_population,
        'pop_age_dist': params['pop_age_dist'],
        'survival_prob_base': params['survival_prob_base'],
        'asfr_base': params['asfr_base'],
        'net_migration_base': params['net_migration_base'],
        'lc_a_x': params['lc_a_x'],
        'lc_b_x': params['lc_b_x'],
        'lc_e_x_t': params['lc_e_x_t'],
        'crude_birth_rate': params['crude_birth_rate'],
        'crude_death_rate': params['crude_death_rate'],
        'net_migration_rate': params['net_migration_rate'],
        'total_fertility_rate': params['total_fertility_rate']
    }

def run_simulation_with_actual_data(params, data):
    fertility_scale = params[0]
    mortality_scale = params[1]
    migration_scale = params[2]
    lee_carter_k_t_std = params[3]
    model_choice = int(round(params[4]))
    trend_strength = params[5]
    lee_carter_error_scale = params[6]
    k_t_realization = params[7]         # Sobol sample for Lee-Carter k_t
    
    current_total_population = data['initial_population']
    current_pop_age_dist = data['pop_age_dist'].copy()  # Make a copy to avoid modifying original
    historical_population_data = data['historical_population']
    
    pop_2050 = 0
    pop_2075 = 0
    
    # Scale parameters with actual UN baseline values
    scaled_asfr = data['asfr_base'] * fertility_scale
    scaled_survival_prob = data['survival_prob_base'] * (2 - mortality_scale)
    # scaled_survival_prob = np.clip(scaled_survival_prob, 0.0, 1.0)
    scaled_net_migration = data['net_migration_base'] * migration_scale
    
    # UN-based trend assumptions (from WPP projections)
    fertility_trend = -0.01 * trend_strength  # Slight decline
    mortality_trend = -0.005 * trend_strength  # Gradual improvement
    migration_trend = 0.001 * trend_strength   # Small increase
    
    for year_idx in range(1, 52):  # 2024 to 2075
        current_year = 2024 + year_idx
        
        # Apply time trends to parameters
        years_from_now = current_year - 2024
        current_fertility_scale = fertility_scale * (1 + fertility_trend) ** years_from_now
        current_mortality_scale = mortality_scale * (1 + mortality_trend) ** years_from_now
        current_migration_scale = migration_scale * (1 + migration_trend) ** years_from_now
        
        current_asfr = data['asfr_base'] * current_fertility_scale
        current_survival = transform_survival_probabilities(
            data['survival_prob_base'], 
            current_mortality_scale
        )
        # current_survival = np.clip(current_survival, 0.0, 1.0)
        current_migration = data['net_migration_base'] * current_migration_scale
        
        if model_choice == 0:  # Cohort-Component
            current_pop_age_dist = cohort_component_model(
                current_pop_age_dist,
                current_survival,
                current_asfr,
                current_migration
            )
            current_total_population = np.sum(current_pop_age_dist)
            
        elif model_choice == 1:  # UN WPP
            # Use actual UN-based rates
            crude_birth_rate = data['crude_birth_rate'] * current_fertility_scale
            crude_death_rate = data['crude_death_rate'] * current_mortality_scale
            net_migration_rate = data['net_migration_rate'] * current_migration_scale
            
            total_births = current_total_population * crude_birth_rate / 1000
            total_deaths = current_total_population * crude_death_rate / 1000
            total_migration = current_total_population * net_migration_rate / 1000
            
            current_total_population = un_world_population_prospects_model(
                current_total_population, total_deaths, total_births, total_migration
            )
            
        elif model_choice == 2:  # Lee-Carter with Sobol-sampled k_t
            # Convert Sobol sample to k_t using truncated normal
            k_t_simulated = sobol_to_truncated_normal(
                k_t_realization, 
                mean=-0.5 * lee_carter_k_t_std, 
                std=lee_carter_k_t_std,
                lower=-3*lee_carter_k_t_std,  # ±3 sigma bounds
                upper=3*lee_carter_k_t_std
            )
            
            lc_e_x_t = get_lee_carter_error_pattern(lee_carter_error_scale, len(data['lc_a_x']))
            
            baseline_log_mortality = np.log(1 - data['survival_prob_base'] + 1e-9)
            predicted_log_mortality = baseline_log_mortality + data['lc_b_x'] * k_t_simulated + lc_e_x_t
            
            # Convert back with careful bounds
            predicted_mortality = np.exp(predicted_log_mortality)
            
            # Apply realistic age-specific constraints
            max_mortality_by_age = np.array([
                0.1 if age < 5 else 
                0.05 if age < 60 else
                0.2 if age < 80 else
                0.5 for age in range(101)
            ])
            
            lee_carter_survival_prob = 1 - predicted_mortality
            
            if np.any(lee_carter_survival_prob < 0) or np.any(lee_carter_survival_prob > 1):
                print(f"Warning: Invalid survival probabilities in iteration {i}")
                print(f"Min: {lee_carter_survival_prob.min()}, Max: {lee_carter_survival_prob.max()}")
            
            # Recalculate with adjusted mortality
            crude_death_rate = np.sum((1 - lee_carter_survival_prob) * current_pop_age_dist) / current_total_population * 1000
            total_deaths = current_total_population * crude_death_rate / 1000
            
            crude_birth_rate = data['crude_birth_rate'] * current_fertility_scale
            total_births = current_total_population * crude_birth_rate / 1000
            
            net_migration_rate = data['net_migration_rate'] * current_migration_scale
            total_migration = current_total_population * net_migration_rate / 1000
            
            current_total_population = un_world_population_prospects_model(
                current_total_population, total_deaths, total_births, total_migration
            )
            
        elif model_choice == 3:  # Lotka
            # Use actual fertility data
            r_val = lotka_intrinsic_growth_rate_model(
                np.cumprod(np.insert(current_survival[:-1], 0, 1.0)),
                current_asfr
            )
            current_total_population = current_total_population * np.exp(r_val)
            

        # Record population at target years
        if current_year == 2050:
            pop_2050 = current_total_population
        if current_year == 2075:
            pop_2075 = current_total_population
    
    return pop_2050, pop_2075

if __name__ == '__main__':
    # Load data once at the beginning
    print("Loading demographic data...")
    demographic_data = load_actual_demographic_data()
    print("Data loaded successfully")
    
    # Generate samples
    param_values = sobol_sample.sample(problem, 1024, calc_second_order=True, scramble=False)
    
    Y_2050 = []
    Y_2075 = []
    for i, row in enumerate(param_values):
        print(f"Running simulation {i+1}/{len(param_values)}")
        p2050, p2075 = run_simulation_with_actual_data(row, demographic_data)
        Y_2050.append(p2050)
        Y_2075.append(p2075)
    
    # Convert to millions for easier interpretation
    Y_2050 = np.array(Y_2050) / 1e6
    Y_2075 = np.array(Y_2075) / 1e6
    
    # Save results
    results_df = pd.DataFrame({
        'fertility_scale': param_values[:, 0],
        'mortality_scale': param_values[:, 1],
        'migration_scale': param_values[:, 2],
        'lee_carter_k_t_std': param_values[:, 3],
        'model_choice': param_values[:, 4],
        'trend_strength': param_values[:, 5],
        'lee_carter_error_scale': param_values[:, 6],
        'k_t_realization': param_values[:, 7],
        'population_2050_millions': Y_2050,
        'population_2075_millions': Y_2075
    })
    
    results_df.to_csv('./Data/simulation_results_actual.csv', index=False)
    print("Simulation results saved")
    
    # Compare with UN projections
    un_data = pd.read_csv("./Data/world_population_combined_1950_2100.csv")
    un_2050 = un_data[un_data["Year"] == 2050]["Total Population, as of 1 July (thousands)"].values[0]
    un_2075 = un_data[un_data["Year"] == 2075]["Total Population, as of 1 July (thousands)"].values[0]
    
    print(f"\nUN Projections:")
    print(f"2050: {un_2050:,.0f} thousand ({un_2050/1000:,.1f} million)")
    print(f"2075: {un_2075:,.0f} thousand ({un_2075/1000:,.1f} million)")
    
    print(f"\nSimulation Results Range:")
    print(f"2050: {Y_2050.min():.1f} - {Y_2050.max():.1f} million")
    print(f"2075: {Y_2075.min():.1f} - {Y_2075.max():.1f} million")
    
    # Perform Sobol analysis
    problem_sobol = {k: v for k, v in problem.items() if k != 'num_vars'}
    Si_2050 = sobol.analyze(problem_sobol, Y_2050, print_to_console=False)
    Si_2075 = sobol.analyze(problem_sobol, Y_2075, print_to_console=False)
    
    print("\nSensitivity Analysis for 2050:")
    for i, name in enumerate(problem['names']):
        print(f"{name}: First-order = {Si_2050['S1'][i]:.3f}, Total-order = {Si_2050['ST'][i]:.3f}")
    
    print("\nSensitivity Analysis for 2075:")
    for i, name in enumerate(problem['names']):
        print(f"{name}: First-order = {Si_2075['S1'][i]:.3f}, Total-order = {Si_2075['ST'][i]:.3f}")
        
    # Save Sobol' results to a file
    with open("sobol_results.txt", "w") as f:
        f.write("Sobol' Analysis Results\n\n")
        f.write("--- 2050 Projections ---\n")
        f.write("First-order indices:\n")
        f.write(str(Si_2050['S1']))
        f.write("\nTotal-order indices:\n")
        f.write(str(Si_2050['ST']))
        f.write("\n\n--- 2075 Projections ---\n")
        f.write("First-order indices:\n")
        f.write(str(Si_2075['S1']))
        f.write("\nTotal-order indices:\n")
        f.write(str(Si_2075['ST']))
    print("Sobol' analysis results saved to sobol_results.txt")

    simdec_data_2050 = pd.DataFrame(param_values, columns=problem['names'])
    simdec_data_2050['population_2050'] = Y_2050

    simdec_data_2075 = pd.DataFrame(param_values, columns=problem['names'])
    simdec_data_2075['population_2075'] = Y_2075


    # Create input DataFrames for SimDec
    inputs_df_2050 = pd.DataFrame(param_values, columns=problem['names'])
    inputs_df_2075 = pd.DataFrame(param_values, columns=problem['names'])
    
    out_2050 = pd.Series(Y_2050, name='population_2050')
    out_2075 = pd.Series(Y_2075, name='population_2075')

    # Also generate scatter plots for each input vs output for visual inspection
    model_choice_labels = {0: 'Cohort-Component',
        1: 'UN WPP',
        2: 'Lee-Carter',
        3: 'Lotka'}
    
    for param_name in problem['names']:
        if param_name == 'model_choice':
            data_2050 = simdec_data_2050.iloc[::16]
            data_2050.loc[:,'model_choice'] = data_2050['model_choice'].round().astype(int)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=param_name, y='population_2050', data=data_2050.iloc[::16], alpha=0.5)
            # plt.title(f'2050 Population vs. {param_name}')
            plt.xticks(ticks=list(model_choice_labels.keys()), labels=list(model_choice_labels.values()))
            plt.xlabel(param_name)
            plt.ylabel('Population 2050')
            plt.tight_layout()
            plt.savefig(f'scatter_2050_{param_name}.png')
            plt.close()
            
            data_2075 = simdec_data_2075.iloc[::16]
            data_2075.loc[:,'model_choice'] = data_2075['model_choice'].round().astype(int)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=param_name, y='population_2075', data=data_2075.iloc[::16], alpha=0.5)
            # plt.title(f'2075 Population vs. {param_name}')
            plt.xticks(ticks=list(model_choice_labels.keys()), labels=list(model_choice_labels.values()))
            plt.xlabel(param_name)
            plt.ylabel('Population 2075')
            plt.tight_layout()
            plt.savefig(f'scatter_2075_{param_name}.png')
            plt.close()
        else:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=param_name, y='population_2050', data=simdec_data_2050.iloc[::16], alpha=0.5)
            # plt.title(f'2050 Population vs. {param_name}')
            plt.xlabel(param_name)
            plt.ylabel('Population 2050')
            plt.tight_layout()
            plt.savefig(f'scatter_2050_{param_name}.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=param_name, y='population_2075', data=simdec_data_2075.iloc[::16], alpha=0.5)
            # plt.title(f'2075 Population vs. {param_name}')
            plt.xlabel(param_name)
            plt.ylabel('Population 2075')
            plt.tight_layout()
            plt.savefig(f'scatter_2075_{param_name}.png')
            plt.close()
    print("Scatter plots for input parameters vs. output saved.")
