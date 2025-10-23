import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import random
from scipy.stats import qmc
import seaborn as sns

# Constants and initial conditions
T0 = 1978  # Initial year
TF = 2080  # Final year
TARGET_POPULATION = 700e6  # 700 million

AGE_GROUPS = np.arange(0, 91) # 0 to 90+
NUM_AGE_GROUPS = len(AGE_GROUPS)
TIME_STEPS = np.arange(T0, TF + 1) # From 1978 to 2080
NUM_TIME_STEPS = len(TIME_STEPS)

# Pre-process death rates for faster lookup
def preprocess_rates(rates_dict):
    ages = np.array(list(rates_dict.keys()))
    values = np.array(list(rates_dict.values()))
    # Create an interpolation function for ages not directly in the dictionary
    return interp1d(ages, values, kind='nearest', bounds_error=False, fill_value=(values[0], values[-1]))

# Extracted data
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

fertility_rates_rudong_1973 = {
    15: 0.8,
    16: 5.8,
    17: 11.0,
    18: 28.8,
    19: 73.2,
    20: 111.2,
    21: 188.9,
    22: 209.6,
    23: 208.7,
    24: 196.9,
    25: 139.5,
    26: 115.3,
    27: 88.7,
    28: 84.6,
    29: 49.1,
    30: 30.4,
    31: 30.6,
    32: 21.7,
    33: 20.9,
    34: 13.1,
    35: 25.1,
    36: 15.2,
    37: 10.5,
    38: 12.0,
    39: 4.0,
    40: 4.0 # Assuming 40+ means 40 and above, using 4.0 for 40
}

fertility_rates_tianjin_1978 = {
    22: 0.6,
    23: 7.5,
    24: 39.1,
    25: 116.0,
    26: 143.2,
    27: 175.9,
    28: 212.3,
    29: 205.8,
    30: 101.4,
    31: 85.2,
    32: 57.3,
    33: 27.0,
    34: 17.6,
    35: 8.1,
    36: 2.0
}

proportion_in_total_fertility_rudong = {
    15: 0.5,
    16: 3.4,
    17: 6.4,
    18: 16.6,
    19: 42.3,
    20: 64.3,
    21: 109.2,
    22: 121.2,
    23: 120.6,
    24: 113.8,
    25: 80.6,
    26: 66.6,
    27: 51.3,
    28: 48.9,
    29: 28.4,
    30: 17.6,
    31: 17.7,
    32: 12.5,
    33: 12.1,
    34: 7.6,
    35: 14.5,
    36: 8.8,
    37: 6.1,
    38: 6.9,
    39: 4.0,
    40: 4.0 # Assuming 40+ means 40 and above, using 4.0 for 40
}

proportion_in_total_fertility_tianjin = {
    22: 0.5,
    23: 6.3,
    24: 32.6,
    25: 96.7,
    26: 119.3,
    27: 146.6,
    28: 176.9,
    29: 171.4,
    30: 84.5,
    31: 71.0,
    32: 47.7,
    33: 22.5,
    34: 14.7,
    35: 6.7,
    36: 1.7
}

proportion_females_by_age = {
    0: 48.43,
    1: 48.92,
    2: 48.51,
    3: 48.81,
    4: 48.89,
    5: 48.92,
    6: 49.49,
    7: 48.83,
    8: 48.75,
    9: 48.71,
    10: 48.69,
    11: 48.72,
    12: 48.79,
    13: 48.70,
    14: 49.44,
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
    38: 47.24,
    39: 46.41,
    40: 46.24,
    41: 46.59,
    42: 46.78,
    43: 46.93,
    44: 46.34,
    45: 46.77,
    46: 46.57,
    47: 46.80,
    48: 46.85,
    49: 47.23,
    50: 47.23 # Assuming 50+ is represented by 50
}

sex_ratio = {
    1975: {"women": 0.4876, "men": 0.5124},
    1976: {"women": 0.4873, "men": 0.5127},
    1977: {"women": 0.4877, "men": 0.5123},
    1978: {"women": 0.4872, "men": 0.5128}
}

proportion_female_infants_zero_age = {
    1953: 0.4881,
    1964: 0.4906,
    1975: 0.4835,
    1978: 0.4843
}

# Pre-process rates into interpolation functions
interp_death_rates_males = preprocess_rates(age_specific_death_rates_males)
interp_death_rates_females = preprocess_rates(age_specific_death_rates_females)

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
    return proportion_female_infants_zero_age.get(year, proportion_female_infants_zero_age[1978]) / 100.0

def population_dynamics_step_optimized(X_t, beta_t, current_year, mortality_evolution_factor, newborn_female_proportion, fertility_age_shift_factor, interp_fertility_proportion, interp_female_proportion_by_age):
    X_t_plus_1 = np.zeros_like(X_t)

    # Births (age 0 population)
    reproductive_ages = np.arange(15, 50) # Ages 15-49
    valid_reproductive_ages = reproductive_ages[reproductive_ages < NUM_AGE_GROUPS]

    # Apply fertility age shift factor
    shifted_fertility_proportions = interp_fertility_proportion(valid_reproductive_ages) / 1000.0
    if fertility_age_shift_factor != 0:
        # Simple shift: move fertility to older ages by adjusting the index
        # fertility is shifted by 5 years (e.g., ages 20-54 instead of 15-49)
        shifted_fertility_proportions_older = interp_fertility_proportion(valid_reproductive_ages + 5) / 1000.0
        shifted_fertility_proportions = (1 - fertility_age_shift_factor) * shifted_fertility_proportions + \
                                        fertility_age_shift_factor * shifted_fertility_proportions_older

    female_population_in_age_groups = X_t[valid_reproductive_ages] * (interp_female_proportion_by_age(valid_reproductive_ages) / 100.0)
    total_births = np.sum(female_population_in_age_groups * beta_t * shifted_fertility_proportions)
    X_t_plus_1[0] = total_births

    # Aging and Deaths for other age groups
    death_rates_male = interp_death_rates_males(AGE_GROUPS) * (1 - mortality_evolution_factor * (current_year - T0) / (TF - T0))
    death_rates_female = interp_death_rates_females(AGE_GROUPS) * (1 - mortality_evolution_factor * (current_year - T0) / (TF - T0))
    
    # Use newborn_female_proportion to weight male/female death rates
    avg_death_rates = (death_rates_male * (1 - newborn_female_proportion) + death_rates_female * newborn_female_proportion)
    
    X_t_plus_1[1:] = X_t[:-1] * (1 - avg_death_rates[:-1])
    # For the oldest age group (90+), they stay in the same group with deaths
    X_t_plus_1[NUM_AGE_GROUPS - 1] += X_t[NUM_AGE_GROUPS - 1] * (1 - avg_death_rates[NUM_AGE_GROUPS - 1])

    return X_t_plus_1

# Equation 8.4-1: Population Dynamics Model (simulation over time)
def simulate_population_optimized(initial_X, beta_trajectory, mortality_evolution_factor, newborn_female_proportion, fertility_age_shift_factor, interp_fertility_proportion, interp_female_proportion_by_age):
    population_history = np.zeros((NUM_TIME_STEPS, NUM_AGE_GROUPS))
    population_history[0] = initial_X
    current_X = initial_X

    for i in range(NUM_TIME_STEPS - 1):
        year = TIME_STEPS[i]
        beta_t = beta_trajectory[i]
        current_X = population_dynamics_step_optimized(current_X, beta_t, year, mortality_evolution_factor, newborn_female_proportion, fertility_age_shift_factor, interp_fertility_proportion, interp_female_proportion_by_age)
        population_history[i + 1] = current_X
        
    return population_history

# Performance Index J(T)
def performance_index_optimized(X_history, X_star_history):
    J = np.sum((X_history - X_star_history)**2) + np.sum((X_history[:, 0] - X_star_history[0])**2)
    return J

# Constraints
def calculate_social_dependency_ratio_optimized(population_distribution):
    dependent_young = np.sum(population_distribution[0:15])
    dependent_old = np.sum(population_distribution[65:])
    productive_age = np.sum(population_distribution[15:65])
    
    if productive_age == 0:
        return float("inf")
    return (dependent_young + dependent_old) / productive_age

def calculate_aging_index_optimized(population_distribution):
    total_pop = np.sum(population_distribution)
    elderly_pop = np.sum(population_distribution[65:])
    if total_pop == 0:
        return 0.0
    return elderly_pop / total_pop

# Objective function for scipy.optimize.minimize
def objective_function_optimized(beta_array, initial_X, X_star_history, penalty_factor, mortality_evolution_factor, newborn_female_proportion, fertility_age_shift_factor, interp_fertility_proportion, interp_female_proportion_by_age):
    population_history = simulate_population_optimized(initial_X, beta_array, mortality_evolution_factor, newborn_female_proportion, fertility_age_shift_factor, interp_fertility_proportion, interp_female_proportion_by_age)
    J = performance_index_optimized(population_history, X_star_history)

    # Add penalty for constraint violations
    penalty = 0.0

    # Peak population constraint
    total_populations = np.sum(population_history, axis=1)
    violations = total_populations[total_populations > 1.2e9]
    if violations.size > 0:
        penalty += penalty_factor * np.sum((violations - 1.2e9)**2)

    # Social dependency ratio constraint
    sdr_values = np.array([calculate_social_dependency_ratio_optimized(pop_dist) for pop_dist in population_history])
    sdr_violations = sdr_values[sdr_values > 1.0]
    if sdr_violations.size > 0:
        penalty += penalty_factor * np.sum((sdr_violations - 1.0)**2)

    # Aging index constraint
    aging_index_values = np.array([calculate_aging_index_optimized(pop_dist) for pop_dist in population_history])
    aging_index_violations = aging_index_values[aging_index_values > 0.7]
    if aging_index_violations.size > 0:
        penalty += penalty_factor * np.sum((aging_index_violations - 0.7)**2)
            
    return J + penalty

def solve_population_trajectory_with_constraints_scipy_optimized(initial_pyramid_type, fertility_rate_source, mortality_evolution_factor, newborn_female_proportion, optimizer_tolerance, optimizer_max_iterations, penalty_factor, fertility_age_shift_factor):
    initial_tfr = 2.16
    beta_initial_array = np.array([initial_tfr for _ in TIME_STEPS])

    initial_population_distribution = parse_population_data_optimized("./Data/china_population_1978.csv")
    if initial_pyramid_type == 1:
        # Example of a younger population pyramid
        initial_population_distribution[0:20] *= 1.2 # Increase younger population
        initial_population_distribution[40:] *= 0.8 # Decrease older population
        initial_population_distribution = initial_population_distribution * (958e6 / np.sum(initial_population_distribution))
    elif initial_pyramid_type == 2:
        # Example of an older population pyramid
        initial_population_distribution[0:20] *= 0.8 # Decrease younger population
        initial_population_distribution[40:] *= 1.2 # Increase older population
        initial_population_distribution = initial_population_distribution * (958e6 / np.sum(initial_population_distribution))
    # Select fertility rate source
    if fertility_rate_source == 0:
        interp_fertility_proportion = preprocess_rates(proportion_in_total_fertility_rudong)
    else:
        interp_fertility_proportion = preprocess_rates(proportion_in_total_fertility_tianjin)

    interp_female_proportion_by_age = preprocess_rates(proportion_females_by_age)

    # Target population distribution
    X_star_history = np.array([
    10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,  # 0–9
    10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,  # 10–19
    10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,  # 20–29
    10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,  # 30–39
    10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,  # 40–49
    9000000, 9000000, 9000000, 9000000, 9000000,                                                     # 50–54
    8000000, 8000000, 8000000, 8000000, 8000000,                                                     # 55–59
    7000000, 7000000, 7000000, 7000000, 7000000,                                                     # 60–64
    6000000, 6000000, 6000000, 6000000, 6000000,                                                     # 65–69
    5000000, 5000000, 5000000, 5000000, 5000000,                                                     # 70–74
    4000000, 4000000, 4000000, 4000000, 4000000,                                                     # 75–79
    2000000, 2000000, 2000000, 2000000, 2000000,                                                     # 80–84
    1000000, 1000000, 1000000, 1000000, 1000000,                                                     # 85–89
    500000                                                                               # 90+
])

    # Bounds for beta (TFR)
    bounds = [(1, 2.16) for _ in TIME_STEPS]

    # Optimization using L-BFGS-B
    result = minimize(
        objective_function_optimized,
        beta_initial_array,
        args=(initial_population_distribution, X_star_history, penalty_factor, mortality_evolution_factor, newborn_female_proportion, fertility_age_shift_factor, interp_fertility_proportion, interp_female_proportion_by_age),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': optimizer_max_iterations, 'ftol': optimizer_tolerance}
    )

    optimized_beta_array = result.x
    population_history = simulate_population_optimized(initial_population_distribution, optimized_beta_array, mortality_evolution_factor, newborn_female_proportion, fertility_age_shift_factor, interp_fertility_proportion, interp_female_proportion_by_age)
    
    # Calculate output variables
    total_population_2080 = np.sum(population_history[-1])
    years_aging_index_exceeded = np.sum(np.array([calculate_aging_index_optimized(pop_dist) for pop_dist in population_history]) > 0.7)
    years_dependency_ratio_exceeded = np.sum(np.array([calculate_social_dependency_ratio_optimized(pop_dist) for pop_dist in population_history]) > 1.0)
    years_upper_threshold_exceeded = np.sum(np.sum(population_history, axis=1) > 1.2e9)

    return optimized_beta_array, population_history, X_star_history, total_population_2080, years_aging_index_exceeded, years_dependency_ratio_exceeded, years_upper_threshold_exceeded

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

    # Generate Sobol sequence
    sampler = qmc.Sobol(d=num_dimensions, scramble=False)
    sample = sampler.random_base2(m=int(np.log2(num_samples)))[sample_index]

    # Map Sobol sample to parameter ranges
    parameters = {}
    i = 0

    # initial_pyramid_type
    parameters['initial_pyramid_type'] = int(np.round(sample[i] * (ranges['initial_pyramid_type'][1] - ranges['initial_pyramid_type'][0]) + ranges['initial_pyramid_type'][0]))
    i += 1

    # fertility_rate_source
    parameters['fertility_rate_source'] = int(np.round(sample[i] * (ranges['fertility_rate_source'][1] - ranges['fertility_rate_source'][0]) + ranges['fertility_rate_source'][0]))
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
    # Define parameters for analysis
    N_SIMULATIONS = 256 
    
    OUTPUT_VARIABLES = [
        'total_population_2080',
        'social_index_ratio',
        'aging_index',
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
            
        optimized_beta, pop_history, _, total_pop, aging_years, sdr_years, threshold_years = solve_population_trajectory_with_constraints_scipy_optimized(
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
        results['outputs']['social_index_ratio'].append(calculate_social_dependency_ratio_optimized(pop_history[-1]))
        results['outputs']['aging_index'].append(calculate_aging_index_optimized(pop_history[-1]))
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

    # UNCERTAINTY VISUALIZATIONS
    print("\nGenerating uncertainty visualizations...")
    
    # 1. Population distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(results_df['total_population_2080'], kde=True, bins=10, ax=axes[0,0])
    sns.histplot(results_df['social_index_ratio'], kde=True, bins=10, ax=axes[0,1])
    sns.histplot(results_df['aging_index'], kde=True, bins=10, ax=axes[1,0])
    sns.histplot(results_df['years_above_1.2B'], kde=True, bins=10, ax=axes[1,1])
    plt.tight_layout()
    plt.savefig('./Figures/uncertainty_distributions.png')
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
    plt.savefig('./Figures/population_trajectories.png')
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
    plt.savefig('./Figures/fertility_trajectories.png')
    plt.close()

    # SUMMARY STATISTICS
    print("\nFinal Results Summary:")
    print(f"Mean 2080 Population: {np.mean(results_df['total_population_2080']/1e6):.2f} million")
    print(f"Mean Aging Index: {np.mean(results_df['aging_index']):.3f}")
    print(f"Mean Fertility Distance: {np.mean(results_df['fertility_distance']):.2f}")

    # Save results
    results_df.to_csv('simulation_results.csv', index=False)
    print(f"\nTotal runtime: {time.time()-start_time:.2f} seconds")
