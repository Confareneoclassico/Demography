import numpy as np
from scipy.optimize import root_scalar

def cohort_component_model(population_age_t, survival_prob, fertility_rates, net_migration_age, female_ratio=0.5):
    """
    Implementation of Cohort-Component Model.
    
    Args:
        population_age_t (np.array): Population at age x at time t
        survival_prob (np.array): Survival probability from age x to x+1
        fertility_rates (np.array): Age-specific fertility rates
        net_migration_age (np.array): Net migration at age x
        female_ratio (float): Proportion of population that is female
    
    Returns:
        np.array: Population at age x at time t+1
    """
    num_ages = len(population_age_t)
    population_age_t_plus_1 = np.zeros(num_ages)
    
    # Calculate births (only from female population of reproductive age)
    reproductive_ages = np.arange(15, 50)  # Standard reproductive age range
    female_pop_reproductive = population_age_t[reproductive_ages] * female_ratio
    total_births = np.sum(female_pop_reproductive * fertility_rates[reproductive_ages])
    
    # Handle newborns (age 0)
    # Assume half of births are female, half male for simplicity
    # import pdb; pdb.set_trace()
    population_age_t_plus_1[0] = total_births * survival_prob[0] + net_migration_age #[0]
    
    # Handle other ages (1 to max age-1)
    for age in range(1, num_ages):
        if age < num_ages - 1:
            # Survive population from previous age
            survived = population_age_t[age-1] * survival_prob[age-1]
        else:
            # For the oldest age group, include survival from previous and current
            survived = (population_age_t[age-1] + population_age_t[age]) * survival_prob[age-1]
        
        population_age_t_plus_1[age] = survived + net_migration_age #[age]
    
    return population_age_t_plus_1

def un_world_population_prospects_model(population_t, deaths_t, births_t, migration_t):
    """UN WPP model"""
    return population_t - deaths_t + births_t + migration_t

def lee_carter_model(m_x_t, a_x, b_x, k_t, e_x_t):
    """Lee-Carter model"""
    return a_x + b_x * k_t + e_x_t

def lotka_intrinsic_growth_rate_model(l_x, m_x, r_initial_guess=0.01, max_iter=1000, tol=1e-6):
    """
    Improved Lotka's model with better numerical stability
    """
    ages = np.arange(len(l_x))
    
    def equation_to_solve(r):
        return np.sum(np.exp(-r * ages) * l_x * m_x) - 1
    
    # Use robust root finding method
    try:
        result = root_scalar(equation_to_solve, x0=r_initial_guess, x1=r_initial_guess+0.01, 
                           method='secant', maxiter=max_iter, xtol=tol)
        return result.root
    except:
        print('Fallback to simple method if root_scalar fails')
        r = r_initial_guess
        for _ in range(max_iter):
            f_r = equation_to_solve(r)
            h = 1e-7
            f_prime_r = (equation_to_solve(r + h) - f_r) / h
            
            if abs(f_prime_r) < 1e-10:
                break
                
            r_new = r - f_r / f_prime_r
            if abs(r_new - r) < tol:
                return r_new
            r = r_new
            
        return r

# Example Usage (Illustrative, not for actual execution without data)
if __name__ == '__main__':
    # Cohort-Component Model Example
    pop_t = np.array([1000, 900, 800, 700, 600]) # Population at time t for ages 0-4
    surv_prob = np.array([0.99, 0.98, 0.97, 0.96, 0.95]) # Survival from x-1 to x
    births = np.array([100, 0, 0, 0, 0]) # New births for age 0
    deaths = np.array([10, 5, 4, 3, 2]) # Deaths for each age group
    migration = np.array([5, 2, 1, 0, 0]) # Net migration for each age group
    
    pop_t_plus_1 = cohort_component_model(pop_t, surv_prob, births, deaths, migration)
    print("Cohort-Component Model (P_t+1):", pop_t_plus_1)

    # UN World Population Prospects Model Example
    total_pop_t = 8_000_000_000
    total_deaths = 50_000_000
    total_births = 130_000_000
    total_migration = 10_000_000
    
    total_pop_t_plus_1 = un_world_population_prospects_model(total_pop_t, total_deaths, total_births, total_migration)
    print("UN WPP Model (P_t+1):", total_pop_t_plus_1)

    # Lee-Carter Model Example
    log_m_x_t = np.array([-5.0, -4.5, -4.0, -3.5, -3.0])
    a_x_val = np.array([-5.1, -4.6, -4.1, -3.6, -3.1])
    b_x_val = np.array([0.1, 0.09, 0.08, 0.07, 0.06])
    k_t_val = 1.0
    e_x_t_val = np.array([0.01, 0.005, 0.002, 0.001, 0.0005])
    
    predicted_log_m_x_t = lee_carter_model(log_m_x_t, a_x_val, b_x_val, k_t_val, e_x_t_val)
    print("Lee-Carter Model (ln(m_x(t))):", predicted_log_m_x_t)

    # Lotka's Intrinsic Growth Rate Model Example
    l_x_val = np.array([1.0, 0.98, 0.95, 0.90, 0.80, 0.65, 0.45, 0.25, 0.10, 0.01]) # Survival probabilities
    m_x_val = np.array([0.0, 0.0, 0.05, 0.15, 0.20, 0.18, 0.10, 0.03, 0.0, 0.0]) # Age-specific fertility rates
    
    r_val = lotka_intrinsic_growth_rate_model(l_x_val, m_x_val)
    print("Lotka's Intrinsic Growth Rate (r):", r_val)

