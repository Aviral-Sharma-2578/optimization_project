# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from scipy.optimize import linprog
import time

# Problem data
x_coef_list = [
    [83.16, 122.22, 82.62, 133.50, 81.81, 65.70, 52.89],
    [97.20, 98.28, 75.48, 147.00, 40.50, 70.20, 110.94],
    [123.84, 99.12, 76.16, 130.00, 79.92, 38.40, 101.48],
    [75.60, 123.48, 42.84, 195.00, 72.90, 88.20, 59.34],
    [139.32, 107.10, 78.54, 121.50, 36.45, 65.70, 99.33],
    [83.88, 92.82, 124.10, 134.50, 75.87, 69.90, 48.59]
]

x_coefs = pd.DataFrame(x_coef_list, index=range(1, 7), columns=range(1, 8))
farm_demands = pd.Series([36, 42, 34, 50, 27, 30, 43], index=range(1, 8))
capacities = pd.Series([80, 90, 110, 120, 100, 120], index=range(1, 7))
y_coefs = pd.Series([220, 240, 260, 275, 240, 230], index=range(1, 7))

# Function to generate neighbors
def neighbour_founder(solution):
    neighbour_solutions = []
    for i in range(len(solution)):
        solution_copy = solution.copy()
        solution_copy[i] = 1 - solution_copy[i]
        neighbour_solutions.append(solution_copy)
    return neighbour_solutions

# Simplex solver
def simplexsolver(facilities):
    coefs = x_coefs.loc[facilities, :].to_numpy().flatten()
    facility_number = len(facilities)
    lessorequal_coef = np.zeros((facility_number, facility_number * 7), dtype="float64")
    for i in range(facility_number):
        lessorequal_coef[i, i * 7:(i + 1) * 7] = farm_demands.values
    lessorequal_rhs = capacities[facilities].values
    equal_coef = np.zeros((7, facility_number * 7), dtype="float64")
    for i in range(7):
        equal_coef[i, i::7] = 1
    equal_rhs = np.ones(7)
    result = linprog(
        coefs,
        A_ub=lessorequal_coef,
        b_ub=lessorequal_rhs,
        A_eq=equal_coef,
        b_eq=equal_rhs,
        method="revised simplex",
    )
    return result

# Function to find the best solution among feasible neighbors
def better_solution_founder(feasible_set):
    solution_to_return = []
    objective_value_of_solution_to_return = np.inf
    for solution in feasible_set:
        if np.sum(solution) > 0:  # Check if the solution is non-trivial
            facilities = [i + 1 for i, j in enumerate(solution) if j == 1]
            simplex_result = simplexsolver(facilities)
            if simplex_result.success:
                objective_value = simplex_result.fun + np.sum(y_coefs[facilities])
                if objective_value < objective_value_of_solution_to_return:
                    solution_to_return = solution
                    objective_value_of_solution_to_return = objective_value
    return solution_to_return, objective_value_of_solution_to_return

# Regular Tabu Search
def tabu_search(tabu_tenure, tabu_list_length):
    current_sol = []
    while True:
        current_sol = [random.choice([0, 1]) for _ in range(6)]
        if np.sum(np.multiply(current_sol, capacities)) > np.sum(farm_demands):
            break
    tabu_list = []
    best_sol, best_sol_cost = better_solution_founder([current_sol])
    h = 0
    while True:
        h += 1
        neighbour_solutions = neighbour_founder(current_sol)
        feasible_set = [
            sol for sol in neighbour_solutions
            if np.sum(np.multiply(sol, capacities)) >= np.sum(farm_demands)
        ]
        potential_sol, potential_sol_cost = better_solution_founder(feasible_set)
        index_to_tabu = [
            idx for idx, (a, b) in enumerate(zip(current_sol, potential_sol)) if a != b
        ][0]
        tabu_list.append([index_to_tabu, tabu_tenure])
        tabu_list = [[idx, tenure - 1] for idx, tenure in tabu_list if tenure > 0]
        if len(tabu_list) > tabu_list_length:
            tabu_list.pop(0)
        current_sol = potential_sol
        if potential_sol_cost < best_sol_cost:
            best_sol, best_sol_cost = potential_sol, potential_sol_cost
        elif potential_sol_cost == best_sol_cost:
            break
    return best_sol, best_sol_cost

# Tabu Search with Lagrangian Relaxation
def lagrangian_tabu_search(tabu_tenure, tabu_list_length, max_iter=50, step_size=0.1):
    lambdas = np.zeros(7)
    current_sol = []
    while True:
        current_sol = [random.choice([0, 1]) for _ in range(6)]
        if np.sum(np.multiply(current_sol, capacities)) > np.sum(farm_demands):
            break
    tabu_list = []
    best_sol, best_sol_cost = better_solution_founder([current_sol])
    h = 0
    while h < max_iter:
        h += 1
        neighbour_solutions = neighbour_founder(current_sol)
        feasible_set = [
            sol for sol in neighbour_solutions
            if np.sum(np.multiply(sol, capacities)) >= np.sum(farm_demands)
        ]
        potential_sol, potential_sol_cost = better_solution_founder(feasible_set)
        index_to_tabu = [
            idx for idx, (a, b) in enumerate(zip(current_sol, potential_sol)) if a != b
        ][0]
        tabu_list.append([index_to_tabu, tabu_tenure])
        tabu_list = [[idx, tenure - 1] for idx, tenure in tabu_list if tenure > 0]
        if len(tabu_list) > tabu_list_length:
            tabu_list.pop(0)
        current_sol = potential_sol
        if potential_sol_cost < best_sol_cost:
            best_sol, best_sol_cost = potential_sol, potential_sol_cost
        lagrange_update = np.sum(current_sol) - 1
        lambdas = lambdas + step_size * lagrange_update
        step_size /= (1 + h)
    return best_sol, best_sol_cost

# Compare Regular Tabu Search and Lagrangian Tabu Search
def compare_methods(tabu_tenure, tabu_list_length):
    # Regular Tabu Search
    print("Running Regular Tabu Search...")
    start_time = time.time()
    best_sol_tabu, best_cost_tabu = tabu_search(tabu_tenure, tabu_list_length)
    elapsed_tabu = time.time() - start_time
    print(f"Tabu Search: Best cost = {best_cost_tabu}, Computational time = {elapsed_tabu:.4f} seconds")
    
    # Lagrangian Tabu Search
    print("\nRunning Tabu Search with Lagrangian Relaxation...")
    start_time = time.time()
    best_sol_lagrangian_tabu, best_cost_lagrangian_tabu = lagrangian_tabu_search(tabu_tenure, tabu_list_length)
    elapsed_lagrangian = time.time() - start_time
    print(f"Lagrangian Tabu Search: Best cost = {best_cost_lagrangian_tabu}, Computational time = {elapsed_lagrangian:.4f} seconds")

# Run the comparison
compare_methods(tabu_tenure=5, tabu_list_length=10)
