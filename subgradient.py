import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Define constants
numiter = 10
Aineq = np.array([[1, 1, 0, 0, 0],  # Adjusted for λ
                  [0, 0, 1, 1, 0],  # Adjusted for λ
                  [0, 0, 0, 0, 1]]) # Add a row for λ, ensuring it's unconstrained
bineq = np.array([1, 1, 1])  # Adjust bounds to match the new Aineq
lb = np.zeros(5)  # Now including λ
ub = np.ones(5)   # Now including λ

# Function to run the subgradient method
def subgradient_algorithm(alpha_decay_factor):
    lambdak = 0.0  # Initial value
    alphak = 1.0   # Initial step size
    lambdavals = np.zeros(numiter)

    for iter in range(numiter):
        # Negate the coefficients for minimization
        c = np.array([-1 * (16 - 8 * lambdak),  # Coefficients for x1
                      -1 * (10 - 2 * lambdak),  # Coefficients for x2
                      -1 * (0 - lambdak),        # Coefficients for x3
                      -1 * (4 - 4 * lambdak),    # Coefficients for x4
                      10 * lambdak])             # Coefficient for 10λ
        
        # Solve the linear programming problem
        res = linprog(c, A_ub=Aineq, b_ub=bineq, bounds=list(zip(lb, ub)), method='highs')
        xk = res.x[:4]  # Only take the first 4 variables

        b_Ax = 10 - np.dot(np.array([8, 2, 2, 4]), xk)
        lambdak = max(0, lambdak - alphak * b_Ax)
        lambdavals[iter] = lambdak
        
        # Update step size
        alphak /= alpha_decay_factor

    return lambdavals

# Run the subgradient algorithm with different decay factors
lambdavals1 = subgradient_algorithm(1.0)  # alpha_k = 1
lambdavals2 = subgradient_algorithm(2.0)  # alpha_k = alpha_{k-1}/2
lambdavals3 = subgradient_algorithm(3.0)  # alpha_k = alpha_{k-1}/3

# Plot the results
plt.plot(lambdavals1, 'ro-', label='alpha_k = 1')
plt.plot(lambdavals2, 'b*-', label='alpha_k = alpha_{k-1}/2')
plt.plot(lambdavals3, 'g+-', label='alpha_k = alpha_{k-1}/3')
plt.xlabel('Iterations (k)')
plt.ylabel('lambda^{(k)}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
plt.grid()
plt.title('Subgradient Algorithm Results')
plt.show()
