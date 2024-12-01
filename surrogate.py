import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Define constants
numiter = 10
Aineq = np.array([[1, 1, 0, 0, 0],  # Adjusted for λ
                  [0, 0, 1, 1, 0],  # Adjusted for λ
                  [0, 0, 0, 0, 1]]) # Add row for λ, ensuring it's unconstrained
bineq = np.array([1, 1, 1])  # Adjust bounds to match the new Aineq
lb = np.zeros(5)  # Now including λ
ub = np.ones(5)   # Now including λ

# Function to run the surrogate subgradient method
def surrogate_subgradient_method(alpha_decay_factor):
    lambdak = 0.1  # Slightly positive initial value for lambda to kick-start updates
    alphak = 1.0   # Initial step size
    lambdavals = np.zeros(numiter)

    # Step 0: Solve initial LP to get x^0
    c_initial = np.array([-1 * (16 - 8 * lambdak), 
                          -1 * (10 - 2 * lambdak), 
                          -1 * (0 - lambdak), 
                          -1 * (4 - 4 * lambdak), 
                          10 * lambdak])
    res_initial = linprog(c_initial, A_ub=Aineq, b_ub=bineq, bounds=list(zip(lb, ub)), method='highs')
    xk = res_initial.x[:4]  # Initial x^0, only the first 4 variables

    for iter in range(numiter):
        # Step 1: Calculate surrogate subgradient
        b_Ax = 10 - np.dot(np.array([8, 2, 2, 4]), xk)
        gk_surrogate = b_Ax  # Surrogate subgradient

        # Debug: Print b_Ax and gk_surrogate
        print(f"Iteration {iter}: b - Ax = {b_Ax}, gk_surrogate = {gk_surrogate}")

        # Update the multipliers
        lambdak = lambdak - alphak * gk_surrogate
        lambdavals[iter] = lambdak

        # Debug: Print updated lambda
        print(f"Iteration {iter}: Updated lambda = {lambdak}")

        # Step 2: Perform approximate optimization to find x^{k+1}
        # Use a gradient-based approach to ensure decrease in L(lambda^{k+1}, x)
        reduction_achieved = False
        step_size = 0.1  # Initial adjustment step size
        max_steps = 100  # Limit number of steps in case of slow convergence

        # Calculate initial surrogate dual value with x^k
        L_tilde_current = np.sum([(16 - 8 * lambdak) * xk[0],
                                  (10 - 2 * lambdak) * xk[1],
                                  (0 - lambdak) * xk[2],
                                  (4 - 4 * lambdak) * xk[3]]) + 10 * lambdak

        for _ in range(max_steps):
            # Compute the gradient (subgradient) of L with respect to x
            gradient = np.array([
                -(16 - 8 * lambdak),  # Partial derivative with respect to x1
                -(10 - 2 * lambdak),  # Partial derivative with respect to x2
                -(0 - lambdak),       # Partial derivative with respect to x3
                -(4 - 4 * lambdak)    # Partial derivative with respect to x4
            ])
            
            # Update x in the direction opposite to the gradient
            x_candidate = np.clip(xk - step_size * gradient, 0, 1)  # Keep x within bounds

            # Compute surrogate dual with candidate x^{k+1}
            L_tilde_candidate = np.sum([(16 - 8 * lambdak) * x_candidate[0],
                                        (10 - 2 * lambdak) * x_candidate[1],
                                        (0 - lambdak) * x_candidate[2],
                                        (4 - 4 * lambdak) * x_candidate[3]]) + 10 * lambdak

            # Check if the candidate x^{k+1} achieves the required reduction
            if L_tilde_candidate < L_tilde_current:
                xk = x_candidate  # Accept the new x^{k+1}
                L_tilde_current = L_tilde_candidate  # Update current surrogate dual value
                reduction_achieved = True
                break  # Exit loop once reduction is achieved
            else:
                # If no improvement, reduce the step size
                step_size *= 0.9  # Decrease step size for finer adjustments

            # Break if step size becomes very small to avoid infinite loop
            if step_size < 1e-6:
                break

        # Step 3: Update the step size alphak
        alphak = alphak / alpha_decay_factor if np.linalg.norm(gk_surrogate) != 0 else alphak

    return lambdavals

# Run the surrogate subgradient algorithm with different decay factors
lambdavals1 = surrogate_subgradient_method(1.0)  # alpha_k = 1
lambdavals2 = surrogate_subgradient_method(2.0)  # alpha_k = alpha_{k-1}/2
lambdavals3 = surrogate_subgradient_method(3.0)  # alpha_k = alpha_{k-1}/3

# Plot the results
plt.plot(lambdavals1, 'ro-', label='alpha_k = 1')
plt.plot(lambdavals2, 'b*-', label='alpha_k = alpha_{k-1}/2')
plt.plot(lambdavals3, 'g+-', label='alpha_k = alpha_{k-1}/3')
plt.xlabel('Iterations (k)')
plt.ylabel('lambda^{(k)}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
plt.grid()
plt.title('Surrogate Subgradient Method Results with Debugging')
plt.show()
