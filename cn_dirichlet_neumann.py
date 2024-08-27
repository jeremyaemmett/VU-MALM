import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

def crank_nicolson_diffusion(variable_diffusivity, domain_length, num_points, time_steps, boundary_flux):
    dx = domain_length / (num_points - 1)
    dt = 1.0  # Time step, can be adjusted
    alpha = dt / (dx ** 2)

    # Initial condition
    u = np.zeros(num_points)
    u_old = np.zeros(num_points)

    # Construct tridiagonal matrix for Crank-Nicolson scheme
    diagonal = np.ones(num_points)
    A = sp.diags([1 - 0.5 * alpha * variable_diffusivity[:-1], -2 * (1 + alpha * variable_diffusivity), 1 - 0.5 * alpha * variable_diffusivity[1:]], [-1, 0, 1], shape=(num_points, num_points)).tocsc()

    for t in range(time_steps):
        # Flux boundary condition
        u[0] = u[1] - boundary_flux * dx

        # Solve the linear system using sparse matrix solver
        b = 0.5 * alpha * variable_diffusivity[:-1] * (u[:-2] - 2 * u[1:-1] + u[2:]) + (1 + 0.5 * alpha * variable_diffusivity) * u[1:-1] + 0.5 * alpha * variable_diffusivity[1:] * (u[2:] - 2 * u[1:-1] + u[:-2])
        u[1:-1] = splinalg.spsolve(A, b)

    return u

# Example usage
domain_length = 1.0
num_points = 100
time_steps = 1000
boundary_flux = 1.0  # Boundary flux
variable_diffusivity = np.linspace(1.0, 0.5, num_points)  # Variable diffusivity, linear for example
solution = crank_nicolson_diffusion(variable_diffusivity, domain_length, num_points, time_steps, boundary_flux)

# Plot the solution
x = np.linspace(0, domain_length, num_points)
plt.plot(x, solution)
plt.xlabel('x')
plt.ylabel('Concentration')
plt.title('1D Crank-Nicolson Diffusion with Flux Boundary Condition')
plt.grid(True)
plt.show()
