import numpy as np
from scipy.linalg import solve_banded
from matplotlib import pyplot as plt

n_dx = 50
n_tx = 3600
alpha = 1.
dx = 1.
dt = 1.

flux = 0.1

T = np.full(n_dx, 20.)
T[0] = 50
x = np.arange(n_dx)*dx
r = alpha * dt /2 /dx**2

rho = ((2.0*flux)/alpha) * dx * r

a = np.zeros((3, n_dx-1))

a[0, 1:] = -r
a[1, :] = (1 + 2*r)
a[1, 48] = (1+r)

a[2, :-1] = -r
a[2, 48] = 0

b = np.zeros(n_dx-1)

fig, ax = plt.subplots()

for i in range(0, n_tx+1):
    if i == 0 or i % 300 == 0: ax.plot(x, T)
    b[0] = r*T[0] + (1-2*r)*T[1] + r*T[2]# + rho*T[1] + (1+rho)*T[0]
    b[1:-1] = r*T[1:-2] + (1-2*r)*T[2:-1] + r*T[3:]
    b[0] = b[0] + r*T[0]
    b[48] = r * T[48-1] + (1-r) * T[48]
    T[1:] = solve_banded((1,1), a, b)
plt.show()

