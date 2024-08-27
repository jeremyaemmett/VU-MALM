import params
import numpy
import diffusion
import transport
from matplotlib import pyplot
numpy.set_printoptions(precision=3)
from matplotlib import pyplot as plt


def cn_diffusion(U, fsrf, diffs, dt, N):

    dt_cn = dt / N

    # Flux
    f_vec = 0.0 * U
    f_vec[0] = fsrf * dt_cn

    n_act = 150

    sigma_us_act = (diffs * dt_cn) / float((2. * dx * dx))
    alpha = (diffs * dt_cn) / float(dx * dx)
    alpha = 0.0*alpha + 0.07

    ### A array -1: 1-23(24) | 0: 0, 1-22(23), 23 | 1: 0-22(23)
    A_u = numpy.diagflat([-alpha[i] for i in range(1, n_act)], -1) + \
          numpy.diagflat([2. + alpha[0]] + [2. * (1. + alpha[i]) for i in range(1, n_act - 1)] + [2. + alpha[-1]]) + \
          numpy.diagflat([-alpha[i] for i in range(n_act - 1)], 1)

    ### B array -1: 1-22(23) | 0: 0, 1-22(23), 23 | 1: 0-22(23)
    B_u = numpy.diagflat([alpha[i] for i in range(1, n_act)], -1) + \
          numpy.diagflat([2. - alpha[0]] + [2. * (1. - alpha[i]) for i in range(1, n_act - 1)] + [2. - alpha[-1]]) + \
          numpy.diagflat([alpha[i] for i in range(n_act - 1)], 1)

    U_record = []

    U_record.append(U)

    for ti in range(1, N):  # [days]

        U_new = numpy.linalg.solve(A_u, B_u.dot(U))# + f_vec
        U = U_new

    return(U)

L = 1.                                          # Domain length [m]
J = 150                                         # Number of grid points
dx = float(L)/float(J-1)                        # Layer thickness [m]
x_grid = numpy.array([j*dx for j in range(J)])

dt = 1.0/24.0                                   # main computational time step [days]

temp = 20.0
ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = transport.equilibrium_concentration(temp)

no_high = 10
# U array
U = numpy.array([2. for i in range(0,no_high)] + [0.1 for i in range(no_high,J)]) # Initial concentrations
U = 0.0 * U
U[0:5] = 0.5*ceq_o2
#U[0] = 0.0
D_u = 0.1 # D_u = Diffusivity [m^2/day]

d_ch4_air, d_o2_air, d_co2_air, d_h2_air, d_ch4_water, d_o2_water, d_co2_water, d_h2_water = diffusion.diffusivity(temp)
diffs = d_o2_water + 0.0*U
diffs[10:] = 0.1*d_o2_water
fsrf = 0.000 # mol/m3/day

fig, ax = plt.subplots()

days = 30.0
for n in range(0,int(days/dt)):
    f_diff_ch4, f_diff_o2, f_diff_co2, f_diff_h2 = diffusion.surface_flux(temp, U[0], U[0], U[0], U[0])
    fsrf = f_diff_o2/dx
    if n*dt % 1 == 0: ax.plot(x_grid, U, color='green', linewidth=0.5 + 2.0 * n / int(days/dt))
    U = cn_diffusion(U, fsrf, diffs, dt, 25)
    print(n*dt, numpy.sum(U))

plt.show()