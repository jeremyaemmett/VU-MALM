import params
import numpy as np
import diffusion
import transport
from matplotlib import pyplot
np.set_printoptions(precision=3)
from matplotlib import pyplot as plt


def cn_diffusion(U, fsrf, diffs, n_act):

    dt_cn = params.dt/params.diff_n_dt

    # Flux
    f_vec = 0.0 * U
    f_vec[0] = fsrf * dt_cn

    sigma_us_act = (diffs * dt_cn) / float((2. * params.dz * params.dz))

    ### A array -1: 1-23(24) | 0: 0, 1-22(23), 23 | 1: 0-22(23)
    A_u = np.diagflat([-sigma_us_act[i] for i in range(1, n_act)], -1) + \
          np.diagflat([1. + sigma_us_act[0]] + [1. + 2. * sigma_us_act[i] for i in range(1, n_act - 1)] + [1. + sigma_us_act[-1]]) + \
          np.diagflat([-sigma_us_act[i] for i in range(n_act - 1)], 1)

    ### B array -1: 1-22(23) | 0: 0, 1-22(23), 23 | 1: 0-22(23)
    B_u = np.diagflat([sigma_us_act[i] for i in range(1, n_act)], -1) + \
          np.diagflat([1. - sigma_us_act[0]] + [1. - 2. * sigma_us_act[i] for i in range(1, n_act - 1)] + [1. - sigma_us_act[-1]]) + \
          np.diagflat([sigma_us_act[i] for i in range(n_act - 1)], 1)

    U_record = []

    U_record.append(U)

    for ti in range(1, params.diff_n_dt):  # [days]

        sum1 = np.sum(U)
        U_new = np.linalg.solve(A_u, B_u.dot(U)) + f_vec
        sum2 = np.sum(U_new - f_vec)
        U = U_new * (sum1/sum2)

    return(U)

L = 1.                                          # Domain length [m]
x_grid = np.array([j*params.dz for j in range(params.nz)])

dt = params.dt                                   # main computational time step [days]

temp = 20.0
ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = transport.equilibrium_concentration(temp)

no_high = 10
# U array
U = np.zeros(params.nz) # Initial concentrations
U = 0.0 * U
U[0] = 0.5*ceq_h2

diffs, eff_diffs = np.zeros([8, params.nz]), np.zeros([4, params.nz])
# In layer l, calculate water and air diffusivities for each gas:
# 0 ch4_air, 1 o2_air, 2 co2_air, 3 h2_air, # 4 ch4_wat, 5 o2_wat, 6 co2_wat, 7 h2_wat
diffs[0], diffs[1], diffs[2], diffs[3], diffs[4], diffs[5], diffs[6], diffs[7] = \
    diffusion.diffusivity(20.0 + np.zeros(params.nz))
eff_diffs = np.zeros(params.nz)
eff_diffs = diffs[1]
eff_diffs[10:] = diffs[5,10:]

dt_cn = dt / params.diff_n_dt
#h2 = cn_diffusion(h2, df['f_diff_ch4'], eff_diffs[0, :], n_dx)
df = {}

fig, ax = plt.subplots()

days = 30.0
for n in range(0,int(days/dt)):
    temp = 20 + n%20
    n_act = 15 + n%8
    df['f_diff_ch4'], df['f_diff_o2'], df['f_diff_co2'], df['f_diff_h2'] = diffusion.surface_flux(temp, U[0], U[0], U[0], U[0])
    fsrf = df['f_diff_o2']/params.dz
    if n*dt % 1 == 0: ax.plot(x_grid, U, color='green', linewidth=0.5 + 2.0 * n / int(days/dt))

    U_act, eff_diffs_act = U[0:n_act], eff_diffs[0:n_act]
    U_act = cn_diffusion(U_act, fsrf, eff_diffs_act, n_act)
    U[0:n_act] = U_act

    print(n_act, U[0], temp, df['f_diff_o2'], n*dt, np.sum(U))

plt.show()