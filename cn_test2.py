import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from plotly.subplots import make_subplots
import plotly.io as pio
import transport
import forcing
import params
from copy import copy
import init

def cn_solver(T_record,n_dx,dx,n_tx,dt,diff,x_atmos):

    # Diffusion coefficients
    r = diff * dt /2 /dx**2

    # Banded 'AB' array (rows = diagonal values of the tri-diagonal 'A' array)
    a = np.zeros((3, n_dx - 1))
    a[0, 1:] = -r[0:(n_dx - 2)]  # j-1
    a[1, :] = (1 + 2 * r[0:(n_dx - 1)])  # j
    a[1, (n_dx - 2)] = (1 + r[n_dx - 2])  # j
    a[2, :-1] = -r[1:(n_dx - 1)]  # j+1
    a[2, (n_dx - 2)] = 0  # j+1

    # Initialize the 'B' array
    b = np.zeros(n_dx - 1)

    # Loop within the main computational time step
    T = T_record[0:n_dx]
    T_records=[]
    T[0] = x_atmos
    for i in range(1, n_tx + 1):

        # Calculate the indices of the 'B' array with corresponding r (diff) coefficients
        ri = r[1]
        b[0] = ri * T[0] + (1 - 2 * ri) * T[1] + ri * T[2]
        ri = r[2:-1]
        b[1:-1] = ri * T[1:-2] + (1 - 2 * ri) * T[2:-1] + ri * T[3:]
        ri = r[0]
        b[0] = b[0] + ri * T[0]
        ri = r[(n_dx - 2)]
        b[(n_dx - 2)] = ri * T[(n_dx - 2) - 1] + (1 - ri) * T[(n_dx - 2)]

        # Solve the linear system 'Ax = B' for x (unknown values at the next time step)
        T[1:] = solve_banded((1, 1), a, b)

        # Record the solution to account for freeze/thaw (which vertically moves the lower boundary level)
        T_record[0:n_dx] = T

    return(T_record)


pio.templates.default='plotly_dark'

temp_data_interp, water_data_interp, waterIce_data_interp, soilOrganic_data_interp, soilMineral_data_interp, \
air_data_interp, ph_data_interp, time_data = forcing.forcing_data()
doy=237
temp_data_interp_mean, water_data_interp_mean, waterIce_data_interp_mean, soilOrganic_data_interp_mean, \
air_data_interp_mean, ph_data_interp_mean = forcing.doy_forcing_data(doy - 1, temp_data_interp, water_data_interp,
                                                                     waterIce_data_interp, soilOrganic_data_interp,
                                                                     soilMineral_data_interp, air_data_interp,
                                                                     ph_data_interp, time_data)

temps = temp_data_interp_mean
moists = 0.01 * waterIce_data_interp_mean
airs = 0.01 * air_data_interp_mean

f_d_w = params.f_d_w
f_d_a = params.f_d_a

d_ch4_eff_mean=[]
d_co2_eff_mean=[]
d_o2_eff_mean=[]
d_h2_eff_mean=[]
for i in range(0,len(temps)):
    d_ch4_air, d_o2_air, d_co2_air, d_h2_air, d_ch4_water, d_o2_water, d_co2_water, d_h2_water = \
        transport.diffusivity(temps[i], f_d_w, f_d_a)

    # Calculate an effective diffusivity weighted by water and air volume
    d_ch4_eff = (moists[i] * d_ch4_water + airs[i] * d_ch4_air) / (moists[i] + airs[i])
    d_co2_eff = (moists[i] * d_co2_water + airs[i] * d_co2_air) / (moists[i] + airs[i])
    d_o2_eff = (moists[i] * d_o2_water + airs[i] * d_o2_air) / (moists[i] + airs[i])
    d_h2_eff = (moists[i] * d_h2_water + airs[i] * d_h2_air) / (moists[i] + airs[i])

    d_ch4_eff_mean = np.append(d_ch4_eff_mean,d_ch4_eff)
    d_co2_eff_mean = np.append(d_co2_eff_mean,d_co2_eff)
    d_o2_eff_mean = np.append(d_o2_eff_mean,d_o2_eff)
    d_h2_eff_mean = np.append(d_h2_eff_mean,d_h2_eff)


d_ch4_eff_mean = np.mean(d_ch4_eff_mean[d_ch4_eff_mean > 0.0])
d_co2_eff_mean = np.mean(d_co2_eff_mean[d_co2_eff_mean > 0.0])
d_o2_eff_mean = np.mean(d_o2_eff_mean[d_o2_eff_mean > 0.0])
d_h2_eff_mean = np.mean(d_h2_eff_mean[d_h2_eff_mean > 0.0])

fig, ax = plt.subplots()
ax.set_ylim(-0.6, 0.0)

# Space/Time stepping
n_dx = 24
dx = 0.025
n_tx = 25
dt = 1/n_tx

# Initial concentration
T_record = np.full(n_dx, 0.0)
x = np.arange(n_dx)*dx

layer_thicknesses, depths = init.define_layers()
print(depths)
print(x)

for j in range(0,15):

    if j <= 3:
        n_dx = n_dx-4
    if 10 > j > 3:
        n_dx = n_dx+2

    diff = np.linspace(1.0,0.1,n_dx)
    atmos_x = 0.047
    r = diff * dt /2 /dx**2
    print(r)
    #stop

    t0 = copy(T_record)
    T_record = cn_solver(T_record, n_dx, dx, n_tx, dt, diff, atmos_x)
    tf = copy(T_record)
    print(j,tf-t0)

    ax.plot(T_record,-x)

plt.show()




