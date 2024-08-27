import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import transport
import forcing
import params
import init

def diffusion_peatland_explicit(j,d_x_eff,dt,dy,x_air,ccn,nrlayers):

    flux = 0.0
    rho = d_x_eff * dt / dy ** 2.0
    if j == 0:
        next = x_air
    else:
        next = ccn[j - 1]
    if j == nrlayers - 1:
        prev = ccn[j]
    else:
        prev = ccn[j + 1]
    ccn[j] = rho * next + (1.0 - 2.0 * rho) * ccn[j] + rho * prev
    if j < nrlayers - 1:
        gradient = ccn[j + 1] - 2 * ccn[j] + ccn[j - 1]
        flux = rho * gradient

    return(ccn, flux)

doy = 180

temp_data_interp, water_data_interp, waterIce_data_interp, soilOrganic_data_interp, soilMineral_data_interp, \
air_data_interp, ph_data_interp, time_data = forcing.forcing_data()
temp_data_interp_mean, water_data_interp_mean, waterIce_data_interp_mean, soilOrganic_data_interp_mean, \
air_data_interp_mean, ph_data_interp_mean = forcing.doy_forcing_data(doy - 1, temp_data_interp, water_data_interp,
waterIce_data_interp, soilOrganic_data_interp,soilMineral_data_interp, air_data_interp, ph_data_interp, time_data)
temps = temp_data_interp_mean
moists = 0.01 * waterIce_data_interp_mean
airs = 0.01 * air_data_interp_mean

layer_thicknesses, depths = init.define_layers()

methaneair, o2air, co2air, h2air = transport.equilibrium_concentration(20.0)

f_d_w = params.f_d_w
f_d_a = params.f_d_a

dt = 0.001 # grid size for time (s)
dy = 0.01 # grid size for space (m)
dy = layer_thicknesses[0]
y = depths
nrlayers = len(y)
nt = np.int(1.0/dt)
plt.figure(figsize=(7,5))
ccn = np.zeros((nrlayers,)) # initial condition

#ccn[0] = o2air

for t in range(0,nt):
    for j in range(0,nrlayers):

        d_ch4_air, d_o2_air, d_co2_air, d_h2_air, d_ch4_water, d_o2_water, d_co2_water, d_h2_water = \
            transport.diffusivity(20.0, f_d_w, f_d_a)

        # Calculate an effective diffusivity weighted by water and air volume
        d_ch4_eff = (moists[j] * d_ch4_water + airs[j] * d_ch4_air) / (moists[j] + airs[j])
        d_co2_eff = (moists[j] * d_co2_water + airs[j] * d_co2_air) / (moists[j] + airs[j])
        d_o2_eff = (moists[j] * d_o2_water + airs[j] * d_o2_air) / (moists[j] + airs[j])
        d_h2_eff = (moists[j] * d_h2_water + airs[j] * d_h2_air) / (moists[j] + airs[j])

        #print(j==nrlayers-1)
        #rho = d_o2_eff * dt / dy**2.0
        #if j == 0:
        #    next = o2air
        #else:
        #    next = ccn[j-1]
        #if j == nrlayers-1:
        #    prev = ccn[j]
        #else:
        #    prev = ccn[j+1]
        #ccn[j] = rho * next + (1.0-2.0*rho)*ccn[j] + rho*prev
        #if j < nrlayers-1:
        #    gradient = ccn[j + 1] - 2 * ccn[j] + ccn[j - 1]
        #    flux = rho * gradient

        ccn, flux = diffusion_peatland_explicit(j,d_o2_eff,dt,dy,o2air,ccn,nrlayers)
#for V in Vout:
#    plt.plot(V,-y,'k')

plt.plot(ccn,-y,'k')
plt.xlabel('Concentration (mol/m$^3$/day)',fontsize=12)
plt.ylabel('Depth (m)',fontsize=12)
plt.ylim(-0.6,0.0)
plt.xlim(0.0,20.0)

print(ccn)

plt.show()