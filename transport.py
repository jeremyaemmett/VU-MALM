from scipy.linalg import solve_banded
from copy import copy
import numpy as np
import params

def partial_pressure():

    ch4_vol = 0.000179
    o2_vol = 20.946
    co2_vol = 0.0412
    h2_vol = 0.00005

    ch4_gmol = 16.042
    o2_gmol = 31.999
    co2_gmol = 44.010
    h2_gmol = 2.016

    ch4_mol = (ch4_vol/100.0) * 100.0 * (1.0/ch4_gmol)
    o2_mol = (o2_vol / 100.0) * 100.0 * (1.0 / o2_gmol)
    co2_mol = (co2_vol / 100.0) * 100.0 * (1.0 / co2_gmol)
    h2_mol = (h2_vol / 100.0) * 100.0 * (1.0 / h2_gmol)

    air_pa = 101.325e3
    ch4_pp = (ch4_vol/100.0) * air_pa
    o2_pp = (o2_vol/100.0) * air_pa
    co2_pp = (co2_vol/100.0) * air_pa
    h2_pp = (h2_vol/100.0) * air_pa

    return(ch4_pp,o2_pp,co2_pp,h2_pp)

def equilibrium_concentration(layer_temp):

    theta = 273.15
    ts = layer_temp+273.15

    h_ch4 = (1.3e-3) * np.exp(1700.0 * ((1.0/ts)-(1.0/theta)))
    h_o2 = (1.3e-3) * np.exp(1500.0 * ((1.0/ts) - (1.0 / theta)))
    h_co2 = (3.4e-2) * np.exp(2400.0 * ((1.0/ts) - (1.0 / theta)))
    h_h2 = (7.8e-4) * np.exp(530.0 * ((1.0/ts) - (1.0 / theta)))

    pp_ch4,pp_o2,pp_co2,pp_h2 = partial_pressure()

    ceq_ch4 = pp_ch4 * h_ch4
    ceq_o2 = pp_o2 * h_o2
    ceq_co2 = pp_co2 * h_co2
    ceq_h2 = pp_h2 * h_h2

    return(ceq_ch4, ceq_o2, ceq_co2, ceq_h2)


def surface_flux(layer_temp,ch4,co2,o2,h2):

    u_10 = 0.0
    n = -0.5
    phi_600 = 2.07 + 0.215 * u_10**1.7
    phi = 298.0

    sc_ch4 = 1898.0 - 110.1*layer_temp + 2.834*(layer_temp**2) - 0.02791*(layer_temp**3)
    sc_o2 = 1800.6 - 120.1 * layer_temp + 3.7818 * (layer_temp ** 2) - 0.047608 * (layer_temp ** 3)
    sc_co2 = 1911.0 - 113.7 * layer_temp + 2.967 * (layer_temp ** 2) - 0.02943 * (layer_temp ** 3)
    sc_h2 = 629.95 - 34.691 * layer_temp + 0.8681 * (layer_temp ** 2) - 0.0084 * (layer_temp ** 3)

    phi_ch4 = phi_600*(sc_ch4/600.0)**n
    phi_o2 = phi_600 * (sc_o2 / 600.0) ** n
    phi_co2 = phi_600 * (sc_co2 / 600.0) ** n
    phi_h2 = phi_600 * (sc_h2 / 600.0) ** n

    ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = equilibrium_concentration(layer_temp)

    f_diff_ch4 = -1.0 * phi_ch4 * (ch4 - ceq_ch4)
    #print(ch4, ceq_ch4,f_diff_ch4)
    #stop
    f_diff_o2 = -1.0 * phi_o2 * (o2 - ceq_o2)
    f_diff_co2 = -1.0 * phi_co2 * (co2 - ceq_co2)
    f_diff_h2 = -1.0 * phi_h2 * (h2 - ceq_h2)

    if layer_temp < 0.0:
        f_diff_ch4 = 0.0
        f_diff_o2 = 0.0
        f_diff_co2 = 0.0
        f_diff_h2 = 0.0

    #print(ch4,ceq_ch4,layer_temp)

    return(f_diff_ch4,f_diff_o2,f_diff_co2,f_diff_h2)

def diffusivity(layer_temp,f_d_w,f_d_a):

    layer_temp_k = layer_temp + 273.15
    sec_to_day = 8.64e4

    # Reference temperatures
    theta = 273.15
    phi = 298.0

    # Diffusivities in air
    d_ch4_air = sec_to_day * f_d_a * (1.9e-5) * (layer_temp_k/phi)**1.82
    d_o2_air = sec_to_day * f_d_a * (1.8e-5) * (layer_temp_k/phi)**1.82
    d_co2_air = sec_to_day * f_d_a * (1.47e-5) * (layer_temp_k/phi)**1.792
    d_h2_air = sec_to_day * f_d_a * (6.68e-5) * (layer_temp_k/phi)**1.82
    if layer_temp < 0.0:
        d_ch4_air = 0.0
        d_o2_air = 0.0
        d_co2_air = 0.0
        d_h2_air = 0.0

    # Diffusivities in water
    sf = 1.0
    d_ch4_water = sec_to_day * f_d_w * (1.5e-9) * (layer_temp_k/theta)
    d_o2_water = sec_to_day * f_d_w * (2.4e-9) * (layer_temp_k/theta)
    d_co2_water = sec_to_day * f_d_w * (1.81e-6) * np.exp(-2032.6/layer_temp_k)
    d_h2_water = sec_to_day * f_d_w * (5.11e-9) * (layer_temp_k/theta)
    if layer_temp < 0.0:
        d_ch4_water = 0.0
        d_o2_water = 0.0
        d_co2_water = 0.0
        d_h2_water = 0.0

    return(d_ch4_air,d_o2_air,d_co2_air,d_h2_air,d_ch4_water,d_o2_water,d_co2_water,d_h2_water)


def cn_solver(T_record,n_dx,dx,n_tx,dt_cn,r,x_atmos):

    # Diffusion coefficients
    #r = diff * dt /2 /dx**2

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
    ####print(T_record)
    T = np.append(x_atmos,T_record[0:n_dx-1])
    ####print(T)
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

        # Record the below-ground solution values (T[0] = c_air gets appended again in the next round through cn_solver)
        T_record[0:n_dx-1] = T[1:]

    return(T_record)

def diffusion_cranknicolson(layer_thicknesses,layer_temps,ch4,co2,o2,h2,ace,moists,airs,dt):

    # Space/Time stepping
    ####ice_line = np.where(layer_temps < 0.0)[0][0]
    ice_line = np.where(layer_temps > 0.0)[0][-1]+1
    #if ice_line == 0:
    #    ice_line = np.where(layer_temps < 0.0)[0][1]
    ####print(np.where(layer_temps > 0.0)[0])
    ####print('ice_line: ',ice_line)
    n_dx = ice_line + 1+1     # Add a 'ghost' layer to represent the surface-air interface (z = 0)
    dx = 0.025              # Thickness of each layer (m)
    dt_cn = 0.001          # CN timestep (days)
    n_tx = int(dt/dt_cn)    # the number of CN dt's in the main computational dt
    # Get air concentrations, used as fixed upper boundaries (mol/m3)
    ch4air, o2air, co2air, h2air = equilibrium_concentration(20.0)
    #print(ch4air,o2air)
    #stop
    f_d_w = params.f_d_w
    f_d_a = params.f_d_a
    #print('ice_line: ',ice_line)
    diffs = np.zeros([8,n_dx])
    eff_diffs = np.zeros([4,n_dx])
    # diffs indices = 0 dch4_air, 1 do2_air, 2 dco2_air, 3 dh2_air, # 4 dch4_wat, 5 do2_wat, 6 dco2_wat, 7 dh2_wat
    for l in range(1,ice_line+1):
        diffs[0,l], diffs[1,l], diffs[2,l], diffs[3,l], diffs[4,l], diffs[5,l], diffs[6,l], diffs[7,l] = \
            diffusivity(layer_temps[l-1], f_d_w, f_d_a)
        #print(layer_temps[l-1])
        ## Calculate an effective diffusivity weighted by water and air volume
        #eff_diffs[0,l] = (moists[l] * diffs[4,l] + airs[l] * diffs[0,l]) / (moists[l] + airs[l])
        #eff_diffs[1,l] = (moists[l] * diffs[5,l] + airs[l] * diffs[1,l]) / (moists[l] + airs[l])
        #eff_diffs[2,l] = (moists[l] * diffs[6,l] + airs[l] * diffs[2,l]) / (moists[l] + airs[l])
        #eff_diffs[3,l] = (moists[l] * diffs[7,l] + airs[l] * diffs[3,l]) / (moists[l] + airs[l])
        # Calculate an effective diffusivity = water diffusivity
        eff_diffs[0,l] = diffs[4,l]
        eff_diffs[1,l] = diffs[5,l]
        eff_diffs[2,l] = diffs[6,l]
        eff_diffs[3,l] = diffs[7,l]
        if layer_temps[l] <= 0.0:
            eff_diffs[:,l] = 0.0
    r_ch4 = eff_diffs[0,:] * dt_cn / 2 / dx**2
    r_o2 = eff_diffs[1,:] * dt_cn / 2 / dx**2
    r_co2 = eff_diffs[2,:] * dt_cn / 2 / dx**2
    r_h2 = eff_diffs[3,:] * dt_cn / 2 / dx**2

    #print(layer_temps)
    #print(ice_line)
    #print(r_ch4)
    #stop

    ch40 = copy(ch4)
    ####print('n_dx: ',n_dx)
    ####print(ch4)
    ch4 = cn_solver(ch4, n_dx, dx, n_tx, dt_cn, r_ch4, ch4air)
    ch4f = copy(ch4)
    ch4_rates = (ch4f-ch40)/dt #mol/m3/day

    o20 = copy(o2)
    o2 = cn_solver(o2, n_dx, dx, n_tx, dt_cn, r_o2, o2air)
    o2_rates = (copy(o2)-o20)/dt #mol/m3/day

    co20 = copy(co2)
    co2 = cn_solver(co2, n_dx, dx, n_tx, dt_cn, r_co2, co2air)
    co2_rates = (copy(co2)-co20)/dt #mol/m3/day

    h20 = copy(h2)
    h2 = cn_solver(h2, n_dx, dx, n_tx, dt_cn, r_h2, h2air)
    h2_rates = (copy(h2)-h20)/dt #mol/m3/day

    ace_rates = 0.0*ch4_rates

    #print(layer_temps)
    f_diff_ch4, f_diff_o2, f_diff_co2, f_diff_h2 = surface_flux(layer_temps[0],ch40[0],co20[0],o20[0],h20[0])
    #print(f_diff_ch4)
    #ch4[0] = ch4[0] + dt * f_diff_ch4
    #co2[0] = co2[0] + dt * f_diff_co2
    #o2[0] = o2[0] + dt * f_diff_o2
    #h2[0] = h2[0] + dt * f_diff_h2

    return(ch4,co2,o2,h2,ace,f_diff_ch4,f_diff_o2,f_diff_co2,f_diff_h2)


def root_fraction(depth):

    depth = 100.0 * depth # Convert depth from [m] to [cm]
    decay_factor = 5.0 # Root depth distribution decay parameter [cm]
    z_max = 50.0 # Maximum depth that roots reach [cm]
    c_term = ((1.0/decay_factor)*np.exp((1.0/decay_factor)*z_max)) / (np.exp((1.0/decay_factor)*z_max) - 1.0)
    f_root = c_term * np.exp(-depth / decay_factor)

    return(f_root)

def root_xsections(layer_thickness, depth):

    rootx_per_biomass = params.a_m_a # Cross-sectional area of root endings per root biomass: 0.085 [m^2/kg]
    ag_biomass = 0.7 # Above-ground biomass (assumed to equal root biomass) [kg/m^2]

    # Density of cross-sectional area of root endings at depth i (m^2/m^3)
    rootx_density = rootx_per_biomass * (root_fraction(depth) / layer_thickness) * ag_biomass

    return(rootx_density)

def plant_transport_rate(dt,layer_thicknesses,depths,layer_temps,ch4,co2,o2,h2,ace,
                q_plant_ch4_profile, q_plant_o2_profile, q_plant_co2_profile, q_plant_h2_profile, q_plant_ace_profile):

    f_d_w = params.f_d_w # Reduction factor for diffusion in water-filled peat: 0.8 []
    f_d_a = params.f_d_a # Reduction factor for diffusion in air-filled peat: 0.8 []

    tortuosity = params.tau # Root tortuosity: 1.5 []

    dt = dt
    dxs = layer_thicknesses
    nx = len(dxs)-1

    # Calculate integrated plant fluxes
    plant_flux_ch4 = 0.0
    plant_flux_o2 = 0.0
    plant_flux_co2 = 0.0
    plant_flux_h2 = 0.0
    plant_flux_ace = 0.0
    for i in range(0, nx+1): # Layer

        # Density of cross-sectional area of root endings at depth i (m^2/m^3)
        d_ch4_air, d_o2_air, d_co2_air, d_h2_air, d_ch4_water, d_o2_water, d_co2_water, d_h2_water \
            = diffusivity(layer_temps[i],f_d_w,f_d_a)
        rootx_density = root_xsections(layer_thicknesses[i],depths[i])

        ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = equilibrium_concentration(layer_temps[i])

        # Plant transport rate in layer i. Positive = layer -> atmos transport. Negative = layer <- atmos [mol/m^3/d]
        q_plant_ch4 = (d_ch4_air / tortuosity) * rootx_density * ((ch4[i]-ceq_ch4) / depths[i])
        q_plant_o2 = (d_o2_air / tortuosity) * rootx_density * ((o2[i]-ceq_o2) / depths[i])
        q_plant_co2 = (d_co2_air / tortuosity) * rootx_density * ((co2[i]-ceq_co2) / depths[i])
        q_plant_h2 = (d_h2_air / tortuosity) * rootx_density * ((h2[i]-ceq_h2) / depths[i])
        q_plant_ace = 0.0

        # If the layer's frozen, there's no plant transport
        if layer_temps[i] <= 0.0:
            q_plant_ch4 = 0.0
            q_plant_o2 = 0.0
            q_plant_co2 = 0.0
            q_plant_h2 = 0.0
            q_plant_ace = 0.0

        # Record the plant transport rates in a profile
        q_plant_ch4_profile[i] = q_plant_ch4
        q_plant_o2_profile[i] = q_plant_o2
        ##if i == 5:
        ##    print('raw: ',q_plant_o2_profile[i])
        ##    print('ceq: ',ceq_o2,layer_temp)
        q_plant_co2_profile[i] = q_plant_co2
        q_plant_h2_profile[i] = q_plant_h2
        q_plant_ace_profile[i] = q_plant_ace

        # Sum the plant fluxes in each layer to yield total plant fluxes [mol/m^2/d] + [m] * [mol/m^3/d] = [mol/m^2/d]
        plant_flux_ch4 = plant_flux_ch4 + (layer_thicknesses[i] * q_plant_ch4)
        plant_flux_o2 = plant_flux_o2 + (layer_thicknesses[i] * q_plant_o2)
        plant_flux_co2 = plant_flux_co2 + (layer_thicknesses[i] * q_plant_co2)
        plant_flux_h2 = plant_flux_h2 + (layer_thicknesses[i] * q_plant_h2)
        plant_flux_ace = 0.0

        #print(i,layer_temp,plant_flux_ch4)

        #if i == 1:
            #print(ch4[i])
            #print(q_plant_h2,h2[i],dt,h2[i],ceq_h2)

    #print('flux: ',plant_flux_ch4)

    return(ch4, co2, o2, h2, ace,
           q_plant_ch4_profile, q_plant_o2_profile, q_plant_co2_profile, q_plant_h2_profile, q_plant_ace_profile,
           plant_flux_ch4, plant_flux_co2, plant_flux_o2, plant_flux_h2, plant_flux_ace)



