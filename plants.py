from scipy.linalg import solve_banded
from copy import copy
import numpy as np
import diffusion
import params

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
    #ag_biomass = 0.0

    # Density of cross-sectional area of root endings at depth i (m^2/m^3)
    rootx_density = rootx_per_biomass * (root_fraction(depth) / layer_thickness) * ag_biomass

    return(rootx_density)


def plant_transport_rate2(dt, layer_thicknesses, depths, layer_temps, root_profile, pr, pf, ca, satpct):

    root_profile = 0.1 * root_profile

    f_d_w = params.f_d_w # Reduction factor for diffusion in water-filled peat: 0.8 []
    f_d_a = params.f_d_a # Reduction factor for diffusion in air-filled peat: 0.8 []

    tortuosity = params.tau # Root tortuosity: 1.5 []

    dt = dt
    dxs = layer_thicknesses
    nx = len(dxs)-1

    # Calculate per-layer and integrated plant fluxes
    for i in range(0, nx+1): # Layer

        tortuosity = params.tau
        if satpct[i] > params.saturation_threshold: tortuosity = 3.0 * params.tau

        # Density of cross-sectional area of root endings at depth i (m^2/m^3)
        d_ch4_air, d_o2_air, d_co2_air, d_h2_air, d_ch4_water, d_o2_water, d_co2_water, d_h2_water \
            = diffusivity(layer_temps[i],f_d_w,f_d_a)
        rootx_density = root_xsections(layer_thicknesses[i],depths[i])
        total_root_biomass = params.dz * np.sum(root_profile)
        f_root_i = (params.dz * root_profile[i]) / total_root_biomass
        rootx_density = total_root_biomass * params.a_m_a * (f_root_i/params.dz)
        #rootx_density = 0.01 * rootx_density
        geomfac = 1.0

        ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = equilibrium_concentration(20.0)

        aero_factor = (1.0 - satpct[i] / 100.0)
        anaero_factor = 1.0 - aero_factor
        #aero_factor = 1.0

        # Diffusion rates across a water-to-air interface (for root endings in saturated soil)
        ch4_layer_rate_w2a, co2_layer_rate_w2a, o2_layer_rate_w2a, h2_layer_rate_w2a = \
            diffusion.surface_flux(layer_temps[i], ca['ch4'][i], ca['co2'][i], ca['o2'][i], ca['h2'][i])
        ch4_layer_rate_w2a, co2_layer_rate_w2a, o2_layer_rate_w2a, h2_layer_rate_w2a = \
        -(rootx_density / tortuosity) * ch4_layer_rate_w2a, -(rootx_density / tortuosity) * co2_layer_rate_w2a, \
        -(rootx_density / tortuosity) * o2_layer_rate_w2a, -(rootx_density / tortuosity) * h2_layer_rate_w2a

        # Diffusion rates in air (for root endings in dry soil)
        ch4_layer_rate_a, co2_layer_rate_a, o2_layer_rate_a, h2_layer_rate_a = \
        aero_factor * (d_ch4_air / tortuosity) * rootx_density * ((ca['ch4'][i]-ceq_ch4) / depths[i]), \
        aero_factor * (d_co2_air / tortuosity) * rootx_density * ((ca['co2'][i] - ceq_co2) / depths[i]), \
        aero_factor * (d_o2_air / tortuosity) * rootx_density * ((ca['o2'][i]-ceq_o2) / depths[i]), \
        aero_factor * (d_h2_air / tortuosity) * rootx_density * ((ca['h2'][i]-ceq_h2) / depths[i])

        # Diffusion rate = Mean of the above diffusion rates, weighted by saturation
        ch4_layer_rate = aero_factor * ch4_layer_rate_a + anaero_factor * ch4_layer_rate_w2a
        co2_layer_rate = aero_factor * ch4_layer_rate_a + anaero_factor * ch4_layer_rate_w2a
        o2_layer_rate = aero_factor * ch4_layer_rate_a + anaero_factor * ch4_layer_rate_w2a
        h2_layer_rate = aero_factor * ch4_layer_rate_a + anaero_factor * ch4_layer_rate_w2a
        ace_layer_rate = 0.0

        # Diffusion rate = Mean of the above diffusion rates, weighted by saturation
        ch4_layer_rate = ch4_layer_rate_w2a
        co2_layer_rate = ch4_layer_rate_w2a
        o2_layer_rate = ch4_layer_rate_w2a
        h2_layer_rate = ch4_layer_rate_w2a
        ace_layer_rate = 0.0

        #aero_factor = (1.0 - satpct[i] / 100.0)
        # Plant transport rate in layer i. Positive = layer -> atmos transport. Negative = layer <- atmos [mol/m^3/d]
        #ch4_layer_rate = aero_factor * (d_ch4_air / tortuosity) * rootx_density * ((ca['ch4'][i]-ceq_ch4) / depths[i])
        #o2_layer_rate = aero_factor * (d_o2_air / tortuosity) * rootx_density * ((ca['o2'][i]-ceq_o2) / depths[i])
        #co2_layer_rate = aero_factor * (d_co2_air / tortuosity) * rootx_density * ((ca['co2'][i]-ceq_co2) / depths[i])
        #h2_layer_rate = aero_factor * (d_h2_air / tortuosity) * rootx_density * ((ca['h2'][i]-ceq_h2) / depths[i])
        #ace_layer_rate = 0.0

        #if i == 0: print('compare: ', o2_layer_rate, -1.0 * abs(ca['o2'][i] - ceq_o2) / params.dt, ca['o2'][i], ceq_o2)

        # Record the plant rate profiles in the 'pr' dictionary
        pr['q_plant_ch4_profile'][i] = ch4_layer_rate
        pr['q_plant_o2_profile'][i] = o2_layer_rate
        pr['q_plant_co2_profile'][i] = co2_layer_rate
        pr['q_plant_h2_profile'][i] = h2_layer_rate
        pr['q_plant_ace_profile'][i] = ace_layer_rate

        for di in range(0, len(list(pr['q_plant_ch4_profile']))):
            if 0.0 < layer_temps[di] < 1.0:
                pr['q_plant_ch4_profile'][di] = layer_temps[di] * pr['q_plant_ch4_profile'][di]
                pr['q_plant_o2_profile'][di] = layer_temps[di] * pr['q_plant_ch4_profile'][di]
                pr['q_plant_co2_profile'][di] = layer_temps[di] * pr['q_plant_ch4_profile'][di]
                pr['q_plant_h2_profile'][di] = layer_temps[di] * pr['q_plant_ch4_profile'][di]

        test_pr = -0.0005

        #pr['q_plant_ch4_profile'][i] = test_pr
        #pr['q_plant_o2_profile'][i] = test_pr
        #pr['q_plant_co2_profile'][i] = test_pr
        #pr['q_plant_h2_profile'][i] = test_pr
        #pr['q_plant_ace_profile'][i] = 0.0

        # If the layer's frozen, there's no plant transport
        if layer_temps[i] <= 0.0:
            pr['q_plant_ch4_profile'][i], pr['q_plant_o2_profile'][i], \
            pr['q_plant_co2_profile'][i], pr['q_plant_h2_profile'][i] = 0.0, 0.0, 0.0, 0.0

        # If the layer's saturated, reduce plant transport
        #if satpct[i] == 100.0:
        #    pr['q_plant_ch4_profile'][i], pr['q_plant_o2_profile'][i], pr['q_plant_co2_profile'][i], pr['q_plant_h2_profile'][i] = \
        #    0.1*pr['q_plant_ch4_profile'][i], 0.1*pr['q_plant_o2_profile'][i], 0.1*pr['q_plant_co2_profile'][i], 0.1*pr['q_plant_h2_profile'][i]
    #stop
    # Sum the plant fluxes in each layer to yield total plant fluxes [mol/m^2/d] + [m] * [mol/m^3/d] = [mol/m^2/d]
    pf['plant_flux_ch4'] = layer_thicknesses[0] * np.sum(pr['q_plant_ch4_profile'])
    pf['plant_flux_o2'] = layer_thicknesses[0] * np.sum(pr['q_plant_o2_profile'])
    pf['plant_flux_co2'] = layer_thicknesses[0] * np.sum(pr['q_plant_co2_profile'])
    pf['plant_flux_h2'] = layer_thicknesses[0] * np.sum(pr['q_plant_h2_profile'])
    pf['plant_flux_ace'] = 0.0

    #print(pr['q_plant_o2_profile'][0])

    return(pr, pf)


def diffusivity(layer_temp,f_d_w,f_d_a):

    layer_temp_k = layer_temp + 273.15
    sec_to_day = 8.64e4

    # Reference temperatures
    theta = 273.15
    phi = 298.0

    # Diffusivities in air
    d_ch4_air = sec_to_day * (1.9e-5) * (layer_temp_k/theta)**1.82
    d_o2_air = sec_to_day * (1.8e-5) * (layer_temp_k/theta)**1.82
    d_co2_air = sec_to_day * (1.47e-5) * (layer_temp_k/theta)**1.792
    d_h2_air = sec_to_day * (6.68e-5) * (layer_temp_k/theta)**1.82
    if layer_temp < 0.0:
        d_ch4_air = 0.0
        d_o2_air = 0.0
        d_co2_air = 0.0
        d_h2_air = 0.0

    # Diffusivities in water
    sf = 1.0
    d_ch4_water = sec_to_day * (1.5e-9) * (layer_temp_k/phi)
    d_o2_water = sec_to_day * (2.4e-9) * (layer_temp_k/phi)
    d_co2_water = sec_to_day * (1.81e-6) * np.exp(-2032.6/layer_temp_k)
    d_h2_water = sec_to_day * (5.11e-9) * (layer_temp_k/phi)
    if layer_temp < 0.0:
        d_ch4_water = 0.0
        d_o2_water = 0.0
        d_co2_water = 0.0
        d_h2_water = 0.0

    return(d_ch4_air,d_o2_air,d_co2_air,d_h2_air,d_ch4_water,d_o2_water,d_co2_water,d_h2_water)


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