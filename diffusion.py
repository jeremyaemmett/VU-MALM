from scipy.linalg import solve_banded
from copy import copy
import numpy as np
import params

# test2

def cn_diffusion(U, fsrf, diffs, n_act, print_flag, dt):

    dt_cn = dt/params.diff_n_dt

    # Flux
    f_vec = 0.0 * U
    f_vec[0] = fsrf

    sigma_us_act = (diffs * dt_cn) / float((2. * params.dz * params.dz))

    ### A array -1: 1-23(24) | 0: 0, 1-22(23), 23 | 1: 0-22(23)
    A_u = np.diagflat([-sigma_us_act[i] for i in range(1, n_act)], -1) + \
          np.diagflat([1. + sigma_us_act[0]] + [1. + 2. * sigma_us_act[i] for i in range(1, n_act - 1)] + [1. + sigma_us_act[-1]]) + \
          np.diagflat([-sigma_us_act[i] for i in range(n_act - 1)], 1)

    A_u_banded = \
        np.array([[0] + [-sigma_us_act[i] for i in range(1, n_act)],
                  [1. + sigma_us_act[0]] + [1. + 2. * sigma_us_act[i] for i in range(1, n_act - 1)] + [1. + sigma_us_act[-1]],
                  [-sigma_us_act[i] for i in range(n_act - 1)] + [0]])
    #A_u_banded = np.transpose(A_u_banded)

    ### B array -1: 1-22(23) | 0: 0, 1-22(23), 23 | 1: 0-22(23)
    B_u = np.diagflat([sigma_us_act[i] for i in range(1, n_act)], -1) + \
          np.diagflat([1. - sigma_us_act[0]] + [1. - 2. * sigma_us_act[i] for i in range(1, n_act - 1)] + [1. - sigma_us_act[-1]]) + \
          np.diagflat([sigma_us_act[i] for i in range(n_act - 1)], 1)

    B_u_banded = \
        np.array([[0] + [sigma_us_act[i] for i in range(1, n_act)],
                  [1. - sigma_us_act[0]] + [1. - 2. * sigma_us_act[i] for i in range(1, n_act - 1)] + [1. - sigma_us_act[-1]],
                  [sigma_us_act[i] for i in range(n_act - 1)] + [0]])

    B_u_dot_u = np.transpose(np.array(np.dot(B_u, U)))

    U_record = []

    U_record.append(U)

    if print_flag: print('U before:   ', np.sum(U) * params.dz)
    U[0] = U[0] + dt * fsrf
    if print_flag: print('U after1:   ', np.sum(U) * params.dz)

    sum1 = np.sum(U)
    for ti in range(1, params.diff_n_dt):  # [days]

        U_new = np.linalg.solve(A_u, B_u.dot(U))# + f_vec
        #U_new = solve_banded((1, 1), A_u_banded, B_u_dot_u) + f_vec

    sum2 = np.sum(U_new)
    U = U_new * (sum1/sum2)

    if print_flag: print('U after2:    ', np.sum(U) * params.dz)
    if print_flag: print('flux added: ', fsrf*params.dz)

    return(U)


def partial_pressure():

    ch4_vol, o2_vol, co2_vol, h2_vol = 0.000179, 20.946, 0.0412, 0.00005

    ch4_gmol, o2_gmol, co2_gmol, h2_gmol = 16.042, 31.999, 44.010, 2.016

    air_pa = 101.325e3
    ch4_pp = (ch4_vol/100.0) * air_pa
    o2_pp = (o2_vol/100.0) * air_pa
    co2_pp = (co2_vol/100.0) * air_pa
    h2_pp = (h2_vol/100.0) * air_pa

    return(ch4_pp, o2_pp, co2_pp, h2_pp)


def equilibrium_concentration(layer_temp):

    layer_temp = 20.0 + 0.0*layer_temp
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


def schmidt_number(layer_temp):

    sc_ch4 = 1898.0 - 110.1 * layer_temp + 2.834 * (layer_temp ** 2) - 0.02791 * (layer_temp ** 3)
    sc_o2 = 1800.6 - 120.1 * layer_temp + 3.7818 * (layer_temp ** 2) - 0.047608 * (layer_temp ** 3)
    sc_co2 = 1911.0 - 113.7 * layer_temp + 2.967 * (layer_temp ** 2) - 0.02943 * (layer_temp ** 3)
    sc_h2 = 629.95 - 34.691 * layer_temp + 0.8681 * (layer_temp ** 2) - 0.0084 * (layer_temp ** 3)

    return(sc_ch4, sc_o2, sc_co2, sc_h2)


def transfer_velocity(sc_ch4, sc_co2, sc_o2, sc_h2):

    u_10 = 0.0
    n = -0.5
    phi_600 = 2.07 + 0.215 * u_10 ** 1.7

    # Calculate transfer velocities, converting phi [cm/hr] to phi [m/day] with a 0.24x factor
    phi_ch4 = 0.24 * phi_600 * (sc_ch4 / 600.0) ** n
    phi_o2 = 0.24 * phi_600 * (sc_o2 / 600.0) ** n
    phi_co2 = 0.24 * phi_600 * (sc_co2 / 600.0) ** n
    phi_h2 = 0.24 * phi_600 * (sc_h2 / 600.0) ** n

    return(phi_ch4, phi_co2, phi_o2, phi_h2)


def surface_flux(layer_temp,ch4,co2,o2,h2):

    sc_ch4, sc_o2, sc_co2, sc_h2 = schmidt_number(layer_temp)

    phi_ch4, phi_co2, phi_o2, phi_h2 = transfer_velocity(sc_ch4, sc_co2, sc_o2, sc_h2)

    ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = equilibrium_concentration(layer_temp)
    #ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = equilibrium_concentration(20.0)

    f_diff_ch4 = -1.0 * phi_ch4 * (ch4 - ceq_ch4)
    #if ch4 < ceq_ch4 and f_diff_ch4 < 0.0: print('flux: ', ch4, ceq_ch4, ch4 - ceq_ch4, f_diff_ch4)
    #print('flux: ', ch4, ceq_ch4, f_diff_ch4)
    f_diff_o2 = -1.0 * phi_o2 * (o2 - ceq_o2)
    f_diff_co2 = -1.0 * phi_co2 * (co2 - ceq_co2)
    f_diff_h2 = -1.0 * phi_h2 * (h2 - ceq_h2)

    if layer_temp < 0.0:
        f_diff_ch4, f_diff_o2, f_diff_co2, f_diff_h2 = 0.0, 0.0, 0.0, 0.0

    return(f_diff_ch4,f_diff_o2,f_diff_co2,f_diff_h2)


def diffusivity(layer_temps):

    layer_temps_k = layer_temps + 273.15
    sec_to_day = 8.64e4

    # Reference temperatures
    theta = 273.15
    phi = 298.0

    # Diffusivities in air. Convert from m^2/s to m^2/day with the sec_to_day factor.
    d_ch4_air = sec_to_day * params.f_d_a * (1.9e-5) * (layer_temps_k/theta)**1.82
    d_o2_air = sec_to_day * params.f_d_a * (1.8e-5) * (layer_temps_k/theta)**1.82
    d_co2_air = sec_to_day * params.f_d_a * (1.47e-5) * (layer_temps_k/theta)**1.792
    #d_h2_air = sec_to_day * params.f_d_a * (6.68e-5) * (layer_temps_k/phi)**1.82
    d_h2_air = sec_to_day * params.f_d_a * (1.8e-5) * (layer_temps_k / theta) ** 1.82

    d_ch4_air[layer_temps < 0.0] = 0.0
    d_o2_air[layer_temps < 0.0] = 0.0
    d_co2_air[layer_temps < 0.0] = 0.0
    d_h2_air[layer_temps < 0.0] = 0.0

    # Diffusivities in water. Convert from m^2/s to m^2/day with the sec_to_day factor.
    d_ch4_water = sec_to_day * params.f_d_w * (1.5e-9) * (layer_temps_k/phi)
    d_o2_water = sec_to_day * params.f_d_w * (2.4e-9) * (layer_temps_k/phi)
    d_co2_water = sec_to_day * params.f_d_w * (1.81e-6) * np.exp(-2032.6/layer_temps_k)
    d_h2_water = sec_to_day * params.f_d_w * (5.11e-9) * (layer_temps_k/phi)

    d_ch4_water[layer_temps < 0.0] = 0.0
    d_o2_water[layer_temps < 0.0] = 0.0
    d_co2_water[layer_temps < 0.0] = 0.0
    d_h2_water[layer_temps < 0.0] = 0.0

    return(d_ch4_air,d_o2_air,d_co2_air,d_h2_air,d_ch4_water,d_o2_water,d_co2_water,d_h2_water)


def effective_diffusivity(satpct, layer_temps, n_layers, n_times):

    diffs, eff_diffs = np.zeros([8, n_layers, n_times]), np.zeros([4, n_layers, n_times])

    # In layer l, calculate water and air diffusivities for each gas:
    # 0 ch4_air, 1 o2_air, 2 co2_air, 3 h2_air, # 4 ch4_wat, 5 o2_wat, 6 co2_wat, 7 h2_wat
    diffs[0], diffs[1], diffs[2], diffs[3], diffs[4], diffs[5], diffs[6], diffs[7] = \
        diffusivity(layer_temps)

    # Calculate effective diffusivities accounting for layer saturation
    ww, aw = 0.01 * satpct, 1.0 - 0.01 * satpct # Weights representing relative water and air volumes, for averaging
    if params.diff_mode == 'mixed':
        eff_diffs[0] = (ww * diffs[4] + aw * diffs[0]) / (ww + aw)
        eff_diffs[1] = (ww * diffs[5] + aw * diffs[1]) / (ww + aw)
        eff_diffs[2] = (ww * diffs[6] + aw * diffs[2]) / (ww + aw)
        eff_diffs[3] = (ww * diffs[7] + aw * diffs[3]) / (ww + aw)
    if params.diff_mode == 'air':  # 0% saturation
        eff_diffs[0], eff_diffs[1], eff_diffs[2], eff_diffs[3] = \
            diffs[0], diffs[1], diffs[2], diffs[3]
    if params.diff_mode == 'water':  # 100% saturation
        eff_diffs[0], eff_diffs[1], eff_diffs[2], eff_diffs[3] = \
            diffs[4], diffs[5], diffs[6], diffs[7]
    eff_diffs[:, layer_temps <= 0.0] = 0.0

    return(eff_diffs)


def diffusion_cranknicolson2(dt, layer_temps, satpct, eff_diffs_ch4, eff_diffs_o2, eff_diffs_co2, eff_diffs_h2, dr, df, ca):

    ch4, co2, o2, h2 = np.copy(ca['ch4']), np.copy(ca['co2']), np.copy(ca['o2']), np.copy(ca['h2'])
    ch4_0, co2_0, o2_0, h2_0 = copy(ca['ch4']), copy(ca['co2']), copy(ca['o2']), copy(ca['h2'])

    # Number of unfrozen layers
    n_act = len(np.where(layer_temps > 0.0)[0])

    # Calculate surface fluxes
    df['f_diff_ch4'], df['f_diff_o2'], df['f_diff_co2'], df['f_diff_h2'] = \
    surface_flux(layer_temps[0],ch4[0],co2[0],o2[0],h2[0])

    ch4_act, eff_diffs_act_ch4 = ch4[0:n_act], eff_diffs_ch4[0:n_act]
    o2_act, eff_diffs_act_o2 = o2[0:n_act], eff_diffs_o2[0:n_act]
    co2_act, eff_diffs_act_co2 = co2[0:n_act], eff_diffs_co2[0:n_act]
    h2_act, eff_diffs_act_h2 = h2[0:n_act], eff_diffs_h2[0:n_act]

    ch4[0:n_act] = cn_diffusion(ch4_act, df['f_diff_ch4'] / params.dz, eff_diffs_act_ch4, n_act, False, dt)
    o2[0:n_act] = cn_diffusion(o2_act, df['f_diff_o2'] / params.dz, eff_diffs_act_o2, n_act, False, dt)
    co2[0:n_act] = cn_diffusion(co2_act, df['f_diff_co2'] / params.dz, eff_diffs_act_co2, n_act, False, dt)
    h2[0:n_act] = cn_diffusion(h2_act, df['f_diff_h2'] / params.dz, eff_diffs_act_h2, n_act, False, dt)

    ch4_f, co2_f, o2_f, h2_f = copy(ch4), copy(co2), copy(o2), copy(h2)

    # Calculate diffusion rate profiles based on the predicted changes in concentrations
    dr['q_diff_ch4_profile'], dr['q_diff_co2_profile'], dr['q_diff_o2_profile'], dr['q_diff_h2_profile'] = \
        (ch4_f-ch4_0)/dt, (co2_f-co2_0)/dt, (o2_f-o2_0)/dt, (h2_f-h2_0)/dt

    for di in range(0, len(list(dr['q_diff_ch4_profile']))):
        if 0.0 < layer_temps[di] < 1.0:
            dr['q_diff_ch4_profile'][di] = layer_temps[di] * dr['q_diff_ch4_profile'][di]
            dr['q_diff_co2_profile'][di] = layer_temps[di] * dr['q_diff_co2_profile'][di]
            dr['q_diff_o2_profile'][di] = layer_temps[di] * dr['q_diff_o2_profile'][di]
            dr['q_diff_h2_profile'][di] = layer_temps[di] * dr['q_diff_h2_profile'][di]

#    ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = equilibrium_concentration(layer_temps)

#    pos_diff_idx = np.where(dr['q_diff_ch4_profile'] > 0.0)[0]
#    if len(pos_diff_idx) > 0: dr['q_diff_ch4_profile'][pos_diff_idx] = np.min([dr['q_diff_ch4_profile'][pos_diff_idx], abs(ca['ch4'][pos_diff_idx] - ceq_ch4[pos_diff_idx]) / params.dt])
#    pos_diff_idx = np.where(dr['q_diff_co2_profile'] > 0.0)[0]
#    if len(pos_diff_idx) > 0: dr['q_diff_co2_profile'][pos_diff_idx] = np.min([dr['q_diff_co2_profile'][pos_diff_idx], abs(ca['co2'][pos_diff_idx] - ceq_co2[pos_diff_idx]) / params.dt])
#    pos_diff_idx = np.where(dr['q_diff_o2_profile'] > 0.0)[0]
#    if len(pos_diff_idx) > 0: dr['q_diff_o2_profile'][pos_diff_idx] = np.min([dr['q_diff_o2_profile'][pos_diff_idx], abs(ca['o2'][pos_diff_idx] - ceq_o2[pos_diff_idx]) / params.dt])
#    pos_diff_idx = np.where(dr['q_diff_h2_profile'] > 0.0)[0]
#    if len(pos_diff_idx) > 0: dr['q_diff_h2_profile'][pos_diff_idx] = np.min([dr['q_diff_h2_profile'][pos_diff_idx], abs(ca['h2'][pos_diff_idx] - ceq_h2[pos_diff_idx]) / params.dt])

    # Prevent diffusion in frozen layers
    #dr['q_diff_ch4_profile'][layer_temps <= 0.0], dr['q_diff_co2_profile'][layer_temps <= 0.0], \
    #dr['q_diff_o2_profile'][layer_temps <= 0.0], dr['q_diff_h2_profile'][layer_temps <= 0.0] = 0.0, 0.0, 0.0, 0.0

    return(dr, df)
