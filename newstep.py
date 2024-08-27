import init
import data
import params
import plants
import forcing
import microbes
import transport
import numpy as np
from copy import copy
from datetime import datetime
from datetime import timedelta

def loss_vs_available(x_profile,q_diff_x_profile,q_plant_x_profile,x_c_cons_profile,x_c_prod_profile,i,dt):

    # List of any negative terms at layer i ([-diffusion, -plant, microbial consumption])
    x_neg_list = np.append(abs((q_diff_x_profile[i])[q_diff_x_profile[i] < 0.0]),abs((q_plant_x_profile[i])[q_plant_x_profile[i] < 0.0]))
    x_neg_list = np.append(x_neg_list, x_c_cons_profile[i])
    # List of any positive terms at layer i ([+diffusion, +plant, microbial production])
    x_pos_list = np.append(abs((q_diff_x_profile[i])[q_diff_x_profile[i] >= 0.0]),abs((q_plant_x_profile[i])[q_plant_x_profile[i] >= 0.0]))
    x_pos_list = np.append(x_pos_list, x_c_prod_profile[i])

    # Sum of the negative terms
    lost = np.sum(x_neg_list)
    gained = np.sum(x_pos_list)
    # Amount that's available to be consumed = amount present + amount simultaneously added
    available = x_profile[i] + gained

    # Scale down the amount diffused by the amount available
    q_diff_x_profile[i] = (available * abs(q_diff_x_profile[i]) / lost) / dt
    # Scale down the amount plant-transported...
    q_plant_x_profile[i] = (available * abs(q_plant_x_profile[i]) / lost) / dt
    # Scale down the amount consumed by microbes...
    x_c_cons_profile[i] = (available * x_c_cons_profile[i] / lost) /dt

    return(lost,available,q_diff_x_profile,q_plant_x_profile,x_c_cons_profile)

def truncate_rates(dt, ch4_profile, o2_profile, co2_profile, h2_profile, ace_profile,
                   ch4_c_prod_profile, co2_c_prod_profile, o2_c_prod_profile, h2_c_prod_profile, ace_c_prod_profile,
                   ch4_c_prod_acmg_profile, ch4_c_prod_h2mg_profile, ace_c_prod_aesp_profile, ace_c_prod_hoag_profile,
                   co2_c_prod_ansp_profile, co2_c_prod_aesp_profile, co2_c_prod_acmg_profile, h2_c_prod_ansp_profile,
                   o2_c_prod_none_profile,
                   ch4_c_cons_profile, co2_c_cons_profile, o2_c_cons_profile, h2_c_cons_profile, ace_c_cons_profile,
                   ch4_c_cons_mtox_profile, ace_c_cons_acmg_profile, co2_c_cons_hoag_profile, co2_c_cons_h2mg_profile,
                   h2_c_cons_hoag_profile, h2_c_cons_h2mg_profile, o2_c_cons_mtox_profile, o2_c_cons_aero_profile,
                   o2_c_cons_aesp_profile,
                   q_diff_ch4_profile, q_diff_co2_profile, q_diff_o2_profile, q_diff_h2_profile,
                   q_plant_ch4_profile, q_plant_o2_profile, q_plant_co2_profile, q_plant_h2_profile, q_plant_ace_profile):

    # Dummy profiles - acetate can't be diffused or plant-transported
    q_diff_ace_profile, q_plant_ace_profile = 0.0*q_diff_ch4_profile, 0.0*q_diff_ch4_profile

    # 'Test' update the quantities
    ch4_rate_sum = (q_diff_ch4_profile - q_plant_ch4_profile + ch4_c_prod_profile - ch4_c_cons_mtox_profile)
    o2_rate_sum = (q_diff_o2_profile - q_plant_o2_profile - o2_c_cons_aero_profile - 2.0*ch4_c_cons_mtox_profile)
    co2_rate_sum = (q_diff_co2_profile - q_plant_co2_profile + co2_c_prod_profile - co2_c_cons_profile)
    h2_rate_sum = (q_diff_h2_profile - q_plant_h2_profile + h2_c_prod_profile - h2_c_cons_profile)
    ace_rate_sum = (ace_c_prod_profile - ace_c_cons_profile)
    ch4_test, o2_test, co2_test, h2_test, ace_test = ch4_rate_sum*dt, o2_rate_sum*dt, co2_rate_sum*dt, h2_rate_sum*dt, ace_rate_sum*dt

    # Record the indices where the 'test' would remove more material than is present in the reservoir
    ch4_truncatable = (ch4_test < 0.0) & (abs(ch4_test) > ch4_profile)
    o2_truncatable = (o2_test < 0.0) & (abs(o2_test) > o2_profile)
    co2_truncatable = (co2_test < 0.0) & (abs(co2_test) > co2_profile)
    h2_truncatable = (h2_test < 0.0) & (abs(h2_test) > h2_profile)
    ace_truncatable = (ace_test < 0.0) & (abs(ace_test) > ace_profile)

    for i in range(0,len(ch4_profile)):

        # If rates should be truncated in layer i (the condition above is met), then truncate them

        # CH4
        if ch4_truncatable[i]:
            lost, available, q_diff_ch4_profile, q_plant_ch4_profile, ch4_c_cons_profile = loss_vs_available(
                ch4_profile,q_diff_ch4_profile,q_plant_ch4_profile,ch4_c_cons_profile,ch4_c_prod_profile,i,dt)
            ch4_c_cons_mtox_profile[i] = (available * ch4_c_cons_mtox_profile[i] / lost) / dt

        # O2
        if o2_truncatable[i]:
            lost, available, q_diff_o2_profile, q_plant_o2_profile, o2_c_cons_profile = loss_vs_available(
                o2_profile, q_diff_o2_profile, q_plant_o2_profile, o2_c_cons_profile, o2_c_prod_profile, i,dt)
            o2_c_cons_aero_profile[i] = (available * o2_c_cons_aero_profile[i] / lost) / dt
            o2_c_cons_mtox_profile[i] = (available * 2.0*o2_c_cons_mtox_profile[i] / lost) / dt

        # CO2
        if co2_truncatable[i]:
            lost, available, q_diff_co2_profile, q_plant_co2_profile, co2_c_cons_profile = loss_vs_available(
                co2_profile, q_diff_co2_profile, q_plant_co2_profile, co2_c_cons_profile, co2_c_prod_profile, i,dt)
            co2_c_cons_hoag_profile[i] = (available * co2_c_cons_hoag_profile[i] / lost) / dt
            co2_c_cons_h2mg_profile[i] = (available * co2_c_cons_h2mg_profile[i] / lost) / dt

        # H2
        if h2_truncatable[i]:
            lost, available, q_diff_h2_profile, q_plant_h2_profile, h2_c_cons_profile = loss_vs_available(
                h2_profile, q_diff_h2_profile, q_plant_h2_profile, h2_c_cons_profile, h2_c_prod_profile, i,dt)
            h2_c_cons_hoag_profile[i] = (available * h2_c_cons_hoag_profile[i] / lost) / dt
            h2_c_cons_h2mg_profile[i] = (available * h2_c_cons_h2mg_profile[i] / lost) / dt

        # Ace
        if ace_truncatable[i]:
            lost, available, q_diff_ace_profile, q_plant_ace_profile, ace_c_cons_profile = loss_vs_available(
                ace_profile, q_diff_ace_profile, q_plant_ace_profile, ace_c_cons_profile, ace_c_prod_profile, i,dt)
            ace_c_cons_acmg_profile[i] = (available * ace_c_cons_acmg_profile[i] / lost) / dt

    return(ch4_c_prod_profile, co2_c_prod_profile, o2_c_prod_profile, h2_c_prod_profile, ace_c_prod_profile,
                   ch4_c_prod_acmg_profile, ch4_c_prod_h2mg_profile, ace_c_prod_aesp_profile, ace_c_prod_hoag_profile,
                   co2_c_prod_ansp_profile, co2_c_prod_aesp_profile, co2_c_prod_acmg_profile, h2_c_prod_ansp_profile,
                   o2_c_prod_none_profile,
                   ch4_c_cons_profile, co2_c_cons_profile, o2_c_cons_profile, h2_c_cons_profile, ace_c_cons_profile,
                   ch4_c_cons_mtox_profile, ace_c_cons_acmg_profile, co2_c_cons_hoag_profile, co2_c_cons_h2mg_profile,
                   h2_c_cons_hoag_profile, h2_c_cons_h2mg_profile, o2_c_cons_mtox_profile, o2_c_cons_aero_profile,
                   o2_c_cons_aesp_profile,
                   q_diff_ch4_profile, q_diff_co2_profile, q_diff_o2_profile, q_diff_h2_profile,
                   q_plant_ch4_profile, q_plant_o2_profile, q_plant_co2_profile, q_plant_h2_profile, q_plant_ace_profile)


def adaptive_timestep(ca,cr,dr,pr,elapsed,temps,dt):

    # Compute net chemistry rates as the sum of the truncated rates
    ch4_rate_sum = (dr['q_diff_ch4_profile'] - pr['q_plant_ch4_profile'] + cr['ch4_c_prod'] - cr['ch4_c_cons_mtox'])
    o2_rate_sum = (dr['q_diff_o2_profile'] - pr['q_plant_o2_profile'] - cr['o2_c_cons_aero'] - 2.0*cr['ch4_c_cons_mtox'])
    co2_rate_sum = (dr['q_diff_co2_profile'] - pr['q_plant_co2_profile'] + cr['co2_c_prod'] - cr['co2_c_cons'])
    h2_rate_sum = (dr['q_diff_h2_profile'] - pr['q_plant_h2_profile'] + cr['h2_c_prod'] - cr['h2_c_cons'])
    ace_rate_sum = (cr['ace_c_prod'] - cr['ace_c_cons'])

    quantities = np.concatenate((ca['ch4'], ca['o2'], ca['co2'], ca['h2'], ca['ace']))
    rates = np.concatenate((ch4_rate_sum, o2_rate_sum, co2_rate_sum, h2_rate_sum, ace_rate_sum))

    approx_dt = 0.5*np.min(quantities/abs(rates))

    dl = [1]
    start=1
    for i in range(5):
        dl.append(start*2)
        start = start*2
    dl = np.array(dl)
    dts = dt/dl

    if len((dts[dts < approx_dt])) == 0:
        clean_dt = dts[-1]
    else:
        clean_dt = dts[dts < approx_dt][0]
    if np.max(rates) == 0.0:
        clean_dt = dt

    if elapsed % dt + clean_dt > dt:
        clean_dt = dt - (elapsed % dt)

    return(clean_dt)


def adaptive_timestep2(ca, cr, dr, pr, forcingt, elapsed, dt):

    ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = plants.equilibrium_concentration(20.0 + 0.0 * forcingt['temp_data'])
    # Projected transport rates, and the differences between current concentrations and equilibrium concentrations
    ch4_transport_sum, ch4_eq_diff = dr['q_diff_ch4_profile']-pr['q_plant_ch4_profile'], (ceq_ch4-ca['ch4'])
    o2_transport_sum, o2_eq_diff = dr['q_diff_o2_profile']-pr['q_plant_o2_profile'], (ceq_o2-ca['o2'])
    co2_transport_sum, co2_eq_diff = dr['q_diff_co2_profile']-pr['q_plant_co2_profile'], (ceq_co2-ca['co2'])
    h2_transport_sum, h2_eq_diff = dr['q_diff_h2_profile']-pr['q_plant_h2_profile'], (ceq_h2-ca['h2'])

    transport_sums = [ch4_transport_sum, o2_transport_sum, co2_transport_sum, h2_transport_sum]
    eq_diffs = [ch4_eq_diff, o2_eq_diff, co2_eq_diff, h2_eq_diff]

    best_dt = params.dt
    count_stack = []
    for c in range(0, len(transport_sums)): # For each gas...

        counts = []
        test_dts = [params.dt, params.dt/2.0, params.dt/4.0, params.dt/8.0, params.dt/16.0, params.dt/32.0, params.dt/120.0,
                    params.dt/64.0, params.dt/128.0, params.dt/256.0, params.dt/512.0] #, params.dt/1024.0]
                    #params.dt/2048.0, params.dt/4096.0, params.dt/8192.0, params.dt/16384.0]

        for dt_test in test_dts: # Try each test dt from largest to smallest

            eq_diff = eq_diffs[c]/dt_test
            true_count = 0
            for l in range(0, params.nz):
                test1, test2, test3 = dt * transport_sums[c][l] > 0.0, dt * transport_sums[c][l] > eq_diff[l], eq_diff[l] > 0.0
                test4 = test1 and test2 and test3
                #test4 = test1 and test2
                if test4:
                    true_count += 1
                    #print(l, ca['ch4'][l], eq_diff[0])

            counts.append(true_count) # Record the number of failed layer tests at each test dt
        #print(counts)

        count_stack.append(counts) # Stack the layer test lists for each gas

    count_stack = np.array(count_stack)
    max_counts = np.max(count_stack, axis=0)
#    if min(max_counts) != 0:
#        print('no good dts')
#        stop
    # The best test dt is is the one where there were the least number of failed layer tests, ideally zero
    # Prioritize the smallest dt in the case of ties
    #print(max_counts)
    #print(' ')
    if min(max_counts) != 0: best_dt_idx = np.where(max_counts == min(max_counts))[0][-1]
    if min(max_counts) == 0: best_dt_idx = np.where(max_counts == min(max_counts))[0][0]
#    print(best_dt_idx)
    best_dt = test_dts[best_dt_idx]
    #print(max_counts, best_dt_idx, best_dt, params.dt/8.0, 0.5/24.0/8.0)
    #print(best_dt)

    if max(forcingt['temp_data']) <= 0.0: best_dt = params.dt # Proceed at the max dt if all layers are frozen

    # If the new dt would overshoot the next write interval, and the
    if elapsed % params.write_dt + best_dt > params.write_dt and params.write_dt - elapsed % params.write_dt > 1e-10:
        best_dt = params.write_dt - elapsed % params.write_dt

#    print(' ')
#    print('best_dt: ', best_dt)
#    print('increase by: ', best_dt * ch4_transport_sum[0])

    return(best_dt)


def activity_level(temps, activity_levels):

    unfrozen_indices = np.where(temps > 0.0)[0]
    frozen_indices = np.where(temps <= 0.0)[0]
    activity_levels[unfrozen_indices] = activity_levels[unfrozen_indices] + 0.01
    activity_levels[activity_levels > 1.0] = 1.0
    activity_levels[frozen_indices] = 0.0

    return(activity_levels)


def truncate_rates2(dt, cr, dr, pr, ca):

    # Dummy profiles - acetate can't be diffused or plant-transported
    dr['q_diff_ace_profile'], pr['q_plant_ace_profile'] = 0.0*dr['q_diff_ch4_profile'], 0.0*dr['q_diff_ch4_profile']

    ## 'Test' update the quantities
    # Net Rates [mol/m3/d]
    ch4_rate_sum = (dr['q_diff_ch4_profile'] - pr['q_plant_ch4_profile'] + cr['ch4_c_prod'] - cr['ch4_c_cons_mtox'])
    o2_rate_sum = (dr['q_diff_o2_profile'] - pr['q_plant_o2_profile'] - cr['o2_c_cons_aero'] - 2.0*cr['ch4_c_cons_mtox'])
    co2_rate_sum = (dr['q_diff_co2_profile'] - pr['q_plant_co2_profile'] + cr['co2_c_prod'] - cr['co2_c_cons'])
    h2_rate_sum = (dr['q_diff_h2_profile'] - pr['q_plant_h2_profile'] + cr['h2_c_prod'] - cr['h2_c_cons'])
    ace_rate_sum = (cr['ace_c_prod'] - cr['ace_c_cons'])
    # Expected change over dt [mol/m3]
    ch4_test, o2_test, co2_test, h2_test, ace_test = ch4_rate_sum*dt, o2_rate_sum*dt, co2_rate_sum*dt, h2_rate_sum*dt, ace_rate_sum*dt

    # Record the indices where the 'test' would remove more material than is present in the reservoir
    ch4_truncatable = (ch4_test < 0.0) & (abs(ch4_test) > ca['ch4'])
    o2_truncatable = (o2_test < 0.0) & (abs(o2_test) > ca['o2'])
    co2_truncatable = (co2_test < 0.0) & (abs(co2_test) > ca['co2'])
    h2_truncatable = (h2_test < 0.0) & (abs(h2_test) > ca['h2'])
    ace_truncatable = (ace_test < 0.0) & (abs(ace_test) > ca['ace'])

    for i in range(0,len(ca['ch4'])):

        # If rates should be truncated in layer i (the condition above is met), then truncate them

        # CH4
        if ch4_truncatable[i]:
            lost, available, dr['q_diff_ch4_profile'], pr['q_plant_ch4_profile'], cr['ch4_c_cons'] = loss_vs_available(
                ca['ch4'],dr['q_diff_ch4_profile'],pr['q_plant_ch4_profile'],cr['ch4_c_cons'],cr['ch4_c_prod'],i,dt)
            cr['ch4_c_cons_mtox'][i] = (available * cr['ch4_c_cons_mtox'][i] / lost) / dt

        # O2
        if o2_truncatable[i]:
            lost, available, dr['q_diff_o2_profile'], pr['q_plant_o2_profile'], cr['o2_c_cons'] = loss_vs_available(
                ca['ch4'],dr['q_diff_o2_profile'],pr['q_plant_o2_profile'],cr['o2_c_cons'],cr['o2_c_prod'],i,dt)
            cr['o2_c_cons_aero'][i] = (available * cr['o2_c_cons_aero'][i] / lost) / dt
            cr['o2_c_cons_mtox'][i] = (available * cr['o2_c_cons_mtox'][i] / lost) / dt

        # CO2
        if co2_truncatable[i]:
            lost, available, dr['q_diff_co2_profile'], pr['q_plant_co2_profile'], cr['co2_c_cons'] = loss_vs_available(
                ca['ch4'],dr['q_diff_co2_profile'],pr['q_plant_co2_profile'],cr['co2_c_cons'],cr['co2_c_prod'],i,dt)
            cr['co2_c_cons_hoag'][i] = (available * cr['co2_c_cons_hoag'][i] / lost) / dt
            cr['co2_c_cons_h2mg'][i] = (available * cr['co2_c_cons_h2mg'][i] / lost) / dt

        # H2
        if h2_truncatable[i]:
            lost, available, dr['q_diff_h2_profile'], pr['q_plant_h2_profile'], cr['h2_c_cons'] = loss_vs_available(
                ca['ch4'],dr['q_diff_h2_profile'],pr['q_plant_h2_profile'],cr['h2_c_cons'],cr['h2_c_prod'],i,dt)
            cr['h2_c_cons_hoag'][i] = (available * cr['h2_c_cons_hoag'][i] / lost) / dt
            cr['h2_c_cons_h2mg'][i] = (available * cr['h2_c_cons_h2mg'][i] / lost) / dt

        # Ace
        if ace_truncatable[i]:
            lost, available, dr['q_diff_ace_profile'], pr['q_plant_ace_profile'], cr['ace_c_cons'] = loss_vs_available(
                ca['ch4'],dr['q_diff_ace_profile'],pr['q_plant_ace_profile'],cr['ace_c_cons'],cr['ace_c_prod'],i,dt)
            cr['ace_c_cons_acmg'][i] = (available * cr['ace_c_cons_acmg'][i] / lost) / dt

    return(cr, dr, pr)

def update_chemistry(dr, pr, cr, ca, dt, forcingt):

    ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = plants.equilibrium_concentration(20.0 + 0.0 * forcingt['temp_data'])
    ch4_transport_sum = np.minimum(dr['q_diff_ch4_profile']-pr['q_plant_ch4_profile'], (ceq_ch4-ca['ch4'])/params.dt)
    o2_transport_sum = np.minimum(dr['q_diff_o2_profile']-pr['q_plant_o2_profile'], (ceq_o2-ca['o2'])/params.dt)
    co2_transport_sum = np.minimum(dr['q_diff_co2_profile']-pr['q_plant_co2_profile'], (ceq_co2-ca['co2'])/params.dt)
    h2_transport_sum = np.minimum(dr['q_diff_h2_profile']-pr['q_plant_h2_profile'], (ceq_h2-ca['h2'])/params.dt)

    transport_sum = dr['q_diff_o2_profile']-pr['q_plant_o2_profile']

    #print(test1 > 0.0 and test1 > test2)

    ch4_chemistry_sum, o2_chemistry_sum, co2_chemistry_sum, h2_chemistry_sum, ace_chemistry_sum = \
        cr['ch4_c_prod'] - cr['ch4_c_cons_mtox'], - 1.0 * cr['o2_c_cons_aero'] - 2.0 * cr['ch4_c_cons_mtox'], \
        cr['co2_c_prod'] - cr['co2_c_cons'], cr['h2_c_prod'] - cr['h2_c_cons'], cr['ace_c_prod'] - cr['ace_c_cons']

    # Compute net chemistry rates as the sum of the truncated rates
    ch4_rate_sum = (dr['q_diff_ch4_profile'] - pr['q_plant_ch4_profile'] + cr['ch4_c_prod'] - cr['ch4_c_cons_mtox'])
    o2_rate_sum = (dr['q_diff_o2_profile'] - pr['q_plant_o2_profile'] - cr['o2_c_cons_aero'] - 2.0 * cr['ch4_c_cons_mtox'])
    co2_rate_sum = (dr['q_diff_co2_profile'] - pr['q_plant_co2_profile'] + cr['co2_c_prod'] - cr['co2_c_cons'])
    h2_rate_sum = (dr['q_diff_h2_profile'] - pr['q_plant_h2_profile'] + cr['h2_c_prod'] - cr['h2_c_cons'])
    ace_rate_sum = (cr['ace_c_prod'] - cr['ace_c_cons'])

    #ch4_rate_sum = ch4_transport_sum + ch4_chemistry_sum
    #o2_rate_sum = o2_transport_sum + o2_chemistry_sum
    #co2_rate_sum = co2_transport_sum + co2_chemistry_sum
    #h2_rate_sum = h2_transport_sum + h2_chemistry_sum
    #ace_rate_sum = ace_chemistry_sum

    #print('parts: ', dr['q_diff_o2_profile'][0], pr['q_plant_o2_profile'][0], cr['o2_c_cons_aero'][0], 2.0 * cr['ch4_c_cons_mtox'][0])
    #print('predict: ', ca['o2'][0] + dt * o2_rate_sum[0])

    # Modify the chemical quantities using the net rates
    #print('1, 2: ', params.dz * np.nansum(ca['ch4']), params.dz * np.nansum(ca['ch4']) + params.dz * np.nansum(dt * ch4_rate_sum))

#    print(ca['ch4'][0], dt * ch4_rate_sum[0], dt * (dr['q_diff_ch4_profile'][0] - pr['q_plant_ch4_profile'][0]))
    ca['ch4'] = ca['ch4'] + dt * ch4_rate_sum
    ca['o2'] = ca['o2'] + dt * o2_rate_sum
    ca['co2'] = ca['co2'] + dt * co2_rate_sum
    ca['h2'] = ca['h2'] + dt * h2_rate_sum
    ca['ace'] = ca['ace'] + dt * ace_rate_sum

    #if ca['ch4'][0] > ceq_ch4[0]:
    #    print(ca['ch4'][0], ceq_ch4[0])
    #    stop

    # Prevent quantities from going negative (should be redundant because of rate truncation)
    ca['ch4'][ca['ch4'] < 0.0], ca['ace'][ca['ace'] < 0.0], ca['co2'][ca['co2'] < 0.0], \
    ca['h2'][ca['h2'] < 0.0], ca['o2'][ca['o2'] < 0.0] = 1e-10, 1e-10, 1e-10, 1e-10, 1e-10

    return(ca)

def write_output(ga, gr, pr, pf, dr, df, ca, cr):

    # Write output for the last simulated year
    if years - 1 > (elapsed - doy) / 365 >= years - 2 and elapsed % dt == 0.0:
        print(elapsed % 365.0)
        if write_flag == True:
            test = init.write_output(test_datetime,
                                     ca['ch4'], ca['co2'], ca['o2'], ca['h2'], ca['ace'], ca['c'], ca['doc'],
                                     ga['acemethanogen_genes'], ga['h2_methanogen_genes'],
                                     ga['homoacetogen_genes'], ga['methanotroph_genes'],
                                     df['diff_flux_ch4'], df['diff_flux_co2'], df['diff_flux_h2'], df['diff_flux_o2'],
                                     pf['plant_flux_ch4'], pf['plant_flux_o2'], pf['plant_flux_co2'],
                                     pf['plant_flux_h2'], pf['plant_flux_ace'],
                                     temp_data_interp_mean, 0.01 * waterIce_data_interp_mean, ph_profile,
                                     ch4_c_prod, co2_c_prod, o2_c_prod_profile, h2_c_prod_profile,
                                     ace_c_prod_profile,
                                     ch4_c_cons_profile, co2_c_cons_profile, o2_c_cons_profile, h2_c_cons_profile,
                                     ace_c_cons_profile,
                                     ch4_c_prod_acmg_profile, ch4_c_prod_h2mg_profile,
                                     ace_c_prod_aesp_profile, ace_c_prod_hoag_profile,
                                     co2_c_prod_ansp_profile, co2_c_prod_aesp_profile, co2_c_prod_acmg_profile,
                                     h2_c_prod_ansp_profile,
                                     o2_c_prod_none_profile,
                                     ch4_c_cons_mtox_profile,
                                     ace_c_cons_acmg_profile,
                                     co2_c_cons_hoag_profile, co2_c_cons_h2mg_profile,
                                     h2_c_cons_hoag_profile, h2_c_cons_h2mg_profile,
                                     o2_c_cons_mtox_profile, o2_c_cons_aero_profile, o2_c_cons_aesp_profile,
                                     acemethanogen_gene_rate_profile, h2_methanogen_gene_rate_profile,
                                     homoacetogen_gene_rate_profile, methanotroph_gene_rate_profile,
                                     acemethanogen_gene_prod_profile, h2_methanogen_gene_prod_profile,
                                     homoacetogen_gene_prod_profile, methanotroph_gene_prod_profile,
                                     acemethanogen_gene_mort_profile, h2_methanogen_gene_mort_profile,
                                     homoacetogen_gene_mort_profile, methanotroph_gene_mort_profile,
                                     q_diff_ch4_profile, q_diff_co2_profile, q_diff_o2_profile, q_diff_h2_profile,
                                     q_diff_ace_profile,
                                     q_plant_ch4_profile, q_plant_co2_profile, q_plant_o2_profile, q_plant_h2_profile,
                                     q_plant_ace_profile)

            return()
