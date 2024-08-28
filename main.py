import init
import data
import plants
import params
import output
import output2
import newstep
import forcing
import pathways
import microbes
import cryogrid
import openpyxl
import transport
import diffusion
import conversions
import numpy as np
from copy import copy
from datetime import datetime
from datetime import timedelta
# 11/17/2023

# Starting date/time
test_datetime0 = params.test_datetime

# Vertical grid
layer_thicknesses, depths = init.define_layers()

#if params.standardize_flag:
#    standardize.standardize()

# Read and interpolate standardized data to the model grid.
site_data = data.read_site_data(params.site, depths, layer_thicknesses)

if params.plot_data_flag: # Optionally plot
    print('Plotting site data')
    data.plot_VvT(site_data, depths)
    data.plot_VvZ(site_data, depths)

# Find fixed-profile data closest to the start date
#kcpool_date = init.most_complete_date(site_data['site_doc'], '_docmolm3')
#cpool_date = init.most_complete_date(site_data['site_comp'], '_socmolm2')
roots_date = init.most_complete_date(site_data['site_roots'], '_roots')

# Assimilate forcing data from CryoGrid
print('Assimilating forcing data')
forcing_profiles, equilibrium_profiles = forcing.all_forcing()

print('Initializing variables') # Initialize dictionaries and storage arrays
ga, gr = init.define_microbes(depths)
pr, pf = init.define_plants(depths)
dr, df = init.define_diffusion(depths)
ca, cr, crd2 = init.define_chemistry(depths, site_data)
ph_profile, temp_profile = init.define_physical(depths)

# Prepare the output file headers
if params.write_flag:
    output_files = output2.prepare_output_files(layer_thicknesses,depths,ca,ga,df,pf,cr,gr,dr,pr,crd2,
                                                forcing_profiles, equilibrium_profiles)
#if params.autotune_flag:
#    if params.params_dict['gen'] == 0: autotune_file = output2.make_autotune_file(0, params.params_dict)

# Computer time of run start
t0 = datetime.now()

# Main computational time loop
print('Begin')
#
elapsed = 0.0 # Seconds elapsed
dt = params.dt # Initial time step
write_count = 0 # Write interval counter
year_count = 0 # Year counter
t = 0 # Time step counter
test_datetime = test_datetime0
doy = float(test_datetime.timetuple().tm_yday)+float(test_datetime.timetuple().tm_hour)/24.0 # Initial day-of-year
# Variables for model spin-up and conservation checks
year_ch4 = []
integrated_ch4_net, integrated_ch4_prod, integrated_ch4_cons, integrated_ch4_flux = 0.0, 0.0, 0.0, 0.0
#
while year_count < params.years:

    # Fractional day of year
    doy = float(test_datetime.timetuple().tm_yday)+float(test_datetime.timetuple().tm_hour)/24.0

    # Retrieve forcing profiles for the day-of-year [1 x n_layers]
    forcing_t = forcing.profile_t(forcing_profiles, doy)
    #forcing_t['satpct_data'][:] = 100.0
    eq = forcing.profile_t(equilibrium_profiles, doy)

    # DOC is derived from measured DOC/SOC ratio, measured SOC, and temperature & moisture forcing profiles
    ca['doc'] = params.k_cpool * \
                (site_data['site_comp'][params.site + '_cpoolmolm2_V'] / (depths[1]-depths[0])) * \
                pathways.temp_facs(forcing_t['temp_data'], params.doc_prod_q10)[0] * 0.01 * forcing_t['satpct_data']

    # Microbial chemistry
    #crd2['activity_level'] = newstep.activity_level(forcing_t['temp_data'], crd2['activity_level'])
    if params.microbe_flag and np.max(forcing_t['temp_data']) > 0.0:
        ga, gr, cr, crd2 = microbes.microbes2(dt, forcing_t, layer_thicknesses, ga, ph_profile, cr, gr, ca, crd2)

    # Compute diffusive transport rates of gases
    df['diff_flux_ch4'], df['diff_flux_o2'], df['diff_flux_co2'], df['diff_flux_h2'] = 0.0, 0.0, 0.0, 0.0
    if params.diffusion_flag and len(np.where(forcing_t['temp_data'] > 0.0)[0]) > 1:
        dr, df = diffusion.diffusion_cranknicolson2(dt, forcing_t['temp_data'], forcing_t['satpct_data'],
                                                    forcing_t['effdiff_data_ch4'], forcing_t['effdiff_data_o2'],
                                                    forcing_t['effdiff_data_co2'], forcing_t['effdiff_data_h2'],
                                                    dr, df, ca)

    #print(' ')
    #if t == 100: stop

    # Compute plant root transport rates of gases
    pf['plant_flux_ch4'], pf['plant_flux_co2'], pf['plant_flux_o2'], pf['plant_flux_h2']  = 0.0, 0.0, 0.0, 0.0
    if params.plant_flag and len(np.where(forcing_t['temp_data'] > 0.0)[0]) > 1:
        pr, pf = plants.plant_transport_rate2(dt, layer_thicknesses, depths, forcing_t['temp_data'],
                 site_data['site_roots'][params.site + '_' + roots_date + '_roots_VvZ'], pr, pf, ca,
                                              forcing_t['satpct_data'])

    # Truncate computed rates wherever net removal exceeds availability
    #cr, dr, pr = newstep.truncate_rates2(dt, cr, dr, pr, ca)

    # Modify the time step for stability
    dt = newstep.adaptive_timestep2(ca, cr, dr, pr, forcing_t, elapsed, dt)

    # Update the model chemistry in unfrozen layers
    if np.max(forcing_t['temp_data']) > 0.0: ca = newstep.update_chemistry(dr, pr, cr, ca, dt, forcing_t)

    # Check if the variables are writable
    write_timestep = elapsed % params.write_dt == 0
    write_year = params.years > year_count >= params.years - 1
    print_details = test_datetime.month == 8 and test_datetime.day == 15 and test_datetime.hour == 0 and \
                    test_datetime.minute == 0 and test_datetime.second == 0 and test_datetime.microsecond == 0

    # After Dec 31, loop the calendar date back to Jan 1 and the day of year to 1
    if t > 0:
        doy = float(test_datetime.timetuple().tm_yday)+float(test_datetime.timetuple().tm_hour)/24.0
        if print_details: # or test_datetime == datetime(2022, 6, 1, 0, 0, 0):
            gap = '                              '
            idx_00cm = 0
            idx_15cm = np.where(abs(depths - 0.15) == np.min(abs(depths - 0.15)))[0][0]
            idx_30cm = np.where(abs(depths - 0.30) == np.min(abs(depths - 0.30)))[0][0]
            idx_45cm = np.where(abs(depths - 0.45) == np.min(abs(depths - 0.45)))[0][0]
            depth_indices = [idx_00cm, idx_15cm, idx_30cm, idx_45cm]
            print(cr['ch4_c_rate'][0])
            print('Year #: ', year_count, test_datetime, eq['ch4'][0], eq['o2'][0], eq['co2'][0], eq['h2'][0])
            print(gap, '*'+str(df['f_diff_ch4']-pf['plant_flux_ch4'])+'*', df['f_diff_o2']-pf['plant_flux_o2'], df['f_diff_co2']-pf['plant_flux_co2'], df['f_diff_h2']-pf['plant_flux_h2'])
            print(ca['ch4'][0], eq['ch4'][0], df['f_diff_ch4'])
            print(gap, pf['plant_flux_ch4'], pf['plant_flux_o2'], pf['plant_flux_co2'], pf['plant_flux_h2'])
            print(gap, df['f_diff_ch4'], df['f_diff_o2'], df['f_diff_co2'], df['f_diff_h2'])
            print(gap + '-')
            for di in depth_indices: print(gap, ca['ch4'][di], ca['o2'][di], ca['co2'][di], ca['h2'][di], ca['doc'][di])
            print(gap + '-')
            for di in depth_indices: print(gap, "{:.7e}".format(ga['acemethanogen_genes'][di]),
                                                "{:.7e}".format(ga['methanotroph_genes'][di]),
                                                "{:.7e}".format(ga['homoacetogen_genes'][di]),
                                                "{:.7e}".format(ga['h2_methanogen_genes'][di]),
                                                "{:.7e}".format(ga['totmethanogen_genes'][di]))
            year_ch4.append(params.dz * np.nansum(ca['ch4']))
            if len(year_ch4) > 1:
                print('Year change CH4 [%]: ', 100 * (year_ch4[-1] - year_ch4[-2]) / year_ch4[-2])
                print('Net CH4 Production [mol/m2/year]: ', integrated_ch4_net)
                print('Change in CH4: ', year_ch4[-1] - year_ch4[-2])
                print('CH4 Diffusive Flux [mol/m2/year]: ', integrated_ch4_flux)
                print('Net + Flux [mol/m2/year]        : ', integrated_ch4_net + integrated_ch4_flux)
            integrated_ch4_net, integrated_ch4_prod, integrated_ch4_cons, integrated_ch4_flux = 0.0, 0.0, 0.0, 0.0
            #stop
        if write_timestep and test_datetime.second > 0: print('wt: ########################', test_datetime.second)
        # Write output
        if write_year and write_timestep and params.autotune_flag:
            print(df['f_diff_ch4']-pf['plant_flux_ch4'], write_count, test_datetime)
            output2.write_autotune_data(params.autotune_file, -1.0 * (df['f_diff_ch4']-pf['plant_flux_ch4']), -1.0 * df['f_diff_ch4'], pf['plant_flux_ch4'], write_count, test_datetime)
        if write_year and write_timestep and params.write_flag:
            test = output2.write_output(output_files, test_datetime,
                                        ca, ga, df, pf,
                                        forcing_t, eq, ph_profile,
                                        cr, gr, dr, pr, crd2,
                                        depths, write_count)
        write_count += 1

    if test_datetime.day in [1, 15] and test_datetime.hour == 0 and test_datetime.minute == 0 and test_datetime.second == 0:
        print(year_count, test_datetime, '*'+str(df['f_diff_ch4']-pf['plant_flux_ch4'])+'*', df['f_diff_ch4'], pf['plant_flux_ch4'], '| dt = ', str(dt*60*24) + ' min', '| diff_dt = ', str(dt*60*24 / params.diff_n_dt) + ' min')
        #print(ca['ace'][0:3], cr['ch4_c_prod_acmg'][0:3], cr['ch4_c_prod_h2mg'][0:3], cr['ch4_c_cons'][0:3])

    # Add dt to the elapsed time
    elapsed = elapsed + dt
    # Adjust elapsed to fall on a write_interval, if it negligibly differs from a write interval
    if abs(elapsed % params.write_dt - params.write_dt) < 1e-10: elapsed = round(elapsed, 2)
    if (test_datetime0 + timedelta(days=elapsed)).year > test_datetime0.year: elapsed, year_count = elapsed - 365.0, year_count + 1
    # Add elapsed to the starting datetime
    test_datetime = test_datetime0 + timedelta(days=elapsed)

    t += 1

# Compute the run duration
tf = datetime.now()
print(tf - t0)
