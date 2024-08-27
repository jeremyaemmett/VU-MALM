import os
import csv
import init
import random
import params
import subprocess
import numpy as np
import conversions
from os import listdir
from datetime import date
from datetime import datetime
from datetime import timedelta
from data import read_site_data
import matplotlib.pyplot as plt
from os.path import isfile, join
from matplotlib import ticker, cm


def rr2(min_val, max_val):

    rand_val = random.uniform(min_val, max_val)
    rand_formatted_val = float(format(rand_val, ".4f"))

    return rand_formatted_val


def format_axis2(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return()


def make_autotune_file(model, gen, params_dict):

    param_keys = list(params_dict.keys())
    fname = ''
    for p in range(0, len(param_keys)):
        param_val = str(params_dict[param_keys[p]])
        param_val = param_val.replace('.', 'p')
        if p < len(param_keys)-1: fname += param_val + '_'
        if p == len(param_keys)-1: fname += param_val
    fname = 'C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + params.site + '/' + params.autotune_name + '/' + str(gen) + '/' + fname + '_' + str(model) + '.csv'
    with open(fname, "a") as current_file:
        writer = csv.writer(current_file)
        writer.writerow(list(params_dict.values()))

    return(fname)


def datestrings2datetimes(datestrings):

    date_format, date_objects = '%Y-%m-%d %H:%M:%S', []
    for date_str in datestrings:

        #print(date_str, date_format)
        date_objects.append(datetime.strptime(date_str, date_format))

    return(date_objects)


def mutate(param_value, fraction):

    # Randomly adjust the parameter to be between <1 - fraction> and <1 + fraction> of its original value
    param_value = rr2((1.0 - fraction) * param_value, (1.0 + fraction) * param_value)

    return(param_value)


def find_keys_with_substring(dict, substr):

    data_dict_keys = np.array(list(dict.keys()))
    output_array = np.zeros_like(data_dict_keys, dtype=bool)
    val_indices = np.char.find(data_dict_keys, substr)
    output_array[np.where(val_indices != -1)] = True
    VvT_vals = data_dict_keys[output_array]

    return(VvT_vals)


def mid_datetime(datetime_list):

    a, b = datetime_list[0], datetime_list[-1]
    c = (b - a) / 2

    return(a + c)

# 4/3/2024
mode = params.autotune_mode
sims_per_generation = params.sims_per_generation
sims_per_first_generation = params.sims_per_first_generation
n_generations = params.n_generations

tunepath = 'C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/'

p_ranges = [[0.0001, 0.02], [0.3, 0.7], [0.01, 0.1], [0.3, 0.7], [1.0, 5.0],
            [1.0, 5.0], [1.0, 5.0], [0.01, 0.1], [0.01, 0.1], [0.3, 0.7], [20.0, 30.0], [0.1, 0.2], [5.0, 15.0],
            [0.01, 0.3]]
p_mutate = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

layer_thicknesses, depths = init.define_layers()
site_data = read_site_data(params.site, depths, layer_thicknesses)
flux_data_dict = find_keys_with_substring(site_data['site_fluxes'], '_VvT')
flux_time_dict = find_keys_with_substring(site_data['site_fluxes'], '_T')

print(flux_data_dict)

meas_mean_fluxes, meas_sdev_fluxes, meas_mid_dates = [], [], []
for f in range(0, len(flux_data_dict)):
    meas_mean_fluxes.append(np.mean(site_data['site_fluxes'][flux_data_dict[f]]))
    meas_sdev_fluxes.append(np.std(site_data['site_fluxes'][flux_data_dict[f]]))
    meas_mid_dates.append(mid_datetime(site_data['site_fluxes'][flux_time_dict[f]]))
meas_mean_fluxes, meas_sdev_fluxes, meas_mid_dates = np.array(meas_mean_fluxes), np.array(meas_sdev_fluxes), \
                                                     np.array(meas_mid_dates)

param_names = ['k_cpool', 'v_ace_cons_max', 'k_ace_prod_ch4', 'k_ch4_prod', 'doc_prod_q10', 'ace_prod_q10',
               'ch4_prod_q10', 'k_ch4_oxid_ch4', 'k_ch4_oxid_o2', 'v_ch4_oxid_max', 'a_m_a', 'k_aer', 'k_aer_doc',
               'k_aer_o2']
pavgs, pmins, pmaxs = [np.array([])] * len(param_names), [np.array([])] * len(param_names), [np.array([])] * len(
    param_names)
colors2 = cm.Set2(np.linspace(0, 1, len(param_names)))

if mode == 'plot':

    # chemical plot
    months_short = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    fig = plt.figure(figsize=(16.0, 8.0))
    gsc = fig.add_gridspec(2, 2, width_ratios=np.ones(2), height_ratios=np.ones(2),
                           left=0.1, right=0.9, bottom=0.05, top=0.95,
                           wspace=0.4, hspace=0.3)
    ax = fig.add_subplot(gsc[0, 0])
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.patch.set_facecolor('gainsboro')
    ax.patch.set_alpha(0.5)
    ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 12, 1)])
    ax.set_xticklabels(months_short, fontsize = 12.0)
    ax.set_ylabel('mol/m$^2$/day')
    ax.set_title('Seasonal CH$_4$ Flux Evolution')
    plt.xticks(fontsize=12.0)
    plt.yticks(fontsize=12.0)
    format_axis2(ax)
    ax2 = fig.add_subplot(gsc[1, 0])
    ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.patch.set_facecolor('gainsboro')
    ax2.patch.set_alpha(0.5)
    ax2.set_xlim([-0.5, n_generations - 0.5])
    #ax2.set_yscale('log')
    ax2.set_xlabel('Generation #', fontsize = 12.0)
    ax2.set_ylabel('Parameter Values', fontsize = 12.0)
    ax2.set_title('Parameter Evolution', fontsize = 12.0)
    plt.xticks(fontsize=12.0)
    plt.yticks(fontsize=12.0)
    format_axis2(ax2)
    ax3 = fig.add_subplot(gsc[0, 1])
    ax3.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.patch.set_facecolor('gainsboro')
    ax3.patch.set_alpha(0.5)
    ax3.set_ylabel('mol/m$^2$/day', fontsize = 12.0)
    ax3.set_title('CH$_4$ Flux Evolution (on date)', fontsize = 12.0)
    plt.xticks(fontsize=12.0)
    plt.yticks(fontsize=12.0)
    ax3.set_xlim([-1.0, n_generations + 1])
    ax3.set_xlabel('Generation #', fontsize = 12.0)
    format_axis2(ax3)

    directories = next(os.walk(tunepath + params.site + '/' + params.autotune_name))[1]
    #colors = ['red', 'orange', 'green', 'blue', 'purple', 'red', 'orange', 'green', 'blue', 'purple']
    colors = cm.rainbow(np.linspace(0, 1, len(directories)))

    meas_colors = ['red', 'green', 'blue']
    for m in range(0, len(meas_mid_dates)):
        ax.plot([meas_mid_dates[m] + timedelta(days = 8.0)], [meas_mean_fluxes[m]], marker = '<', color = meas_colors[m], markersize = 15.0, alpha = 0.5)
        ax.plot([meas_mid_dates[m], meas_mid_dates[m]], [meas_mean_fluxes[m] - (meas_sdev_fluxes[m])/8.0, meas_mean_fluxes[m] + (meas_sdev_fluxes[m])/8.0], linestyle = '-', color=meas_colors[m], linewidth = 10.0, alpha = 0.3)
        ax3.plot([n_generations - 0.1 + 0.6], [meas_mean_fluxes[m]], marker = '<', markersize = 15.0, color=meas_colors[m], alpha = 0.5)
        lower, upper = meas_mean_fluxes[m] - meas_sdev_fluxes[m] / 8.0, meas_mean_fluxes[m] + meas_sdev_fluxes[m] / 8.0
        ax3.plot([n_generations, n_generations], [lower, upper], linestyle = '-', linewidth = 10.0, color = meas_colors[m], alpha = 0.3)
        #ax.text(meas_mid_dates[m], 1.15 * upper, str(meas_mid_dates[m].day) + '-' + str(meas_mid_dates[m].month) + '-' + str(meas_mid_dates[m].year),
        #         ha='center', va='center', fontsize=10, weight='bold', color=meas_colors[m], rotation = 25,
        #         bbox=dict(facecolor='gainsboro', alpha=0.0, boxstyle='round', edgecolor = 'gainsboro'))

    for d in range(0, len(directories)):
        directory_path = tunepath + params.site + '/' + params.autotune_name + '/' + directories[d] + '/'
        files = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
        fluxes_stack, fluxes_date_stack, params_stack = [], [], []

        flux_avgs, flux_mins, flux_maxs = [[],[]], [[],[]], [[],[]]
        for file in files:
            with open(directory_path + file, newline='') as csvfile: data = list(csv.reader(csvfile))

            file_list = []
            meas_colors = ['red', 'green', 'blue']
            for m in range(0, len(meas_mid_dates)):

                dates = np.array(datestrings2datetimes(np.array(data[2:])[:, 1]))
                fluxes = [-float(x) for x in np.array(data[2:])[:, 2]]

                n_records = int(365.0 / params.write_dt) - 1
                dates = dates[0:n_records]
                fluxes = fluxes[0:n_records]

                fluxes_smooth = np.convolve(fluxes, np.ones(28) / 28, mode='same')
                test_date, test_range = meas_mid_dates[m], timedelta(days=5)
                test_date_low, test_date_high = test_date - test_range, test_date + test_range
                test_date_low_idx = np.where(abs(dates - test_date_low) == np.min(abs(dates - test_date_low)))[0][0]
                test_date_high_idx = np.where(abs(dates - test_date_high) == np.min(abs(dates - test_date_high)))[0][0]
                fluxes_date = np.array(fluxes_smooth[test_date_low_idx:test_date_high_idx])

                flux_avgs[m].append(fluxes_date.mean(axis=0))
                flux_mins[m].append(fluxes_date.min(axis=0))
                flux_maxs[m].append(fluxes_date.max(axis=0))

                file_list.append(fluxes_date)
            fluxes_stack.append(np.array(fluxes))
            print(file, len(dates), len(fluxes))
            params_stack.append([float(v) for v in data[0]])
            #ax.plot(dates, np.convolve(fluxes, np.ones(28) / 28, mode='same'), color = 'lightgray', alpha = 0.3)
        for m in range(0, len(meas_mid_dates)):
            gen_avg, gen_min, gen_max = np.mean(flux_avgs[m]), np.min(flux_mins[m]), np.max(flux_maxs[m])
            ax3.plot([d], [gen_avg], marker='o', markersize=7.5, color=meas_colors[m])
            ax3.plot([d, d], [gen_min, gen_max], linestyle='-', color=meas_colors[m])
            ax3.plot([d-0.2, d+0.2], [gen_min, gen_min], linestyle='-', color=meas_colors[m])
            ax3.plot([d-0.2, d+0.2], [gen_max, gen_max], linestyle='-', color=meas_colors[m])
        print(len(fluxes_stack))
        fluxes_stack = np.array(fluxes_stack)
        print(fluxes_stack.shape)
        fluxes_date_stack = np.array(fluxes_date_stack) # n_files, n_meas, n_times
        print(len(fluxes_stack[1]))
        params_stack = np.array(params_stack)
        avg_fluxes, min_fluxes, max_fluxes = fluxes_stack.mean(axis = 0), fluxes_stack.min(axis = 0), fluxes_stack.max(axis = 0)
        print(len(avg_fluxes))
        avg_params, min_params, max_params = params_stack.mean(axis=0), params_stack.min(axis=0), params_stack.max(axis=0)

        for p in range(0, len(param_names)):
            pavgs[p] = np.append(pavgs[p], avg_params[p])
            pmins[p] = np.append(pmins[p], min_params[p])
            pmaxs[p] = np.append(pmaxs[p], max_params[p])

        flux_tseries = np.convolve(avg_fluxes, np.ones(12) / 12, mode='same')
        #flux_tseries = avg_fluxes
        flux_tseries[flux_tseries > 0.03] = 0.0
        #flux_tseries[flux_tseries > (np.mean(flux_tseries) + 2.0 * np.std(flux_tseries))] = 0.0
        ax.plot(dates, flux_tseries, marker=None, c=colors[d], linestyle = '-', linewidth = 1.0, alpha = 0.5)
        if d == len(directories) - 1: ax.plot(dates, flux_tseries, marker=None, c=colors[d], linestyle='-', linewidth=3.0, alpha=0.8)
        lower, upper = min_fluxes, max_fluxes
        min_fluxes[min_fluxes > 0.03] = 0.0
        max_fluxes[max_fluxes > 0.03] = 0.0
        #if d == len(directories)-1: ax.fill_between(dates, lower, upper, alpha=0.2, color=colors[d])
    for p in range(0, np.shape(pavgs)[0]):
        print(p, pavgs[p])
        ax2.plot(range(len(directories)), pavgs[p], marker=None, linewidth = 3.0, alpha = 0.8, c=colors2[p])
        lower, upper = pmins[p], pmaxs[p]
        ax2.fill_between(range(len(directories)), lower, upper, alpha = 0.3, color=colors2[p])
    p_lbl_string, p_avg_string = '', ''
    for p in range(0, np.shape(pavgs)[0]):
        ecolor = 'Green'
        if pmins[p][-1] < p_ranges[p][0] or pmaxs[p][-1] > p_ranges[p][1]: ecolor = 'Red'
        txt = str(format(p_ranges[p][0], ".4f")) + ' | ' + str(format(pmins[p][-1], ".4f")) + ' - ' + \
              str(format(pavgs[p][-1], ".4f")) +   ' - ' + str(format(pmaxs[p][-1], ".4f")) + ' | ' + \
              str(format(p_ranges[p][1], ".4f"))
        p_avg_string += ', ' + str(format(pavgs[p][-1], ".4f"))
        ax2.text(1.95 * ax2.get_xlim()[1], ax2.get_ylim()[0] + (p + 0.5) * (ax2.get_ylim()[1]-ax2.get_ylim()[0])/10.0, txt,
                 ha='center', va='center', fontsize=10, weight='bold', color='black',
                 bbox=dict(facecolor=colors2[p], alpha=0.5, boxstyle='round', edgecolor = colors2[p]))
        txt = param_names[p]
        p_lbl_string += ', ' + txt
        ax2.text(1.3 * ax2.get_xlim()[1], ax2.get_ylim()[0] + (p + 0.5) * (ax2.get_ylim()[1]-ax2.get_ylim()[0])/10.0, txt,
                 ha='center', va='center', fontsize=10, weight='bold', color='black',
                 bbox=dict(facecolor=ecolor, alpha=0.2, boxstyle='round', edgecolor = ecolor))
    average_parameter_string = p_lbl_string[2:] + ' = ' + p_avg_string[2:]
    with open(tunepath + params.site + '/' + params.autotune_name + '/avg_param_vals.txt', "w") as current_file:
        current_file.write(average_parameter_string)
    ax3.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])

if mode == 'run':

    # Make a tune folder
    if not os.path.exists(tunepath + params.site + '/' + params.autotune_name):
        os.mkdir(tunepath + params.site + '/' + params.autotune_name)

    gen = 0
    for g in range(0, n_generations):

        genpath, prevgenpath = tunepath + params.site + '/' + params.autotune_name + '/' + str(gen) + '/', \
                               tunepath + params.site + '/' + params.autotune_name + '/' + str(gen - 1) + '/'

        # Make a new generation folder
        if not os.path.exists(genpath):
            os.mkdir(genpath)

        if gen == 0: # Generation zero

            for n in range(0, sims_per_first_generation):

                # New model parameter array
                new_params = np.zeros(len(param_names))

                # Randomize parameters within specified ranges
                new_params[0], new_params[1], new_params[2], new_params[3], new_params[4], \
                new_params[5], new_params[6], new_params[7], new_params[8], new_params[9],\
                new_params[10], new_params[11], new_params[12], new_params[13] = \
                rr2(p_ranges[0][0], p_ranges[0][1]), rr2(p_ranges[1][0], p_ranges[1][1]), \
                rr2(p_ranges[2][0], p_ranges[2][1]), rr2(p_ranges[3][0], p_ranges[3][1]), \
                rr2(p_ranges[4][0], p_ranges[4][1]), rr2(p_ranges[5][0], p_ranges[5][1]), \
                rr2(p_ranges[6][0], p_ranges[6][1]), 0.05, \
                0.02, 0.5, rr2(p_ranges[10][0], p_ranges[10][1]), rr2(p_ranges[11][0], p_ranges[11][1]), \
                rr2(p_ranges[12][0], p_ranges[12][1]), rr2(p_ranges[13][0], p_ranges[13][1])

                # Fill the params_dict dictionary
                params_dict = {}
                params_dict['k_cpool'] = new_params[0]
                params_dict['v_ace_cons_max'] = new_params[1]
                params_dict['k_ace_prod_ch4'] = new_params[2]
                params_dict['k_ch4_prod'] = new_params[3]
                params_dict['doc_prod_q10'] = new_params[4]
                params_dict['ace_prod_q10'] = new_params[5]
                params_dict['ch4_prod_q10'] = new_params[6]
                params_dict['k_ch4_oxid_ch4'] = new_params[7]
                params_dict['k_ch4_oxid_o2'] = new_params[8]
                params_dict['v_ch4_oxid_max'] = new_params[9]
                params_dict['a_m_a'] = new_params[10]
                params_dict['k_aer'] = new_params[11]
                params_dict['k_aer_doc'] = new_params[12]
                params_dict['k_aer_o2'] = new_params[13]

                # Make a file for the simulation, with parameter values in the header line, in a generation 0 folder
                file = make_autotune_file(n, gen, params_dict)

        if gen > 0:

            # List the previous generation's output files
            files = [f for f in listdir(prevgenpath) if isfile(join(prevgenpath, f))]

            # Probability distribution for model selection
            bin = 5.0
            bins = (bin * (np.arange(len(files))+1))[::-1]
            ranges = np.cumsum(bins)

            # Read the flux time series in each file, record the 10-day average flux centered on the test date
            diffs_list = []
            for m in range(0, len(meas_mid_dates)):
                avg_fluxes = []
                for file in files:
                    # meas_mid_dates
                    # meas_mean_fluxes
                    with open(prevgenpath + file, newline='') as csvfile:
                        data = list(csv.reader(csvfile))
                    dates = np.array(datestrings2datetimes(np.array(data[2:])[:,1]))
                    #print(file)
                    #print(dates)
                    fluxes = [-float(x) for x in np.array(data[2:])[:,2]]
                    test_date, test_range = meas_mid_dates[m], timedelta(days = 5)
                    test_date_low, test_date_high = test_date - test_range, test_date + test_range
                    test_date_low_idx = np.where(abs(dates - test_date_low) == np.min(abs(dates - test_date_low)))[0][0]
                    test_date_high_idx = np.where(abs(dates - test_date_high) == np.min(abs(dates - test_date_high)))[0][0]
                    avg_fluxes.append(np.nanmean(fluxes[test_date_low_idx:test_date_high_idx]))

                avg_fluxes = np.array(avg_fluxes)

                # Rank the models based on the closeness of their max fluxes to the target flux (closest to furthest)
                diffs, files = abs(avg_fluxes - meas_mean_fluxes[m]), np.array(files)
                diffs_list.append(diffs)

            mean_diffs = np.array(diffs_list).mean(axis = 0)

            sorted_idx = np.argsort(mean_diffs)
            sorted_mean_diffs = mean_diffs[sorted_idx]
            sorted_files = files[sorted_idx]

            for n in range(0, sims_per_generation):

                # New model parameter array
                new_params = np.zeros(len(param_names))
                #print('sim: ', n)
                while np.nanmin(new_params) == 0.0: # Until all of the parameters have been selected...
                    #print('test')
                    # Randomly select which parameter to copy
                    rand_param_idx = np.random.randint(len(new_params))
                    #print(rand_param_idx)

                    # Randomly select a model from which to copy the parameter, biased by the probability distribution
                    test = np.random.randint(np.nanmax(ranges))
                    idx = np.where(ranges > test)[0][0]
                    idx = np.random.randint(2)

                    # Read and copy the parameter
                    with open(prevgenpath + sorted_files[idx], newline='') as csvfile:
                        data = list(csv.reader(csvfile))
                    param_value = float(data[0][rand_param_idx])

                    # Mutate the parameter, but constrain the new value to its allowable range
                    if random.randint(1,5) == 5 and p_mutate[rand_param_idx] == 1:
                        try_param_value = mutate(param_value, 0.50)
                        while not (p_ranges[rand_param_idx][0] < try_param_value < p_ranges[rand_param_idx][1]):
                            try_param_value = mutate(param_value, 0.50)
                            #print(p_ranges[rand_param_idx][0], try_param_value, p_ranges[rand_param_idx][1])
                        param_value = try_param_value

                    new_params[rand_param_idx] = param_value
                    #print(new_params)

                    # Fill the params_dict dictionary until the new_params array is full. Many overwrites should occur.
                    params_dict = {}
                    params_dict['k_cpool'] = new_params[0]
                    params_dict['v_ace_cons_max'] = new_params[1]
                    params_dict['k_ace_prod_ch4'] = new_params[2]
                    params_dict['k_ch4_prod'] = new_params[3]
                    params_dict['doc_prod_q10'] = new_params[4]
                    params_dict['ace_prod_q10'] = new_params[5]
                    params_dict['ch4_prod_q10'] = new_params[6]
                    params_dict['k_ch4_oxid_ch4'] = new_params[7]
                    params_dict['k_ch4_oxid_o2'] = new_params[8]
                    params_dict['v_ch4_oxid_max'] = new_params[9]
                    params_dict['a_m_a'] = new_params[10]
                    params_dict['k_aer'] = new_params[11]
                    params_dict['k_aer_doc'] = new_params[12]
                    params_dict['k_aer_o2'] = new_params[13]

                # Make a file for the simulation, with parameter values in the header line, in the appropriate gen folder
                file = make_autotune_file(n, gen, params_dict)

        if gen == 0:
            #sims_per_first_generation = 1
            for n in range(0, sims_per_first_generation): subprocess.run(['python', 'main.py'])
        if gen > 0:
            for n in range(0, sims_per_generation): subprocess.run(['python', 'main.py'])

        gen += 1

plt.savefig('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + params.site + '/' + params.autotune_name + '/evolution.png',bbox_inches='tight')

