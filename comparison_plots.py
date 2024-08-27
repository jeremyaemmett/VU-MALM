from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyrolite.util.plot.style import ternary_color
from matplotlib.collections import PatchCollection
from pyrolite.util.synthetic import normal_frame
from matplotlib.ticker import LogFormatter
from IPython.display import display, Latex
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib import ticker, cm
from os.path import isfile, join
from data import read_site_data
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from matplotlib import cm
from datetime import date
import matplotlib as mpl
from os import listdir
from copy import copy
from numpy import ma
import pandas as pd
import numpy as np
import matplotlib
import transport
import diffusion
import pathways
import params
import math
import init
import csv
import os


def format_axis(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(' ')

    return()


def read_1d_output(file):

    with open(file, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    # Read unit, time data (x), and 1D values (v)
    unit = data[0][0]
    x, v = [], []
    for i in range(0, len(data[2:])):
        x.append(data[2:][i][1])
        v.append(-float(data[2:][i][2]))
    x, v = np.array(x), np.array(v)
    t = datestrings2datetimes(x)

    return(unit, t, v)


def read_2d_output(file, v_temp):

    with open(file, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    unit = np.array(data)[0][0]
    #print(file, np.shape(data))
    x, y, v = np.array(data)[2:-1, 1], np.array(data)[1, 2:-1], np.transpose(np.array(data)[2:-1, 2:-1])
    y, v = y.astype(float), v.astype(float)
    v[v_temp < 0.0] = np.nan
    t = datestrings2datetimes(x)

    return(unit, t, y, v)


def datestrings2datetimes(datestrings):

    date_format, date_objects = '%Y-%m-%d %H:%M:%S', []
    for date_str in datestrings: date_objects.append(datetime.strptime(date_str, date_format))

    return(date_objects)


def closest_time_index(model_times, meas_time):

    time_differences = abs(np.array([(i - meas_time) for i in np.array(model_times)]))
    closest_index = np.where(time_differences == np.min(time_differences))[0][0]

    return(closest_index)


def find_keys_with_substring(dict, substr):

    data_dict_keys = np.array(list(dict.keys()))
    output_array = np.zeros_like(data_dict_keys, dtype=bool)
    val_indices = np.char.find(data_dict_keys, substr)
    output_array[np.where(val_indices != -1)] = True
    VvT_vals = data_dict_keys[output_array]

    return(VvT_vals)


def find_elements_with_substring(list, substr):

    list = np.array(list)
    output_array = np.zeros_like(list, dtype=bool)
    val_indices = np.char.find(list, substr)
    output_array[np.where(val_indices != -1)] = True
    elements = np.sort(list[output_array])

    return(elements)


def datestrings2datetimes(datestrings):

    date_format, date_objects = '%Y-%m-%d %H:%M:%S', []
    for date_str in datestrings: date_objects.append(datetime.strptime(date_str, date_format))

    return(date_objects)


def rmse(predictions, targets):

    return np.sqrt(np.nanmean((predictions - targets) ** 2))


def compare_slice(axis, site_data_key, data_unit, model_output_path, title, xlabel, ylabel, data_factor):

    format_axis(axis)

    axis.set_ylim([max(depths), min(depths)])
    axis.set_title(title, weight='bold', fontsize = 16)
    axis.set_xlabel(xlabel, fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14)
    axis.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Read 1D data profiles
    key_list = find_keys_with_substring(site_data[site_data_key], '_VvZ')
    plotable_data_keys = find_elements_with_substring(key_list, data_unit)

    datetimes = []
    for p in plotable_data_keys:
        condensed_date = p.split('_')[-3]
        day, month, year = condensed_date[0:2], condensed_date[2:4], condensed_date[4:]
        datetimes.append(datetime(int(year), int(month), int(day)))

    # Read 2D model output
    o2_units, o2_ts, o2_ys, o2_vs = [], [], [], []
    for p in range(0, len(datetimes)):
        o2_unit, o2_t, o2_y, o2_v = read_2d_output(model_output_path, v_temp)
        o2_units.append(o2_unit)
        o2_ts.append(o2_t)
        o2_ys.append(o2_y)
        o2_vs.append(o2_v)
    # PLot sliced model output at each datetime
    colors = ['Red', 'Green', 'Blue']
    ratios = [0.2, 0.5, 0.8]
    widths = [6.0, 4.0, 2.0]
    mins, maxs = [], []
    for p in range(0, len(datetimes)):
        datetime2 = datetimes[p]
        idx = closest_time_index(np.array(o2_ts[p]), datetime2)
        meas_data = data_factor * site_data[site_data_key][plotable_data_keys[p]]
        meas_data = meas_data[0:-1]
        depths2 = depths[0:-1]
        model_data = o2_vs[p][:, idx]
        meas_data[np.isnan(model_data)] = np.nan
        mins.append(np.nanmin([meas_data, model_data]))
        maxs.append(np.nanmax([meas_data, model_data]))
        axis.set_xlim(np.nanmin(mins), np.nanmax(maxs))
        absdiffs_data, diffs_data = abs(meas_data - model_data), meas_data - model_data
        root_mean_square = rmse(model_data, meas_data)
        axis.text(np.nanmin(mins) + ratios[p]*(np.nanmax(maxs) - np.nanmin(mins)), 0.55, \
                    str(datetime2).split(' ')[0] + '\nRMSE: ' + '{:.2f}'.format(root_mean_square), \
                    ha='center', va='center', fontsize=12, weight = 'bold', color = colors[p], \
                    bbox=dict(boxstyle="round", fc='white', ec=colors[p], lw=2, alpha = 0.8))
        axis.plot(meas_data, depths2, color=colors[p], linewidth=widths[p], linestyle='--', alpha = 0.5)
        axis.plot(model_data, depths2, color=colors[p], linewidth=widths[p], alpha = 0.5)
        for f in range(1,len(depths2)):
            norm = 0.5*abs(meas_data[f-1] - model_data[f-1])/np.nanmax(absdiffs_data)
            #if p == 0: axis.fill_betweenx([depths2[f-1], depths[f]], [model_data[f-1], model_data[f]], [meas_data[f-1], meas_data[f]], color=cm.Reds(norm), alpha = 0.5, edgecolor=None)
            #if p == 1: axis.fill_betweenx([depths2[f-1], depths[f]], [model_data[f-1], model_data[f]], [meas_data[f-1], meas_data[f]], color=cm.Greens(norm), alpha = 0.5, edgecolor=None)
            #if p == 2: axis.fill_betweenx([depths2[f - 1], depths[f]], [model_data[f - 1], model_data[f]], [meas_data[f - 1], meas_data[f]], color=cm.Blues(norm), alpha = 0.5, edgecolor=None)
            if p == 0: axis.fill_betweenx([depths2[f-1], depths[f]], [model_data[f-1], model_data[f]], [meas_data[f-1], meas_data[f]], color='red', alpha = 0.3, edgecolor=None)
            if p == 1: axis.fill_betweenx([depths2[f-1], depths[f]], [model_data[f-1], model_data[f]], [meas_data[f-1], meas_data[f]], color='green', alpha = 0.3, edgecolor=None)
            if p == 2: axis.fill_betweenx([depths2[f - 1], depths[f]], [model_data[f - 1], model_data[f]], [meas_data[f - 1], meas_data[f]], color='blue', alpha = 0.3, edgecolor=None)

#unit, t, v = read_1d_output(mypath + '/' + cdir)

# Vertical grid
layer_thicknesses, depths = init.define_layers()

# Get temperature profile data to mask values where T < 0C
with open('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/en/temp_data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
x_temp, y_temp, v_temp = np.array(data)[2:-1,1], np.array(data)[1,2:-1], np.transpose(np.array(data)[2:-1,2:-1])
y_temp, v_temp = y_temp.astype(float), v_temp.astype(float)

fig = plt.figure(figsize=(18, 18))
gsc = fig.add_gridspec(3, 3, width_ratios=[1,1,1], height_ratios=[1,1,1],
                      left=0.1, right=0.9, bottom=0.05, top=0.95,
                      wspace=0.4, hspace=0.3)

# Read all site data
site_data = read_site_data(params.site, depths)

temp_list = list(site_data['site_temps'].keys())
print(temp_list)
temp05_list = np.sort(find_elements_with_substring(temp_list ,'temps05_VvT'))
temp15_list = np.sort(find_elements_with_substring(temp_list ,'temps15_VvT'))
temp25_list = np.sort(find_elements_with_substring(temp_list ,'temps25_VvT'))
temp05_tlist = np.sort(find_elements_with_substring(temp_list ,'_dtimes'))
temp15_tlist = np.sort(find_elements_with_substring(temp_list ,'_dtimes'))
temp25_tlist = np.sort(find_elements_with_substring(temp_list ,'_dtimes'))
temp_ts = datestrings2datetimes(x_temp)
datetimes = []
temp05s, temp15s, temp25s = [], [], []

temp_ax = fig.add_subplot(gsc[2,0])
temp_ax.set_title('Temperatures at z = 5, 15, 25 cm', fontweight = 'bold', fontsize = 16)
temp_ax.set_xlim([datetime(2022, 6, 22), datetime(2022, 6, 27)])
temp_ax.set_ylim([-2, 17.5])
plt.xticks(rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
format_axis(temp_ax)
temp_ax2 = fig.add_subplot(gsc[2,1])
temp_ax2.set_title(' ')
temp_ax2.set_xlim([datetime(2022, 8, 6), datetime(2022, 8, 10)])
temp_ax2.set_ylim([-2, 17.5])
plt.xticks(rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
format_axis(temp_ax2)

temp05_datas = []
temp05_times = []
for p in temp05_list:
    temp05_datas.append(site_data['site_temps'][p])
for p in temp05_tlist:
    temp05_times.append(site_data['site_temps'][p])
temp15_datas = []
temp15_times = []
for p in temp15_list:
    temp15_datas.append(site_data['site_temps'][p])
for p in temp15_tlist:
    temp15_times.append(site_data['site_temps'][p])
temp25_datas = []
temp25_times = []
for p in range(0,len(temp05_list)):
    temp_ax.plot(site_data['site_temps'][temp05_tlist[p]], site_data['site_temps'][temp05_list[p]], color = 'red', linewidth = 6.0, alpha = 0.5, linestyle = '--')
    temp_ax2.plot(site_data['site_temps'][temp05_tlist[p]], site_data['site_temps'][temp05_list[p]], color='red', linewidth=6.0, alpha=0.5, linestyle = '--')
for p in range(0,len(temp15_list)):
    temp_ax.plot(site_data['site_temps'][temp15_tlist[p]], site_data['site_temps'][temp15_list[p]], color = 'green', linewidth = 6.0, alpha = 0.5, linestyle = '--')
    temp_ax2.plot(site_data['site_temps'][temp15_tlist[p]], site_data['site_temps'][temp15_list[p]], color='green', linewidth=6.0, alpha=0.5, linestyle = '--')
for p in range(0,len(temp25_list)):
    temp_ax.plot(site_data['site_temps'][temp25_tlist[p]], site_data['site_temps'][temp25_list[p]], color = 'blue', linewidth = 6.0, alpha = 0.5, linestyle = '--')
    temp_ax2.plot(site_data['site_temps'][temp25_tlist[p]], site_data['site_temps'][temp25_list[p]], color = 'blue', linewidth = 6.0, alpha = 0.5, linestyle = '--')

interp05, interp15, interp25 = [], [], []
for i in range(0, np.shape(v_temp)[1]):
    temp_profile = v_temp[:,i]
    interp05.append(np.interp([0.05], y_temp, temp_profile))
    interp15.append(np.interp([0.15], y_temp, temp_profile))
    interp25.append(np.interp([0.25], y_temp, temp_profile))
temp_ts = datestrings2datetimes(x_temp)

temp_ax.plot(temp_ts, interp05, color = 'red')
temp_ax.plot(temp_ts, interp15, color = 'green')
temp_ax.plot(temp_ts, interp25, color = 'blue')

temp_ax2.plot(temp_ts, interp05, color = 'red')
temp_ax2.plot(temp_ts, interp15, color = 'green')
temp_ax2.plot(temp_ts, interp25, color = 'blue')

for p in temp05_list:
    condensed_date = p.split('_')[-3]
    day, month, year = condensed_date[0:2], condensed_date[2:4], condensed_date[4:]
    datetime2 = datetime(int(year), int(month), int(day))
    idx = closest_time_index(np.array(temp_ts), datetime2)

# O2 profiles
o2_ax = fig.add_subplot(gsc[0,0])
compare_slice(o2_ax,
              'site_o2umolL', 'molm3',
              'C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/o2.csv',
              'O$_2$', 'mol/m$^3$', 'Depth (m)', 100.0)

# DOC profiles
doc_ax = fig.add_subplot(gsc[0,1])
compare_slice(doc_ax,
              'site_doc', 'docmolm3',
              'C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/doc.csv',
              'DOC', 'mol/m$^3$', 'Depth (m)', 1.0)

# mcrA profiles
mcra_ax = fig.add_subplot(gsc[1,0])
compare_slice(mcra_ax,
              'site_mcra', 'mcracopm3',
              'C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ga/acemethanogen_genes.csv',
              'Methanogens', 'mol C/m$^3$', 'Depth (m)', 1e-14)

# pmoA1 profiles
pmoa1_ax = fig.add_subplot(gsc[1,1])
compare_slice(pmoa1_ax,
              'site_pmoa1', 'pmoa1copm3',
              'C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ga/methanotroph_genes.csv',
              'Methanotrophs$^1$', 'mol C/m$^3$', 'Depth (m)', 8e-14)

# pmoA2 profiles
pmoa2_ax = fig.add_subplot(gsc[1,2])
compare_slice(pmoa2_ax,
              'site_pmoa2', 'pmoa2copm3',
              'C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ga/methanotroph_genes.csv',
              'Methanotrophs$^2$', 'mol C/m$^3$', 'Depth (m)', 8e-14)


plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/fancy/comparisons.png',bbox_inches='tight')

