from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatter
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from matplotlib import ticker, cm
from os.path import isfile, join
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
import transport
import pathways
import params
import math
import init
import csv
import os


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_range(vals):

    max_val = abs(np.nanmax(vals))
    min_val = abs(np.nanmin(vals))
    com_val = np.nanmax([min_val, max_val])

    return(com_val)


def plot_params(ax, file, ylabel, xlabel, hspace, wspace, xsize_fac, ysize_fac):

    ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
    ax.set_title(file.split('/')[-2:][0] + "['" + (file.split('/')[-2:][1]).split('.')[0] + "']", fontsize=14,
                 fontweight="bold")
    fig.patch.set_facecolor('ghostwhite')
    fig.set_size_inches(xsize_fac * (grid + 1), ysize_fac * (grid - 1), forward=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Flux [mol/m2/day]')
    ax.set_xlabel(' ')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    ax.set_xticklabels(months)

    return()


def cbar_ticks(vals):

    com_val = plot_range(vals)

    ticks = [0.0, 0.5, 1.0]
    if com_val > 0.0:
        com_order_of_mag = math.floor(math.log(com_val, 10))
        ticks = (10 ** com_order_of_mag) * np.array(range(10))
        if com_val <  3*10**com_order_of_mag:
            ticks = ((10**com_order_of_mag)/2) * np.array(range(20))
        if com_val < 2*10**com_order_of_mag:
            ticks = ((10**com_order_of_mag)/4) * np.array(range(40))

    ticks = np.concatenate((-np.flip(ticks)[0:-1], ticks))

    return(ticks)

def data_dimensions(file):

    with open(file, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    n_dimensions = len(np.shape(np.array(data)))

    return(n_dimensions)


def read_1d_output(file):
    print(file)
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

    unit = np.array(data)[0, 0]
    x, y, v = np.array(data)[2:-1, 1], np.array(data)[1, 2:-1], np.transpose(np.array(data)[2:-1, 2:-1])
    y, v = y.astype(float), v.astype(float)
    v[v_temp < 0.0] = np.nan
    t = datestrings2datetimes(x)

    return(unit, t, y, v)


def datestrings2datetimes(datestrings):

    date_format, date_objects = '%Y-%m-%d %H:%M:%S', []
    for date_str in datestrings: date_objects.append(datetime.strptime(date_str, date_format))

    return(date_objects)


def VvZvT_2_ColIntVvT(VvZvT):

    ColIntVvT = np.nansum(VvZvT, axis=0)

    return(ColIntVvT)


def VvT_2_TimeIntVvT(VvT, T, nT):

    # Time integrated, column-integrated values
    integration_dates, integration_vals = [], []
    for j in range(0, int(len(VvT) / nT)):
        start, end = j * nT, j * nT + nT
        start_date, end_date = T[start], T[end]
        mid_date = datetime.fromtimestamp((start_date.timestamp() + end_date.timestamp()) / 2)
        integration_dates.append(mid_date)
        integration_vals.append(np.sum(column_integrated[start:end]))

    return(integration_dates, integration_vals)


# Find all model output directories
path = params.main_directory + 'modelOutput/' + params.site + '/'
directories = [f for f in listdir(path)]

# Make separate lists for 1D and 2D data
directories_1d, directories_2d = [], []
for d in range(0,len(directories)):
    if directories[d][-1] != 'f': directories_2d.append(directories[d])
for d in range(0,len(directories)):
    if directories[d][-1] == 'f': directories_1d.append(directories[d])

# Plots are saved in a <site_name> directory. Make sure there is one.
if not os.path.exists(params.main_directory + 'modelPlots/' + params.site):
    os.mkdir(params.main_directory + 'modelPlots/' + params.site)

# Get temperature profile data to mask values where T < 0C
with open('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/en/temp_data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
x_temp, y_temp, v_temp = np.array(data)[2:-1,1], np.array(data)[1,2:-1], np.transpose(np.array(data)[2:-1,2:-1])
y_temp, v_temp = y_temp.astype(float), v_temp.astype(float)

# For each output directory...
for d in range(0,len(directories_1d)):

    # Make a plot directory corresponding to the output file directory
    if not os.path.exists(params.main_directory+'modelPlots/'+params.site+'/'+directories_1d[d]+'/'):
        os.mkdir(params.main_directory+'modelPlots/'+params.site+'/'+directories_1d[d]+'/')

    # Find all files in the output directory
    mypath = params.main_directory + 'modelOutput/' + params.site + '/' + directories_1d[d]
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Configure plot grid based on anticipated number of plots (one plot per output file)
    n_subplots = len(onlyfiles)
    grid = int(np.ceil(np.sqrt(n_subplots)))
    if n_subplots > 5:
        fig, axes = plt.subplots(grid-1,grid+1)
    else:
        fig, axes = plt.subplots(1, n_subplots)

    # For each output file...
    for f in range(0, len(onlyfiles)):
        file = mypath + '/' + onlyfiles[f]

        # Read unit, time data (x), and 1D values (y)
        unit, t, v = read_1d_output(file)

        # Make a cumulative 1D data time series
        v_cumes = []
        for i in range(0, len(v)): v_cumes.append(np.sum(v[0:i]))

        # Make subplot
        ax = fig.axes[f]
        ax2 = ax.twinx()
        plot_params(ax, file, ylabel = 'Flux [mol/m2/day]', xlabel = ' ', hspace = 0.3, wspace = 0.3, xsize_fac = 6.0, ysize_fac = 2.0)
        ax2.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'])
        ax.plot(t, np.convolve(v, np.ones(28)/28, mode='same'))
        ax2.plot(t, v_cumes, linewidth=3.0, color = 'gray')

    plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/' + directories_1d[d] + '/' + directories_1d[d] + '.png',bbox_inches='tight')

# For each output directory...
for d in range(0,len(directories_2d)):

    # Make a plot directory corresponding to the output file directory
    if not os.path.exists(params.main_directory+'modelPlots/'+params.site+'/'+directories_2d[d]+'/'):
        os.mkdir(params.main_directory+'modelPlots/'+params.site+'/'+directories_2d[d]+'/')

    # Find all files in the output directory
    mypath = params.main_directory + 'modelOutput/' + params.site + '/' + directories_2d[d]
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Configure plot grid based on anticipated number of plots (one plot per output file)
    n_subplots = len(onlyfiles)
    grid = int(np.ceil(np.sqrt(n_subplots)))

    # For each applicable plot mode (Time vs Depth vs Z, Column-Integrated vs Time, Time-Column-Integrated vs Time)...
    if directories_2d[d][-1] == 'a':
        modes = [0]
    else:
        modes = [0, 1, 2]
    for mode in modes:

        # Refresh plot grids to prevent ax overlap between plot modes
        if n_subplots > 4:
            fig, axes = plt.subplots(grid - 1, grid + 1)
        else:
            fig, axes = plt.subplots(1, n_subplots)

        # For each output file...
        for f in range(0,len(onlyfiles)):

            file = mypath+'/'+onlyfiles[f]

            if data_dimensions(file) == 2:
                # Read unit, time data (x), depth data (y), and 2D values (z)
                unit, t, y, v = read_2d_output(file, v_temp)

            # Configure plot panels
            ax = fig.axes[f]
            plot_params(ax, file, ylabel = 'Depths [m]', xlabel = ' ', hspace = 0.3, wspace = 0.3, xsize_fac = 6.0, ysize_fac = 5.0)

            # Column-integrated values
            column_integrated = VvZvT_2_ColIntVvT(v)

            # Time integrated, column-integrated values
            integration_dates, integration_vals = VvT_2_TimeIntVvT(column_integrated, t, 112)

            # Plots
            #
            # Time vs Depth vs V plots
            if mode == 0:
                ax.set_ylim([0.45, 0.025])
                c3 = ax.contourf(t, y, v, levels=100, cmap='rainbow', vmin = -plot_range(v), vmax = plot_range(v))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('bottom', size='5%', pad=0.5)
                cbar = fig.colorbar(c3,orientation='horizontal',cax=cax, ticks = cbar_ticks(v), ax=ax)
                cbar.ax.tick_params(axis="both", rotation=30)
                cbar.set_label(unit)
            #
            # Column-integrated vs Time plots
            if mode == 1 and directories_2d[d][-1] == 'r':
                ax.set_ylim([min(column_integrated), max(column_integrated)])
                ax.plot(t, np.convolve(column_integrated, np.ones(28)/28, mode='same'), linewidth = 2.0, color = 'gray')
            #
            # Time-integrated, Column-integrated vs Time bar charts
            if mode == 2 and directories_2d[d][-1] == 'r':
                ax.bar(integration_dates, integration_vals, width = 20, color='gray')

        # Remove un-used plot spaces
        n_plot_spaces = (grid - 1) * (grid + 1)
        empty_plot_spaces = n_plot_spaces - len(onlyfiles)
        n_fig_axes = len(fig.axes)
        for ax in fig.axes:
            if not ax.has_data():
                ax.remove()

        # Save plot
        plt.savefig(params.main_directory + 'modelPlots/' + params.site+ '/' + directories_2d[d] + '/' + directories_2d[d] + str(mode) + '.png',bbox_inches='tight')




