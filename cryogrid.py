from mpl_toolkits.axes_grid1 import make_axes_locatable
from openpyxl import load_workbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import timedelta
from openpyxl import Workbook
from datetime import datetime
from matplotlib import text
from matplotlib import cm
from datetime import date
from copy import copy
import conversions
import numpy as np
import diffusion
import pathways
import openpyxl
import forcing
import params
import shutil
import plants
import init
import data
import csv
import os


def read_csv_array(filename):

    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    return(data)


def profile_plots(axis,data,tempdata,satdata,levels,xlabel,timedata,depthdata,color_scheme,fig,temp_nan,secondary):

    axis.set_xlim([150, 320])

    X, Y = np.meshgrid(timedata, depthdata)
    data2 = copy(data)
    ###data2[tempdata < 0.0] = np.nan
    #c = axis.contourf(X, Y, data2, levels=levels, cmap=color_scheme)
    temp_mask = 1.0 + 0.0 * tempdata
    temp_mask[tempdata > 0.0] = np.nan
    c = axis.contourf(X, Y, data2, cmap=color_scheme, levels=100)
    axis.set_ylim([-0.60, 0.0])
    axis.set_xlabel('Day of Year',fontsize=24)
    divider = make_axes_locatable(axis)
    if secondary == False:
        cax = divider.append_axes('bottom', size='5%', pad=1.2)
        cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(axis="both", labelsize=18, rotation=45)
        cbar.set_label(xlabel, rotation=0, fontsize=22, fontweight="bold")
    if secondary:
        cax = divider.append_axes('bottom', size='5%', pad=-2.3)
        fig.add_axes(cax)
        cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(axis="both", labelsize=18, rotation=45)
        cbar.set_label(xlabel, rotation=0, fontsize=22, fontweight="bold")
        cbar.set_ticks([])
    axis.xaxis.set_tick_params(labelsize=20)
    axis.yaxis.set_tick_params(labelsize=20)
    axis.set_ylabel('Depth (m)',fontsize=22)

    axis.contourf(X, Y, temp_mask, colors = 'white', hatches = '/', alpha = 0.5)
    if temp_nan:
        axis.contourf(X, Y, temp_mask, colors='white')
    c = axis.contour(X, Y, tempdata, linewidths=2.0, levels=[-400,0.0],colors='black')

    c = axis.contour(X, Y, satdata, linewidths=4.0, levels=[0.0, 99.0],colors='white',alpha=0.5)
    c = axis.contour(X, Y, satdata, linewidths=2.0, levels=[0.0, 99.0],colors='blue',alpha=0.5,linestyles='--')

    return()


def forcing_t(cg_forcing, t):

    forcing = {}

    closest_index = np.where(abs(cg_forcing['time_data']-t) == min(abs(cg_forcing['time_data']-t)))[0][0]

    for key in cg_forcing:
        key_dimensions = len(np.shape(cg_forcing[key]))
        if key_dimensions == 2:  # Only consider time-space profiles, not e.g. depth_data or time_data
            forcing[key] = cg_forcing[key][:,closest_index]

    forcing['time_data'] = cg_forcing['time_data'][closest_index]
    forcing['depth_data'] = cg_forcing['depth_data']

    return(forcing)


def cggrid2modelgrid(cg_forcing):

    # New CryoGrid forcing dictionary, with profiles interpolated to [n_model_layers x n_CryoGrid_timesteps]
    cg_forcing2 = cg_forcing.copy()

    # n_CryoGrid_timesteps
    first_key = list(cg_forcing.keys())[0]
    ncgt = np.shape(cg_forcing[first_key])[1]

    # n_model_layers
    layer_thicknesses, depths = init.define_layers()
    depths=-1.0*depths

    # Interpolate the data to the model depth grid
    for key in cg_forcing:
        key_dimensions = len(np.shape(cg_forcing[key]))

        ir = 14 # Number of CryoGrid layers to utilize for interpolation (from the surface)

        if key_dimensions == 1: # Only interpolate time or space profiles, not e.g. depth_data or time_data
            data_interp = cg_forcing[key]

        if key_dimensions == 2: # Only interpolate time x space profiles, not e.g. depth_data or time_data
            data_interp = np.zeros((len(depths), ncgt))
            for i in range(0, np.shape(data_interp)[1]):
                xnew, data_interp[:, i] = forcing.interp2grid(
                                                 cg_forcing['depth_data'][0:ir], cg_forcing[key][0:ir, i], depths)

        if key_dimensions == 3:  # Only interpolate type x time x space profiles, not e.g. depth_data or time_data
            data_interp = np.zeros( (np.shape(cg_forcing[key])[0] , len(depths) , ncgt))
            for k in range(0, np.shape(data_interp)[0]):
                for i in range(0, np.shape(data_interp)[2]):
                    xnew, data_interp[k, :, i] = forcing.interp2grid(
                                                 cg_forcing['depth_data'][0:ir], cg_forcing[key][k, 0:ir, i], depths)

        cg_forcing2[key] = data_interp

    # The model depths are the new depths
    cg_forcing2['depth_data'] = depths

    return(cg_forcing2)


def cryogrid():

    main_directory = params.main_directory
    cg_directory = params.cg_directory
    forcing_profiles = 0

    print('Reading processed CryoGrid output')

    make_plots = True

    ### Cryogrid output

    cg_forcing = {} # Temperatures and composition fractions
    cg_ceq = {}      # Gas equilibrium concentrations derived from temperatures

    # Read the Cryogrid output into arrays with dimension [cg-depths, cg_times]
    temp_data = np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/T.txt'), dtype='float64')
    water_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/water.txt'),dtype='float64')
    waterIce_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/waterIce.txt'),dtype='float64')
    ice_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/ice.txt'),dtype='float64')
    soilOrganic_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/soilOrganic.txt'), dtype='float64')
    soilOrganic_data = np.full(np.shape(temp_data), soilOrganic_data)
    soilMineral_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/soilMineral.txt'), dtype='float64')
    soilMineral_data = np.full(np.shape(temp_data), soilMineral_data)
    air_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/air.txt'),dtype='float64')
    air_data = 100.0 - waterIce_data - soilOrganic_data - soilMineral_data
    sat_data = 100.0 * water_data / (water_data + air_data)
    porosity_data = 100.0 - (soilOrganic_data + soilMineral_data)

    # At every time interval s, set saturation = 100% if saturation > threshold,
    # ensuring saturation at and below the water table depth.

    #for s in range(0, np.shape(sat_data)[1]):
    #    sat_data_profile = sat_data[:, s]
    #    sat_data_profile[sat_data_profile > 70.0] = 100.0
    #    water_table_index = np.where(sat_data_profile == 100.)[0]
    #    if len(np.where(sat_data_profile == 100.)[0]) > 0:
    #        sat_data_profile[(water_table_index[0]):] = 100.0
    #    sat_data[:, s] = sat_data_profile

    # Sum the arrays to verify that soil composition adds to 1.0 (100%)
    ssum_data = ice_data + soilOrganic_data + soilMineral_data + air_data

    # Read the Cryogrid time data, and shift values to start at 0 (days)
    time_data = (np.array(read_csv_array(main_directory+
                'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/TIMESTAMP.txt'), dtype='float64'))[0]
    time_data = time_data - time_data[0]

    # Read the Cryogrid depth data, and shift values to place the surface at 0 (meters)
    depth_data = np.array(read_csv_array(main_directory+
                'CryoGrid/CryoGridCommunity_results/'+params.site+'/clean_output/z.txt'), dtype='float64')
    depth_data = depth_data - depth_data[0]
    depth_data = (np.transpose(depth_data))[0]
    depth_data = depth_data[0:-1]
    pos_depths = abs(depth_data)

    cg_forcing['temp_data'] = temp_data
    cg_forcing['water_data'] = water_data
    cg_forcing['waterIce_data'] = waterIce_data
    cg_forcing['ice_data'] = ice_data
    cg_forcing['soilOrganic_data'] = soilOrganic_data
    cg_forcing['soilMineral_data'] = soilMineral_data
    cg_forcing['air_data'] = air_data
    cg_forcing['time_data'] = time_data
    cg_forcing['depth_data'] = depth_data
    cg_forcing['satpct_data'] = sat_data

    cg_forcing['effdiff_data_ch4'] = \
            diffusion.effective_diffusivity(sat_data, temp_data, np.shape(sat_data)[0], np.shape(sat_data)[1])[0]
    cg_forcing['effdiff_data_o2'] = \
            diffusion.effective_diffusivity(sat_data, temp_data, np.shape(sat_data)[0], np.shape(sat_data)[1])[1]
    cg_forcing['effdiff_data_co2'] = \
            diffusion.effective_diffusivity(sat_data, temp_data, np.shape(sat_data)[0], np.shape(sat_data)[1])[2]
    cg_forcing['effdiff_data_h2'] = \
            diffusion.effective_diffusivity(sat_data, temp_data, np.shape(sat_data)[0], np.shape(sat_data)[1])[3]

    # New CryoGrid forcing dictionary, with profiles interpolated to [n_model_layers x n_CryoGrid_timesteps]
    cg_forcing2 = cggrid2modelgrid(cg_forcing)

    # Derived gas equilibrium concentrations, from CryoGrid temperatures
    cg_ceq['ch4'] = diffusion.equilibrium_concentration(cg_forcing2['temp_data'])[0]
    cg_ceq['o2'] = diffusion.equilibrium_concentration(cg_forcing2['temp_data'])[1]
    cg_ceq['co2'] = diffusion.equilibrium_concentration(cg_forcing2['temp_data'])[2]
    cg_ceq['h2'] = diffusion.equilibrium_concentration(cg_forcing2['temp_data'])[3]
    cg_ceq['time_data'] = time_data
    cg_ceq['depth_data'] = depth_data

    if make_plots == True:

        # Configure plot panels
        fig, axes = plt.subplots(2, 3)
        fig.patch.set_facecolor('ivory')
        fig.set_size_inches(20.0, 12.5, forward=True)
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[0, 2]
        ax4 = axes[1, 0]
        ax5 = axes[1, 1]
        ax6 = axes[1, 2]

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_facecolor('ivory')

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_facecolor('ivory')

        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_facecolor('ivory')

        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.set_facecolor('ivory')

        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.set_facecolor('ivory')

        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.set_facecolor('ivory')

        # Plot the Cryogrid data in the upper panel row
        profile_plots(ax1, temp_data, temp_data, sat_data, np.linspace(-10, 18, 15), 'Temperature [\u00b0C]', time_data,
                                 depth_data, cm.rainbow, fig, False, False)
        profile_plots(ax6, porosity_data, temp_data, sat_data, np.linspace(75, 100, 21), 'Porosity [%]', time_data,
                                 depth_data, cm.rainbow, fig, True, False)
        profile_plots(ax4, soilOrganic_data, temp_data, sat_data, np.linspace(0, 20, 21), 'Soil Organic [%]', time_data,
                                 depth_data, cm.Oranges, fig, True, False)
        profile_plots(ax5, soilMineral_data, temp_data, sat_data, np.linspace(0, 2, 21), 'Soil Mineral [%]', time_data,
                                 depth_data, cm.Wistia, fig, True, False)
        profile_plots(ax2, water_data, temp_data, sat_data, np.linspace(0, 100, 21), 'Liquid Water [%]', time_data,
                                 depth_data, cm.Blues, fig, False, False)
        profile_plots(ax3, sat_data, temp_data, sat_data,
                                 np.linspace(0, 100, 21), 'Pore Saturation [%]',
                                 time_data, depth_data, cm.rainbow, fig, False, False)

        plt.tight_layout()

        file_string = 'lateral'+str(params.lateral)+str(params.water_table_depth)+'m'+'_'+\
                      'rainfrac'+str(params.rain_fraction)+'_'+'snowfrac'+str(params.snow_fraction)+'_'+\
                      'z0'+str(params.z0)+'m'+'_'+'rootdepth'+str(params.rootDepth)+'m'+'_'+'evapdepth'+\
                      str(params.evaporationDepth)+'m'+'_'+'resdist'+str(params.distance_reservoir)+'m'+'_'+\
                      'conlength'+str(params.reservoir_contact_length)+'m'+'_'+'saturated'+str(params.saturated)
        file_string = file_string.replace('.', 'p')

        # Save the plots
        print('Plotting processed Cryogrid output: ', main_directory+
              'CryoGrid/CryoGridCommunity_results/'+params.site+'/plots/'+file_string+'.png')
        plt.savefig(main_directory+'CryoGrid/CryoGridCommunity_results/'+params.site+'/plots/'+file_string+'.png')

    return(cg_forcing2, cg_ceq)