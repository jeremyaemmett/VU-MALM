import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import text
import matplotlib.dates as mdates
import params
import os
import diffusion
from copy import copy
import pathways
from openpyxl import load_workbook
import openpyxl
import shutil
from openpyxl import Workbook
import csv
import init
import data
import plants
import conversions
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from datetime import timedelta
from datetime import date

# test4 8/27/2024

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

def interp2grid(depths, values, newgrid):
    #print(np.shape(newgrid), np.shape(np.flip(depths)), np.shape(np.flip(values)))
    ynew = np.interp(newgrid, np.flip(depths), np.flip(values))
    #from IPython import embed
    #embed()
    return (newgrid, ynew)

def interp2grid2(depths, values, newgrid):

    ynew = np.interp(newgrid, depths, values)
    #from IPython import embed
    #embed()
    return (newgrid, ynew)

def read_csv_header(filename, column_idx, var_type, header_lines):
    with open(filename) as f:
        reader = csv.reader(f)
        if header_lines != 0:
            for h in range(0,header_lines):
                header = next(reader)
        vals = []
        for row in reader:
            if var_type == 'string':
                val = row[column_idx]
            if var_type == 'integer':
                val = int(row[column_idx])
            if var_type == 'float':
                if row[column_idx] == '':
                    val = -9999.0
                else:
                    val = float(row[column_idx])
            vals.append(val)
    return vals

def comparison(doy,profile1,profile2,depths,axis,ax1):

    make_comp_plots = True

    if make_comp_plots == True:

        axis.plot(doy + 0.5*(profile1/profile2)/2.0-50, -depths, color='white', linewidth=8.0, linestyle='solid')
        axis.plot(doy + 0.5*(profile1/profile2)/2.0-50, -depths, color='black', linewidth=3.0, linestyle='solid')
        x_sd = abs(0.01*profile1/profile2 - 1.0)
        sd = np.sqrt(np.mean(x_sd))
        axis.plot([(doy + 0.5 * (100.0) / 2.0 - 50), doy + 0.5 * (100.0) / 2.0 - 50], [-0.42, -0.05], color='gray',linewidth=10.0, linestyle='solid',alpha=0.4)
        axis.plot([(doy + 0.5*(100.0)/2.0-50),doy + 0.5*(100.0)/2.0-50],[-0.42,-0.4], color='gray', linewidth=3.0,linestyle='solid')
        axis.plot([doy + 0.5*(0.0) / 2.0-50, doy + 0.5*(0.0) / 2.0-50], [-0.42, -0.4], color='gray', linewidth=3.0, linestyle='solid')
        axis.plot([doy + 0.5*(50.0) / 2.0-50, doy + 0.5*(50.0) / 2.0-50], [-0.42, -0.4], color='gray', linewidth=3.0, linestyle='solid')
        axis.plot([doy + 0.5*(150.0) / 2.0-50, doy + 0.5*(150.0) / 2.0-50], [-0.42, -0.4], color='gray', linewidth=3.0, linestyle='solid')
        axis.plot([doy + 0.5*(200.0) / 2.0-50, doy + 0.5*(200.0) / 2.0-50], [-0.42, -0.4], color='gray', linewidth=3.0, linestyle='solid')
        axis.plot([doy + 0.5*(0.0) / 2.0-50, doy + 0.5*(0.0) / 2.0-50], [-0.40, -0.05], color='gray', linewidth=3.0, linestyle='solid')
        axis.plot([doy + 0.5*(0.0) / 2.0-50, doy + 0.5*(200.0) / 2.0-50], [-0.40, -0.40], color='gray', linewidth=3.0, linestyle='solid')
        axis.annotate(' ',(doy + 0.5*(0.0) / 2.0-50, -0.46),color='gray',fontsize=24,ha='center',rotation=0,weight='bold')
        axis.annotate(' ', (doy + 0.5*(50.0) / 2.0-50, -0.46), color='gray', fontsize=24, ha='center',rotation=0,weight='bold')
        axis.annotate('100%', (doy + 0.5*(100.0) / 2.0-50, -0.46), color='gray', fontsize=24, ha='center',rotation=0,weight='bold')
        axis.annotate(' ', (doy + 0.5*(150.0) / 2.0-50, -0.46), color='gray', fontsize=24, ha='center',rotation=0,weight='bold')
        axis.annotate(' ', (doy + 0.5*(200.0) / 2.0-50, -0.46), color='gray', fontsize=24, ha='center',rotation=0,weight='bold')
        ax1.annotate(r'$\frac{Sim}{Obs}$', (doy + 0.5*(160.0) / 2.0-50, -0.25), color='black', fontsize=34, ha='center', weight='bold',rotation=0)
        axis.annotate('\u03C3 = {}'.format(np.round(sd,2)), (doy + 0.5*(120.0) / 2.0-50, -0.37), color='darkgray', fontsize=24, ha='center', weight='bold',rotation=0)
    return()


def soil_temp_data():

    filename = 'C:/Users/Jeremy/Desktop/data/CR1000_C1_C2.dat'

    times = np.array(read_csv_header(filename, 0, 'string',4))

    for i in range(0, len(times)):
        t_test = times[i]
        times[i] = t_test[0:4] + t_test[5:7] + t_test[8:10] + t_test[11:13] + t_test[14:16] + t_test[17:19]
    datetimes = [datetime.strptime(str(int(t)), '%Y%m%d%H%M%S') for t in times]
    record = np.array(read_csv_header(filename, 1, 'integer',4))
    batt_volt_avg = np.array(read_csv_header(filename, 2, 'float',4))
    ptemp_avg = np.array(read_csv_header(filename, 3, 'float',4))
    soiltemp1_avg = np.array(read_csv_header(filename, 4, 'float',4))
    soiltemp2_avg = np.array(read_csv_header(filename, 5, 'float',4))
    soiltemp3_avg = np.array(read_csv_header(filename, 6, 'float',4))
    stmp_1_d_005_avg = np.array(read_csv_header(filename, 10, 'float',4))
    stmp_1_d_015_avg = np.array(read_csv_header(filename, 11, 'float',4))
    stmp_1_d_025_avg = np.array(read_csv_header(filename, 12, 'float',4))
    stmp_2_d_005_avg = np.array(read_csv_header(filename, 19, 'float',4))
    stmp_2_d_015_avg = np.array(read_csv_header(filename, 20, 'float',4))
    stmp_2_d_025_avg = np.array(read_csv_header(filename, 21, 'float',4))

    return(times,datetimes,record,batt_volt_avg,ptemp_avg,soiltemp1_avg,soiltemp2_avg,soiltemp3_avg,
           stmp_1_d_005_avg,stmp_1_d_015_avg,stmp_1_d_025_avg,stmp_2_d_005_avg,stmp_2_d_015_avg,stmp_2_d_025_avg)

def closest_soil_temps(test_datetime,datetimes,stmp_1_d_005_avg,stmp_1_d_015_avg,stmp_1_d_025_avg,
                       stmp_2_d_005_avg,stmp_2_d_015_avg,stmp_2_d_025_avg):

    dts = np.array([abs(test_datetime - x) for x in datetimes])
    closest_index = (np.where(dts == min(dts)))[0][0]
    stmp_1s = [stmp_1_d_005_avg[closest_index], stmp_1_d_015_avg[closest_index], stmp_1_d_025_avg[closest_index]]
    stmp_2s = [stmp_2_d_005_avg[closest_index], stmp_2_d_015_avg[closest_index], stmp_2_d_025_avg[closest_index]]

    return(stmp_1s,stmp_2s)

def forcing_profiles(doy, depth_data, temp_data, water_data, waterIce_data, ice_data, air_data, soilOrganic_data,
                     soilMineral_data, depths):

    # Get cross-sections for the specified day of year
    ir = 14
    water_xsection = water_data[:,doy*4]
    waterIce_xsection = waterIce_data[:,doy*4]
    ice_xsection = ice_data[:,doy*4]
    air_xsection = air_data[:,doy*4]
    soilOrganic_xsection = soilOrganic_data[:,doy*4]
    soilMineral_xsection = soilMineral_data[:,doy*4]
    temp_xsection = temp_data[:,doy*4]
    porosity_xsection = 100.0 - (soilOrganic_data[:,doy*4] + soilMineral_data[:,doy*4])

    new_grid, water_xsection_new = interp2grid(depth_data[0:ir], water_xsection[0:ir], -depths)
    new_grid, waterIce_xsection_new = interp2grid(depth_data[0:ir], waterIce_xsection[0:ir], -depths)
    new_grid, ice_xsection_new = interp2grid(depth_data[0:ir], ice_xsection[0:ir], -depths)
    new_grid, air_xsection_new = interp2grid(depth_data[0:ir], air_xsection[0:ir], -depths)
    new_grid, soilOrganic_xsection_new = interp2grid(depth_data[0:ir], soilOrganic_xsection[0:ir], -depths)
    new_grid, soilMineral_xsection_new = interp2grid(depth_data[0:ir], soilMineral_xsection[0:ir], -depths)
    new_grid, temp_xsection_new = interp2grid(depth_data[0:ir], temp_xsection[0:ir], -depths)
    new_grid, porosity_xsection_new = interp2grid(depth_data[0:ir], porosity_xsection[0:ir], -depths)

    return(temp_xsection_new, water_xsection_new, waterIce_xsection_new, ice_xsection_new, air_xsection_new,
           soilOrganic_xsection_new, soilMineral_xsection_new, porosity_xsection_new)


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
                xnew, data_interp[:, i] = interp2grid(cg_forcing['depth_data'][0:ir], cg_forcing[key][0:ir, i], depths)

        if key_dimensions == 3:  # Only interpolate type x time x space profiles, not e.g. depth_data or time_data
            data_interp = np.zeros( (np.shape(cg_forcing[key])[0] , len(depths) , ncgt))
            for k in range(0, np.shape(data_interp)[0]):
                for i in range(0, np.shape(data_interp)[2]):
                    xnew, data_interp[k, :, i] = interp2grid(
                                                 cg_forcing['depth_data'][0:ir], cg_forcing[key][k, 0:ir, i], depths)

        cg_forcing2[key] = data_interp

    # The model depths are the new depths
    cg_forcing2['depth_data'] = depths

    return(cg_forcing2)


def profile_t(cg_profile, t):

    profile = {}

    # Closest time index in the CryoGrid output to the model datetime
    closest_index = np.where(abs(cg_profile['time_data'] - t) == min(abs(cg_profile['time_data'] - t)))[0][0]

    for key in cg_profile:
        key_dimensions = len(np.shape(cg_profile[key]))
        if key_dimensions == 2:  # Depth x Time data
            profile[key] = cg_profile[key][:, closest_index]
        if key_dimensions == 3:  # Type x Depth x Time data
            profile[key] = cg_profile[key][:, :, closest_index]

    profile['time_data'] = cg_profile['time_data'][closest_index]
    profile['depth_data'] = cg_profile['depth_data']

    # If there's a 'satpct_data' key, make sure all saturation values at or below the water table depth = 100.0
    #if 'satpct_data' in list(cg_profile.keys()):
    #    water_table_index = np.where(profile['satpct_data'] == 100.)[0][0]
    #    if len(np.where(profile['satpct_data'] == 100.)[0]) > 0:
    #        cg_profile['satpct_data'][water_table_index:] = 100.0

    return (profile)


def forcing_data():

    make_plots = True

    ### Cryogrid output [cg_depths, cg_times]

    temp_data = np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\T.txt'),dtype='float64')
    water_data = 100.0*np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\water.txt'),dtype='float64')
    waterIce_data = 100.0*np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\waterIce.txt'),dtype='float64')
    ice_data = 100.0*np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\ice.txt'),dtype='float64')
    soilOrganic_data = 100.0*np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\soilOrganic.txt'),dtype='float64')
    soilOrganic_data = np.full(np.shape(temp_data),soilOrganic_data)
    soilMineral_data = 100.0*np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\soilMineral.txt'),dtype='float64')
    soilMineral_data = np.full(np.shape(temp_data),soilMineral_data)
    air_data = 100.0*np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\air.txt'),dtype='float64')
    air_data = np.full(np.shape(temp_data),air_data)

    # Sum the arrays to verify that soil composition adds to 1.0 (100%)
    ssum_data = ice_data + soilOrganic_data + soilMineral_data + air_data

    # Read the Cryogrid time data, and shift values to start at 0 (days)
    time_data = (np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\TIMESTAMP.txt'),dtype='float64'))[0]
    time_data = time_data - time_data[0]

    # Read the Cryogrid depth data, and shift values to place the surface at 0 (meters)
    depth_data = np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\z.txt'),dtype='float64')
    depth_data = depth_data - depth_data[0]
    depth_data = (np.transpose(depth_data))[0]
    depth_data = depth_data[0:-1]

    if make_plots == True:
        # Configure plot panels
        fig, axes = plt.subplots(3,3)
        #fig, axes = plt.subplots(2,2)
        fig.patch.set_facecolor('ivory')
        fig.set_size_inches(20.0, 20.0, forward=True)
        ax1 = axes[0,0]
        ax2 = axes[0,1]
        ax3 = axes[0,2]
        #ax3 = axes[1,0]
        ax4 = axes[1,0]
        #ax4 = axes[1,1]
        ax5 = axes[1,1]
        ax6 = axes[1,2]
        ax7 = axes[2,0]
        ax8 = axes[2,1]
        ax9 = axes[2,2]

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

        # Plot the Cryogrid data in the upper panel row
        #test = profile_plots(ax2,100.0 - (soilOrganic_data + soilMineral_data),temp_data,np.linspace(50, 100, 21),'Water [%]',time_data,depth_data,cm.GnBu,fig)
        #test = profile_plots(ax4,soilOrganic_data,temp_data,np.linspace(0, 20, 21),'soilOrganic [%]',time_data,depth_data,cm.rainbow,fig)
        c_pool = params.dz * (conversions.kg2mol(1500.0, 'c'))  # mol/m2
        dz = 0.05
        f_moist = waterIce_data/100.0
        t = temp_data
        #doc = pathways.doc_concentration(k_cpool=params.k_cpool, c_pool=c_pool, dz=dz, doc_prod_q10=params.doc_prod_q10,f_moist=f_moist, t=t)
        #doc = doc/40.0
        test = profile_plots(ax1,temp_data,temp_data,np.linspace(-10, 18, 15),'Temp [\u00b0C]',time_data,depth_data,cm.rainbow,fig)
        test = profile_plots(ax2, water_data, temp_data, np.linspace(0, 100, 21),'Water [%]', time_data, depth_data, cm.GnBu, fig)
        profile_plots(ax3, soilOrganic_data, temp_data, np.linspace(0, 20, 21),'Organic Carbon [%]', time_data, depth_data, cm.copper_r, fig)
        #profile_plots(ax4, doc, temp_data, np.linspace(0, 3, 21), 'Dissolved OC [mol/m$^3$]', time_data, depth_data, cm.pink_r, fig)
        test = profile_plots(ax5,soilMineral_data,temp_data,np.linspace(0, 2, 21),'soilMineral [%]',time_data,depth_data,cm.rainbow,fig)
        test = profile_plots(ax6,air_data,temp_data,np.linspace(0, 100, 21),'air [%]',time_data,depth_data,cm.rainbow,fig)
        test = profile_plots(ax7,waterIce_data,temp_data,np.linspace(0, 100, 21),'water + ice [%]',time_data,depth_data,cm.GnBu,fig)
        test = profile_plots(ax8,ice_data, temp_data, np.linspace(0, 100, 21), 'ice [%]', time_data, depth_data,cm.GnBu, fig)
        test = profile_plots(ax9, waterIce_data + soilOrganic_data + soilMineral_data + air_data, temp_data, np.linspace(50, 160, 21), 'waterIce + soilOrganic + \n soilMineral + air [%]', time_data, depth_data,cm.rainbow, fig)

    ## Interpolate the Cryogrid data to the microbe model's depth grid [n_layers x n_Cryogrid_timesteps]

    # Number of Cryogrid timesteps
    ncgt = np.shape(temp_data)[1]

    layer_thicknesses, depths = init.define_layers()

    depths=-1.0*depths
    temp_data_interp = np.zeros((len(depths),ncgt))
    water_data_interp = np.zeros((len(depths),ncgt))
    waterIce_data_interp = np.zeros((len(depths),ncgt))
    soilOrganic_data_interp = np.zeros((len(depths),ncgt))
    soilMineral_data_interp = np.zeros((len(depths),ncgt))
    air_data_interp = np.zeros((len(depths),ncgt))

    # Number of Cryogrid layers from (top-down) to utilize for interpolation
    ir = 14

    for i in range(0,np.shape(temp_data_interp)[1]):
        xnew, temp_data_interp[:,i] = interp2grid(depth_data[0:ir],temp_data[0:ir,i],depths)
        xnew, water_data_interp[:, i] = interp2grid(depth_data[0:ir], water_data[0:ir, i], depths)
        xnew, waterIce_data_interp[:, i] = interp2grid(depth_data[0:ir], waterIce_data[0:ir, i], depths)
        xnew, soilOrganic_data_interp[:, i] = interp2grid(depth_data[0:ir], soilOrganic_data[0:ir, i], depths)
        xnew, soilMineral_data_interp[:, i] = interp2grid(depth_data[0:ir], soilMineral_data[0:ir, i], depths)
        xnew, air_data_interp[:, i] = interp2grid(depth_data[0:ir], air_data[0:ir, i], depths)

    ### Field Data

    # Read the fieldwork data into arrays with dimension [n_depth-measurements x 1]. Carbon, Density/VWC,
    # and pH measurements have different depth grids (e.g. 'depth1') since they were collected from different cores.

    soc_filename = 'C:/Users/Jeremy/Desktop/model_data/SOC_palsa_high_aug_2022.txt'
    depth1 = np.flip(np.array(read_csv_header(soc_filename, 0, 'float',1)))
    before = np.flip(np.array(read_csv_header(soc_filename, 1, 'float',1)))
    after = np.flip(np.array(read_csv_header(soc_filename, 2, 'float',1)))
    foil = np.flip(np.array(read_csv_header(soc_filename, 3, 'float',1)))

    density_filename = 'C:/Users/Jeremy/Desktop/model_data/density_palsa_high_aug_2022.txt'
    depth2 = np.flip(np.array(read_csv_header(density_filename, 0, 'float',1)))
    wet = np.flip(np.array(read_csv_header(density_filename, 1, 'float',1)))
    tray = np.flip(np.array(read_csv_header(density_filename, 2, 'float',1)))
    dry = np.flip(np.array(read_csv_header(density_filename, 3, 'float',1)))
    diameter = np.flip(np.array(read_csv_header(density_filename, 4, 'float',1)))
    length = np.flip(np.array(read_csv_header(density_filename, 5, 'float',1)))
    volume = length*np.pi*(diameter/2.0)**2

    ph_filename = 'C:/Users/Jeremy/Desktop/model_data/pH_palsa_high_aug_2022.txt'
    depth3 = np.flip(np.array(read_csv_header(ph_filename, 0, 'float',1)))
    ph = np.flip(np.array(read_csv_header(ph_filename, 1, 'float',1)))

    # Interpolate the fieldwork data to the microbe model's depth grid [n_layers x 1]
    layer_thicknesses, depths = init.define_layers()
    new_grid, before = interp2grid2(depth1,before,depths)
    new_grid, after = interp2grid2(depth1,after,depths)
    new_grid, foil = interp2grid2(depth1,foil,depths)
    new_grid, wet = interp2grid2(depth2,wet,depths)
    new_grid, dry = interp2grid2(depth2,dry,depths)
    new_grid, tray = interp2grid2(depth2,tray,depths)
    new_grid, volume = interp2grid2(depth2,volume,depths)
    new_grid, ph = interp2grid2(depth3,ph,depths)

    # Correct for the weights of foil and tray containers
    before = before - foil
    after = after - foil
    wet = wet - tray
    dry = dry - tray

    # Bulk density of dry soil
    rho_soil = dry / volume # g-dry / cm3

    # Porosity of soil, assuming solid particle density = 2.6 g/cm^3
    porosity = 1.0 - (rho_soil / 2.6)

    # Mass ratio of SOC to dry soil, and minerals to dry soil
    soc_dry_g_g = (before - after) / before # g C/g dry soil
    min_dry_g_g = after / before # g Min/g dry soil

    # Fraction of dry soil composed of SOC and minerals
    v_ratio = (min_dry_g_g * 1.4) / (soc_dry_g_g * 2.6)
    v_frac_min = v_ratio / (v_ratio + 1.0)
    v_frac_soc = 1.0 - v_frac_min

    # Volumes of SOC and minerals
    v_solid = (1.0 - porosity)
    #v_min = np.round(v_frac_min * v_solid, 2)
    v_min = v_frac_min * v_solid
    v_soc = np.round(v_frac_soc * v_solid, 2)

    # Volumetric water content (VWC) of wet soil
    gwc = (wet - dry) / dry # g/g
    vwc = gwc * (rho_soil / 1.0)

    # Convert mass ratios of SOC and minerals [g/g-dry soil], to volume densities [kg/m^3]
    soc_g_cm3 = soc_dry_g_g * rho_soil
    min_g_cm3 = min_dry_g_g * rho_soil
    soc_kg_m3 = soc_g_cm3 * 1000.0
    min_kg_m3 = min_g_cm3 * 1000.0

    # Convert volume density of SOC [kg/m^3] to [mol/m^3]
    soc_mol_m3 = conversions.kg2mol(soc_kg_m3,'c')

    # Get cross-sections for the specified day of year
    doy = 237
    temp_xsection, water_xsection, waterIce_xsection, ice_xsection, air_xsection, soilOrganic_xsection, \
    soilMineral_xsection, porosity_xsection = forcing_profiles(doy, depth_data, temp_data, water_data, waterIce_data,
                                                ice_data,air_data, soilOrganic_data, soilMineral_data, depths)

    if make_plots == True:
        #test = comparison(doy, 100.0*waterIce_xsection, 100.0*vwc, depths, ax2, ax1)
        test = comparison(doy,water_xsection,vwc,depths,ax2,ax1)

        #test = comparison(doy,100.0*porosity_xsection,100.0*porosity,depths,ax3,ax1)

        #test = comparison(doy,100.0*soilOrganic_xsection,100.0*v_soc,depths,ax4,ax1)

        #test = comparison(doy,100.0*soilMineral_xsection_new,100.0*v_min,depths,ax5)

    test_datetime = datetime(2022, 8, 7, 0, 0, 0)

    new_grid, temp_xsection_new = interp2grid(depth_data[0:ir], temp_xsection[0:ir], [-0.05,-0.15,-0.25])
    times, datetimes, record, batt_volt_avg, ptemp_avg, soiltemp1_avg, soiltemp2_avg, soiltemp3_avg, \
    stmp_1_d_005_avg, stmp_1_d_015_avg, stmp_1_d_025_avg, stmp_2_d_005_avg, stmp_2_d_015_avg, stmp_2_d_025_avg = soil_temp_data()
    stmp_1s,stmp_2s = closest_soil_temps(test_datetime,datetimes,stmp_1_d_005_avg,stmp_1_d_015_avg,stmp_1_d_025_avg,
                       stmp_2_d_005_avg,stmp_2_d_015_avg,stmp_2_d_025_avg)
    if make_plots == True:
        test = comparison(doy,100.0*temp_xsection_new,stmp_1s,np.array([0.05,0.15,0.25]),ax1,ax1)

    doy = 174
    temp_xsection = temp_data[:, doy * 4]
    new_grid, temp_xsection_new = interp2grid(depth_data[0:ir], temp_xsection[0:ir], [-0.05,-0.15,-0.25])
    test_datetime = datetime(2022, 6, 23, 0, 0, 0)
    times, datetimes, record, batt_volt_avg, ptemp_avg, soiltemp1_avg, soiltemp2_avg, soiltemp3_avg, \
    stmp_1_d_005_avg, stmp_1_d_015_avg, stmp_1_d_025_avg, stmp_2_d_005_avg, stmp_2_d_015_avg, stmp_2_d_025_avg = soil_temp_data()
    stmp_1s,stmp_2s = closest_soil_temps(test_datetime,datetimes,stmp_1_d_005_avg,stmp_1_d_015_avg,stmp_1_d_025_avg,
                       stmp_2_d_005_avg,stmp_2_d_015_avg,stmp_2_d_025_avg)
    if make_plots == True:
        test = comparison(174,100.0*temp_xsection_new,stmp_1s,np.array([0.05,0.15,0.25]),ax1,ax1)

    if make_plots == True:
        plt.tight_layout()
        #print('plot test')
        plt.savefig('C:/Users/Jeremy/Desktop/cgOutput/palsa_high/plots.png')

    soilOrganic_data_interp = soc_mol_m3

    # Interpolated field measurement data NOT handled by Cryogrid. User can leave these unchanging in time (by default),
    # or can manually evolve them in time). [n_layers x n_Cryogrid_timesteps]
    root_data_interp = np.array(np.zeros((len(layer_thicknesses),ncgt)))
    ph_data_interp = np.array(np.zeros((len(layer_thicknesses),ncgt)))
    for i in range(0, len(layer_thicknesses)): # Layer
        root_data_interp[i,:] = plants.root_xsections(layer_thicknesses[i], depths[i])
        ph_data_interp[i,:] = ph[i]

    return(temp_data_interp, water_data_interp, waterIce_data_interp,
           soilOrganic_data_interp, soilMineral_data_interp,
           air_data_interp,
           ph_data_interp,
           root_data_interp,
           time_data)

def doy_forcing_data(doy,temp_data_interp,water_data_interp,waterIce_data_interp,soilOrganic_data_interp,
                     soilMineral_data_interp,air_data_interp,ph_data_interp,root_data_interp,time_data):

    # Find the consecutive time array indices corresponding to the desired day of year (doy)
    doy_idx1 = np.where(np.floor(time_data)==doy)[0][0]
    doy_idx2 = np.where(np.floor(time_data)==doy)[0][-1]

    # Daily-average of the profiles on the desired day of year
    temp_data_interp_mean = np.mean(temp_data_interp[:, doy_idx1:doy_idx2 + 1], axis=1)
    water_data_interp_mean = np.mean(water_data_interp[:, doy_idx1:doy_idx2 + 1], axis=1)
    waterIce_data_interp_mean = np.mean(waterIce_data_interp[:, doy_idx1:doy_idx2 + 1], axis=1)
    #soilOrganic_data_interp_mean = np.mean(soilOrganic_data_interp[:, doy_idx1:doy_idx2 + 1], axis=1)
    soilOrganic_data_interp_mean = soilOrganic_data_interp
    air_data_interp_mean = np.mean(air_data_interp[:, doy_idx1:doy_idx2 + 1], axis=1)
    ph_data_interp_mean = np.mean(ph_data_interp[:, doy_idx1:doy_idx2 + 1], axis=1)
    root_data_interp_mean = np.mean(root_data_interp[:, doy_idx1:doy_idx2 + 1], axis=1)

    return(temp_data_interp_mean, water_data_interp_mean, waterIce_data_interp_mean, soilOrganic_data_interp_mean,
           air_data_interp_mean, ph_data_interp_mean, root_data_interp_mean)

#temp_data_interp, water_data_interp, waterIce_data_interp, soilOrganic_data_interp, soilMineral_data_interp, \
#air_data_interp, ph_data_interp, root_data_interp, time_data = forcing_data()

#test = doy_forcing_data(237, temp_data_interp, water_data_interp, waterIce_data_interp, soilOrganic_data_interp,
#                        soilMineral_data_interp, air_data_interp, ph_data_interp, root_data_interp, time_data)

#test = forcing_data()

def all_forcing():

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
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/T.txt'), dtype='float64')
    water_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/water.txt'),dtype='float64')
    waterIce_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/waterIce.txt'),dtype='float64')
    ice_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/ice.txt'),dtype='float64')
    soilOrganic_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/soilOrganic.txt'), dtype='float64')
    soilOrganic_data = np.full(np.shape(temp_data), soilOrganic_data)
    soilMineral_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/soilMineral.txt'), dtype='float64')
    soilMineral_data = np.full(np.shape(temp_data), soilMineral_data)
    air_data = 100.0 * np.array(read_csv_array(params.main_directory+
                    'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/air.txt'),dtype='float64')
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
                'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/TIMESTAMP.txt'), dtype='float64'))[0]
    time_data = time_data - time_data[0]

    # Read the Cryogrid depth data, and shift values to place the surface at 0 (meters)
    depth_data = np.array(read_csv_array(main_directory+
                'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/clean_output/z.txt'), dtype='float64')
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

    n_rows, n_cols = np.shape(cg_forcing2['effdiff_data_ch4'])[0], np.shape(cg_forcing2['effdiff_data_ch4'])[1]

    for i in range(0, n_rows):
        for j in range(0, n_cols):
            if 0.0 < cg_forcing2['temp_data'][i, j] < 1.0: cg_forcing2['effdiff_data_ch4'][i, j] = \
                (cg_forcing2['temp_data'][i, j] / 1.0) * cg_forcing2['effdiff_data_ch4'][i, j]

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
                      'conlength'+str(params.reservoir_contact_length)+'m'
        file_string = file_string.replace('.', 'p')

        # Make a plot directory if it doesn't already exist
        if not os.path.exists(main_directory+'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/plots/'):
            os.mkdir(main_directory+'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/plots/')
        # Save the plots
        print('Plotting processed Cryogrid output: ', main_directory+'CryoGrid/CryoGridCommunity_results/'+params.site+
              '/'+params.site+'_'+params.cryogrid_name+'/plots/')
        plt.savefig(main_directory+'CryoGrid/CryoGridCommunity_results/'+params.site+'/'+params.site+'_'+params.cryogrid_name+'/plots/'+file_string+'.png')

    return(cg_forcing2, cg_ceq)


