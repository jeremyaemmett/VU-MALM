import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import text
import matplotlib.dates as mdates
import params
from copy import copy
import pathways
import csv
import init
import conversions
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from datetime import timedelta
from datetime import date

def read_csv_array(filename):

    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    return(data)

def profile_plots(axis,data,tempdata,levels,xlabel,timedata,depthdata,color_scheme,fig):

    axis.set_xlim([121,319])

    X, Y = np.meshgrid(timedata, depthdata)
    data2 = copy(data)
    ###data2[tempdata < 0.0] = np.nan
    c = axis.contourf(X, Y, data2, levels=levels, cmap=color_scheme)
    axis.set_ylim([-0.5,0.0])
    axis.set_xlabel('Day of Year',fontsize=28)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('bottom', size='5%', pad=1.2)
    cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(axis="both",labelsize=24,rotation=45)
    cbar.set_label(xlabel, rotation=0,fontsize=28,fontweight="bold")
    axis.xaxis.set_tick_params(labelsize=26)
    axis.yaxis.set_tick_params(labelsize=26)
    axis.set_ylabel('Depth (m)',fontsize=28)
    #axis.set_xticklabels(['        May','        Jun','        Jul','        Aug','        Sep','        Oct',' '])

    c = axis.contour(X, Y, tempdata, linestyles='dashed', linewidths=2.0, levels=[-400,0.0],colors='black')
    return()

make_plots = False

### Cryogrid output

# Read the Cryogrid output into arrays with dimension [cg-depths, cg_times]
temp_data = np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\T.txt'), dtype='float64')
water_data = 100.0 * np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\water.txt'), dtype='float64')
waterIce_data = 100.0 * np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\waterIce.txt'),dtype='float64')
ice_data = 100.0 * np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\ice.txt'), dtype='float64')
soilOrganic_data = 100.0 * np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\soilOrganic.txt'),dtype='float64')
soilOrganic_data = np.full(np.shape(temp_data), soilOrganic_data)
soilMineral_data = 100.0 * np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\soilMineral.txt'),dtype='float64')
soilMineral_data = np.full(np.shape(temp_data), soilMineral_data)
air_data = 100.0 * np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\air.txt'), dtype='float64')
air_data = np.full(np.shape(temp_data), air_data)

# Sum the arrays to verify that soil composition adds to 1.0 (100%)
ssum_data = ice_data + soilOrganic_data + soilMineral_data + air_data

# Read the Cryogrid time data, and shift values to start at 0 (days)
time_data = (np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\TIMESTAMP.txt'), dtype='float64'))[0]
time_data = time_data - time_data[0]

# Read the Cryogrid depth data, and shift values to place the surface at 0 (meters)
depth_data = np.array(read_csv_array(r'C:\Users\Jeremy\Desktop\cgOutput\palsa_high\z.txt'), dtype='float64')
depth_data = depth_data - depth_data[0]
depth_data = (np.transpose(depth_data))[0]
depth_data = depth_data[0:-1]

if make_plots == True:
    # Configure plot panels
    fig, axes = plt.subplots(3,3)
    fig.patch.set_facecolor('ivory')
    fig.set_size_inches(20.0, 20.0, forward=True)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[0,2]
    ax4 = axes[1,0]
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
    test = profile_plots(ax1, temp_data, temp_data, np.linspace(-10, 18, 15), 'Temp [\u00b0C]', time_data, depth_data,cm.rainbow, fig)
    test = profile_plots(ax2,waterIce_data,temp_data,np.linspace(0, 100, 21),'waterIce [%]',time_data,depth_data,cm.rainbow,fig)
    test = profile_plots(ax3,100.0 - (soilOrganic_data + soilMineral_data), temp_data, np.linspace(75, 100, 21), 'Porosity [%]', time_data, depth_data, cm.rainbow, fig)
    test = profile_plots(ax4,soilOrganic_data,temp_data,np.linspace(0, 20, 21),'soilOrganic [%]',time_data,depth_data,cm.rainbow,fig)
    test = profile_plots(ax5,soilMineral_data,temp_data,np.linspace(0, 2, 21),'soilMineral [%]',time_data,depth_data,cm.rainbow,fig)
    test = profile_plots(ax6,air_data,temp_data,np.linspace(0, 100, 21),'air [%]',time_data,depth_data,cm.rainbow,fig)
    test = profile_plots(ax7,water_data,temp_data,np.linspace(0, 100, 21),'water [%]',time_data,depth_data,cm.rainbow,fig)
    test = profile_plots(ax8, ice_data, temp_data, np.linspace(0, 100, 21), 'ice [%]', time_data, depth_data,cm.rainbow, fig)
    test = profile_plots(ax9, waterIce_data + soilOrganic_data + soilMineral_data + air_data, temp_data, np.linspace(50, 150, 21), 'waterIce + soilOrganic + \n soilMineral + air [%]', time_data, depth_data,cm.rainbow, fig)

    if make_plots == True:
        plt.tight_layout()
        plt.savefig('C:/Users/Jeremy/Desktop/cgOutput/palsa_high/plots.png')