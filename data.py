import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import params
import conversions
from datetime import datetime
from datetime import timedelta
from datetime import date
import netCDF4 as nc4
from datetime import date
from scipy.interpolate import interp1d
import microbes
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# test 8/21/2024

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


def interp2grid(depths, values, newgrid):
    ynew = np.interp(newgrid, np.flip(depths), np.flip(values))
    #from IPython import embed
    #embed()
    return (newgrid, ynew)


def interp2grid2(depths, values, newgrid):

    ynew = np.interp(newgrid, depths, values)
    #from IPython import embed
    #embed()
    return (newgrid, ynew)


def find_unique_dates(site, data_name, extra):

    # Find all files
    file_list = os.listdir(params.main_directory+'siteData/standard/'+site+'/'+data_name+'/')

    # Find all files belonging to the site
    file_list2 = []
    for f in file_list:
        if site in f:
            file_list2.append(f)

    # Find all unique dates among the files belonging to the site
    if len(extra) == 0:
        unique_dates = np.sort(list(set([i.split('.', 1)[0][-8:] for i in file_list2])))
    else:
        unique_dates = np.sort(list(set([i.split('.', 1)[0][-(8+len(extra)):-len(extra)] for i in file_list2])))

    return(unique_dates)


def get_closest_udate(search_keys, udate):

    big = list(search_keys.keys())
    dates = []
    for b in big:
        if 'VvZ' in b.split('_'):
            dates.append(b.split('_')[-3])
    dates = np.sort(list(set(dates)))
    udate2 = datetime(int(udate[4:8]), int(udate[2:4]), int(udate[0:2]))
    datetimes = []
    for d in dates:
        datetimes.append(datetime(int(d[4:8]), int(d[2:4]), int(d[0:2])))
    closest_datetime = min(datetimes, key=lambda x: abs(x - udate2))
    day, month, year = str(closest_datetime.day), str(closest_datetime.month), str(closest_datetime.year)
    if int(day) < 10:
        day = '0' + day
    if int(month) < 10:
        month = '0' + month
    closest_udate = day + month + year

    return(closest_udate)


def get_unique_units(VvZ_vals):

    units = []
    for VvZ_val in VvZ_vals:
        units.append(str(VvZ_val).split('_')[-2])
    unique_units = np.sort(list(set(units)))

    return(unique_units)


def relabel(label):

    labels1 = ['[kg/m^3]','m3m3','umolL','molm3','mgL','copm3','copg','molm2day']
    labels2 = ['[kg/m$^3$]','[m$^3$/m$^3$]','[umol/L]','[mol/m$^3$]','[mg/L]','[copies/m$^3$]','[copies/g]','[mol/m$^2$/day]']

    revised_label = label
    for i in range(0,len(labels1)):
        if labels1[i] in label:
            revised_label = label[0:label.find(labels1[i])]+'\n'+labels2[i]

    return(revised_label)


def plot_VvT(site_data, model_depths):

    if not os.path.exists(params.main_directory + 'siteData/dataPlots/' + params.site):
        os.mkdir(params.main_directory + 'siteData/dataPlots/' + params.site)

    lstyles = ['solid','dotted','dashed','dashdot']

    for data_dict in list(site_data.keys()):
        dates = []
        data_dict_keys = np.array(list(site_data[data_dict].keys()))

        # Find the keys containing value lists
        output_array = np.zeros_like(data_dict_keys, dtype=bool)
        print(data_dict)
        # 'TypeError: string operation on non-string array' indicates that no data was found for this dictionary.
        val_indices = np.char.find(data_dict_keys, '_VvT')
        output_array[np.where(val_indices != -1)] = True
        VvT_vals = data_dict_keys[output_array]

        # Find the keys containing datetime lists
        output_array = np.zeros_like(data_dict_keys, dtype=bool)
        dtime_indices = np.char.find(data_dict_keys, '_T')
        output_array[np.where(dtime_indices != -1)] = True
        VvT_dtimes = data_dict_keys[output_array]

        #print(VvT_dtimes)

        unique_units = get_unique_units(VvT_vals)

        if len(VvT_vals) > 0:
            if len(unique_units) == 1: # Only one data type or unit
                fig, ax = plt.subplots()
                fig.tight_layout(pad=2.0)
                ax.grid(True)
                unit_indices = np.zeros_like(VvT_vals, dtype=bool)
                index_array = np.char.find(np.array(VvT_vals), unique_units[0])
                unit_indices[np.where(index_array != -1)] = True
                VvT_vals_unit = VvT_vals[unit_indices]
                for v in range(0, len(list(VvT_vals_unit))):
                    date = str(VvT_vals_unit[v]).split('_')[-3]
                    date2 = date[:2] + '-' + date[2:]
                    date3 = date2[:5] + '-' + date2[5:]
                    dates = [dates,date3]
                    #axes[a].set_ylim([-5, 30])
                    ax.set_title(params.site, fontsize=12, fontdict=dict(weight='bold'))
                    t_label, v_label = site_data[data_dict]['T_label'], unique_units[0]
                    #print(unique_units[0])
                    ax.set_xlabel(t_label, fontsize=12, fontdict=dict(weight='bold'))
                    v_label = relabel(v_label)
                    ax.set_ylabel(v_label, fontsize=12, fontdict=dict(weight='bold'))
                    l2 = ax.plot(site_data[data_dict][VvT_dtimes[v]],site_data[data_dict][VvT_vals_unit[v]],markersize=8.0 - v * (8.0 / (len(list(VvT_vals)))), marker='o', linestyle='none')
                    ax.set_xlim([datetime(2022, 1, 1, 0, 0, 0), datetime(2022, 12, 31, 0, 0 ,0)])
                    ax.legend(VvT_vals_unit, framealpha=0.0, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=1, fontsize=8.0)
                l3 = ax.plot([datetime(2022, 1, 1, 0, 0, 0), datetime(2022, 12, 31, 0, 0, 0)], [0.0, 0.0], linestyle='--', color='black')
                plt.xticks(rotation=45)
                plt.savefig(params.main_directory + 'siteData/dataPlots/' + params.site + '/' + data_dict + '.png', bbox_inches='tight')
            else: # Multiple data types or units
                fig, axes = plt.subplots(len(unique_units),1)
                fig.tight_layout(pad=2.0)
                for a in range(0, len(unique_units)):
                    unit_indices = np.zeros_like(VvT_vals, dtype=bool)
                    index_array = np.char.find(np.array(VvT_vals), unique_units[a])
                    unit_indices[np.where(index_array != -1)] = True
                    VvT_vals_unit = VvT_vals[unit_indices]
                    for v in range(0, len(list(VvT_vals_unit))):
                        axes[a].grid(True)
                        date = str(VvT_vals_unit[v]).split('_')[-3]
                        date2 = date[:2] + '-' + date[2:]
                        date3 = date2[:5] + '-' + date2[5:]
                        dates = [dates,date3]
                        #axes[a].set_ylim([-5, 30])
                        axes[0].set_title(params.site, fontsize=12, fontdict=dict(weight='bold'))
                        t_label, v_label = site_data[data_dict]['T_label'], unique_units[a]
                        #print(site_data[data_dict].keys())
                        axes[a].set_xlabel(t_label, fontsize=12, fontdict=dict(weight='bold'))
                        if a > 0: axes[a].set_ylabel(' ')
                        v_label = relabel(v_label)
                        axes[a].set_ylabel(v_label, fontsize=12, fontdict=dict(weight='bold'))
                        l2 = axes[a].plot(range(0,len(site_data[data_dict][VvT_vals_unit[v]])),site_data[data_dict][VvT_vals_unit[v]], markersize=8.0-v*(8.0/(len(list(VvT_vals)))), marker='o', linestyle='none')
                        axes[a].legend(VvT_vals_unit, framealpha=0.0, loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1, fontsize=8.0)
                    plt.savefig(params.main_directory + 'siteData/dataPlots/' + params.site + '/' + data_dict + '.png', bbox_inches='tight')



    return()


def plot_VvZ(site_data, model_depths):

    if not os.path.exists(params.main_directory + 'siteData/dataPlots/' + params.site):
        os.mkdir(params.main_directory + 'siteData/dataPlots/' + params.site)

    lstyles = ['solid','dotted','dashed','dashdot']

    for data_dict in list(site_data.keys()):
        dates = []
        data_dict_keys = np.array(list(site_data[data_dict].keys()))

        output_array = np.zeros_like(data_dict_keys, dtype=bool)
        depth_indices = np.char.find(data_dict_keys, '_Z')
        output_array[np.where(depth_indices != -1)] = True
        VvZ_depths = data_dict_keys[output_array]

        output_array = np.zeros_like(data_dict_keys, dtype=bool)
        depth_indices = np.char.find(data_dict_keys, '_VvZ')
        output_array[np.where(depth_indices != -1)] = True
        VvZ_vals = data_dict_keys[output_array]

        unique_units = get_unique_units(VvZ_vals)

        if len(VvZ_vals) > 0:
            if len(unique_units) == 1:
                fig, ax = plt.subplots()
                fig.set_size_inches(5.0 * len(unique_units), 6, forward=True)
                fig.tight_layout(pad=2.0)
                ax.grid(True)
                for v in range(0, len(list(VvZ_vals))):
                    date = str(VvZ_vals[v]).split('_')[-3]
                    date2 = date[:2] + '-' + date[2:]
                    date3 = date2[:5] + '-' + date2[5:]
                    dates = [dates,date3]
                    ax.set_ylim([0.45, 0.0])
                    ax.set_title(data_dict, fontsize=12, fontdict=dict(weight='bold'))
                    z_label, v_label = site_data[data_dict]['Z_label'], site_data[data_dict]['V_label']
                    ax.set_ylabel(z_label, fontsize=12, fontdict=dict(weight='bold'))
                    v_label = relabel(v_label)
                    ax.set_xlabel(v_label, fontsize=12, fontdict=dict(weight='bold'))
                    l2 = ax.plot(site_data[data_dict][VvZ_vals[v]], model_depths, linewidth=8.0-v*(8.0/(len(list(VvZ_vals)))), linestyle='-')
                ax.legend(VvZ_vals, framealpha=0.0)
                plt.savefig(params.main_directory + 'siteData/dataPlots/' + params.site + '/' + data_dict + '.png', bbox_inches='tight')
            else:
                fig, axes = plt.subplots(1, len(unique_units))
                fig.set_size_inches(3.0 * len(unique_units), 6, forward=True)
                fig.tight_layout(pad=2.0)
                ax.grid(True)
                for a in range(0, len(unique_units)):
                    unit_indices = np.zeros_like(VvZ_vals, dtype=bool)
                    index_array = np.char.find(np.array(VvZ_vals), unique_units[a])
                    unit_indices[np.where(index_array != -1)] = True
                    VvZ_vals_unit = VvZ_vals[unit_indices]
                    for v in range(0, len(list(VvZ_vals_unit))):
                        axes[a].grid(True)
                        date = str(VvZ_vals_unit[v]).split('_')[-3]
                        date2 = date[:2] + '-' + date[2:]
                        date3 = date2[:5] + '-' + date2[5:]
                        dates = [dates,date3]
                        axes[a].set_ylim([0.45, 0.0])
                        axes[a].set_title(params.site, fontsize=12, fontdict=dict(weight='bold'))
                        z_label, v_label = site_data[data_dict]['Z_label'], unique_units[a]
                        axes[a].set_ylabel(z_label, fontsize=12, fontdict=dict(weight='bold'))
                        if a > 0: axes[a].set_ylabel(' ')
                        v_label = relabel(v_label)
                        axes[a].set_xlabel(v_label, fontsize=12, fontdict=dict(weight='bold'), rotation=25)
                        l2 = axes[a].plot(site_data[data_dict][VvZ_vals_unit[v]], model_depths, linewidth=8.0-v*(8.0/(len(list(VvZ_vals)))), linestyle='-')
                        axes[a].legend(VvZ_vals_unit, framealpha=0.0, loc='upper center', bbox_to_anchor=(0.5, -0.30-0.05*a), ncol=1, fontsize=12.0)
                    plt.savefig(params.main_directory + 'siteData/dataPlots/' + params.site + '/' + data_dict + '.png', bbox_inches='tight')


    return()


def read_roots(site, model_depths):

    unique_dates = find_unique_dates(site, 'roots', extra = '')

    site_roots = {}
    for udate in unique_dates:

        # Read the root data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/roots/'+site+'_'+udate+'.txt'
        print('     - Root data: ', filename)
        Z_label = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths = np.array(read_csv_header(filename, 0, 'float', 1))
        roots = np.array(read_csv_header(filename, 1, 'float', 1))

        # Interpolate root data to the model depth grid, populate dictionary
        xnew, site_roots[site + '_' + udate + '_roots_VvZ'] = interp2grid2(depths, roots, model_depths)
        site_roots[site + '_' + udate + '_Z'] = depths
        site_roots['Z_label'], site_roots['V_label'] = Z_label, V_label

    return(site_roots)


def read_temps(site, model_depths):

    unique_dates = find_unique_dates(site, 'temps', extra = '_05cm')

    site_temps = {}
    for udate in unique_dates:

        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/temps/'+site+'_'+udate+'_05cm.txt'
        print('     - 5cm temps: ', filename)
        temps05 = np.array(read_csv_header(filename, 1, 'float',1))

        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/temps/'+site+'_'+udate+'_15cm.txt'
        print('     - 15cm temps: ', filename)
        temps15 = np.array(read_csv_header(filename, 1, 'float',1))

        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/temps/'+site+'_'+udate+'_25cm.txt'
        print('     - 25cm temps: ', filename)
        temps25 = np.array(read_csv_header(filename, 1, 'float',1))

        times = np.array(read_csv_header(filename, 0, 'string',1))

        T_label = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label = np.array(read_csv_header(filename, 1, 'string', 0))[0]

        dtimes = []
        for t in times:
            year, day, month, hh, mm, ss = t[0:4], t[4:6], t[6:8], t[8:10], t[10:12], t[12:14]
            dtimes.append(datetime(int(year), int(month), int(day), int(hh), int(mm), int(ss)))
        dtimes = np.array(dtimes)
        mid_date = dtimes[0] + (dtimes[-1] - dtimes[0]) / 2

        site_temps[site + '_' + udate + '_dtimes'] = dtimes
        site_temps[site + '_' + udate + '_mid_date'] = mid_date

        #interpolated_temp_profiles = []
        #for m in range(0, len(temps05)):
        #
        #    # Interpolate temp data to the model depth grid
        #    xnew, interpolated_temp_profile = interp2grid([0.05,0.15,0.25],[temps05[m],temps15[m],temps25[m]],model_depths)
        #    interpolated_temp_profiles.append(interpolated_temp_profile)

        site_temps[site + '_' + udate + '_temps05_VvT'] = temps05
        site_temps[site + '_' + udate + '_temps15_VvT'] = temps15
        site_temps[site + '_' + udate + '_temps25_VvT'] = temps25

        site_temps['T_label'], site_temps['V_label'] = T_label, V_label
        #site_temps[site + '_' + udate + '_mean_temp_depth'] = np.mean(interpolated_temp_profiles,axis=0)

    return(site_temps)


def read_comp(site, model_depths, model_thicks):

    site_soc, site_min, site_air, site_wat, site_comp, site_watertable = {}, {}, {}, {}, {}, {}

    unique_dates = find_unique_dates(site, 'waterpct', extra='')
    for udate in unique_dates:

        # Read the SOC data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/organicpct/'+site+'_'+udate+'.txt'
        print('     - SOC: ', filename)
        Z_label_soc = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label_soc = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths_soc = np.array(read_csv_header(filename, 0, 'float', 1))         # [m]
        soc_m3_m3 = np.array(read_csv_header(filename, 1, 'float', 1))         # [kg/kg]
        #
        # Interpolate SOC data to the depth grid
        depths_soc, soc_m3_m3 = interp2grid2(depths_soc, soc_m3_m3, model_depths)
        site_soc[site + '_' + udate + '_socm3m3_VvZ'] = soc_m3_m3
        site_soc[site + '_' + udate + '_Z'] = depths_soc
        #
        # SOC volume concentration
        site_comp[site + '_' + udate + '_socm3m3_VvZ'] = site_soc[site + '_' + udate + '_socm3m3_VvZ']

        # Read the MIN data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/mineralpct/'+site+'_'+udate+'.txt'
        print('     - MIN: ', filename)
        Z_label_min = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label_min = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths_min = np.array(read_csv_header(filename, 0, 'float', 1))         # [m]
        min_m3_m3 = np.array(read_csv_header(filename, 1, 'float', 1))         # [kg/kg]
        #
        # Interpolate MIN data to the depth grid
        depths_min, min_m3_m3 = interp2grid2(depths_min, min_m3_m3, model_depths)
        site_min[site + '_' + udate + '_minm3m3_VvZ'] = min_m3_m3
        site_min[site + '_' + udate + '_Z'] = depths_min
        #
        # MIN volume concentration
        site_comp[site + '_' + udate + '_minm3m3_VvZ'] = site_min[site + '_' + udate + '_minm3m3_VvZ']

        # Read the AIR data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/airpct/'+site+'_'+udate+'.txt'
        print('     - AIR: ', filename)
        Z_label_air = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label_air = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths_air = np.array(read_csv_header(filename, 0, 'float', 1))         # [m]
        air_m3_m3 = np.array(read_csv_header(filename, 1, 'float', 1))         # [kg/kg]
        #
        # Interpolate AIR data to the depth grid
        depths_air, air_m3_m3 = interp2grid2(depths_air, air_m3_m3, model_depths)
        site_air[site + '_' + udate + '_airm3m3_VvZ'] = air_m3_m3
        site_air[site + '_' + udate + '_Z'] = depths_air
        #
        # AIR volume concentration
        site_comp[site + '_' + udate + '_airm3m3_VvZ'] = site_air[site + '_' + udate + '_airm3m3_VvZ']

        # Read the WATER data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/waterpct/'+site+'_'+udate+'.txt'
        print('     - WATER: ', filename)
        Z_label_water = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label_water = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths_water = np.array(read_csv_header(filename, 0, 'float', 1))         # [m]
        water_m3_m3 = np.array(read_csv_header(filename, 1, 'float', 1))         # [kg/kg]
        #
        # Interpolate WATER data to the depth grid
        depths_water, water_m3_m3 = interp2grid2(depths_water, water_m3_m3, model_depths)
        site_wat[site + '_' + udate + '_watm3m3_VvZ'] = water_m3_m3
        site_wat[site + '_' + udate + '_Z'] = depths_water
        #
        # WATER volume concentration
        site_comp[site + '_' + udate + '_watm3m3_VvZ'] = site_wat[site + '_' + udate + '_watm3m3_VvZ']

        # Sum of soil components should = 1
        print('     - Total derived from the sum of all quantities')
        site_comp[site + '_' + udate + '_total_VvZ'] = site_comp[site + '_' + udate + '_watm3m3_VvZ'] + \
                                                       site_comp[site + '_' + udate + '_airm3m3_VvZ'] + \
                                                       site_comp[site + '_' + udate + '_socm3m3_VvZ'] + \
                                                       site_comp[site + '_' + udate + '_minm3m3_VvZ']

        # Convert SOC m3/m3 to: mol/m3 and mol/m2, for the chemistry dict, and for the DOC = f(SOC kg/m2) calculation
        #
        soc_mol_m3 = conversions.kg2mol(1400.0 * soc_m3_m3, 'c')
        site_soc[site + '_' + udate + '_socmolm3_VvZ'] = soc_mol_m3
        #
        # SOC molar mass concentration
        site_comp[site + '_' + udate + '_socmolm3_VvZ'] = site_soc[site + '_' + udate + '_socmolm3_VvZ']
        #
        soc_mol_m2 = np.sum(model_thicks * soc_mol_m3)
        site_soc[site + '_' + udate + '_socmolm2_V'] = soc_mol_m2
        #
        # SOC column-integrated mass concentration
        site_comp[site + '_' + udate + '_socmolm2_V'] = site_soc[site + '_' + udate + '_socmolm2_V']

    unique_dates = find_unique_dates(site, 'watertable', extra='')
    for udate in unique_dates:

        # Read the WATER TABLE data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/watertable/'+site+'_'+udate+'.txt'
        print('     - WATER TABLE: ', filename)
        Z_label_water = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label_water = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths_watertable = np.array(read_csv_header(filename, 0, 'float', 1))         # [m]
        water_table = np.array(read_csv_header(filename, 1, 'float', 1))         # [kg/kg]
        water_table_depth = depths_watertable[np.where(water_table == 1.)[0][0]]
        #
        # Record the WATER TABLE depth
        site_watertable[site + '_' + udate + '_watertable_VvZ'] = water_table_depth
        site_watertable[site + '_' + udate + '_Z'] = depths_water
        #
        # WATER volume concentration
        site_comp[site + '_' + udate + '_watertable_VvZ'] = site_watertable[site + '_' + udate + '_watertable_VvZ']

    # Average the column-integrated SOC molar mass concentrations, across all available profiles
    # This produces one representative 'cpoolmolm2' factor for the site
    socmolm2_keys = [s for s in list(site_soc.keys()) if 'socmolm2' in s]
    socmolm2_vals = []
    for k in socmolm2_keys:
        socmolm2_vals.append(site_soc[k])
    site_soc[site + '_cpoolmolm2_V'] = np.mean(np.array(socmolm2_vals))
    site_comp[site + '_cpoolmolm2_V'] = site_soc[site + '_cpoolmolm2_V']

    Z_label_comp, V_label_comp = Z_label_water, 'Volumetric %'
    site_comp['Z_label'], site_comp['V_label'] = Z_label_comp, V_label_comp

    return(site_comp)


def read_mcra(site, model_depths):

    unique_dates = find_unique_dates(site, 'mcra', extra='')

    site_mcra = {}
    for udate in unique_dates:

        # Read the mcra data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/mcra/'+site+'_'+udate+'.txt'
        print('     - mcrA: ', filename)
        Z_label = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths = np.array(read_csv_header(filename, 0, 'float', 1))
        mcras = np.array(read_csv_header(filename, 1, 'float', 1)) # copies/g

        # Interpolate gene data to the model depth grid
        depths, site_mcra[site + '_' + udate + '_mcracopg_VvZ'] = interp2grid2(depths,mcras,model_depths)

        site_mcra[site + '_' + udate + '_Z'] = depths
        site_mcra['Z_label'], site_mcra['V_label'] = Z_label, V_label

    return(site_mcra)


def read_pmoa1(site, model_depths):

    unique_dates = find_unique_dates(site, 'pmoa1', extra='')

    site_pmoa1 = {}
    for udate in unique_dates:

        # Read the pmoa1 data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/pmoa1/'+site+'_'+udate+'.txt'
        print('     - pmoa1: ', filename)
        Z_label = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths = np.array(read_csv_header(filename, 0, 'float', 1))
        pmoa1s = np.array(read_csv_header(filename, 1, 'float', 1)) # copies/g

        # Interpolate gene data to the model depth grid
        depths, site_pmoa1[site + '_' + udate + '_pmoa1copg_VvZ'] = interp2grid2(depths,pmoa1s,model_depths)

        site_pmoa1[site + '_' + udate + '_Z'] = depths
        site_pmoa1['Z_label'], site_pmoa1['V_label'] = Z_label, V_label

    return(site_pmoa1)


def read_pmoa2(site, model_depths):

    unique_dates = find_unique_dates(site, 'pmoa2', extra='')

    site_pmoa2 = {}
    for udate in unique_dates:

        # Read the pmoa2 data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/pmoa2/'+site+'_'+udate+'.txt'
        print('     - pmoa2: ', filename)
        Z_label = np.array(read_csv_header(filename, 0, 'string', 0))[0]
        V_label = np.array(read_csv_header(filename, 1, 'string', 0))[0]
        depths = np.array(read_csv_header(filename, 0, 'float', 1))
        pmoa2s = np.array(read_csv_header(filename, 1, 'float', 1)) # copies/g

        # Interpolate gene data to the model depth grid
        depths, site_pmoa2[site + '_' + udate + '_pmoa2copg_VvZ'] = interp2grid2(depths,pmoa2s,model_depths)

        site_pmoa2[site + '_' + udate + '_Z'] = depths
        site_pmoa2['Z_label'], site_pmoa2['V_label'] = Z_label, V_label

    return(site_pmoa2)


def read_albedo(site, model_depths):

    unique_dates = find_unique_dates(site, 'albedo', extra='')

    site_albedo = {}
    for udate in unique_dates:

        # Read the albedo data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/albedo/'+site+'_'+udate+'.txt'
        print('     - Albedo: ', filename)
        albedo = float(np.array(read_csv_header(filename, 0, 'string', 1))[0])

        # Populate the dictionary containing albedo data (site_albedo)
        site_albedo[site + '_' + udate + '_albedo'] = albedo

    return(site_albedo)


def read_elevation(site, model_depths):

    unique_dates = find_unique_dates(site, 'elevation', extra='')

    site_elevation = {}
    for udate in unique_dates:

        # Read the albedo data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/elevation/'+site+'_'+udate+'.txt'
        print('     - Elevation: ', filename)
        elevation = float(np.array(read_csv_header(filename, 0, 'string', 1))[0])

        # Populate the dictionary containing elevation data (site_elevation)
        site_elevation[site + '_' + udate + '_elevation'] = elevation

    return(site_elevation)


def read_latitude(site, model_depths):

    unique_dates = find_unique_dates(site, 'latitude', extra='')

    site_latitude = {}
    for udate in unique_dates:

        # Read the latitude data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/latitude/'+site+'_'+udate+'.txt'
        print('     - Latitude: ', filename)
        latitude = float(np.array(read_csv_header(filename, 0, 'string', 1))[0])

        # Populate the dictionary containing latitude data (site_latitude)
        site_latitude[site + '_' + udate + '_latitude'] = latitude

    return(site_latitude)


def read_longitude(site, model_depths):

    unique_dates = find_unique_dates(site, 'longitude', extra='')

    site_longitude = {}
    for udate in unique_dates:

        # Read the longitude data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/'+site+'/longitude/'+site+'_'+udate+'.txt'
        print('     - Longitude: ', filename)
        longitude = float(np.array(read_csv_header(filename, 0, 'string', 1))[0])

        # Populate the dictionary containing longitude data (site_longitude)
        site_longitude[site + '_' + udate + '_longitude'] = longitude

    return(site_longitude)


def read_fluxes(site, model_depths):

    unique_dates = find_unique_dates(site, 'fluxes', extra='')

    site_fluxes = {}
    for udate in unique_dates:

        # Read the flux data
        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/' + site + '/fluxes/' + site + '_' + udate + '.txt'
        print('     - Fluxes: ', filename)
        dates = np.array(read_csv_header(filename, 0, 'string', 1))
        fluxes = np.array(read_csv_header(filename, 1, 'float', 1))

        mean_flux = np.mean(abs(fluxes))
        dates2 = []
        fluxes2 = []
        for d in range(0,len(dates)):
            if abs(fluxes[d]) < 2.0*mean_flux:
                year, day, month, hour, minute, second = int(dates[d][0:4]), int(dates[d][4:6]), int(dates[d][6:8]), int(dates[d][8:10]), int(dates[d][10:12]), int(dates[d][12:14])
                dates2.append(datetime(year, month, day, hour, minute, second))
                fluxes2.append(fluxes[d])

        # Populate the dictionary containing flux data (site_fluxes)
        site_fluxes['T_label'] = 'Date'
        site_fluxes[site + '_' + udate + '_T'] = dates2
        site_fluxes[site + '_' + udate + '_fluxesmolm2day_VvT'] = fluxes2

    return(site_fluxes)


def read_site_data(site, model_depths, model_thicks):

    # Choose which data to read and store it in a dictionary (site_data)

    print('Reading standardized site data: ', 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/standard/<site>/<data>/<site>_<udate>.txt')

    site_data = {}

    site_data['site_roots'] = read_roots(site, model_depths)
    site_data['site_temps'] = read_temps(site, model_depths)
    site_data['site_comp'] = read_comp(site, model_depths, model_thicks) # Thicknesses needed for SOC unit conversions
    site_data['site_mcra'] = read_mcra(site, model_depths)
    site_data['site_pmoa1'] = read_pmoa1(site, model_depths)
    site_data['site_pmoa2'] = read_pmoa2(site, model_depths)
    site_data['site_albedo'] = read_albedo(site, model_depths)
    site_data['site_elevation'] = read_elevation(site, model_depths)
    site_data['site_latitude'] = read_latitude(site, model_depths)
    site_data['site_longitude'] = read_longitude(site, model_depths)
    site_data['site_fluxes'] = read_fluxes(site, model_depths)

    return(site_data)




