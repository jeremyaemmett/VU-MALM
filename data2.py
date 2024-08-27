import numpy as np
import matplotlib.pyplot as plt
import csv
import params
from datetime import datetime
from datetime import timedelta
from datetime import date
import netCDF4 as nc4
from datetime import date
from scipy.interpolate import interp1d
import microbes
import forcing
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

def read_csv(filename, column_idx, var_type):
    with open(filename) as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)
        header3 = next(reader)
        header4 = next(reader)
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

def soil_temp_data():

    filename = 'C:/Users/Jeremy/Desktop/data/CR1000_C1_C2.dat'

    times = np.array(read_csv(filename, 0, 'string'))
    for i in range(0, len(times)):
        t_test = times[i]
        times[i] = t_test[0:4] + t_test[5:7] + t_test[8:10] + t_test[11:13] + t_test[14:16] + t_test[17:19]
    datetimes = [datetime.strptime(str(int(x)), '%Y%m%d%H%M%S') for x in times]
    record = np.array(read_csv(filename, 1, 'integer'))
    batt_volt_avg = np.array(read_csv(filename, 2, 'float'))
    ptemp_avg = np.array(read_csv(filename, 3, 'float'))
    soiltemp1_avg = np.array(read_csv(filename, 4, 'float'))
    soiltemp2_avg = np.array(read_csv(filename, 5, 'float'))
    soiltemp3_avg = np.array(read_csv(filename, 6, 'float'))
    stmp_1_d_005_avg = np.array(read_csv(filename, 10, 'float'))
    stmp_1_d_015_avg = np.array(read_csv(filename, 11, 'float'))
    stmp_1_d_025_avg = np.array(read_csv(filename, 12, 'float'))
    stmp_2_d_005_avg = np.array(read_csv(filename, 19, 'float'))
    stmp_2_d_015_avg = np.array(read_csv(filename, 20, 'float'))
    stmp_2_d_025_avg = np.array(read_csv(filename, 21, 'float'))

    return(times,datetimes,record,batt_volt_avg,ptemp_avg,soiltemp1_avg,soiltemp2_avg,soiltemp3_avg,
           stmp_1_d_005_avg,stmp_1_d_015_avg,stmp_1_d_025_avg,stmp_2_d_005_avg,stmp_2_d_015_avg,stmp_2_d_025_avg)

def era5_soil_temp_data():

    nc = nc4.Dataset(r'C:\Users\Jeremy\Desktop\churchill\era5_1.nc', 'r')

    lon = nc.variables['longitude'][:]
    lat = nc.variables['latitude'][:]
    time = nc.variables['time'][:]
    skin_temp = nc.variables['skt'][:]
    layer1_temp = nc.variables['stl1']
    layer2_temp = nc.variables['stl2']
    layer3_temp = nc.variables['stl3']
    layer4_temp = nc.variables['stl4']
    layer1_temps = layer1_temp[:, 0, 1] - 273.15
    layer2_temps = layer2_temp[:, 0, 1] - 273.15
    layer3_temps = layer3_temp[:, 0, 1] - 273.15
    layer4_temps = layer4_temp[:, 0, 1] - 273.15
    x = datetime(1900, 1, 1)
    date_times = []
    for i in range(len(time)):
        date_times.append(x + timedelta(hours=int(time[i])))

    return(lon,lat,time,date_times,skin_temp,layer1_temps,layer2_temps,layer3_temps,layer4_temps)


def soil_moist_data():

    filename = 'C:/Users/Jeremy/Desktop/data/CR1000_C1_C2.dat'

    times = np.array(read_csv(filename, 0, 'string'))
    for i in range(0, len(times)):
        t_test = times[i]
        times[i] = t_test[0:4] + t_test[5:7] + t_test[8:10] + t_test[11:13] + t_test[14:16] + t_test[17:19]
    datetimes = [datetime.strptime(str(int(x)), '%Y%m%d%H%M%S') for x in times]
    record = np.array(read_csv(filename, 1, 'integer'))
    batt_volt_avg = np.array(read_csv(filename, 2, 'float'))
    ptemp_avg = np.array(read_csv(filename, 3, 'float'))
    swct_1_d_005_avg = np.array(read_csv(filename, 7, 'float'))
    swct_1_d_015_avg = np.array(read_csv(filename, 8, 'float'))
    swct_1_d_025_avg = np.array(read_csv(filename, 9, 'float'))
    swct_2_d_005_avg = np.array(read_csv(filename, 16, 'float'))
    swct_2_d_015_avg = np.array(read_csv(filename, 17, 'float'))
    swct_2_d_025_avg = np.array(read_csv(filename, 18, 'float'))

    return(times,datetimes,record,batt_volt_avg,ptemp_avg,
           swct_1_d_005_avg,swct_1_d_015_avg,swct_1_d_025_avg,swct_2_d_005_avg,swct_2_d_015_avg,swct_2_d_025_avg)

def air_temp_data():

    nc = nc4.Dataset(r'C:\Users\Jeremy\Desktop\2m_2022_era5.nc', 'r')

    lon = nc.variables['longitude'][:]
    lat = nc.variables['latitude'][:]
    time = nc.variables['time'][:]
    temp = nc.variables['t2m'][:]
    temps = temp[:, 0, 1] - 273.15

    x = datetime(1900, 1, 1)
    date_times = []
    for i in range(len(time)):
        date_times.append(x + timedelta(hours=int(time[i])))

    return(lon,lat,time,temp,temps,date_times)


def closest_soil_temps(test_datetime,datetimes,stmp_1_d_005_avg,stmp_1_d_015_avg,stmp_1_d_025_avg,
                       stmp_2_d_005_avg,stmp_2_d_015_avg,stmp_2_d_025_avg):

    dts = np.array([abs(test_datetime - x) for x in datetimes])
    closest_index = (np.where(dts == min(dts)))[0][0]
    stmp_1s = [stmp_1_d_005_avg[closest_index], stmp_1_d_015_avg[closest_index], stmp_1_d_025_avg[closest_index]]
    stmp_2s = [stmp_2_d_005_avg[closest_index], stmp_2_d_015_avg[closest_index], stmp_2_d_025_avg[closest_index]]

    return(stmp_1s,stmp_2s)


def closest_model_soil_temps(test_datetime,depths):

    f = r'C:/Users/Jeremy/Desktop/data/temp_test.txt'

    # Find the closest file date
    model_dates = read_csv_header(f, 0, 'string', 0)
    model_datetimes = [datetime.strptime(str(int(x)), '%Y%m%d%H%M%S') for x in model_dates]
    dts = np.array([abs(test_datetime - x) for x in model_datetimes])
    closest_index = (np.where(dts == min(dts)))[0][0]

    # Read across the row and append values
    model_temps = []
    for c in range(1, len(depths)+1):
        val = (read_csv_header(f, c, 'float', closest_index))[0]
        model_temps.append(val)
    model_temps = np.array(model_temps)

    return(model_temps)


def closest_era5_soil_temps(test_datetime,datetimes,layer1_temps,layer2_temps,layer3_temps,layer4_temps):

    dts = np.array([abs(test_datetime - x) for x in datetimes])
    closest_index = (np.where(dts == min(dts)))[0][0]
    layer_temps = [layer1_temps[closest_index], layer2_temps[closest_index],
               layer3_temps[closest_index], layer4_temps[closest_index]]

    return(layer_temps)


def closest_soil_moist(test_datetime,datetimes,swct_1_d_005_avg,swct_1_d_015_avg,swct_1_d_025_avg,
                       swct_2_d_005_avg,swct_2_d_015_avg,swct_2_d_025_avg):

    dts = np.array([abs(test_datetime - x) for x in datetimes])
    closest_index = (np.where(dts == min(dts)))[0][0]
    swct_1s = [swct_1_d_005_avg[closest_index], swct_1_d_015_avg[closest_index], swct_1_d_025_avg[closest_index]]
    swct_2s = [swct_2_d_005_avg[closest_index], swct_2_d_015_avg[closest_index], swct_2_d_025_avg[closest_index]]

    return(swct_1s,swct_2s)

def closest_air_temp(test_datetime,datetimes,air_temps):

    dts = np.array([abs(test_datetime - x) for x in datetimes])
    closest_index = (np.where(dts == min(dts)))[0][0]
    air_temp = air_temps[closest_index]

    return(air_temp)

def layer_interpolation_sentek(depths,vals):
    y = vals
    x = np.array([0.0,5.0,15.0,25.0])
    f = interp1d(x,y,fill_value='extrapolate')
    xnew = 100.0*depths
    return(f(xnew))


def layer_interpolation_era5(depths,vals):
    y = vals
    x = np.array([0.0,3.5,17.5,64.0,194.5])
    f = interp1d(x,y,fill_value='extrapolate')
    xnew = 100.0*depths
    return(f(xnew))


def get_closest_data(test_datetime,depths):

    temps = closest_soil_temps(test_datetime,(soil_temp_data())[1],(soil_temp_data())[8],(soil_temp_data())[9],
                          (soil_temp_data())[10],(soil_temp_data())[11],(soil_temp_data())[12],(soil_temp_data())[13])
    moist = closest_soil_moist(test_datetime,(soil_moist_data())[1],(soil_moist_data())[5],(soil_moist_data())[6],
                          (soil_moist_data())[7],(soil_moist_data())[8],(soil_moist_data())[9],(soil_moist_data())[10])
    air = closest_air_temp(test_datetime,(air_temp_data())[5],(air_temp_data())[4])
    era5_temps = closest_era5_soil_temps(test_datetime,(era5_soil_temp_data())[3],
                          (era5_soil_temp_data())[5],(era5_soil_temp_data())[6],
                          (era5_soil_temp_data())[7],(era5_soil_temp_data())[8])

    interp_temps1 = layer_interpolation_sentek(depths,np.concatenate([np.array([air]),np.array(temps[0])]))
    interp_temps2 = layer_interpolation_sentek(depths,np.concatenate([np.array([air]),np.array(temps[1])]))
    interp_moist1 = layer_interpolation_sentek(depths,np.concatenate([np.array([moist[0][0]]),np.array(moist[0])]))
    interp_moist2 = layer_interpolation_sentek(depths,np.concatenate([np.array([moist[1][0]]),np.array(moist[1])]))
    interp_era5_temps = layer_interpolation_era5(depths,np.concatenate([np.array([air]),np.array(era5_temps)]))

    return(interp_temps1,interp_temps2,interp_moist1,interp_moist2,interp_era5_temps)


def read_site_data(site, model_depths):

    site_roots = forcing.read_roots(params.site, model_depths)
    site_temps = forcing.read_temps(params.site, model_depths)
    site_comp = forcing.read_comp(params.site, model_depths)
    site_o2pct = forcing.read_o2pct(params.site, site_comp, model_depths)
    site_o2umolL = forcing.read_o2umolL(params.site, site_comp, model_depths)
    site_doc = forcing.read_doc(params.site, site_comp, model_depths)
    site_mcra = forcing.read_mcra(params.site, site_comp, model_depths)
    site_pmoa1 = forcing.read_pmoa1(params.site, site_comp, model_depths)
    site_pmoa2 = forcing.read_pmoa2(params.site, site_comp, model_depths)
    site_albedo = forcing.read_albedo(params.site, model_depths)
    site_geo = forcing.read_geo(params.site, model_depths)
    site_fluxes = forcing.read_fluxes(params.site, model_depths)

    stop

    return(site_roots, site_comp, site_temps, site_o2pct, site_o2umolL, site_doc, site_genes, site_albedo, site_fluxes,
           site_geo, site_mcra, site_pmoa1, site_pmoa2)




