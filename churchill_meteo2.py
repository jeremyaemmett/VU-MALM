import os
import csv
import random
import shutil
import numpy as np
import pandas as pd
from os import listdir
from matplotlib import cm
from datetime import date
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from os.path import isfile, join


def era5_mm_day_hourly_uncorrected2corrected(mat_times, era5_mm_day_hourly_uncorrected, monthly_ratios):

    date_times = []
    for d in range(0, len(mat_times)):
        matlab_datenum = mat_times[d]
        dtime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
        date_times.append(dtime)

    era5_mm_day_hourly_corrected = np.copy(era5_mm_day_hourly_uncorrected)
    for i in range(0, len(date_times)):
        month_idx = pd.to_datetime(date_times[i]).month - 1
        year = pd.to_datetime(date_times[i]).year
        era5_mm_day_hourly_corrected[i] = monthly_ratios[month_idx] * era5_mm_day_hourly_uncorrected[i]
    #stop

    return (era5_mm_day_hourly_corrected, date_times)


def churchillua_mm_day_daily2monthly_mm(times_daily, mm_day_daily):

    date_times = []
    for d in range(0, len(date_collection)):
        dtime = pd.to_datetime(times_daily[d])
        date_times.append(dtime)
    mm_day_daily_dataframe = pd.DataFrame({'times': date_times, 'values': mm_day_daily})
    mm_day_monthlysum_dataframe = mm_day_daily_dataframe.resample('M', on='times').sum()
    new_datetimes = np.array(pd.to_datetime((mm_day_monthlysum_dataframe.index.tolist())))
    mm_day_monthlysum = np.array(mm_day_monthlysum_dataframe['values'])

    return(mm_day_monthlysum, new_datetimes)


def monthlysums2monthlymeans(monthly_datetimes, monthly_sums):

    monthlysum_stack = [[], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(0, len(monthly_datetimes)):
        month_idx = pd.to_datetime(monthly_datetimes[i]).month
        year = pd.to_datetime(monthly_datetimes[i]).year
        monthlysum_stack[month_idx - 1].append(monthly_sums[i])

    monthlysum_monthlymeans = []
    for i in range(0, 12):
        monthlysum_monthlymeans.append(np.mean(monthlysum_stack[i]))

    return(monthlysum_monthlymeans, monthlysum_stack)


def format_axis1(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xlabel(' ')
    ax.set_facecolor('whitesmoke')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5, axis='y')

    return()

def format_axis2(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xlabel(' ')
    ax.set_facecolor('whitesmoke')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    return()

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

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

def mm_day_hourly2daily_mm(mat_times, mm_day_hourly):

    mm_hr_hourly = (1. / 24.) * mm_day_hourly
    date_times = []
    for d in range(0, len(mat_times)):
        matlab_datenum = mat_times[d]
        dtime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
        date_times.append(dtime)

    mm_hr_hourly_dataframe = pd.DataFrame({'times': date_times, 'values': mm_hr_hourly})
    mm_hr_dailysum_dataframe = mm_hr_hourly_dataframe.resample('D', on='times').sum()
    new_datetimes = np.array(pd.to_datetime((mm_hr_dailysum_dataframe.index.tolist())))
    mm_hr_dailysum = np.array(mm_hr_dailysum_dataframe['values'])

    return(mm_hr_dailysum, new_datetimes)


def mm_day_hourly2monthly_mm(mat_times, mm_day_hourly):

    mm_hr_hourly = (1. / 24.) * mm_day_hourly
    date_times = []
    for d in range(0, len(mat_times)):
        matlab_datenum = mat_times[d]
        dtime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
        date_times.append(dtime)

    mm_hr_hourly_dataframe = pd.DataFrame({'times': date_times, 'values': mm_hr_hourly})
    print(mm_hr_hourly_dataframe)
    mm_hr_dailysum_dataframe = mm_hr_hourly_dataframe.resample('M', on='times').sum()
    new_datetimes = np.array(pd.to_datetime((mm_hr_dailysum_dataframe.index.tolist())))
    mm_hr_dailysum = np.array(mm_hr_dailysum_dataframe['values'])

    return(mm_hr_dailysum, new_datetimes)

#file = 'C:/Users/Jeremy/Desktop/churchill_meteo/en_climate_hourly_MB_5060605_01-2022_P1H.csv'

#rain_mm_hr = np.array(read_csv_header(filename, 0, 'float', 1))

#path = "C:/Users/Jeremy/Desktop/churchill_meteo/CHURCHILL_UA/"
#filenames = find_csv_filenames(path)
#for name in filenames:
#  year = (read_csv_header(path + name, 5, 'integer', 1))[0]
#  month = (read_csv_header(path + name, 6, 'integer', 1))[0]
#  new_name = str(year) + '_daily_meteo.csv'
#  dest = shutil.copyfile(path + name, path + 'renamed/' + new_name)

fig = plt.figure(figsize=(12.0, 8.0))
gsc = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1,1,1])
ax1 = fig.add_subplot(gsc[0, 0])
ax2 = fig.add_subplot(gsc[1, 0])
ax3 = fig.add_subplot(gsc[0, 1])
ax4 = fig.add_subplot(gsc[1, 1])
ax5 = fig.add_subplot(gsc[0, 2])
ax6 = fig.add_subplot(gsc[1, 2])
ax7 = fig.add_subplot(gsc[2, 0])
ax8 = fig.add_subplot(gsc[2, 1])
ax9 = fig.add_subplot(gsc[2, 2])
ax1.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax1.set_ylim(0, 40)
ax2.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax2.set_ylim(0, 40)
ax4.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax7.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax8.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax7.set_ylim(0, 40)
ax4.set_ylim(0, 150)
ax5.set_ylim(0, 80)
ax6.set_ylim(0, 80)
ax9.set_ylim(0, 80)
ax1.set_title('mm/day')
ax1.set_ylabel('CHURCHILL_UA')
ax2.set_ylabel('mm/day')
ax2.set_ylabel('ERA5')
ax5.set_ylabel('mm/day')
ax7.set_ylabel('ERA5 Corrected')
ax5.set_title('mm/month (mean)')
ax3.set_title('mm/month')
ax3.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax3.set_ylim(0,130)
ax4.set_ylim(0,130)
ax8.set_ylim(0,130)
#ax4.set_ylabel('CHURCHILL_UA Snow [mm/day]')
#ax4.set_xlabel('ERA5 Snow [mm/day]')
#ax3.set_aspect('equal')
#ax4.set_aspect('equal')
format_axis1(ax1)
format_axis1(ax2)
format_axis1(ax3)
format_axis1(ax4)
format_axis1(ax5)
format_axis1(ax6)
format_axis1(ax7)
format_axis1(ax8)
format_axis1(ax9)

path = "C:/Users/Jeremy/Desktop/churchill_meteo/CHURCHILL_UA/"
filenames = find_csv_filenames(path)
rain_collection, snow_collection, date_collection = [], [], []
for name in filenames:

    rain = np.array(read_csv_header(path + name, 19, 'float', 1)) # mm/day
    snow = np.array(read_csv_header(path + name, 21, 'float', 1)) # cm/day

    rain[rain < 0.0] = 0.0
    snow[snow < 0.0] = 0.0

    rain_collection.append(rain)
    snow_collection.append(snow)

    year = read_csv_header(path + name, 5, 'integer', 1)
    month = read_csv_header(path + name, 6, 'integer', 1)
    day = read_csv_header(path + name, 7, 'integer', 1)

    date_time = []
    for i in range(0, len(year)):
        date_time.append(date(year[i], month[i], day[i]))
    date_collection.append(date_time)

    #ax1.plot(date_time, rain, color = 'salmon', alpha = 0.5)
    #ax1.plot(date_time, snow, color = 'cornflowerblue', alpha = 0.5)

rain_collection = np.concatenate(rain_collection) # mm/day (daily accumulated)
snow_collection = np.concatenate(snow_collection) # mm/day (daily accumulated)
date_collection = np.concatenate(date_collection) # Day of Year

kernel = 14
ax1.plot(date_collection, rain_collection, color = 'salmon', alpha = 0.5)
ax1.plot(date_collection, np.convolve(rain_collection, np.ones(kernel) / kernel, mode = 'same'), color = 'red', alpha = 0.5)
ax1.plot(date_collection, snow_collection, color = 'cornflowerblue', alpha = 0.5)
ax1.plot(date_collection, np.convolve(snow_collection, np.ones(kernel) / kernel, mode = 'same'), color = 'blue', alpha = 0.5)

print('CHURCHILL_UA: ', np.sum(rain_collection))

filename = 'C:/Users/Jeremy/Desktop/era5_precip_palsa.csv'
era5_rain_mm_day_hourly = np.array(read_csv_header(filename, 0, 'float', 1))
era5_snow_mm_day_hourly = np.array(read_csv_header(filename, 1, 'float', 1))
era5_matlab_times = np.array(read_csv_header(filename, 2, 'float', 1))

# Convert ERA5 mm/day (hourly accumulated) to mm/day (daily accumulated)
era5_rain_mm_hr_dailysum, era5_datetimes_dailysum = mm_day_hourly2daily_mm(era5_matlab_times, era5_rain_mm_day_hourly)
era5_snow_mm_hr_dailysum, era5_datetimes_dailysum = mm_day_hourly2daily_mm(era5_matlab_times, era5_snow_mm_day_hourly)

# Convert CHURCHILL_UA mm/day (daily accumulated) to mm/month (monthly accumulated)
churchillua_rain_mm_day_monthlysum, churchillua_datetimes_monthlysum = churchillua_mm_day_daily2monthly_mm(date_collection, rain_collection)
churchillua_snow_mm_day_monthlysum, churchillua_datetimes_monthlysum = churchillua_mm_day_daily2monthly_mm(date_collection, snow_collection)

# Convert ERA5 mm/day (hourly accumulated) to mm/month (monthly accumulated)
era5_rain_mm_hr_monthlysum, era5_datetimes_monthlysum = mm_day_hourly2monthly_mm(era5_matlab_times, era5_rain_mm_day_hourly)
era5_snow_mm_hr_monthlysum, era5_datetimes_monthlysum = mm_day_hourly2monthly_mm(era5_matlab_times, era5_snow_mm_day_hourly)

kernel = 14
#avg_chemicals1 = np.convolve(avg_chemicals1, np.ones(kernel) / kernel, mode='same')[kernel:-kernel]
ax2.plot(era5_datetimes_dailysum, era5_rain_mm_hr_dailysum, color = 'salmon', alpha = 0.5)
ax2.plot(era5_datetimes_dailysum, np.convolve(era5_rain_mm_hr_dailysum, np.ones(kernel) / kernel, mode = 'same'), color = 'red', alpha = 0.5)
ax2.plot(era5_datetimes_dailysum, era5_snow_mm_hr_dailysum, color = 'cornflowerblue', alpha = 0.5)
ax2.plot(era5_datetimes_dailysum, np.convolve(era5_snow_mm_hr_dailysum, np.ones(kernel) / kernel, mode = 'same'), color = 'blue', alpha = 0.5)

year = 2003
month = 5

print(date_collection)
print(era5_datetimes_dailysum)

for i in range(0, len(date_collection)):
    print(date_collection[i])

plt.show()


