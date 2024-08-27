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
    print('ratio: ', monthly_ratios)
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
        print(year)
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

fig = plt.figure(figsize=(16.0, 6.0))
gsc = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1,1,1])
ax1 = fig.add_subplot(gsc[0, 1])
ax2 = fig.add_subplot(gsc[1, 1])
ax3 = fig.add_subplot(gsc[0, 2])
ax4 = fig.add_subplot(gsc[1, 2])
ax5 = fig.add_subplot(gsc[0, 3])
ax6 = fig.add_subplot(gsc[1, 3])
ax7 = fig.add_subplot(gsc[2, 1])
ax8 = fig.add_subplot(gsc[2, 2])
ax9 = fig.add_subplot(gsc[2, 3])
ax10 = fig.add_subplot(gsc[1, 4])
ax11 = fig.add_subplot(gsc[1, 0])
ax12 = fig.add_subplot(gsc[2, 0])
ax1.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax1.set_ylim(0, 6)
ax2.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax2.set_ylim(0, 6)
ax4.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax7.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax8.set_xlim([datetime(2002, 1, 1), datetime(2008, 1, 1)])
ax11.set_xlim([datetime(2002, 1, 1), datetime(2003, 1, 1)])
ax12.set_xlim([datetime(2002, 1, 1), datetime(2003, 1, 1)])
ax7.set_ylim(0, 6)
ax11.set_ylim(0, 6)
ax12.set_ylim(0, 6)
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
ax10.set_ylim(0,3)
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
format_axis1(ax10)
format_axis1(ax11)
format_axis1(ax12)

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

kernel = 30
#ax1.plot(date_collection, rain_collection, color = 'salmon', alpha = 0.5)
ax1.plot(date_collection, np.convolve(rain_collection, np.ones(kernel) / kernel, mode = 'same'), color = 'red', alpha = 0.5)
#ax1.plot(date_collection, snow_collection, color = 'cornflowerblue', alpha = 0.5)
ax1.plot(date_collection, np.convolve(snow_collection, np.ones(kernel) / kernel, mode = 'same'), color = 'blue', alpha = 0.5)

print('CHURCHILL_UA: ', np.sum(rain_collection))

filename = 'C:/Users/Jeremy/Desktop/era5_precip_palsa.csv'
era5_rain_mm_day_hourly = np.array(read_csv_header(filename, 0, 'float', 1))
era5_snow_mm_day_hourly = np.array(read_csv_header(filename, 1, 'float', 1))
era5_matlab_times = np.array(read_csv_header(filename, 2, 'float', 1))
era5_date_times = []
for d in range(0, len(era5_matlab_times)):
    matlab_datenum = era5_matlab_times[d]
    dtime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
    era5_date_times.append(dtime)

kernel = 24 * 30
ax11.plot(era5_date_times, np.convolve(era5_rain_mm_day_hourly, np.ones(kernel) / kernel, mode = 'same'), color = 'salmon')
ax11.plot(era5_date_times, np.convolve(era5_snow_mm_day_hourly, np.ones(kernel) / kernel, mode = 'same'), color = 'cornflowerblue')

# Convert ERA5 mm/day (hourly accumulated) to mm/day (daily accumulated)
era5_rain_mm_hr_dailysum, era5_datetimes_dailysum = mm_day_hourly2daily_mm(era5_matlab_times, era5_rain_mm_day_hourly)
era5_snow_mm_hr_dailysum, era5_datetimes_dailysum = mm_day_hourly2daily_mm(era5_matlab_times, era5_snow_mm_day_hourly)

# Convert CHURCHILL_UA mm/day (daily accumulated) to mm/month (monthly accumulated)
churchillua_rain_mm_day_monthlysum, churchillua_datetimes_monthlysum = churchillua_mm_day_daily2monthly_mm(date_collection, rain_collection)
churchillua_snow_mm_day_monthlysum, churchillua_datetimes_monthlysum = churchillua_mm_day_daily2monthly_mm(date_collection, snow_collection)

# Convert ERA5 mm/day (hourly accumulated) to mm/month (monthly accumulated)
era5_rain_mm_hr_monthlysum, era5_datetimes_monthlysum = mm_day_hourly2monthly_mm(era5_matlab_times, era5_rain_mm_day_hourly)
era5_snow_mm_hr_monthlysum, era5_datetimes_monthlysum = mm_day_hourly2monthly_mm(era5_matlab_times, era5_snow_mm_day_hourly)

kernel = 30
#avg_chemicals1 = np.convolve(avg_chemicals1, np.ones(kernel) / kernel, mode='same')[kernel:-kernel]
#ax2.plot(era5_datetimes_dailysum, era5_rain_mm_hr_dailysum, color = 'salmon', alpha = 0.5)
ax2.plot(era5_datetimes_dailysum, np.convolve(era5_rain_mm_hr_dailysum, np.ones(kernel) / kernel, mode = 'same'), color = 'red', alpha = 0.5)
#ax2.plot(era5_datetimes_dailysum, era5_snow_mm_hr_dailysum, color = 'cornflowerblue', alpha = 0.5)
ax2.plot(era5_datetimes_dailysum, np.convolve(era5_snow_mm_hr_dailysum, np.ones(kernel) / kernel, mode = 'same'), color = 'blue', alpha = 0.5)

ax3.fill_between(churchillua_datetimes_monthlysum, np.array(churchillua_rain_mm_day_monthlysum) + np.array(churchillua_snow_mm_day_monthlysum), color = 'gray', alpha = 0.5)
ax3.fill_between(churchillua_datetimes_monthlysum, churchillua_rain_mm_day_monthlysum, color = 'salmon', alpha = 1.0)
ax3.fill_between(churchillua_datetimes_monthlysum, churchillua_snow_mm_day_monthlysum, color = 'cornflowerblue', alpha = 0.5)
ax3.text(datetime(2005, 1, 1), 110, round(np.sum(churchillua_rain_mm_day_monthlysum), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='salmon', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax3.text(datetime(2005, 1, 1), 95, round(np.sum(churchillua_snow_mm_day_monthlysum), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='cornflowerblue', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax3.text(datetime(2005, 1, 1), 80, round(np.sum(np.array(churchillua_rain_mm_day_monthlysum) + np.array(churchillua_snow_mm_day_monthlysum)), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='gray', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
print(' ')
test = churchillua_rain_mm_day_monthlysum
for i in test:
    print(i)
print(np.sum(churchillua_rain_mm_day_monthlysum))

era5_aligned_idx1, era5_aligned_idx2 = 12*(2002 - 1950), 12*(2008 - 1950)

ax4.fill_between(era5_datetimes_monthlysum[era5_aligned_idx1:era5_aligned_idx2], np.array(era5_rain_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2]) + np.array(era5_snow_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2]), color = 'gray', alpha = 0.5)
ax4.fill_between(era5_datetimes_monthlysum[era5_aligned_idx1:era5_aligned_idx2], era5_rain_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2], color = 'salmon', alpha = 1.0)
ax4.fill_between(era5_datetimes_monthlysum[era5_aligned_idx1:era5_aligned_idx2], era5_snow_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2], color = 'cornflowerblue', alpha = 0.5)
ax4.text(datetime(2005, 1, 1), 110, round(np.sum(era5_rain_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2]), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='salmon', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax4.text(datetime(2005, 1, 1), 95, round(np.sum(era5_snow_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2]), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='cornflowerblue', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax4.text(datetime(2005, 1, 1), 80, round(np.sum(np.array(era5_rain_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2]) + np.array(era5_snow_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2])), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='gray', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))

churchillua_rain_mm_day_monthlysum_monthlymeans, churchillua_rain_mm_day_monthlysum_stack = monthlysums2monthlymeans(churchillua_datetimes_monthlysum, churchillua_rain_mm_day_monthlysum)
churchillua_snow_mm_day_monthlysum_monthlymeans, churchillua_rain_mm_day_monthlysum_stack = monthlysums2monthlymeans(churchillua_datetimes_monthlysum, churchillua_snow_mm_day_monthlysum)

era5_rain_mm_hr_monthlysum_monthlymeans, era5_rain_mm_hr_monthlysum_stack = monthlysums2monthlymeans(era5_datetimes_monthlysum[era5_aligned_idx1:era5_aligned_idx2], era5_rain_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2])
era5_snow_mm_hr_monthlysum_monthlymeans, era5_snow_mm_hr_monthlysum_stack = monthlysums2monthlymeans(era5_datetimes_monthlysum[era5_aligned_idx1:era5_aligned_idx2], era5_snow_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2])

#print(churchillua_rain_mm_day_monthlysum_stack)
#print(era5_rain_mm_hr_monthlysum_stack)

#for i in range(0, 12):
#    ratio = np.array(churchillua_rain_mm_day_monthlysum_stack[i]) / np.array(era5_rain_mm_hr_monthlysum_stack[i])
#    print(i, ratio)

#stop

#print('1: ', churchillua_rain_mm_day_monthlysum_monthlymeans)
ax5.plot(range(12), churchillua_rain_mm_day_monthlysum_monthlymeans, color = 'salmon')
ax5.plot(range(12), churchillua_snow_mm_day_monthlysum_monthlymeans, color = 'cornflowerblue')
ax5.plot(range(12), np.array(churchillua_rain_mm_day_monthlysum_monthlymeans) + np.array(churchillua_snow_mm_day_monthlysum_monthlymeans), color = 'gray', linestyle = '--')
ax5.fill_between(range(12), churchillua_rain_mm_day_monthlysum_monthlymeans, color='salmon', alpha=0.5)
ax5.fill_between(range(12), churchillua_snow_mm_day_monthlysum_monthlymeans, color='cornflowerblue', alpha=0.5)
ax5.text(1.5, 60, round(np.sum(churchillua_rain_mm_day_monthlysum_monthlymeans), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='salmon', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax5.text(1.5, 50, round(np.sum(churchillua_snow_mm_day_monthlysum_monthlymeans), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='cornflowerblue', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax5.text(1.5, 40, round(np.sum(np.array(churchillua_rain_mm_day_monthlysum_monthlymeans) + np.array(churchillua_snow_mm_day_monthlysum_monthlymeans)), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='gray', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))

ax6.plot(range(12), era5_rain_mm_hr_monthlysum_monthlymeans, color = 'salmon')
ax6.plot(range(12), era5_snow_mm_hr_monthlysum_monthlymeans, color = 'cornflowerblue')
ax6.plot(range(12), np.array(era5_rain_mm_hr_monthlysum_monthlymeans) + np.array(era5_snow_mm_hr_monthlysum_monthlymeans), color = 'gray', linestyle = '--')
ax6.fill_between(range(12), era5_rain_mm_hr_monthlysum_monthlymeans, color='salmon', alpha=0.5)
ax6.fill_between(range(12), era5_snow_mm_hr_monthlysum_monthlymeans, color='cornflowerblue', alpha=0.5)
ax6.text(1.5, 60, round(np.sum(era5_rain_mm_hr_monthlysum_monthlymeans), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='salmon', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax6.text(1.5, 50, round(np.sum(era5_snow_mm_hr_monthlysum_monthlymeans), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='cornflowerblue', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
ax6.text(1.5, 40, round(np.sum(np.array(era5_rain_mm_hr_monthlysum_monthlymeans) + np.array(era5_snow_mm_hr_monthlysum_monthlymeans)), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='gray', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))

monthly_rain_ratios = np.array(churchillua_rain_mm_day_monthlysum_monthlymeans) / np.array(era5_rain_mm_hr_monthlysum_monthlymeans)
monthly_snow_ratios = np.array(churchillua_snow_mm_day_monthlysum_monthlymeans) / np.array(era5_snow_mm_hr_monthlysum_monthlymeans)
monthly_rain_ratios[np.isnan(monthly_rain_ratios)] = 1.0
monthly_snow_ratios[np.isnan(monthly_snow_ratios)] = 1.0

#monthly_rain_ratios = 0.0 * monthly_rain_ratios + 0.95

ax10.plot(range(12), monthly_rain_ratios, color = 'salmon')
ax10.plot(range(12), monthly_snow_ratios, color = 'cornflowerblue')
ax10.plot(range(12), 0.0 * np.array(range(12)) + 1.0, color = 'black')

#
era5_rain_mm_day_hourly_corrected, era5_mm_day_hourly_datetimes = era5_mm_day_hourly_uncorrected2corrected(era5_matlab_times, era5_rain_mm_day_hourly, monthly_rain_ratios)
era5_snow_mm_day_hourly_corrected, era5_mm_day_hourly_datetimes = era5_mm_day_hourly_uncorrected2corrected(era5_matlab_times, era5_snow_mm_day_hourly, monthly_snow_ratios)

kernel = 24 * 30
ax12.plot(era5_date_times, np.convolve(era5_rain_mm_day_hourly_corrected, np.ones(kernel) / kernel, mode = 'same'), color = 'salmon')
ax12.plot(era5_date_times, np.convolve(era5_snow_mm_day_hourly_corrected, np.ones(kernel) / kernel, mode = 'same'), color = 'cornflowerblue')

#
era5_rain_mm_hr_dailysum_corrected, era5_datetimes_dailysum = mm_day_hourly2daily_mm(era5_matlab_times, era5_rain_mm_day_hourly_corrected)
era5_snow_mm_hr_dailysum_corrected, era5_datetimes_dailysum = mm_day_hourly2daily_mm(era5_matlab_times, era5_snow_mm_day_hourly_corrected)
#
kernel = 30
#ax7.plot(era5_datetimes_dailysum, era5_rain_mm_hr_dailysum_corrected, color = 'salmon', alpha = 0.5)
ax7.plot(era5_datetimes_dailysum, np.convolve(era5_rain_mm_hr_dailysum_corrected, np.ones(kernel) / kernel, mode = 'same'), color = 'red', alpha = 0.5)
#ax7.plot(era5_datetimes_dailysum, era5_snow_mm_hr_dailysum_corrected, color = 'cornflowerblue', alpha = 0.5)
ax7.plot(era5_datetimes_dailysum, np.convolve(era5_snow_mm_hr_dailysum_corrected, np.ones(kernel) / kernel, mode = 'same'), color = 'blue', alpha = 0.5)
#
# # Convert ERA5 mm/day (hourly accumulated) to mm/month (monthly accumulated)
# era5_rain_mm_hr_monthlysum_corrected, era5_datetimes_monthlysum = mm_day_hourly2monthly_mm(era5_matlab_times, era5_rain_mm_day_hourly_corrected)
# era5_snow_mm_hr_monthlysum_corrected, era5_datetimes_monthlysum = mm_day_hourly2monthly_mm(era5_matlab_times, era5_snow_mm_day_hourly_corrected)
#
# ax8.fill_between(era5_datetimes_monthlysum, np.array(era5_rain_mm_hr_monthlysum_corrected) + np.array(era5_snow_mm_hr_monthlysum_corrected), color = 'gray', alpha = 0.5)
# ax8.fill_between(era5_datetimes_monthlysum, era5_rain_mm_hr_monthlysum_corrected, color = 'salmon', alpha = 1.0)
# ax8.fill_between(era5_datetimes_monthlysum, era5_snow_mm_hr_monthlysum_corrected, color = 'cornflowerblue', alpha = 0.5)
# print(' ')
# test = era5_rain_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2]
# for i in test:
#     print(i)
# print(round(np.sum(era5_rain_mm_hr_monthlysum[era5_aligned_idx1:era5_aligned_idx2]), 2))
# print(' ')
# test = era5_rain_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2]
# for i in test:
#     print(i)
# print(round(np.sum(era5_rain_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2]), 2))
#
# ax8.text(datetime(2005, 1, 1), 110, round(np.sum(era5_rain_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2]), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='salmon', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
# ax8.text(datetime(2005, 1, 1), 95, round(np.sum(era5_snow_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2]), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='cornflowerblue', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
# ax8.text(datetime(2005, 1, 1), 80, round(np.sum(np.array(era5_rain_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2]) + np.array(era5_snow_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2])), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='gray', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
#
# era5_rain_mm_hr_monthlysum_monthlymeans_corrected, era5_rain_mm_hr_monthlysum_stack_corrected = monthlysums2monthlymeans(era5_datetimes_monthlysum[era5_aligned_idx1:era5_aligned_idx2], era5_rain_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2])
# era5_snow_mm_hr_monthlysum_monthlymeans_corrected, era5_snow_mm_hr_monthlysum_stack_corrected = monthlysums2monthlymeans(era5_datetimes_monthlysum[era5_aligned_idx1:era5_aligned_idx2], era5_snow_mm_hr_monthlysum_corrected[era5_aligned_idx1:era5_aligned_idx2])
#
# print('2: ', era5_rain_mm_hr_monthlysum_monthlymeans_corrected)
# ax9.plot(range(12), era5_rain_mm_hr_monthlysum_monthlymeans_corrected, color = 'salmon')
# ax9.plot(range(12), era5_snow_mm_hr_monthlysum_monthlymeans_corrected, color = 'cornflowerblue')
# ax9.plot(range(12), np.array(era5_rain_mm_hr_monthlysum_monthlymeans_corrected) + np.array(era5_snow_mm_hr_monthlysum_monthlymeans_corrected), color = 'gray', linestyle = '--')
# ax9.fill_between(range(12), era5_rain_mm_hr_monthlysum_monthlymeans_corrected, color='salmon', alpha=0.5)
# ax9.fill_between(range(12), era5_snow_mm_hr_monthlysum_monthlymeans_corrected, color='cornflowerblue', alpha=0.5)
# ax9.text(1.5, 60, round(np.sum(era5_rain_mm_hr_monthlysum_monthlymeans_corrected), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='salmon', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
# ax9.text(1.5, 50, round(np.sum(era5_snow_mm_hr_monthlysum_monthlymeans_corrected), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='cornflowerblue', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
# ax9.text(1.5, 40, round(np.sum(np.array(era5_rain_mm_hr_monthlysum_monthlymeans_corrected) + np.array(era5_snow_mm_hr_monthlysum_monthlymeans_corrected)), 2), horizontalalignment='center', verticalalignment='bottom', fontsize=12.0, color='gray', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', alpha=0.0))
# #test
plt.show()


