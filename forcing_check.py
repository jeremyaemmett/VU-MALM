import os
import csv
import random
import numpy as np
from os import listdir
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from os.path import isfile, join

def format_axis(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(' ')
    ax.set_facecolor('whitesmoke')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5, axis='y')

    return()

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

def month_means(rains, snows, dtimes):
    matlab_datenum = dtimes
    datetimes = []
    rain_month_collection = [[], [], [], [], [], [], [], [], [], [], [], []]
    snow_month_collection = [[], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(0, len(dtimes)):
        matlab_datenum = dtimes[i]
        dtime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
        datetimes.append(dtime)
        rain_month_collection[dtime.month - 1].append(rains[i])
        snow_month_collection[dtime.month - 1].append(snows[i])

    hours_per_year = 8760.0
    hours_per_month = 730.0

    rain_month_means = np.zeros(12)
    snow_month_means = np.zeros(12)
    for i in range(0, 12):
        rain_month_means[i] = hours_per_month * np.mean(rain_month_collection[i])
        snow_month_means[i] = hours_per_month * np.mean(snow_month_collection[i])

    return(rain_month_means, snow_month_means, datetimes)

filename = 'C:/Users/Jeremy/Desktop/era5_precip_palsa.csv'

rain_mm_hr = np.array(read_csv_header(filename, 0, 'float', 1))
print((1./365.) * (1./24.)*np.sum(rain_mm_hr))
stop
snow_mm_hr = np.array(read_csv_header(filename, 1, 'float', 1))
date_times = np.array(read_csv_header(filename, 2, 'float', 1))

churchill_rainfall = [7.0, 5.0, 15.0, 16.0, 19.0, 30.0, 46.0, 63.0, 33.0, 36.0, 15.0, 9.0] # mm / month
churchill_snowfall = [42.0, 28.0, 50.0, 47.0, 36.0, 13.0, 0.0, 0.0, 2.0, 33.0, 60.0, 44.0] # mm / month

rain_month_means, snow_month_means, datetimes = month_means(rain_mm_hr, snow_mm_hr, date_times)

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig = plt.figure(figsize=(10.0, 8.0))
gsc = fig.add_gridspec(2, 1, width_ratios=[1], height_ratios=[1,1])

print(rain_month_means)

ax1 = fig.add_subplot(gsc[0])
ax2 = fig.add_subplot(gsc[1])

ax1.set_xlim([datetime(2022, 1, 1), datetime(2022, 12, 31)])

churchill_annual_rainfall = 30.5 * np.sum(churchill_rainfall) / 1000.0
churchill_annual_snowfall = 30.5 * np.sum(churchill_snowfall) / 1000.0

print(churchill_annual_rainfall, churchill_annual_snowfall)

integrated_era5_rainfall = np.sum(rain_month_means) / 1000.0
integrated_era5_snowfall = np.sum(snow_month_means) / 1000.0

print(integrated_era5_rainfall, integrated_era5_snowfall)

monthly_rain_factor = churchill_rainfall / (rain_month_means / 30.5)
monthly_snow_factor = churchill_snowfall / (snow_month_means / 30.5)

print(monthly_rain_factor, monthly_snow_factor)

rain_mm_hr_corrected = []
for i in range(0, len(rain_mm_hr)):
    rain_mm_hr_corrected.append(rain_mm_hr[i] * monthly_rain_factor[datetimes[i].month - 1])
    #print(rain_mm_hr[i], monthly_rain_factor[datetimes[i].month-1])
    #if datetimes[i].month-1 == 2: stop

snow_mm_hr_corrected = []
for i in range(0, len(snow_mm_hr)):
    snow_mm_hr_corrected.append(snow_mm_hr[i] * monthly_snow_factor[datetimes[i].month - 1])

kernel = 24 * 7

ax1.plot(datetimes, np.convolve(rain_mm_hr_corrected, np.ones(kernel) / kernel, mode = 'same'), color = 'red', linestyle = 'dashed')
ax1.plot(datetimes, np.convolve(snow_mm_hr_corrected, np.ones(kernel) / kernel, mode = 'same'), color = 'blue', linestyle = 'dashed')
ax1.plot(datetimes, np.convolve(rain_mm_hr, np.ones(kernel) / kernel, mode = 'same'), color = 'red')
ax1.plot(datetimes, np.convolve(snow_mm_hr, np.ones(kernel) / kernel, mode = 'same'), color = 'blue')
ax1.set_ylim([0, 10])

rain_month_means_corrected, snow_month_means_corrected, datetimes = month_means(rain_mm_hr_corrected, snow_mm_hr_corrected, date_times)

ax2.plot(range(12), rain_month_means / 30.5, color = 'red')
ax2.plot(range(12), snow_month_means / 30.5, color = 'blue')
ax2.plot(range(12), churchill_rainfall, color = 'salmon', linewidth = 5.0) # mm / month
ax2.plot(range(12), churchill_snowfall, color = 'cornflowerblue', linewidth = 5.0) # mm / month
ax2.fill_between(range(12), rain_month_means / 30.5, churchill_rainfall, color = 'red', alpha = 0.1)
ax2.fill_between(range(12),snow_month_means / 30.5, churchill_snowfall, color = 'blue', alpha = 0.1)
ax2.set_ylabel('mm / month')
plt.xticks(range(12), labels = months)

ax2.plot(range(12), rain_month_means_corrected / 30.5, color = 'red', linestyle = '--')
#ax2.plot(range(12), snow_month_means_corrected / 30.5, color = 'green', linestyle = '--')

print(rain_month_means_corrected / 30.5)
print(np.sum(churchill_rainfall))

format_axis(ax1)
format_axis(ax2)

plt.show()