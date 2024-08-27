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
import plants
import conversions
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from datetime import timedelta
from datetime import date

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

    site = 'palsa_high'

    site_temps = {}

    month_list = ['jun','aug']

    for month in month_list:

        filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/temps/temp_'+site+'_'+month+'_2022.txt'

        times = np.array(read_csv_header(filename, 0, 'string',1))
        for i in range(0, len(times)):
            t_test = times[i]
            times[i] = t_test[0:4] + t_test[5:7] + t_test[8:10] + t_test[11:13] + t_test[14:16] + t_test[17:19]
            times[i] = conversions.yyyymmddhhmm2doy(times[i])

        record = np.array(read_csv_header(filename, 1, 'integer',1))
        temp05cm = np.array(read_csv_header(filename, 2, 'float',1))
        temp15cm = np.array(read_csv_header(filename, 3, 'float',1))
        temp25cm = np.array(read_csv_header(filename, 4, 'float',1))

        site_temps[site + '_' + month + '_dayofyear'] = times
        site_temps[site + '_' + month + '_record'] = record
        site_temps[site + '_' + month + '_temp05cm'] = temp05cm
        site_temps[site + '_' + month + '_temp15cm'] = temp15cm
        site_temps[site + '_' + month + '_temp25cm'] = temp25cm

        return(site_temps)

site_temps = soil_temp_data()

print(site_temps['palsa_high_jun_dayofyear'])

#print(test)

#fig, ax = plt.subplots()
#fig.set_size_inches(18.5, 7.5, forward=True)
#ax.plot(test[2],test[5])
#plt.show()