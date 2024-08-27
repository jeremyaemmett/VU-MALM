# Plots raw root profiles and fitted exponential curves

import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

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

filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/raw/roots3.csv'

depths = 0.01*np.array([2.5,5.0,7.5,10.0,12.5,15.0,17.5,20.0,22.5,25.0,27.5,30.0,32.5,35.0,37.5,40.0,42.5])

f1 = np.array(read_csv_header(filename, 6, 'float', 0))[0:17]
f2 = np.array(read_csv_header(filename, 22, 'float', 0))[0:17]
f3 = np.array(read_csv_header(filename, 30, 'float', 0))[0:17]
f4 = np.array(read_csv_header(filename, 38, 'float', 0))[0:17]
f5 = np.array(read_csv_header(filename, 46, 'float', 0))[0:17]
f6 = np.array(read_csv_header(filename, 54, 'float', 0))[0:17]
f7 = np.array(read_csv_header(filename, 62, 'float', 0))[0:17]
f8 = np.array(read_csv_header(filename, 70, 'float', 0))[0:17]
f9 = np.array(read_csv_header(filename, 78, 'float', 0))[0:17]
f10 = np.array(read_csv_header(filename, 14, 'float', 0))[0:17]

p1 = np.array(read_csv_header(filename, 6, 'float', 0))[17:34]
p2 = np.array(read_csv_header(filename, 22, 'float', 0))[17:34]
p3 = np.array(read_csv_header(filename, 30, 'float', 0))[17:34]
p4 = np.array(read_csv_header(filename, 38, 'float', 0))[17:34]
p5 = np.array(read_csv_header(filename, 46, 'float', 0))[17:34]
p6 = np.array(read_csv_header(filename, 54, 'float', 0))[17:34]
p7 = np.array(read_csv_header(filename, 62, 'float', 0))[17:34]
p8 = np.array(read_csv_header(filename, 70, 'float', 0))[17:34]
p9 = np.array(read_csv_header(filename, 78, 'float', 0))[17:34]
p10 = np.array(read_csv_header(filename, 14, 'float', 0))[17:34]

b1 = np.array(read_csv_header(filename, 6, 'float', 0))[34:51]
b2 = np.array(read_csv_header(filename, 22, 'float', 0))[34:51]
b3 = np.array(read_csv_header(filename, 30, 'float', 0))[34:51]
b4 = np.array(read_csv_header(filename, 38, 'float', 0))[34:51]
b5 = np.array(read_csv_header(filename, 46, 'float', 0))[34:51]
b6 = np.array(read_csv_header(filename, 54, 'float', 0))[34:51]
b7 = np.array(read_csv_header(filename, 62, 'float', 0))[34:51]
b8 = np.array(read_csv_header(filename, 70, 'float', 0))[34:51]
b9 = np.array(read_csv_header(filename, 78, 'float', 0))[34:51]
b10 = np.array(read_csv_header(filename, 14, 'float', 0))[34:51]

f1[f1 == 0] = np.nan
f2[f2 == 0] = np.nan
f3[f3 == 0] = np.nan
f4[f4 == 0] = np.nan
f5[f5 == 0] = np.nan
f6[f6 == 0] = np.nan
f7[f7 == 0] = np.nan
f8[f8 == 0] = np.nan
f9[f9 == 0] = np.nan
f10[f10 == 0] = np.nan

p1[p1 == 0] = np.nan
p2[p2 == 0] = np.nan
p3[p3 == 0] = np.nan
p4[p4 == 0] = np.nan
p5[p5 == 0] = np.nan
p6[p6 == 0] = np.nan
p7[p7 == 0] = np.nan
p8[p8 == 0] = np.nan
p9[p9 == 0] = np.nan
p10[p10 == 0] = np.nan

b1[b1 == 0] = np.nan
b2[b2 == 0] = np.nan
b3[b3 == 0] = np.nan
b4[b4 == 0] = np.nan
b5[b5 == 0] = np.nan
b6[b6 == 0] = np.nan
b7[b7 == 0] = np.nan
b8[b8 == 0] = np.nan
b9[b9 == 0] = np.nan
b10[b10 == 0] = np.nan

f_high_block = np.array([[f6],[f7],[f8],[f9],[f10]])
f_high_mean = (np.nanmean(f_high_block,axis=0))[0]
f_low_block = np.array([[f1],[f3],[f4],[f5]])
f_low_mean = np.nanmean(f_low_block,axis=0)[0]

p_high_block = np.array([[p5],[p6],[p7],[p8],[p9],[p10]])
p_high_mean = np.nanmean(p_high_block,axis=0)[0]
p_low_block = np.array([[p1],[p2],[p3],[p4]])
p_low_mean = np.nanmean(p_low_block,axis=0)[0]

b_high_block = np.array([[b4],[b5],[b6],[b7],[b8],[b9],[b10]])
b_high_mean = np.nanmean(b_high_block,axis=0)[0]
b_low_block = np.array([[b1],[b2],[b3]])
b_low_mean = np.nanmean(b_low_block,axis=0)[0]

fig, axes = plt.subplots(2, 3)

ax1 = axes[0,0]
ax1.set_xlim([0,50])
ax1.set_ylim([-0.45,0.0])
ax1.plot(p5,-depths,marker='o',linestyle='none')
ax1.plot(p6,-depths,marker='o',linestyle='none')
ax1.plot(p7,-depths,marker='o',linestyle='none')
ax1.plot(p8,-depths,marker='o',linestyle='none')
ax1.plot(p9,-depths,marker='o',linestyle='none')
ax1.plot(p10,-depths,marker='o',linestyle='none')
#
test = np.polyfit(np.log(depths[p_high_mean > 0]), p_high_mean[p_high_mean > 0], 1)
y = test[0] * np.log(depths) + test[1]
#
ax1.plot(p_high_mean,-depths,marker='x',linestyle='None',color='black')
ax1.plot(y,-depths,linestyle='-',color='black',linewidth=3.0)
#with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\palsa_high_10082023.txt', "w") as myfile1:
#    myfile1.write('depth (m),root mass density (kg/m^3)\n')
#for i in range(0,len(y)):
#    with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\palsa_high_10082023.txt', "a") as myfile1:
#        myfile1.write(
#            f"{depths[i]},{y[i]}\n")

ax2 = axes[0,1]
ax2.set_xlim([-5,50])
ax2.set_ylim([-0.45,0.0])
ax2.plot(f6,-depths,marker='o',linestyle='none')
ax2.plot(f7,-depths,marker='o',linestyle='none')
ax2.plot(f8,-depths,marker='o',linestyle='none')
ax2.plot(f9,-depths,marker='o',linestyle='none')
ax2.plot(f10,-depths,marker='o',linestyle='none')
#
test = np.polyfit(np.log(depths[f_high_mean > 0]), f_high_mean[f_high_mean > 0], 1)
y = test[0] * np.log(depths) + test[1]
#
ax2.plot(f_high_mean,-depths,marker='x',linestyle='None',color='black')
ax2.plot(y,-depths,linestyle='-',color='black',linewidth=3.0)
#with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\fen_high_10082023.txt', "w") as myfile1:
#    myfile1.write('depth (m),root mass density (kg/m^3)\n')
#for i in range(0,len(y)):
#    with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\fen_high_10082023.txt', "a") as myfile1:
#        myfile1.write(
#            f"{depths[i]},{y[i]}\n")

ax3 = axes[0,2]
ax3.set_xlim([-5,50])
ax3.set_ylim([-0.45,0.0])
ax3.plot(b4,-depths,marker='o',linestyle='none')
ax3.plot(b5,-depths,marker='o',linestyle='none')
ax3.plot(b6,-depths,marker='o',linestyle='none')
ax3.plot(b7,-depths,marker='o',linestyle='none')
ax3.plot(b8,-depths,marker='o',linestyle='none')
ax3.plot(b9,-depths,marker='o',linestyle='none')
ax3.plot(b10,-depths,marker='o',linestyle='none')
#
test = np.polyfit(np.log(depths[b_high_mean > 0]), b_high_mean[b_high_mean > 0], 1)
y = test[0] * np.log(depths) + test[1]
y[y < 0] = 0.0
#
ax3.plot(b_high_mean,-depths,marker='x',linestyle='None',color='black')
ax3.plot(y,-depths,linestyle='-',color='black',linewidth=3.0)
#with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\tundra_high_10082023.txt', "w") as myfile1:
#    myfile1.write('depth (m),root mass density (kg/m^3)\n')
#for i in range(0,len(y)):
#    with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\tundra_high_10082023.txt', "a") as myfile1:
#        myfile1.write(
#            f"{depths[i]},{y[i]}\n")

ax4 = axes[1,0]
ax4.set_xlim([-5,50])
ax4.set_ylim([-0.45,0.0])
ax4.plot(p1,-depths,marker='o',linestyle='none')
ax4.plot(p2,-depths,marker='o',linestyle='none')
ax4.plot(p3,-depths,marker='o',linestyle='none')
ax4.plot(p4,-depths,marker='o',linestyle='none')
#
test = np.polyfit(np.log(depths[p_low_mean > 0]), p_low_mean[p_low_mean > 0], 1)
y = test[0] * np.log(depths) + test[1]
#
ax4.plot(p_low_mean,-depths,marker='x',linestyle='None',color='black')
ax4.plot(y,-depths,linestyle='-',color='black',linewidth=3.0)
#with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\palsa_low_10082023.txt', "w") as myfile1:
#    myfile1.write('depth (m),root mass density (kg/m^3)\n')
#for i in range(0,len(y)):
#    with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\palsa_low_10082023.txt', "a") as myfile1:
#        myfile1.write(
#            f"{depths[i]},{y[i]}\n")

ax5 = axes[1,1]
ax5.set_xlim([-5,50])
ax5.set_ylim([-0.45,0.0])
ax5.plot(f1,-depths,marker='o',linestyle='none')
#ax5.plot(f2,-depths,marker='o',linestyle='none')
ax5.plot(f3,-depths,marker='o',linestyle='none')
ax5.plot(f4,-depths,marker='o',linestyle='none')
ax5.plot(f5,-depths,marker='o',linestyle='none')
#
test = np.polyfit(np.log(depths[f_low_mean > 0]), f_low_mean[f_low_mean > 0], 1)
y = test[0] * np.log(depths) + test[1]
#
ax5.plot(f_low_mean,-depths,marker='x',linestyle='None',color='black')
ax5.plot(y,-depths,linestyle='-',color='black',linewidth=3.0)
#with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\fen_low_10082023.txt', "w") as myfile1:
#    myfile1.write('depth (m),root mass density (kg/m^3)\n')
#for i in range(0,len(y)):
#    with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\fen_low_10082023.txt', "a") as myfile1:
#        myfile1.write(
#            f"{depths[i]},{y[i]}\n")

ax6 = axes[1,2]
ax6.set_xlim([-5,50])
ax6.set_ylim([-0.45,0.0])
ax6.plot(b1,-depths,marker='o',linestyle='none')
ax6.plot(b2,-depths,marker='o',linestyle='none')
ax6.plot(b3,-depths,marker='o',linestyle='none')
#ax6.plot(b10,-depths,marker='o',linestyle='none')
#
test = np.polyfit(np.log(depths[b_low_mean > 0]), b_low_mean[b_low_mean > 0], 1)
y = test[0] * np.log(depths) + test[1]
#
ax6.plot(b_low_mean,-depths,marker='x',linestyle='None',color='black')
ax6.plot(y,-depths,linestyle='-',color='black',linewidth=3.0)
#with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\tundra_low_10082023.txt', "w") as myfile1:
#    myfile1.write('depth (m),root mass density (kg/m^3)\n')
#for i in range(0,len(y)):
#    with open(r'C:\Users\Jeremy\Desktop\Churchill_Data\siteData\roots2\tundra_low_10082023.txt', "a") as myfile1:
#        myfile1.write(
#            f"{depths[i]},{y[i]}\n")

plt.show()
