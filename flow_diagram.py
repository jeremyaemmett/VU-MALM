from matplotlib.lines import Line2D
from os.path import isfile, join
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from datetime import date
from math import radians
from os import listdir
import numpy as np
import matplotlib
import params
import math
import csv


def find_keys_with_substring(dict, substr):

    data_dict_keys = np.array(list(dict.keys()))
    output_array = np.zeros_like(data_dict_keys, dtype=bool)
    val_indices = np.char.find(data_dict_keys, substr)
    output_array[np.where(val_indices != -1)] = True
    VvT_vals = data_dict_keys[output_array]

    return(VvT_vals)


def find_elements_with_substring(list, substr):

    list = np.array(list)
    output_array = np.zeros_like(list, dtype=bool)
    val_indices = np.char.find(list, substr)
    output_array[np.where(val_indices != -1)] = True
    elements = list[output_array]

    return(elements)


def draw_circle(axis, coords, radius, color, label, sum):


    circle1 = plt.Circle((coords[0], coords[1]), 1.05*radius, color='gray', alpha=1.0)
    circle2 = plt.Circle((coords[0], coords[1]), radius, color=color, alpha = 1.0)
    axis.add_artist(circle1)
    axis.add_artist(circle2)
    axis.text(coords[0], coords[1]-0.1, label, ha='center', va='center', fontsize=42, weight='bold', color='black')
    sum_color = 'green'
    if sum < 0.0: sum_color = 'red'
    axis.text(coords[0], coords[1]-1.3, '{:.1e}'.format(sum), ha='center', va='center', fontsize=24, weight='bold', color=sum_color)

    return()


def vector_angle(p1, p2):

    angle_rad = math.atan2(p2[1]-p1[1], p2[0]-p1[0])

    return(angle_rad)


def draw_flow_vectors(axis, points, angles, color, values, label_flag):

    values = values[values != 0.0]
    sum = params.write_dt * np.sum(values)
    linear_array, log_array = np.linspace(0.5, 2.0, 1000), np.logspace(-5, -1, 1000)

    if np.nansum(values) > 0.0: values = np.convolve(values, np.ones(28) / 28, mode='same')

    if np.mean(values) < 0.0: color = 'red'

    for a in range(0, len(points)):
        angle1, angle2 = radians(angles[a][0]), radians(angles[a][1])
        coord1, coord2 = np.array(points[a][0]), np.array(points[a][1])
        x1, x2, y1, y2 = coord1[0], coord2[0], coord1[1], coord2[1]
        x1, x2 = x1 + radius * np.cos(angle1), x2 + radius * np.cos(angle2)
        y1, y2 = y1 + radius * np.sin(angle1), y2 + radius * np.sin(angle2)
        arrow_width = np.where(abs(log_array - abs(sum)) == np.min(abs(log_array - abs(sum))))[0][0]
        axis.arrow(x1, y1, x2 - x1, y2 - y1, width=linear_array[arrow_width], length_includes_head=True, head_width = 1.2*linear_array[arrow_width],  color=color, alpha=0.3)
        axis.arrow(x1, y1, x2 - x1, y2 - y1, width=linear_array[arrow_width], length_includes_head=True, head_width=1.2*linear_array[arrow_width], color='black', alpha=1.0, fill=False)
        dist = math.dist([x1, y1], [x2, y2])
        angle = vector_angle([x1, y1], [x2, y2])
        box_angle = angle
        if 2.*np.pi > angle > np.pi: box_angle = (angle + np.pi) % (2.*np.pi)
        if angle < 0.0: box_angle = (angle + np.pi) % (2.*np.pi)
        if angle == np.pi: box_angle = 0.0
        if box_angle > np.pi/2.: box_angle = (box_angle - np.pi)
        #print(np.degrees(angle), np.degrees(box_angle), '{:.1e}'.format(sum))
        mean_x, mean_y = np.mean([x1, x2]), np.mean([y1, y2])
        lx, ly = x1 + 0.75 * dist * np.cos(angle1), y1 + 0.75 * dist * np.sin(angle1)
        if label_flag: axis.text(np.mean([x1, mean_x]), np.mean([y1, mean_y]), '{:.1e}'.format(sum), ha='center', va='center', fontsize=24, color=color, fontweight='bold', rotation = np.degrees(box_angle), bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', edgecolor = color))
        #if label_flag: axis.text(lx, ly, '{:.1e}'.format(sum), ha='center', va='center', fontsize=24, color=color,
        #                         fontweight='bold', rotation=np.degrees(box_angle), bbox=dict(facecolor='white',
        #                         alpha=0.7, boxstyle='round', edgecolor=color))
        x2s, y2s = [], []
        xlows, ylows, xhighs, yhighs = [], [], [], []
        for i in range(0, len(values)):
            idx = np.where(abs(log_array - abs(values[i])) == np.min(abs(log_array - abs(values[i]))))[0][0]
            length = linear_array[idx]
            dist_i = 0.5 * i * (dist / len(values))
            x2s.append(x1 + dist_i * np.cos(angle))
            y2s.append(y1 + dist_i * np.sin(angle))
            xlows.append(x2s[i] - length*np.cos(angle + np.pi/2.))
            ylows.append(y2s[i] - length*np.sin(angle + np.pi/2.))
            xhighs.append(x2s[i] + length*np.cos(angle + np.pi/2.))
            yhighs.append(y2s[i] + length*np.sin(angle + np.pi/2.))
        ###plt.fill(np.append(xlows, xhighs[::-1]), np.append(ylows, yhighs[::-1]), color=color, alpha = 0.5)
        #for i in range(0, len(values)):
        #    if i < len(values) - 1 and i % 28 == 0:
        #        x_low, y_low =
        #        axis.plot([x2s[i] - length*np.cos(angle + np.pi/2.), x2s[i] + length*np.cos(angle + np.pi/2.)],
        #                  [y2s[i] - length*np.sin(angle + np.pi/2.), y2s[i] + length*np.sin(angle + np.pi/2.)],
        #                  color=color, alpha = 1.0)

    return()


def draw_flow_vectors2(axis, points, angles, color, values, direction, label_flag):

    if direction == 'prod': delta_x = 0.5
    if direction == 'cons': delta_x = 0.5

    values = values[values != 0.0]
    sum = params.write_dt * np.sum(values)
    linear_array, log_array = np.linspace(0.5, 2.0, 1000), np.logspace(-4, 0, 1000)

    if np.nansum(values) > 0.0: values = np.convolve(values, np.ones(28) / 28, mode='same')

    if np.mean(values) < 0.0: color = 'red'

    for a in range(0, len(points)):
        angle1, angle2 = radians(angles[a][0]), radians(angles[a][1])
        coord1, coord2 = np.array(points[a][0]), np.array(points[a][1])
        x1, x2, y1, y2 = coord1[0], coord2[0], coord1[1], coord2[1]
        x1, x2 = x1 + 0.9*radius * np.cos(angle1), x2 + radius * np.cos(angle2)
        y1, y2 = y1 + 0.9*radius * np.sin(angle1), y2 + radius * np.sin(angle2)
        arrow_width = np.where(abs(log_array - abs(sum)) == np.min(abs(log_array - abs(sum))))[0][0]
        axis.arrow(x1, y1, x2 - x1, y2 - y1, width=linear_array[arrow_width], length_includes_head=True, head_width = 1.2*linear_array[arrow_width],  color=color, alpha=0.3)
        axis.arrow(x1, y1, x2 - x1, y2 - y1, width=linear_array[arrow_width], length_includes_head=True, head_width=1.2*linear_array[arrow_width], color='black', alpha=1.0, fill=False)
        dist = math.dist([x1, y1], [x2, y2])
        angle = vector_angle([x1, y1], [x2, y2])
        box_angle = angle
        if 2.*np.pi > angle > np.pi: box_angle = (angle + np.pi) % (2.*np.pi)
        if angle < 0.0: box_angle = (angle + np.pi) % (2.*np.pi)
        if angle == np.pi: box_angle = 0.0
        if box_angle == 0.0: box_angle = np.pi/4.
        if box_angle > np.pi/2.: box_angle = (box_angle - np.pi)
        if box_angle == np.pi/2.: box_angle = 0.0
        #print(np.degrees(angle), np.degrees(box_angle), '{:.1e}'.format(sum))
        mean_x, mean_y = np.mean([x1, x2]), np.mean([y1, y2])
        if direction == 'prod':
            box_color = 'green'
            lx, ly = x1 + 0.75 * dist * np.cos(angle), y1 + 0.75 * dist * np.sin(angle)
        if direction == 'cons':
            box_color = 'red'
            lx, ly = x1 + 0.25 * dist * np.cos(angle), y1 + 0.25 * dist * np.sin(angle)
        #if label_flag: axis.text(np.mean([x1, mean_x]), np.mean([y1, mean_y]), '{:.1e}'.format(sum), ha='center', va='center', fontsize=24, color=color, fontweight='bold', rotation = np.degrees(box_angle), bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', edgecolor = color))
        if label_flag: axis.text(lx, ly, '{:.1e}'.format(sum), ha='center',
                                 va='center', fontsize=12 + 10*linear_array[arrow_width], color=box_color, fontweight='bold', rotation=np.degrees(box_angle),
                                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', edgecolor='white'))
        x2s, y2s = [], []
        xlows, ylows, xhighs, yhighs = [], [], [], []
        if direction == 'prod':
            x1 = x1 + 0.5 * dist * np.cos(angle)
            y1 = y1 + 0.5 * dist * np.sin(angle)
        for i in range(0, len(values)):
            idx = np.where(abs(log_array - abs(values[i])) == np.min(abs(log_array - abs(values[i]))))[0][0]
            length = linear_array[idx]
            dist_i = delta_x * i * (dist / len(values))
            x2s.append(x1 + dist_i * np.cos(angle))
            y2s.append(y1 + dist_i * np.sin(angle))
            xlows.append(x2s[i] - length*np.cos(angle + np.pi/2.))
            ylows.append(y2s[i] - length*np.sin(angle + np.pi/2.))
            xhighs.append(x2s[i] + length*np.cos(angle + np.pi/2.))
            yhighs.append(y2s[i] + length*np.sin(angle + np.pi/2.))
        #plt.plot(xlows, ylows, color=box_color, linewidth=5.0)
        #plt.plot(xhighs, yhighs, color=box_color, linewidth=5.0)
        ###plt.fill(np.append(xlows, xhighs[::-1]), np.append(ylows, yhighs[::-1]), color=color, alpha = 0.5)
        #for i in range(0, len(values)):
        #    if i < len(values) - 1 and i % 28 == 0:
        #        x_low, y_low =
        #        axis.plot([x2s[i] - length*np.cos(angle + np.pi/2.), x2s[i] + length*np.cos(angle + np.pi/2.)],
        #                  [y2s[i] - length*np.sin(angle + np.pi/2.), y2s[i] + length*np.sin(angle + np.pi/2.)],
        #                  color=color, alpha = 1.0)

    return()


def read_2d_output(file, v_temp):

    with open(file, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    unit = np.array(data)[0][0]
    #print(file, np.shape(data))
    x, y, v = np.array(data)[2:-1, 1], np.array(data)[1, 2:-1], np.transpose(np.array(data)[2:-1, 2:-1])
    y, v = y.astype(float), v.astype(float)
    v[v_temp < 0.0] = np.nan
    t = datestrings2datetimes(x)

    return(unit, t, y, v)


def read_2d_output_no_nan(file):

    with open(file, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    unit = np.array(data)[0][0]
    #print(file, np.shape(data))
    x, y, v = np.array(data)[2:-1, 1], np.array(data)[1, 2:-1], np.transpose(np.array(data)[2:-1, 2:-1])
    y, v = y.astype(float), v.astype(float)
    t = datestrings2datetimes(x)

    return(unit, t, y, v)


def read_1d_output(file):

    with open(file, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    # Read unit, time data (x), and 1D values (v)
    unit = data[0][0]
    x, v = [], []
    for i in range(0, len(data[2:])):
        x.append(data[2:][i][1])
        v.append(-float(data[2:][i][2]))
    x, v = np.array(x), np.array(v)
    t = datestrings2datetimes(x)

    return(unit, t, v)



def column_totals_and_averages(quantities, times):

    avg_quantities = []
    tot_quantities = []
    avg_times = []
    for i in range(0,len(times)):
        quantities_i = quantities[:, i]
        indices = np.where(~np.isnan(quantities_i))[0]
        average_quantity_i = 0.0
        total_quantity_i = 0.0
        average_time = 0.0
        if np.sum(quantities_i[indices]) > 0:
            average_quantity_i = np.average(quantities_i[indices])
            total_quantity_i = params.dz * np.sum(quantities_i[indices])
            average_time = times[i]
        avg_quantities.append(average_quantity_i)
        tot_quantities.append(total_quantity_i)

    return(tot_quantities, avg_quantities, avg_times)


def format_axis(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(' ')

    return()


def datestrings2datetimes(datestrings):

    date_format, date_objects = '%Y-%m-%d %H:%M:%S', []
    for date_str in datestrings: date_objects.append(datetime.strptime(date_str, date_format))

    return(date_objects)


def node_angle(key):

    doc_pos, ch4_pos, h2_pos, ace_pos, co2_pos, som_pos, o2_pos, aero_pos, leg1_pos, leg2_pos, o2_pos2 = \
        [20, 10], [20, 30], [30, 20], [20, 20], [10, 20], [10, 10], [30, 30], [20, 0], [32.5, -2], [32.5, 15], [30, 10]
    co2df_pos, ch4df_pos, o2df_pos, h2df_pos = [0, 20], [20, 40], [40, 40], [40, 20]
    co2pf_pos, ch4pf_pos, o2pf_pos, h2pf_pos = [0, 20], [20, 40], [40, 40], [40, 20]

    doc_prod_ace_ansp, doc_prod_co2_ansp, doc_prod_h2_ansp = [75, 285], [120, 330], [45, 225]
    doc_prod_ace_aesp, doc_prod_co2_aesp = [105, 255], [150, 300]
    co2_prod_ace_hoag, h2_prod_ace_hoag = [345, 195], [180, 0]
    co2_prod_ch4_h2mg, h2_prod_ch4_h2mg = [60, 210], [135, 315]
    ace_prod_ch4_acmg, ace_prod_co2_acmg = [90, 270], [165, 15]
    ch4_prod_co2_mtox, o2_prod_co2_mtox = [240, 30], [180, 0]
    doc_prod_aero_aero, o2_prod_doc_aero = [270, 90], [180, 0]
    som_prod_doc_break = [315, 135]
    legend_angles = [[90, 270]]
    co2df, ch4df, o2df, h2df = [210, 330], [120, 240], [210, 60], [30, 150]
    co2pf, ch4pf, o2pf, h2pf = [150, 30], [60, 300], [240, 30], [330, 210]

    keys = ['co2_prod_ansp', 'h2_prod_ansp', 'ace_prod_aesp', 'co2_prod_aesp', 'o2_cons_aesp', 'ace_prod_hoag', 'co2_cons_hoag', 'h2_cons_hoag', 'ch4_prod_h2mg', 'co2_cons_h2mg', 'h2_cons_h2mg', 'ch4_cons_mtox', 'o2_cons_mtox', 'ace_cons_acmg', 'ch4_prod_acmg', 'co2_prod_acmg', 'o2_cons_aero']
    if key == 'co2_prod_ansp': pos, angle, color = [doc_pos, co2_pos], doc_prod_co2_ansp, 'gray'
    if key == 'h2_prod_ansp': pos, angle, color = [doc_pos, h2_pos], doc_prod_h2_ansp, 'gray'
    if key == 'ace_prod_ansp': pos, angle, color = [doc_pos, ace_pos], doc_prod_ace_ansp, 'gray'
    if key == 'ace_prod_aesp': pos, angle, color = [doc_pos, ace_pos], doc_prod_ace_aesp, 'black'
    if key == 'co2_prod_aesp': pos, angle, color = [doc_pos, co2_pos], doc_prod_co2_aesp, 'black'
    if key == 'o2_cons_aesp': pos, angle, color = [doc_pos, aero_pos], doc_prod_aero_aero, 'orange'
    if key == 'ace_prod_hoag': pos, angle, color = [co2_pos, ace_pos], co2_prod_ace_hoag, 'lightskyblue'
    if key == 'co2_cons_hoag': pos, angle, color = [co2_pos, ace_pos], co2_prod_ace_hoag, 'lightskyblue'
    if key == 'h2_cons_hoag': pos, angle, color = [h2_pos, ace_pos], h2_prod_ace_hoag, 'lightskyblue'
    if key == 'ch4_prod_h2mg': pos, angle, color = [co2_pos, ch4_pos], co2_prod_ch4_h2mg, 'blue'
    if key == 'co2_cons_h2mg': pos, angle, color = [co2_pos, ch4_pos], co2_prod_ch4_h2mg, 'blue'
    if key == 'h2_cons_h2mg': pos, angle, color = [h2_pos, ch4_pos], h2_prod_ch4_h2mg, 'blue'
    if key == 'co2_prod_mtox': pos, angle, color = [ch4_pos, co2_pos], ch4_prod_co2_mtox, 'magenta'
    if key == 'ch4_cons_mtox': pos, angle, color = [ch4_pos, co2_pos], ch4_prod_co2_mtox, 'magenta'
    if key == 'o2_cons_mtox': pos, angle, color = [o2_pos, ch4_pos], o2_prod_co2_mtox, 'magenta'
    if key == 'ace_cons_acmg': pos, angle, color = [ace_pos, ch4_pos], ace_prod_ch4_acmg, 'purple'
    if key == 'ch4_prod_acmg': pos, angle, color = [ace_pos, ch4_pos], ace_prod_ch4_acmg, 'purple'
    if key == 'co2_prod_acmg': pos, angle, color = [ace_pos, co2_pos], ace_prod_co2_acmg, 'purple'
    if key == 'o2_cons_aero': pos, angle, color = [o2_pos2, doc_pos], o2_prod_doc_aero, 'orange'

    return(pos, angle, color)

unit_diff, t_diff, v_diff = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/df/f_diff_ch4.csv')
unit_plant, t_plant, v_plant = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/pf/plant_flux_ch4.csv')
ch4_dfluxes = v_diff
ch4_pfluxes = v_plant

unit_diff, t_diff, v_diff = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/df/f_diff_co2.csv')
unit_plant, t_plant, v_plant = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/pf/plant_flux_co2.csv')
co2_dfluxes = v_diff
co2_pfluxes = v_plant

unit_diff, t_diff, v_diff = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/df/f_diff_o2.csv')
unit_plant, t_plant, v_plant = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/pf/plant_flux_o2.csv')
o2_dfluxes = v_diff
o2_pfluxes = v_plant

unit_diff, t_diff, v_diff = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/df/f_diff_h2.csv')
unit_plant, t_plant, v_plant = read_1d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/pf/plant_flux_h2.csv')
h2_dfluxes = v_diff
h2_pfluxes = v_plant

# Flow plot
fig = plt.figure(figsize=(25, 35))
gsf = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
                      left=0.1, right=0.9, bottom=0.05, top=0.95,
                      wspace=0.4, hspace=0.3)

flow_ax = fig.add_subplot(gsf[0,0])
flow_ax.set_aspect('equal')
flow_ax.set_xlim([0,40])
flow_ax.set_ylim([-10,40])
flow_ax.plot([0],[0])

radius = 3.0

doc_pos, ch4_pos, h2_pos, ace_pos, co2_pos, som_pos, o2_pos, aero_pos, leg1_pos, leg2_pos, o2_pos2 = \
[20, 10], [20, 30], [30,20], [20,20], [10,20], [10,10], [30,30], [10,10], [32.5, -2], [32.5, 15], [30, 10]
co2df_pos, ch4df_pos, o2df_pos, h2df_pos = [0,20], [20,40], [30,40], [40,20]
co2pf_pos, ch4pf_pos, o2pf_pos, h2pf_pos = [0,20], [20,40], [30,40], [40,20]

circles = [doc_pos, ch4_pos, h2_pos, ace_pos, co2_pos, som_pos, o2_pos]
ccolors = ['gold', 'cyan', 'cyan', 'plum', 'cyan', 'lightsalmon', 'cyan']
clabels = ['DOC', 'CH$_4$', 'H$_2$', 'Ace', 'CO$_2$', 'SOM', 'O$_2$']

doc_prod_ace_ansp, doc_prod_co2_ansp, doc_prod_h2_ansp = [240, 120], [180, 0], [0, 180]
doc_prod_ace_aesp, doc_prod_co2_aesp = [300, 60], [210, 330]
co2_prod_ace_hoag, h2_prod_ace_hoag = [330, 120], [225, 45]
co2_prod_ch4_h2mg, h2_prod_ch4_h2mg = [60, 210], [135, 315]
ace_prod_ch4_acmg, ace_prod_co2_acmg = [90, 270], [150, 300]
ch4_prod_co2_mtox, o2_prod_co2_mtox = [240, 30], [180, 0]
doc_prod_aero_aero, o2_prod_doc_aero = [180, 0], [225, 45]
som_prod_doc_break = [315, 135]
legend_angles = [[90, 270]]
co2df, ch4df, o2df, h2df = [210,330], [120, 240], [240, 120], [30, 150]
co2pf, ch4pf, o2pf, h2pf = [150,30], [60, 300], [300, 60], [330, 210]

ansp_angles = [doc_prod_ace_ansp, doc_prod_co2_ansp, doc_prod_h2_ansp]
aesp_angles = [doc_prod_ace_aesp, doc_prod_co2_aesp]
hoag_angles = [co2_prod_ace_hoag, h2_prod_ace_hoag]
h2mg_angles = [co2_prod_ch4_h2mg, h2_prod_ch4_h2mg]
acmg_angles = [ace_prod_ch4_acmg, ace_prod_co2_acmg]
mtox_angles = [ch4_prod_co2_mtox, o2_prod_co2_mtox]
aero_angles = [doc_prod_aero_aero, o2_prod_doc_aero]
break_angles = [som_prod_doc_break]
co2df_angles, ch4df_angles, o2df_angles, h2df_angles = [co2df], [ch4df], [o2df], [h2df]
co2pf_angles, ch4pf_angles, o2pf_angles, h2pf_angles = [co2pf], [ch4pf], [o2pf], [h2pf]

ansp_points = [[doc_pos, ace_pos], [doc_pos, co2_pos], [doc_pos, h2_pos]]
aesp_points = [[doc_pos, ace_pos], [doc_pos, co2_pos]]
hoag_points = [[co2_pos, ace_pos], [h2_pos, ace_pos]]
h2mg_points = [[co2_pos, ch4_pos], [h2_pos, ch4_pos]]
acmg_points = [[ace_pos, ch4_pos], [ace_pos, co2_pos]]
mtox_points = [[ch4_pos, co2_pos], [o2_pos, ch4_pos]]
aero_points = [[doc_pos, aero_pos], [o2_pos, doc_pos]]
break_points = [[som_pos, doc_pos]]
co2df_points, ch4df_points, o2df_points, h2df_points = \
    [[co2_pos, co2df_pos]], [[ch4_pos, ch4df_pos]], [[o2df_pos, o2_pos]], [[h2_pos, h2df_pos]]
co2pf_points, ch4pf_points, o2pf_points, h2pf_points = \
    [[co2_pos, co2pf_pos]], [[ch4_pos, ch4pf_pos]], [[o2pf_pos, o2_pos]], [[h2_pos, h2pf_pos]]
legend_points = [[leg1_pos, leg2_pos]]

#ansp_ratios = [1.0, 0.5, (1.0/6.0)]
#aesp_ratios = [1.0, 0.5]
#hoag_ratios =

# Get temperature profile data to mask values where T < 0C
with open('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/en/temp_data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
x_temp, y_temp, v_temp = np.array(data)[2:-1,1], np.array(data)[1,2:-1], np.transpose(np.array(data)[2:-1,2:-1])
y_temp, v_temp = y_temp.astype(float), v_temp.astype(float)

ansp_unit, ansp_t, ansp_y, ansp_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/crd2/ansp_ref_rate.csv', v_temp)
ansp_tots = np.array(column_totals_and_averages(ansp_v, ansp_t)[0])
aesp_unit, aesp_t, aesp_y, aesp_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/crd2/aesp_ref_rate.csv', v_temp)
aesp_tots = np.array(column_totals_and_averages(aesp_v, aesp_t)[0])
hoag_unit, hoag_t, hoag_y, hoag_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/crd2/hoag_ref_rate.csv', v_temp)
hoag_tots = np.array(column_totals_and_averages(hoag_v, hoag_t)[0])
h2mg_unit, h2mg_t, h2mg_y, h2mg_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/crd2/h2mg_ref_rate.csv', v_temp)
h2mg_tots = np.array(column_totals_and_averages(h2mg_v, h2mg_t)[0])
mtox_unit, mtox_t, mtox_y, mtox_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/crd2/mtox_ref_rate.csv', v_temp)
mtox_tots = np.array(column_totals_and_averages(mtox_v, mtox_t)[0])
acmg_unit, acmg_t, acmg_y, acmg_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/crd2/acmg_ref_rate.csv', v_temp)
acmg_tots = np.array(column_totals_and_averages(acmg_v, acmg_t)[0])
aero_unit, aero_t, aero_y, aero_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/crd2/aero_ref_rate.csv', v_temp)
aero_tots = np.array(column_totals_and_averages(aero_v, aero_t)[0])

ch4_unit, ch4_t, ch4_y, ch4_v = read_2d_output_no_nan('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/ch4.csv')
ch4_tots = np.array(column_totals_and_averages(ch4_v, ch4_t)[0])
co2_unit, co2_t, co2_y, co2_v = read_2d_output_no_nan('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/co2.csv')
co2_tots = np.array(column_totals_and_averages(co2_v, co2_t)[0])
ace_unit, ace_t, ace_y, ace_v = read_2d_output_no_nan('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/ace.csv')
ace_tots = np.array(column_totals_and_averages(ace_v, ace_t)[0])
o2_unit, o2_t, o2_y, o2_v = read_2d_output_no_nan('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/o2.csv')
o2_tots = np.array(column_totals_and_averages(o2_v, o2_t)[0])
h2_unit, h2_t, h2_y, h2_v = read_2d_output_no_nan('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/h2.csv')
h2_tots = np.array(column_totals_and_averages(h2_v, h2_t)[0])
doc_unit, doc_t, doc_y, doc_v = read_2d_output_no_nan('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/doc.csv')
doc_tots = np.array(column_totals_and_averages(doc_v, doc_t)[0])

mypath = params.main_directory + 'modelOutput/' + params.site + '/cr'
cr_onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

data_dict, pos_dict, angle_dict, color_dict = {}, {}, {}, {}
reactions = ['ansp', 'aesp', 'hoag', 'h2mg', 'mtox', 'acmg', 'aero']
ch4_rates, co2_rates, o2_rates, h2_rates, ace_rates, doc_rates = [], [], [], [], [], []
for reaction in reactions:
    files = find_elements_with_substring(cr_onlyfiles, reaction)
    for i in range(0, len(files)):
        chem, direction = files[i].split('_')[0], files[i].split('_')[2]
        c_unit, c_t, c_y, c_v = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/cr/' + files[i], v_temp)
        c_tots = np.array(column_totals_and_averages(c_v, c_t)[0])
        rate2add = params.write_dt * np.nansum(c_tots)
        if direction == 'cons': rate2add = -1.0 * rate2add
        if chem == 'ch4': ch4_rates.append(rate2add)
        if chem == 'co2': co2_rates.append(rate2add)
        if chem == 'o2': o2_rates.append(rate2add)
        if chem == 'h2': h2_rates.append(rate2add)
        if chem == 'ace': ace_rates.append(rate2add)
        if chem == 'doc': doc_rates.append(rate2add)
        key = chem + '_' + direction + '_' + reaction
        print(key)
        pos, angle, color = node_angle(key)
        print(key, pos, angle)
        data_dict[key], pos_dict, angle_dict = c_tots, pos, angle
        draw_flow_vectors2(flow_ax, [pos], [angle], color, c_tots, direction, True)

print(ch4_rates)

#print(info_dict.keys())

method = 2
if method == 2:
    #draw_flow_vectors(flow_ax, ansp_points, ansp_angles, 'gray', ansp_tots, True)
    #draw_flow_vectors(flow_ax, aesp_points, aesp_angles, 'black', aesp_tots, True)
    #draw_flow_vectors(flow_ax, hoag_points, hoag_angles, 'lightskyblue', hoag_tots, True)
    #draw_flow_vectors(flow_ax, h2mg_points, h2mg_angles, 'blue', h2mg_tots, True)
    #draw_flow_vectors(flow_ax, mtox_points, mtox_angles, 'magenta', mtox_tots, True)
    #draw_flow_vectors(flow_ax, acmg_points, acmg_angles, 'purple', acmg_tots, True)
    #draw_flow_vectors(flow_ax, aero_points, aero_angles, 'orange', aero_tots, True)
    draw_flow_vectors(flow_ax, ch4df_points, ch4df_angles, 'lightgreen', ch4_dfluxes, True)
    draw_flow_vectors(flow_ax, co2df_points, co2df_angles, 'lightgreen', co2_dfluxes, True)
    draw_flow_vectors(flow_ax, o2df_points, o2df_angles, 'lightgreen', o2_dfluxes, True)
    draw_flow_vectors(flow_ax, h2df_points, h2df_angles, 'lightgreen', h2_dfluxes, True)
    draw_flow_vectors(flow_ax, ch4pf_points, ch4pf_angles, 'green', ch4_pfluxes, True)
    draw_flow_vectors(flow_ax, co2pf_points, co2pf_angles, 'green', co2_pfluxes, True)
    draw_flow_vectors(flow_ax, o2pf_points, o2pf_angles, 'green', o2_pfluxes, True)
    draw_flow_vectors(flow_ax, h2pf_points, h2pf_angles, 'green', h2_pfluxes, True)
    #draw_flow_vectors(flow_ax, legend_points, legend_angles, 'gray', np.logspace(-5, -1, 1000), False)

#draw_flow_vectors(flow_ax, break_points, break_angles, 'black')
#draw_flow_vectors(flow_ax, co2f_points, co2f_angles, 'black')
#draw_flow_vectors(flow_ax, ch4f_points, ch4f_angles, 'black')
#draw_flow_vectors(flow_ax, o2f_points, o2f_angles, 'black')
#draw_flow_vectors(flow_ax, h2f_points, h2f_angles, 'black')

circles = [doc_pos, ch4_pos, h2_pos, ace_pos, co2_pos, som_pos, o2_pos]
ccolors = ['gold', 'cyan', 'cyan', 'plum', 'cyan', 'lightsalmon', 'cyan']
clabels = ['DOC', 'CH$_4$', 'H$_2$', 'Ace', 'CO$_2$', 'SOM', 'O$_2$']

draw_circle(flow_ax, ch4_pos, radius, 'cyan', 'CH$_4$', np.sum(np.array(ch4_rates)))
draw_circle(flow_ax, co2_pos, radius, 'cyan', 'CO$_2$', np.sum(np.array(co2_rates)))
draw_circle(flow_ax, o2_pos, radius, 'cornflowerblue', 'O$_2$', np.sum(np.array(o2_rates)))
draw_circle(flow_ax, h2_pos, radius, 'cyan', 'H$_2$', np.sum(np.array(h2_rates)))
draw_circle(flow_ax, ace_pos, radius, 'plum', 'Ace', np.sum(np.array(ace_rates)))
draw_circle(flow_ax, doc_pos, radius, 'gold', 'DOC', np.sum(np.array(doc_rates)))
draw_circle(flow_ax, o2_pos2, radius, 'cornflowerblue', 'O$_2$', 0.0)
draw_circle(flow_ax, som_pos, radius, 'lightsalmon', 'SOC', 0.0)

#for c in range(0,len(circles)):
#
#    coord = circles[c]
#    draw_circle(flow_ax, coord, radius, ccolors[c], clabels[c])

arrow = r"$\rightarrow$"
ansp_eq = 'DOC ' + arrow + ' 0.67 Ace + 0.33 CO$_2$ + 0.11 H$_2$'
aesp_eq = 'DOC ' + arrow + ' Ace + 0.5 CO$_2$'
hoag_eq = '4 H$_2$ + 2 CO$_2$' + arrow + ' Ace + 2 H$_2$O'
h2mg_eq = '4 H$_2$ + CO$_2$' + arrow + ' CH$_4$ + 2 H$_2$O'
mtox_eq = 'CH$_4$ + 2 O$_2$' + arrow + ' CO$_2$ + 2 H$_2$O'
acmg_eq = 'Ace' + arrow + ' CH$_4$ + 2 CO$_2$'

list1 = [Line2D([], [], marker='s', color = x, alpha = 0.5, markersize = 20, linewidth = 0) for x in ['lightgreen', 'green']]
labels1 = ['F$_{diffusion}$', 'F$_{plant}$']
flow_ax.legend(handles = list1, loc = 'lower left', fontsize=32, labels = labels1, frameon = False, ncol = 1, columnspacing=0.8, title='Fluxes [mol/m$^2$/year]', title_fontsize = 36)

test = fig.add_axes([0.9, 0.15, 0.001, 0.001])

list2 = [Line2D([], [], marker='s', color = x, alpha = 0.5, markersize = 20, linewidth = 0) for x in ['gray', 'black', 'lightskyblue', 'blue', 'magenta', 'purple' , 'orange']]
labels2 = ['ANSP: ' + ansp_eq, 'AESP: ' + aesp_eq, 'HOAG: ' + hoag_eq, 'H2MG: ' + h2mg_eq, 'MTOX: ' + mtox_eq, 'ACMG: ' + acmg_eq, 'AERO']
test.legend(handles = list2, loc = 'lower right', fontsize=32, labels = labels2, frameon = False, ncol = 1, columnspacing=0.8, title='Reactions [mol/m$^2$/year]', title_fontsize = 36)

plt.axis('off')

#levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#log_array = np.logspace(-5, -1, 1000)
#legend_base_y, legend_base_x, legend_height = leg1_pos[1], leg1_pos[0], leg2_pos[1] - leg1_pos[1] - 2.0*radius
#for l in range(0, len(levels)):
#    idx = np.where(abs(log_array - levels[l]) == np.min(abs(log_array - levels[l])))[0][0]
#    frac = idx / len(log_array)
#    annotation_y = 0.75 * (legend_base_y + legend_height * frac)
#    flow_ax.text(legend_base_x + 3.0, annotation_y - 0.3 + radius, str(levels[l]), ha='center', va='center', fontsize=32, color='black')
#flow_ax.text(legend_base_x + 1.5, legend_base_y + legend_height + 4.0, 'mol/m$^2$/day', ha='center', va='center', fontsize=32, color='black')

plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/fancy/flow2.png',bbox_inches='tight')

# chemical plot
months_short = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
fig = plt.figure(figsize=(7.5, 5))
gsc = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
                      left=0.1, right=0.9, bottom=0.05, top=0.95,
                      wspace=0.4, hspace=0.3)
ca_ax = fig.add_subplot(gsc[0,0])
ca_ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha = 0.5)
#ca_ax.set_facecolor('gainsboro', alpha = 0.5)
ca_ax.patch.set_facecolor('gainsboro')
ca_ax.patch.set_alpha(0.5)
ca_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
ca_ax.set_xticklabels(months_short)
ca_ax.set_ylabel('$\Delta$' + ' mol/m$^2$', fontsize = 16)
ca_ax.tick_params(axis = 'both', labelsize=12)
ca_ax.set_title('Year-Change\nChemical Column Abundances\n', fontweight='bold', fontsize=14)

ca_ax.plot(doc_t, (doc_tots-doc_tots[0])/10.0, linewidth = 8.0, color = 'black', alpha = 0.5)
ca_ax.plot(doc_t, (doc_tots-doc_tots[0])/10.0, linewidth = 6.0, color = 'gold', alpha = 1.0)

all_array = np.array([[ch4_tots-ch4_tots[0]], [co2_tots-co2_tots[0]], [h2_tots-h2_tots[0]], [o2_tots-o2_tots[0]], [ace_tots-ace_tots[0]]])
ca_ax_min, ca_ax_max = -1.2 * np.nanmax(abs(all_array)), 1.2 * np.nanmax(abs(all_array))
ca_ax.set_ylim([ca_ax_min, ca_ax_max])

ca_ax.plot(ch4_t, ch4_tots-ch4_tots[0], linewidth = 8.0, color = 'black', alpha = 0.5)
ca_ax.plot(ch4_t, ch4_tots-ch4_tots[0], linewidth = 6.0, color = 'powderblue', alpha = 1.0)

ca_ax.plot(ch4_t, co2_tots-co2_tots[0], linewidth = 8.0, color = 'black', alpha = 0.5)
ca_ax.plot(co2_t, co2_tots-co2_tots[0], linewidth = 6.0, color = 'lightskyblue', alpha = 1.0)

ca_ax.plot(h2_t, h2_tots-h2_tots[0], linewidth = 8.0, color = 'black', alpha = 0.5)
ca_ax.plot(h2_t, h2_tots-h2_tots[0], linewidth = 6.0, color = 'cyan', alpha = 1.0)

ca_ax.plot(o2_t, o2_tots-o2_tots[0], linewidth = 8.0, color = 'black', alpha = 0.5)
ca_ax.plot(o2_t, o2_tots-o2_tots[0], linewidth = 6.0, color = 'cornflowerblue')

ca_ax.plot(ace_t, ace_tots-ace_tots[0], linewidth = 8.0, color = 'black', alpha = 0.5)
ca_ax.plot(ace_t, ace_tots-ace_tots[0], linewidth = 6.0, color = 'magenta')

list1 = [Line2D([], [], marker='s', color = x, alpha = 1.0, markersize = 0, linewidth = 8) for x in ['powderblue', 'lightskyblue', 'cyan', 'cornflowerblue', 'magenta', 'gold']]
labels1 = ['CH$_4$', 'CO$_2$', 'H$_2$', 'O$_2$', 'Ace', 'DOC / 10']
ca_ax.legend(handles = list1, loc = 'lower left', fontsize=12, labels = labels1, frameon = True, ncol = 2, columnspacing=0.8, edgecolor='white', labelspacing = 1.0, facecolor = 'white', framealpha = 1.0)

format_axis(ca_ax)

plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/fancy/totals.png',bbox_inches='tight')

# test test test