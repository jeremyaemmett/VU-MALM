from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyrolite.util.plot.style import ternary_color
from matplotlib.collections import PatchCollection
from pyrolite.util.synthetic import normal_frame
from matplotlib.ticker import LogFormatter
from IPython.display import display, Latex
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib import ticker, cm
from os.path import isfile, join
from data import read_site_data
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from matplotlib import cm
from datetime import date
import matplotlib as mpl
from os import listdir
from copy import copy
from numpy import ma
import pandas as pd
import numpy as np
import conversions
import matplotlib
import transport
import diffusion
import pathways
import params
import math
import init
import csv
import os

def log_contour(xvals, yvals, vvals, cmap, axis, n_levels, title, cb_label, xlabel_flag):

    loglvls = np.linspace(np.nanmin(vvals), np.nanmax(vvals), 100)
    if np.nanmax(loglvls) == 0.0: loglvls = [0.0, 1.0]
    #log_min, log_max = np.floor(np.log10(np.nanmin(vvals))), np.floor(np.log10(np.nanmax(vvals)))+1
    #if np.nanmin(vvals) > 0.0 and np.nanmax(vvals)  > 0.0:
    #    loglvls = 10**np.linspace(int(log_min),int(log_max),2*(abs(int(log_min))-abs(int(log_max)))+1)
    #if len(loglvls) == 1: loglvls = np.linspace(np.nanmin(vvals), np.nanmax(vvals), 100)

    axis.set_ylim([0.45, 0.025])
    axis.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
    if xlabel_flag:
        axis.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'], fontsize=12, rotation = 45)
    else:
        axis.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' '], fontsize=14)
    axis.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=14)
    axis.set_ylabel('Depth [m]', fontsize=16)
    axis.set_title(title, fontsize=16)
    axis.patch.set_facecolor('gainsboro')
    axis.patch.set_alpha(0.5)
    format_axis(axis)
    #c3 = axis.contourf(xvals, yvals, vvals, cmap=cmap, levels=log_lvls(vvals, n_levels),
    #                    norm=matplotlib.colors.LogNorm())
    #c3 = axis.contourf(xvals, yvals, vvals, cmap=cmap, levels=loglvls,
    #                   norm=matplotlib.colors.LogNorm())
    c3 = axis.contourf(xvals, yvals, vvals, cmap=cmap, levels=loglvls)
    cbar = fig.colorbar(c3, orientation='vertical', ax=axis, label = cb_label)

    return()


def column_averages(quantities, times):

    avg_quantities = []
    avg_times = []
    for i in range(0,len(times)):
        quantities_i = quantities[:, i]
        indices = np.where(~np.isnan(quantities_i))[0]
        average_quantity_i = 0.0
        average_time = 0.0
        if np.sum(quantities_i[indices]) > 0:
            average_quantity_i = np.average(quantities_i[indices])
            average_time = times[i]
        avg_quantities.append(average_quantity_i)

    return(avg_quantities, avg_times)


def weighted_column_averages(refrates, quantities, times):

    avg_weights = []
    avg_quantities = []
    avg_times = []
    for i in range(0,len(times)):
        refrates_i = refrates[:, i]
        quantities_i = quantities[:, i]
        indices = np.where(~np.isnan(refrates_i))[0]
        average_quantity_i = np.nan
        average_weight_i = np.nan
        average_time = np.nan
        if np.sum(refrates_i[indices]) > 0:
            average_quantity_i = np.average(quantities_i[indices], weights = refrates_i[indices])
            average_weight_i = np.average(refrates_i[indices], weights = refrates_i[indices])
            average_time = times[i]
            condition1 = (times[i].day == 1 or times[i].day == 15) and times[i].hour == 0
            condition2 = 6 <= times[i].month <= 9
            if condition2: avg_quantities.append(average_quantity_i)
            if condition2: avg_weights.append(average_weight_i)
            if condition2: avg_times.append(average_time)

    return(avg_weights, avg_quantities, avg_times)

def plot3d2(reaction, axis, refreaction_dict, microbe_dict, chemical_dict, temps):

    ref_reaction = refreaction_dict[reaction]
    times = refreaction_dict['times']

    ph = 6.0
    ga = {}
    ga['acemethanogen_genes'], ga['h2_methanogen_genes'], ga['homoacetogen_genes'], ga[
        'methanotroph_genes'] = 1, 1, 1, 1
    print(ga.keys())
    al = 1.0

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ca = {}

    if reaction == 'ansp':
        microbes = 1.0 + 0.0 * microbe_dict['acemethanogen']
        chemicals1 = chemical_dict['doc']
        chemicals2 = 0.0 * chemical_dict['doc'] + 1.0
        lvls = np.logspace(-1, -0.1, 25)
        # lvls = np.linspace(0.0, 1.0, 100)
        x_label = 'DOC'
        y_label = 'X'
        leading_string, v_term_string = r'$DOCprodAce  = $', r'$V_{DOCprodAce,max}$'
        sub1_term_string, sub2_term_string = r'$\frac{DOC}{K_{DOCprodAce} + DOC}$', r'$\frac{[O_2]}{K_{AceprodO2} + [O_2]}$'
        temp_factor_string, ph_factor_string = r'$f_T(AceprodQ10)$', r'$f_{pH}$'
        biomass_factor_string, times_string = r'$B_{DOCoxidizers}$', r'$\times$'
        cb_label = r'$\frac{DOCprodAce}{B_{DOCoxidizers}}$'
        vmax, sub1k, sub2k, ftq10 = params.v_doc_prod_ace_max, params.k_doc_prod_ace, params.k_ace_prod_o2, params.ace_prod_q10
    if reaction == 'aesp':
        microbes = 1.0 + 0.0 * microbe_dict['acemethanogen']
        chemicals1 = chemical_dict['doc']
        chemicals2 = chemical_dict['o2']
        lvls = np.logspace(-1, -0.1, 25)
        # lvls = np.linspace(0.0, 1.0, 100)
        x_label = 'DOC'
        y_label = 'O2'
        leading_string, v_term_string = r'$DOCprodAce  = $', r'$V_{DOCprodAce,max}$'
        sub1_term_string, sub2_term_string = r'$\frac{DOC}{K_{DOCprodAce} + DOC}$', r'$\frac{[O_2]}{K_{AceprodO2} + [O_2]}$'
        temp_factor_string, ph_factor_string = r'$f_T(AceprodQ10)$', r'$f_{pH}$'
        biomass_factor_string, times_string = r'$B_{DOCoxidizers}$', r'$\times$'
        cb_label = r'$\frac{DOCprodAce}{B_{DOCoxidizers}}$'
        vmax, sub1k, sub2k, ftq10 = params.v_doc_prod_ace_max, params.k_doc_prod_ace, params.k_ace_prod_o2, params.ace_prod_q10
    if reaction == 'acmg':
        microbes = microbe_dict['acemethanogen']
        chemicals1 = chemical_dict['ace']
        chemicals2 = 0.0 * chemical_dict['ace'] + 1.0
        lvls = np.logspace(-1, 0.2, 100)
        # lvls = np.linspace(0.0,1.0,100)
        x_label = 'Acetate'
        y_label = 'X'
        leading_string, v_term_string = r'$AceprodCH4  = $', r'$V_{Acecons,max}$'
        sub1_term_string, sub2_term_string = r'$\frac{Ace}{K_{AceprodCH4} + Ace}$', r'$\frac{[X]}{K_{X} + [X]}$'
        temp_factor_string, ph_factor_string = r'$f_T(CH4prodQ10)$', r'$f_{pH}$'
        biomass_factor_string, times_string = r'$B_{Acemethanogens}$', r'$\times$'
        cb_label = r'$\frac{AceprodCH4}{B_{Acemethanogens}}$'
        vmax, sub1k, sub2k, ftq10 = params.v_ace_cons_max, params.k_ace_prod_ch4, 0.0, params.ch4_prod_q10
    if reaction == 'h2mg':
        microbes = microbe_dict['h2_methanogen']
        chemicals1 = chemical_dict['co2']
        chemicals2 = chemical_dict['h2']
        lvls = np.logspace(-4, -2, 100)
        # lvls = np.linspace(0.0, 1.0, 100)
        x_label = 'CO2'
        y_label = 'H2'
        leading_string, v_term_string = r'$H2prodCH4  = $', r'$V_{H2prodCH4,max}$'
        sub1_term_string, sub2_term_string = r'$\frac{H2}{K_{H2prodCH4} + H2}$', r'$\frac{[CO_2]}{K_{CO2prodCH4} + [CO_2]}$'
        temp_factor_string, ph_factor_string = r'$f_{T2}$', r'$f_{pH}$'
        biomass_factor_string, times_string = r'$B_{H2methanogens}$', r'$\times$'
        cb_label = r'$\frac{H2prodCH4}{B_{H2methanogens}}$'
        vmax, sub1k, sub2k, ftq10 = params.v_h2_prod_ch4_max, params.k_h2_prod_ch4, params.k_co2_prod_ch4, 1.0
    if reaction == 'hoag':
        microbes = microbe_dict['homoacetogen']
        chemicals1 = chemical_dict['co2']
        chemicals2 = chemical_dict['h2']
        lvls = np.logspace(-1, -0.1, 100)
        # lvls = np.linspace(0.0, 1.0, 100)
        x_label = 'CO2'
        y_label = 'H2'
        leading_string, v_term_string = r'$H2prodAce  = $', r'$V_{H2prodAce,max}$'
        sub1_term_string, sub2_term_string = r'$\frac{H2}{K_{H2prodAce} + H2}$', r'$\frac{[CO_2]}{K_{CO2prodAce} + [CO_2]}$'
        temp_factor_string, ph_factor_string = r'$f_{T1}$', r'$f_{pH}$'
        biomass_factor_string, times_string = r'$B_{Homoacetogens}$', r'$\times$'
        cb_label = r'$\frac{H2prodAce}{B_{Homoacetogens}}$'
        vmax, sub1k, sub2k, ftq10 = params.v_h2_prod_ace_max, params.k_h2_prod_ace, params.k_co2_prod_ace, 1.0
    if reaction == 'mtox':
        microbes = microbe_dict['methanotroph']
        chemicals1 = chemical_dict['ch4']
        chemicals2 = chemical_dict['o2']
        lvls = np.logspace(-1, -0.1, 100)
        # lvls = np.linspace(0.0, 1.0, 100)
        x_label = 'CH4'
        y_label = 'O2'
        leading_string, v_term_string = r'$Roxid,CH4  = $', r'$V_{CH4oxid,max}$'
        sub1_term_string, sub2_term_string = r'$\frac{CH4}{K_{CH4oxidCH4} + CH4}$', r'$\frac{[O_2]}{K_{CH4oxidO2} + [O_2]}$'
        temp_factor_string, ph_factor_string = r'$f_T(CH4oxidQ10)$', r'$f_{pH}$'
        biomass_factor_string, times_string = r'$B_{Methanotrophs}$', r'$\times$'
        cb_label = r'$\frac{Roxid,CH4}{B_{Methanotrophs}}$'
        vmax, sub1k, sub2k, ftq10 = params.v_ch4_oxid_max, params.k_ch4_oxid_ch4, params.k_ch4_oxid_o2, params.ch4_oxid_q10

    fig.text(.32, .67, leading_string, horizontalalignment='center', verticalalignment='center', fontsize=14)
    fig.text(.42, .67, v_term_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'red')
    fig.text(.42, .64, '(' + str(vmax) + ')', horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'red')
    fig.text(.48, .67, times_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'black')
    fig.text(.535, .67, sub1_term_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'green')
    fig.text(.535, .64, '(' + str(sub1k) + ')', horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'green')
    fig.text(.59, .67, times_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'black')
    fig.text(.64, .67, sub2_term_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'green')
    fig.text(.64, .64, '(' + str(sub2k) + ')', horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'green')
    fig.text(.69, .67, times_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'black')
    fig.text(.755, .67, temp_factor_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'blue')
    fig.text(.755, .64, '(' + str(ftq10) + ')', horizontalalignment='center', verticalalignment='center', fontsize=14, color='blue')
    fig.text(.82, .67, times_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'black')
    fig.text(.84, .67, ph_factor_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'gold')
    fig.text(.86, .67, times_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'black')
    fig.text(.915, .67, biomass_factor_string, horizontalalignment='center', verticalalignment='center', fontsize=14, color = 'purple')

    avg_rates, avg_microbes, avg_times = weighted_column_averages(ref_reaction, microbes, times)
    if reaction == 'ansp' or reaction == 'aesp': avg_microbes = np.array(avg_microbes) + 1.0
    avg_rates, avg_chemicals1, avg_times = weighted_column_averages(ref_reaction, chemicals1, times)
    avg_rates, avg_chemicals2, avg_times = weighted_column_averages(ref_reaction, chemicals2, times)
    avg_rates, avg_temps, avg_times = weighted_column_averages(ref_reaction, temps, times)

    kernel = 2*28
    avg_chemicals1 = np.convolve(avg_chemicals1, np.ones(kernel) / kernel, mode='same')[kernel:-kernel]
    avg_chemicals2 = np.convolve(avg_chemicals2, np.ones(kernel) / kernel, mode='same')[kernel:-kernel]
    avg_temps = np.convolve(avg_temps, np.ones(kernel) / kernel, mode='same')[kernel:-kernel]
    avg_microbes = np.convolve(avg_microbes, np.ones(kernel) / kernel, mode='same')[kernel:-kernel]
    avg_rates = np.convolve(avg_rates, np.ones(kernel) / kernel, mode='same')[kernel:-kernel]

    top_temp = np.nanmax(avg_temps)
    bot_temp = np.nanmin(avg_temps)

    axis.axes.set_xlim3d(left=np.nanmin(avg_chemicals1), right=np.nanmax(avg_chemicals1))
    axis.axes.set_ylim3d(bottom=np.nanmin(avg_chemicals2), top=np.nanmax(avg_chemicals2))
    axis.axes.set_zlim3d(bottom=bot_temp, top=top_temp)

    # Set up survey vectors
    xvec = np.linspace(np.nanmin(avg_chemicals1), np.nanmax(avg_chemicals1), 101)
    yvec = np.linspace(np.nanmin(avg_chemicals2), np.nanmax(avg_chemicals2), 101)
    zvec = np.linspace(bot_temp, top_temp, 101)
    x1, x2, x3 = np.meshgrid(xvec, yvec, zvec)
    x, y, z = np.meshgrid(xvec, yvec, zvec)

    if reaction == 'ansp':
        ca['doc'], t, ga['ansp_genes'] = x1, x3, 0.0*x1 + 1.0
        prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
            pathways.ansp(ca, t, ph, 1.0)
    if reaction == 'aesp':
        ca['doc'], ca['o2'], t, ga['aesp_genes'] = x1, x2, x3, 0.0*x1 + 1.0
        prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
            pathways.aesp(ca, t, ph, 1.0)
    if reaction == 'mtox':
        ca['ch4'], ca['o2'], t, ga['methanotroph_genes'] = x1, x2, x3, 0.0*x1 + 1.0
        prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
            pathways.mtox(ca, t, ph, ga, 1.0)
    if reaction == 'h2mg':
        ca['co2'], ca['h2'], t, ga['h2_methanogen_genes'] = x1, x2, x3, 0.0 * x1 + 1.0
        prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
            pathways.h2mg(ca, t, ph, ga, 1.0)
    if reaction == 'hoag':
        ca['co2'], ca['h2'], t, ga['homoacetogen_genes'] = x1, x2, x3, 0.0 * x1 + 1.0
        prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
            pathways.h2mg(ca, t, ph, ga, 1.0)
    if reaction == 'acmg':
        x2 = 0.0*x1 + 1.0
        ca['ace'], t, ga['acemethanogen_genes'] = x1, x3, 0.0 * x1 + 1.0
        prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
            pathways.acmg(ca, t, ph, ga, 1.0)

    lvls = np.logspace(np.log10(np.nanmin(ref_rate)), np.log10(np.nanmax(ref_rate)), 100)

    #test = axis.contourf(x1[:, :, -1], x2[:, :, 0], ref_rate[:, :, 0], zdir='z', offset=0.0, cmap='coolwarm', alpha=0.4, levels=lvls, antialiased=True, linewidths=3.0)
    #test = axis.contourf(ref_rate[0, :, :], x2[0, :, :], x1[0, :, :], ref_rate[0, :, :], zdir='x', offset=0.0165, cmap='coolwarm', alpha=0.4, levels=lvls, antialiased=True, linewidths=3.0)
    print('ref_rate: ', ref_rate[0, :, :])

    arr = ref_rate
    xmax, ymax, zmax = max(xvec), max(yvec), max(zvec)
    xmin, ymin, zmin = min(xvec), min(yvec), min(zvec)
    vmin, vmax = np.min(arr), np.max(arr)

    #axis.contourf(x[:, :, 0], y[:, :, 0], arr[:, :, -1].T,
    #            zdir='z', offset=zmax, levels=lvls, cmap='coolwarm')
    #axis.contourf(x[0, :, :].T, arr[:, 0, :].T, z[0, :, :].T,
    #            zdir='y', offset=ymin, levels=lvls, cmap='coolwarm')
    #axis.contourf(arr[-1, :, :].T, y[:, 0, :].T, z[:, 0, :].T,
    #            zdir='x', offset=xmax, levels=lvls, cmap='coolwarm')
    print(arr[:, -1, :].T)
    p3d1 = axis.contourf(x[:, :, 0], y[:, :, 0], arr[:, :, 0].T,
                zdir='z', offset=zmin, levels=lvls, cmap='coolwarm', alpha=0.3)
    p3d2 = axis.contourf(x[0, :, :].T, arr[:, -1, :].T, z[0, :, :].T,
                zdir='y', offset=ymax, levels=lvls, cmap='coolwarm', alpha=0.3)
    p3d3 = axis.contourf(arr[0, :, :].T, y[:, 0, :].T, z[:, 0, :].T,
                zdir='x', offset=xmin, levels=lvls, cmap='coolwarm', alpha=0.3)

    cbar = fig.colorbar(p3d3, orientation='vertical', ax=axis, shrink = 0.5, pad = 0.1)
    cbar.ax.set_title(cb_label + '\n' + ' ', fontsize = 14)


    print('shapes: ', np.shape(x1), np.shape(x2), np.shape(x3))

    avg_rates = np.array(avg_rates)

    min_mic, max_mic = np.nanmin(avg_microbes), np.nanmax(avg_microbes)
    #axis.plot(avg_chemicals1, avg_chemicals2, avg_temps, color='gray', linewidth = 3.0)
    axis.set_xlabel(' \n' + x_label + ' [mol/m$^3$]', color = 'green')
    axis.set_ylabel(' \n' + y_label + ' [mol/m$^3$]', color = 'green')
    axis.set_zlabel(' \n' + 'Temperature [C]', color = 'blue')

    plot_trajectories = True

    for v in range(0, len(avg_rates)):

        norm = avg_rates[v] / np.nanmax(avg_rates)
        mic_size = 200.0 * (avg_microbes[v] - min_mic) / (max_mic - min_mic)
        if reaction == 'ansp' or reaction == 'aesp': mic_size = 200.0
        time_condition = avg_times[v].day == 15 and avg_times[v].hour == 0
        if v < len(avg_rates) - 10 and time_condition:
            x1, x2 = avg_chemicals1[v], avg_chemicals1[v+10]
            y1, y2 = avg_chemicals2[v], avg_chemicals2[v + 10]
            z1, z2 = avg_temps[v], avg_temps[v + 10]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            arrow_height = distance/2.0
        alpha_val = 0.2 + v/len(avg_rates)
        if alpha_val > 1.0: alpha_val = 1.0
        if v < len(avg_rates) - 1 and plot_trajectories: axis.plot([avg_chemicals1[v], avg_chemicals1[v+1]], [avg_chemicals2[v], avg_chemicals2[v+1]],
                        [avg_temps[v], avg_temps[v+1]], color='black', linewidth=1.0, alpha=alpha_val)

        # print(norm, np.nanmax(lvls))
        #axis.scatter(avg_chemicals1[v], avg_chemicals2[v], avg_temps[v], marker='o', c=cm.rainbow(norm), cmap='jet',
        #                alpha=0.0, s=mic_size)
        label_time_condition = avg_times[v].day == 1 and avg_times[v].hour == 0
        if (label_time_condition) and plot_trajectories: axis.scatter(avg_chemicals1[v], avg_chemicals2[v], avg_temps[v], marker='o',
                                            facecolors=None, edgecolors='black', alpha=0.8, s=mic_size)
        if (label_time_condition or time_condition) and plot_trajectories: axis.scatter(avg_chemicals1[v], avg_chemicals2[v], avg_temps[v], marker='o',
                                            c=cm.rainbow(norm), cmap='jet', alpha=0.8, s=mic_size, edgecolors = 'purple', linewidths = 3.0)
        if (label_time_condition or time_condition) and plot_trajectories: axis.plot([avg_chemicals1[v], avg_chemicals1[v]],
                                               [avg_chemicals2[v], avg_chemicals2[v]], [bot_temp, avg_temps[v]],
                                               linestyle='--', color='black', alpha=0.5*alpha_val)
        if (label_time_condition or time_condition) and plot_trajectories: axis.plot([avg_chemicals1[v], avg_chemicals1[v]],
                                               [avg_chemicals2[v], np.nanmax(avg_chemicals2)],
                                               [avg_temps[v], avg_temps[v]], linestyle='--', color='black', alpha=0.5*alpha_val)
        if (label_time_condition) and plot_trajectories: axis.text(avg_chemicals1[v], avg_chemicals2[v], avg_temps[v],
                                               months[avg_times[v].month - 1] + '\n', horizontalalignment='center',
                                               verticalalignment='bottom', fontsize=14.0, color='black',
                                               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round',
                                                         alpha=0.0))

    return()


def log_lvls(array, n_levels):

    min, max = np.nanmin(array), np.nanmax(array)

    #print('min: ', min, 'max: ', max)
    if min == 0.0 and max > 0.0:
        log_max = math.log(max, 10)
        log_min = log_max - 1
        lvls = np.logspace(log_min, log_max, n_levels)
    if min > 0.0 and max > 0.0:
        log_min, log_max = math.log(min, 10), math.log(max, 10)
        lvls = np.logspace(log_min, log_max, n_levels)
    if min < 0.0 and max > 0.0: # Use linear levels
        lvls = np.linspace(min, max, n_levels)
    if min == max: lvls = [0.0, 1.0]

    #if min == max: max = 1.5*min

    #lvls = np.arange(min, max, abs(max-min)/100.0)

    return(lvls)


def format_axis(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(' ')

    return()


def format_axis2(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return()


def offset_datetime_list(datetime_list, days):

    offset_list = []
    for l in range(0,len(datetime_list)):
        offset_list.append(datetime_list[l]+timedelta(days=days))

    return(offset_list)


def VvZvT_2_ColIntVvT(VvZvT):

    ColIntVvT = params.dz * np.nansum(VvZvT, axis=0)

    return(ColIntVvT)


def VvT_2_TimeIntVvT(VvT, T, nT):

    # Time integrated, column-integrated values
    integration_dates, integration_vals = [], []
    for j in range(0, int(len(VvT) / nT)):
        start, end = j * nT, j * nT + nT
        start_date, end_date = T[start], T[end]
        mid_date = datetime.fromtimestamp((start_date.timestamp() + end_date.timestamp()) / 2)
        integration_dates.append(mid_date)
        integration_vals.append(np.sum(VvT[start:end]))

    return(integration_dates, integration_vals)


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} " if plt.rcParams["text.usetex"] else f"{s}"


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


def remove_outliers(array, nstd):

    std = np.nanstd(array)
    #print('max: ', np.nanmax(array))
    #print('std: ',std)
    array[(array-np.nanmean(array)) > nstd*std] = np.nan
    #print('max: ',np.nanmax(array))


    return(array)


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


def datestrings2datetimes(datestrings):

    date_format, date_objects = '%Y-%m-%d %H:%M:%S', []
    for date_str in datestrings: date_objects.append(datetime.strptime(date_str, date_format))

    return(date_objects)


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


def plot_range(vals):

    max_val = abs(np.nanmax(vals))
    min_val = abs(np.nanmin(vals))
    com_val = np.nanmax([min_val, max_val])

    return(com_val)


def abc_to_rgb(A=0.0,B=0.0,C=0.0):
    ''' Map values A, B, C (all in domain [0,1]) to
    suitable red, green, blue values.'''
    return (min(B+C,1.0),min(A+C,1.0),min(A+B,1.0))


def draw_rectangle(center, label, axis, color):

    width, height = 35, 15
    x1, x2, y1, y2 = center[0] - width / 2., center[0] + width / 2., center[1] - height / 2., center[1] + height / 2.
    axis.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=3.0, alpha=0.5)
    axis.annotate(label, xy=(center[0], center[1]), ha='center', va='center')

    return()


#print(site_data['site_fluxes'][flux_time_dict[0]])

# Vertical grid
layer_thicknesses, depths = init.define_layers()

# Find all model output directories
path = params.main_directory + 'modelOutput/' + params.site + '/'
directories = [f for f in listdir(path)]

# Make separate lists for 1D and 2D data
directories_1d, directories_2d = [], []
for d in range(0,len(directories)):
    if directories[d] == 'cr': directories_2d.append(directories[d])
for d in range(0,len(directories)):
    if directories[d][-1] == 'f': directories_1d.append(directories[d])

#fig, ax = plt.subplots()

chemical = 'ch4'
v_list = []
# For each output directory (flux type)
for d in range(0, len(directories_1d)):

    # Find all files in the output directory
    mypath = params.main_directory + 'modelOutput/' + params.site + '/' + directories_1d[d]
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Stack the diffusive flux and plant flux data
    cdir = ''
    for f in range(0,len(onlyfiles)):
        if chemical in onlyfiles[f]:
            cdir = onlyfiles[f]
            print(onlyfiles[f])
    unit, t, v = read_1d_output(mypath + '/' + cdir)
    v_list.append(v)

v_tot = np.sum(np.asarray(v_list), axis=0)

#ax.plot(t, np.convolve(v_tot, np.ones(28)/28, mode='same'))

# Get temperature profile data to mask values where T < 0C
with open('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/en/temp_data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
x_temp, y_temp, v_temp = np.array(data)[2:-1,1], np.array(data)[1,2:-1], np.transpose(np.array(data)[2:-1,2:-1])
y_temp, v_temp = y_temp.astype(float), v_temp.astype(float)

# Get moisture profile data
waterIce_unit, x_waterIce, y_waterIce, v_waterIce = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/en/waterIce_data.csv', v_temp)
y_waterIce, v_waterIce = y_waterIce.astype(float), v_waterIce.astype(float)

# Get SOC profile data
soc_unit, x_soc, y_soc, v_soc = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/en/soilOrganic_data.csv', v_temp)
y_soc, v_soc = y_soc.astype(float), v_soc.astype(float)

# Saturation % data
satpct_unit, x_satpct, y_satpct, v_satpct = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/en/satpct_data.csv', v_temp)
y_satpct, v_satpct = y_satpct.astype(float), v_satpct.astype(float)

# Acetate data
ace_unit, x_ace, y_ace, v_ace = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/ace.csv', v_temp)
y_ace, v_ace = y_ace.astype(float), v_ace.astype(float)

# CH4 data
ch4_unit, x_ch4, y_ch4, v_ch4 = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/ch4.csv', v_temp)
y_ch4, v_ch4 = y_ch4.astype(float), v_ch4.astype(float)

# CO2 data
co2_unit, x_co2, y_co2, v_co2 = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/co2.csv', v_temp)
y_co2, v_co2 = y_co2.astype(float), v_co2.astype(float)

# H2 data
h2_unit, x_h2, y_h2, v_h2 = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/h2.csv', v_temp)
y_h2, v_h2 = y_h2.astype(float), v_h2.astype(float)

# O2 data
o2_unit, x_o2, y_o2, v_o2 = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/o2.csv', v_temp)
y_o2, v_o2 = y_o2.astype(float), v_o2.astype(float)

# DOC data
doc_unit, x_doc, y_doc, v_doc = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ca/doc.csv', v_temp)
y_doc, v_doc = y_doc.astype(float), v_doc.astype(float)

chemical_dict = {}
chemical_dict['ch4'] = v_ch4
chemical_dict['ace'] = v_ace
chemical_dict['co2'] = v_co2
chemical_dict['o2'] = v_o2
chemical_dict['h2'] = v_h2
chemical_dict['doc'] = v_doc

# Ace methanogen data
acemethanogen_unit, x_acemethanogen, y_acemethanogen, v_acemethanogen = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ga/acemethanogen_genes.csv', v_temp)
y_acemethanogen, v_acemethanogen = y_acemethanogen.astype(float), v_acemethanogen.astype(float)

# H2 methanogen data
h2_methanogen_unit, x_h2_methanogen, y_h2_methanogen, v_h2_methanogen = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ga/h2_methanogen_genes.csv', v_temp)
y_h2_methanogen, v_h2_methanogen = y_h2_methanogen.astype(float), v_h2_methanogen.astype(float)

# Homoacetogen data
homoacetogen_unit, x_homoacetogen, y_homoacetogen, v_homoacetogen = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ga/homoacetogen_genes.csv', v_temp)
y_homoacetogen, v_homoacetogen = y_homoacetogen.astype(float), v_homoacetogen.astype(float)

# Methanotroph data
methanotroph_unit, x_methanotroph, y_methanotroph, v_methanotroph = read_2d_output('C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/palsa_low/ga/methanotroph_genes.csv', v_temp)
y_methanotroph, v_methanotroph = y_methanotroph.astype(float), v_methanotroph.astype(float)

microbe_dict = {}
microbe_dict['acemethanogen'] = v_acemethanogen
microbe_dict['h2_methanogen'] = v_h2_methanogen
microbe_dict['homoacetogen'] = v_homoacetogen
microbe_dict['methanotroph'] = v_methanotroph

# Find all chemical rate files ('cr' directory)
mypath = params.main_directory + 'modelOutput/' + params.site + '/cr'
cr_onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
c_prod_list, c_cons_list = [], []  # Files with production/consumption rates via different pathways, for <chemical>
c_net, c_prod, c_cons = '', '', '' # Files with net, total production, and total consumption rates, for <chemical>
for f in range(0,len(cr_onlyfiles)):
    chem, type = cr_onlyfiles[f].split('_')[0], (cr_onlyfiles[f].split('_')[2]).split('.')[0]
    print(chem, type, cr_onlyfiles[f], len(cr_onlyfiles[f].split('_')))
    split_length = len(cr_onlyfiles[f].split('_'))
    if chem == chemical and type == 'rate': c_net = cr_onlyfiles[f]
    if chem == chemical and type == 'prod': c_prod = cr_onlyfiles[f]
    if chem == chemical and type == 'cons': c_cons = cr_onlyfiles[f]
    if chem == chemical and type == 'prod' and split_length == 4: c_prod_list.append(cr_onlyfiles[f])
    if chem == chemical and type == 'cons' and split_length == 4: c_cons_list.append(cr_onlyfiles[f])

# Find all files in the output directory
mypath = params.main_directory + 'modelOutput/' + params.site + '/crd2'
onlyfiles_crd2 = [f for f in listdir(mypath) if isfile(join(mypath, f))]

reaction_list = []
for p in range(0,len(c_prod_list)):
    element = c_prod_list[p]
    reaction_list.append((element.split('.')[0]).split('_')[-1])
for c in range(0, len(c_cons_list)):
    element = c_cons_list[c]
    reaction_list.append((element.split('.')[0]).split('_')[-1])

print(reaction_list)

refreaction_dict = {}

reaction_dictionary = {}
for reaction in reaction_list: # Reaction plot panels

    reaction_dictionary[reaction + '_files'] = find_elements_with_substring(onlyfiles_crd2, reaction)

    react_units, react_ts, react_ys, react_vs = [], [], [], []
    for p in range(0,len(reaction_dictionary[reaction + '_files'])):
        reacti = reaction_dictionary[reaction + '_files'][p]
        reacti_unit, reacti_t, reacti_y, reacti_v = read_2d_output(mypath + '/' + reacti, v_temp)
        if 'ref' in reaction_dictionary[reaction + '_files'][p]: refreaction_dict[reaction] = reacti_v
        react_units.append(reacti_unit)
        react_ts.append(reacti_t)
        react_ys.append(reacti_y)
        react_vs.append(reacti_v)

        reaction_dictionary[reaction+'_units'] = react_units
        reaction_dictionary[reaction+'_ts'] = react_ts
        reaction_dictionary[reaction+'_ys'] = react_ys
        reaction_dictionary[reaction+'_vs'] = react_vs

refreaction_dict['times'] = reacti_t

###

# Find all chemical abundance files ('ca' directory)
mypath = params.main_directory + 'modelOutput/' + params.site + '/ca'
ca_onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
ca_units, ca_ts, ca_ys, ca_vs, ca_titles = [], [], [], [], []
for f in range(0, len(ca_onlyfiles)):
    ca_unit, ca_t, ca_y, ca_v = read_2d_output(mypath + '/' + ca_onlyfiles[f], v_temp)
    ca_units.append(ca_unit)
    ca_ts.append(ca_t)
    ca_ys.append(ca_y)
    ca_vs.append(ca_v)
    ca_titles.append(ca_onlyfiles[f])
ca_index = np.where(np.asarray(ca_titles) == chemical+'.csv')[0][0]

# Find all chemical equilibrium concentration files ('eq' directory)
mypath = params.main_directory + 'modelOutput/' + params.site + '/eq'
eq_onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
eq_units, eq_ts, eq_ys, eq_vs, eq_titles = [], [], [], [], []
for f in range(0, len(eq_onlyfiles)):
    eq_unit, eq_t, eq_y, eq_v = read_2d_output(mypath + '/' + eq_onlyfiles[f], v_temp)
    eq_units.append(eq_unit)
    eq_ts.append(eq_t)
    eq_ys.append(eq_y)
    eq_vs.append(eq_v)
    eq_titles.append(eq_onlyfiles[f])
eq_index = np.where(np.asarray(eq_onlyfiles) == chemical+'.csv')[0][0]

# Files corresponding to specific pathways should not include total production and consumption rate files
#c_prod_list.remove(c_prod)
#c_cons_list.remove(c_cons)

# Read net production rate data
mypath = params.main_directory + 'modelOutput/' + params.site + '/cr'
c_net_unit, c_net_t, c_net_y, c_net_v = read_2d_output(mypath + '/' + c_net, v_temp)
c_net_zmean = params.dz * np.nansum(c_net_v, axis=0)
c_net_0 = params.dz * c_net_v[0,:]
c_net_integration_dates, c_net_integration_vals = VvT_2_TimeIntVvT(VvZvT_2_ColIntVvT(c_net_v), c_net_t, 112)
c_net_integration_dates = offset_datetime_list(c_net_integration_dates, days = -7.5)

# Read total production rate data
c_prod_unit, c_prod_t, c_prod_y, c_prod_v = read_2d_output(mypath + '/' + c_prod, v_temp)
c_prod_integration_dates, c_prod_integration_vals = VvT_2_TimeIntVvT(VvZvT_2_ColIntVvT(c_prod_v), c_prod_t, 112)
c_prod_integration_dates = offset_datetime_list(c_prod_integration_dates, days = 0)
c_prod_units, c_prod_ts, c_prod_ys, c_prod_vs, c_prod_zmeans, c_prod_0s = [], [], [], [], [], []
for p in range(0,len(c_prod_list)):
    c_prodi = c_prod_list[p]
    c_prodi_unit, c_prodi_t, c_prodi_y, c_prodi_v = read_2d_output(mypath + '/' + c_prodi, v_temp)
    c_prodi_zmean = params.dz * np.nansum(c_prodi_v, axis = 0)
    c_prodi_0 = params.dz * c_prodi_v[0,:]
    c_prod_units.append(c_prodi_unit)
    c_prod_ts.append(c_prodi_t)
    c_prod_ys.append(c_prodi_y)
    c_prod_vs.append(c_prodi_v)
    c_prod_zmeans.append(c_prodi_zmean)
    c_prod_0s.append(c_prodi_0)

# Read total consumption rate data
c_cons_unit, c_cons_t, c_cons_y, c_cons_v = read_2d_output(mypath + '/' + c_cons, v_temp)
c_cons_integration_dates, c_cons_integration_vals = VvT_2_TimeIntVvT(VvZvT_2_ColIntVvT(c_cons_v), c_cons_t, 112)
c_cons_integration_dates = offset_datetime_list(c_cons_integration_dates, days = 7.5)
c_cons_units, c_cons_ts, c_cons_ys, c_cons_vs, c_cons_zmeans, c_cons_0s = [], [], [], [], [], []
for c in range(0,len(c_cons_list)):
    c_consi = c_cons_list[c]
    c_consi_unit, c_consi_t, c_consi_y, c_consi_v = read_2d_output(mypath + '/' + c_consi, v_temp)
    c_consi_zmean = params.dz * np.nansum(c_consi_v, axis=0)
    c_consi_0 = params.dz * c_consi_v[0, :]
    c_cons_units.append(c_consi_unit)
    c_cons_ts.append(c_consi_t)
    c_cons_ys.append(c_consi_y)
    c_cons_vs.append(c_consi_v)
    c_cons_zmeans.append(c_consi_zmean)
    c_cons_0s.append(c_consi_0)

# Make a list of time-depth integrated total production rates
c_prod_integration_dates_list, c_prod_integration_vals_list = [], []
prod_labels = []
for p in range(0,len(c_prod_list)):
    print(c_prod_list[p])
    label = str(c_prod_list[p])
    label = label.split('.')[0]
    prod_labels.append(label)
    c_unit, c_t, c_y, c_v = read_2d_output(mypath + '/' + c_prod_list[p], v_temp)
    c_integration_dates, c_integration_vals = VvT_2_TimeIntVvT(VvZvT_2_ColIntVvT(c_v), c_t, 112)
    c_prod_integration_dates_list.append(c_integration_dates)
    c_prod_integration_vals_list.append(c_integration_vals)

# Make a list of time-depth integrated total consumption rates
c_cons_integration_dates_list, c_cons_integration_vals_list = [], []
cons_labels = []
for p in range(0,len(c_cons_list)):
    label = str(c_cons_list[p])
    label = label.split('.')[0]
    cons_labels.append(label)
    c_unit, c_t, c_y, c_v = read_2d_output(mypath + '/' + c_cons_list[p], v_temp)
    c_integration_dates, c_integration_vals = VvT_2_TimeIntVvT(VvZvT_2_ColIntVvT(c_v), c_t, 112)
    c_cons_integration_dates_list.append(c_integration_dates)
    c_cons_integration_vals_list.append(c_integration_vals)

rate_labels = ['Net '+chemical+' prod']+prod_labels+cons_labels

months_short = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
blanks = [' ', ' ', ' ', ' ', ' ', ' ', ' ']

# Environment plots
fig = plt.figure(figsize=(15, 7))
gs0 = fig.add_gridspec(2, 2, width_ratios=[1,1], height_ratios=[1,1],
                      left=0.1, right=0.9, bottom=0.05, top=0.95,
                      wspace=0.4, hspace=0.3)

# Temperature
temp_ax = fig.add_subplot(gs0[0,0])
temp_ax.set_ylim([0.45, 0.025])
temp_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
temp_ax.set_xticklabels(months_short)
temp_ax.set_ylabel('Depth [m]', fontsize=11)
temp_ax.set_title('Temperature [C]')
temp_ax.set_facecolor('gainsboro')
format_axis(temp_ax)
loglvls = np.linspace(0,25,101)
c3 = temp_ax.contourf(x_temp, y_temp, v_temp, levels=loglvls, cmap='rainbow')
cbar = fig.colorbar(c3, orientation='vertical', ax=temp_ax)
cbar.ax.tick_params(labelsize=10)

# waterIce
waterIce_ax = fig.add_subplot(gs0[0,1])
waterIce_ax.set_ylim([0.45, 0.025])
waterIce_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
waterIce_ax.set_xticklabels(months_short)
waterIce_ax.set_ylabel('Depth [m]', fontsize=11)
waterIce_ax.set_title('Soil Moisture [%]')
waterIce_ax.set_facecolor('gainsboro')
format_axis(waterIce_ax)
loglvls = np.linspace(0,100,101)
c3 = waterIce_ax.contourf(x_waterIce, y_waterIce, v_waterIce, levels=loglvls, cmap='rainbow')
cbar = fig.colorbar(c3, orientation='vertical', ax=waterIce_ax)
cbar.ax.tick_params(labelsize=10)

# DOC
doc_ax = fig.add_subplot(gs0[1,0])
doc_ax.set_ylim([0.45, 0.025])
doc_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
doc_ax.set_xticklabels(months_short)
doc_ax.set_ylabel('Depth [m]', fontsize=11)
doc_ax.set_title('DOC [mol/m$^3$]')
doc_ax.set_facecolor('gainsboro')
format_axis(doc_ax)
loglvls = np.linspace(0,np.nanmax(v_doc),101)
c3 = doc_ax.contourf(x_doc, y_doc, v_doc, levels=loglvls, cmap='rainbow')
cbar = fig.colorbar(c3, orientation='vertical', ax=doc_ax)
cbar.ax.tick_params(labelsize=10)

# SOC
soc_ax = fig.add_subplot(gs0[1,1])
soc_ax.set_ylim([0.45, 0.025])
soc_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
soc_ax.set_xticklabels(months_short)
soc_ax.set_ylabel('Depth [m]', fontsize=11)
soc_ax.set_title('SOC [kg/m$^3$], [mol/m$^3$]')
soc_ax.set_facecolor('gainsboro')
format_axis(soc_ax)
v_soc_kgm3 = 1400.0 * (v_soc/100.0)
v_soc_molm3 = conversions.kg2mol(v_soc_kgm3, 'c')
loglvls = np.linspace(0,np.nanmax(v_soc_kgm3),101)
c3 = soc_ax.contourf(x_soc, y_soc, v_soc_kgm3, levels=loglvls, cmap='rainbow')
cbar = fig.colorbar(c3, orientation='vertical', ax=soc_ax)
cbar.ax.tick_params(labelsize=10)
c4 = soc_ax.contour(x_soc, y_soc, v_soc_molm3, levels=[0, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4], color = 'black')
soc_ax.clabel(c4, c4.levels, inline=True, fmt=fmt, fontsize=10)

plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/fancy/env.png',bbox_inches='tight')

nrows = len(c_prod_list) + len(c_cons_list) + 2
ncols = 3

# Main chemical plots
fig = plt.figure(figsize=(18, 7*(nrows/3)))
gs = fig.add_gridspec(nrows, ncols, width_ratios=[2] + [1] * (ncols-1), height_ratios=[3] + [2] * (nrows-1),
                      left=0.1, right=0.9, bottom=0.05, top=0.95,
                      wspace=0.4, hspace=0.3)

# Site data
layer_thicknesses, depths = init.define_layers()
site_data = read_site_data(params.site, depths)
flux_data_dict = find_keys_with_substring(site_data['site_fluxes'], '_VvT')
flux_time_dict = find_keys_with_substring(site_data['site_fluxes'], '_T')

mcra_data_dict = find_keys_with_substring(site_data['site_mcra'], '_mcracopm3_VvZ')
pmoa1_data_dict = find_keys_with_substring(site_data['site_pmoa1'], '_pmoa1copm3_VvZ')
pmoa2_data_dict = find_keys_with_substring(site_data['site_pmoa2'], '_pmoa2copm3_VvZ')
mcra_depth_dict = find_keys_with_substring(site_data['site_mcra'], '_Z')
pmoa1_depth_dict = find_keys_with_substring(site_data['site_pmoa1'], '_Z')
pmoa2_depth_dict = find_keys_with_substring(site_data['site_pmoa2'], '_Z')

o2pct_data_dict = find_keys_with_substring(site_data['site_o2pct'], '_VvZ')
o2umolL_data_dict = find_keys_with_substring(site_data['site_o2umolL'], '_VvZ')
roots_data_dict = find_keys_with_substring(site_data['site_roots'], '_VvZ')
temp_time_dict = find_keys_with_substring(site_data['site_temps'], 'dtimes')
temp05cm_data_dict = find_keys_with_substring(site_data['site_temps'], 'temps05_VvT')
temp15cm_data_dict = find_keys_with_substring(site_data['site_temps'], 'temps15_VvT')
temp25cm_data_dict = find_keys_with_substring(site_data['site_temps'], 'temps25_VvT')

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ca_ch4_index = np.where(np.array(ca_titles) == 'ch4.csv')[0][0]
ca_co2_index = np.where(np.array(ca_titles) == 'co2.csv')[0][0]
ca_o2_index = np.where(np.array(ca_titles) == 'o2.csv')[0][0]
ca_h2_index = np.where(np.array(ca_titles) == 'h2.csv')[0][0]

net_prof_ax = fig.add_subplot(gs[1,1])
c_net_v = remove_outliers(c_net_v, 5.0)
test = log_contour(c_net_t, c_net_y, c_net_v, 'RdYlGn', net_prof_ax, 50, 'Net', 'mol/m$^3$/day', False)

# chemical plot
ca_ax = fig.add_subplot(gs[0,0])
ca_vs[ca_index] = remove_outliers(ca_vs[ca_index], 5.0)
loglvls = log_lvls(ca_vs[ca_index], 100)
test = log_contour(ca_ts[ca_index], ca_ys[ca_index], ca_vs[ca_index], 'rainbow', ca_ax, 1000, ca_onlyfiles[ca_index], 'mol/m$^3$', True)

ch4_schmidts, co2_schmidts, o2_schmidts, h2_schmidts = diffusion.schmidt_number(v_temp[0,:])
ch4_phi, co2_phi, o2_phi, h2_phi = diffusion.transfer_velocity(ch4_schmidts, co2_schmidts, o2_schmidts, h2_schmidts)
phis = [0.0*ch4_phi, 0.0*ch4_phi, ch4_phi, co2_phi, 0.0*ch4_phi, h2_phi, o2_phi]
phi_titles = np.array(['c','doc','ch4','co2','ace','h2','o2'])
phi_index = np.where(phi_titles == chemical)[0][0]

decomp_rate_ax = fig.add_subplot(gs[1:3, 0])
decomp_rate_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
decomp_rate_ax.set_xticklabels(months)
decomp_rate_ax.set_ylabel('Total Rates\n[mol/m$^2/day$]', fontsize=16)
plt.yticks(fontsize=16)
format_axis(decomp_rate_ax)
decomp_rate_ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha = 0.5, axis = 'y')
for d in range(0,len(c_prod_vs)):
    curve = np.convolve(c_prod_zmeans[d], np.ones(28)/28, mode='same')
    decomp_rate_ax.plot(c_prod_ts[d], curve, alpha=0.5, color='Green')
    decomp_rate_ax.fill_between(c_prod_ts[d], curve, color='Green', alpha=0.5)
for d in range(0,len(c_cons_vs)):
    curve = np.convolve(c_cons_zmeans[d], np.ones(28) / 28, mode='same')
    decomp_rate_ax.plot(c_cons_ts[d], -curve, alpha=0.5, color='Red')
    decomp_rate_ax.fill_between(c_cons_ts[d], -curve, color='red', alpha=0.5)
curve = np.convolve(c_net_zmean, np.ones(28) / 28, mode='same')

curve_cumes = []
for i in range(0, len(v)): curve_cumes.append(np.sum(curve[0:i]))
curve_cumes = params.write_dt * np.array(curve_cumes)

decomp_rate_ax.fill_between(c_prod_ts[0], curve, color='gray', alpha=0.5)
decomp_rate_ax.plot(t, np.convolve(v_tot, np.ones(28)/28, mode='same'), color='black', linewidth=3.0, alpha=0.5)
cumf_ax = decomp_rate_ax.twinx()
cumf_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
cumf_ax.set_xticklabels(blanks)
cumf_ax.set_ylabel('\u03A3 F$_{srf}$ [mol/m$^2$]', fontsize=11)
format_axis(cumf_ax)
cumf_ax.spines['bottom'].set_visible(False)

v_cumes = []
for i in range(0, len(v)): v_cumes.append(np.sum(v_tot[0:i]))
v_cumes = params.write_dt * np.array(v_cumes)

print(curve_cumes[-1])
#stop

c3 = cumf_ax.plot(t, v_cumes, linestyle='--', color='black', linewidth=3.0, alpha=0.5)
c4 = cumf_ax.plot(t, curve_cumes, linestyle='--', color='gray', linewidth=6.0, alpha=0.5)
#for k in range(0,len(flux_data_dict)):
#    decomp_rate_ax.plot(site_data['site_fluxes'][flux_time_dict[k]], site_data['site_fluxes'][flux_data_dict[k]], marker='.', linestyle='')
cb = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cm.viridis), ax=decomp_rate_ax, shrink = 0.5)
cb.ax.zorder = -1
ymin, ymax = cumf_ax.get_ylim()[0], cumf_ax.get_ylim()[1]
start_day, width_days = datetime(2022, 11, 15), timedelta(days = 70)
cumf_ax.fill([start_day, start_day + width_days, start_day + width_days, start_day], [ymin, ymin, ymax, ymax], clip_on = False, color='white')


q10_ax = fig.add_subplot(gs[3:, 0])
v_temp_shallow = v_temp[0:1,:]
shallow_temp_avgs = np.mean(v_temp_shallow, axis = 0)
q10_ax.set_xlim([0.0, np.nanmax(shallow_temp_avgs)])
q10_ax.set_ylabel('F$_{srf}$ [mol/m$^2$/day]', fontsize=16)
q10_ax.set_xlabel('Shallow soil temperature [\u2103]', fontsize=16)
plt.yticks(fontsize=16)
v_tots = v_tot[0:-1]
q10_ax.scatter(shallow_temp_avgs,v_tots,marker='o')
q10_ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha = 0.3)
format_axis2(q10_ax)
#
times = refreaction_dict['times']
colors = ['darkorchid','mediumpurple','royalblue','cornflowerblue','powderblue','aquamarine',
          'palegreen','yellowgreen','khaki','sandybrown','lightsalmon','lightcoral']
#
for_q10_legend = fig.add_axes([0.1, 0.35, 0.001, 0.001])
new_handles = [Line2D([], [], c=x, marker = 'o', linewidth = 0.0, markersize=5.0) for x in colors[4:11]]
for_q10_legend.legend(handles=new_handles, loc='upper left', labels=months[4:11], fontsize = 16, ncol = 2)
plt.axis('off')
fake_ax = q10_ax.twinx()
cb = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cm.viridis), ax=q10_ax, shrink = 0.5)
cb.ax.zorder = -1
ymin, ymax = fake_ax.get_ylim()[0], fake_ax.get_ylim()[1]
start_day, width_days = datetime(2022, 11, 15), timedelta(days = 70)
start_day, width_days = 18.0, 5.0
fake_ax.tick_params(axis='y', colors='white')
format_axis(fake_ax)
fake_ax.fill([start_day, start_day + width_days, start_day + width_days, start_day], [ymin, ymin, ymax, ymax], clip_on = False, color='white')
#
for t in range(0,len(shallow_temp_avgs)):
    month = times[t].month - 1
    q10_ax.plot([shallow_temp_avgs[t]], [v_tots[t]], marker = 'o', color=colors[month])

for p in range(0,len(c_prod_list)):
    prod_prof_ax = fig.add_subplot(gs[p+2,1])
    c_prod_vs[p] = remove_outliers(c_prod_vs[p], 5.0)
    loglvls = log_lvls(c_prod_vs[p], 10)
    test = log_contour(c_prod_ts[p], c_prod_ys[p],c_prod_vs[p], 'Greens', prod_prof_ax, 50, c_prod_list[p], 'mol/m$^3$/day', False)

for c in range(0,len(c_cons_list)):
    cons_prof_ax = fig.add_subplot(gs[c+p+3,1])
    c_cons_vs[c] = remove_outliers(c_cons_vs[c], 4.0)
    loglvls = log_lvls(c_cons_vs[c], 10)
    if c == len(c_cons_list)-1:
        month_labels = True
    else:
        month_labels = False
    test = log_contour(c_cons_ts[c], c_cons_ys[c], c_cons_vs[c], 'Reds', cons_prof_ax, 50, c_cons_list[c], 'mol/m$^3$/day', month_labels)

    plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/fancy/' + chemical + '.png',bbox_inches='tight')

#plt.show()

#print(c_prod_list, c_cons_list)

#reaction_list.remove('prod')
#reaction_list.remove('cons')

mypath = params.main_directory + 'modelOutput/' + params.site + '/crd2'
for reaction in reaction_list:
    #reaction = 'mtox'

    print('reaction: ', reaction, reaction_dictionary[reaction + '_files'])

    fig = plt.figure(figsize=(15, 10))
    gs2 = fig.add_gridspec(5, 3, width_ratios=(1, 2, 2), height_ratios=(2, 1, 1, 1, 1),
                           left=0.1, right=0.9, bottom=0.05, top=0.95,
                           wspace=0.5, hspace=0.7)

    for p in range(0,len(reaction_dictionary[reaction + '_files'])): # Contributor contour plots

        if 'ref'  in reaction_dictionary[reaction + '_files'][p]: row, col, cmap, lvls = 0, 1, 'jet', np.logspace(-3, -0, 100)
        #if 'vmax' in react_files[p]: row, col, cmap, lvls = 1, 0, 'Reds', np.arange(0.0, 1.0, 0.01)
        if 'chem' in reaction_dictionary[reaction + '_files'][p]: row, col, cmap, lvls = 1, 0, 'Greens', np.logspace(-1, 0, 100)
        if 'temp' in reaction_dictionary[reaction + '_files'][p]: row, col, cmap, lvls = 2, 0, 'Blues', np.arange(-1, 0, 0.01)
        #if 'ph'   in react_files[p]: row, col, cmap, lvls = 1, 1, 'Oranges', np.arange(0.0, 1.0, 0.01)
        if 'micb' in reaction_dictionary[reaction + '_files'][p]: row, col, cmap, lvls = 3, 0, 'Purples', np.logspace(-3, 0, 100)

        # Exclude ph and vmax profiles, since they're constant
        if 'ph' not in reaction_dictionary[reaction + '_files'][p] and 'vmax' not in reaction_dictionary[reaction + '_files'][p]:

            prod_prof_ax = fig.add_subplot(gs2[row,col])
            if 'micb' in reaction_dictionary[reaction + '_files'][p] or 'ref' in reaction_dictionary[reaction + '_files'][p]:
                month_labels = True
            else:
                month_labels = False
            units = 'mol/m$^3$/day'
            if 'micb' in reaction_dictionary[reaction + '_files'][p] or 'chem' in reaction_dictionary[reaction + '_files'][p]:
                units = ' '
            prod_prof_ax.set_facecolor('gainsboro')
            reaction_dictionary[reaction + '_vs'][p] = remove_outliers(reaction_dictionary[reaction + '_vs'][p], 5.0)
            print(reaction)
            test = log_contour(reaction_dictionary[reaction + '_ts'][p], reaction_dictionary[reaction + '_ys'][p], reaction_dictionary[reaction + '_vs'][p], cmap, prod_prof_ax, 50, reaction_dictionary[reaction + '_files'][p], units, month_labels)

    mean_ax = fig.add_subplot(gs2[0,2])
    mean_ax.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
    #mean_ax.set_ylim([-abs_max, abs_max])
    mean_ax.set_ylabel('[]')
    mean_ax.set_xlabel('. \n . \n . \n .')
    mean_ax.set_title('Rate-weighted mean contribution',fontsize=11)
    mean_ax.spines['bottom'].set_position('zero')


    #print(reaction)

    pspace_ax = fig.add_subplot(gs2[1:,1:], projection = '3d')
    pspace_ax.set_title = ' '
    print('reaction: ', reaction)
    plot3d2(reaction, pspace_ax, refreaction_dict, microbe_dict, chemical_dict, v_temp)

    ref_file = find_elements_with_substring(reaction_dictionary[reaction + '_files'], 'ref')
    test_string = mypath + '/' + ref_file[0]
    ref_unit, ref_t, ref_y, ref_v = read_2d_output(test_string, v_temp)

    modes = ['vmax', 'temp', 'chem', 'ph', 'micb']
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for m in range(0, len(modes)):
        test_file = find_elements_with_substring(reaction_dictionary[reaction + '_files'], modes[m])[0]
        test_unit, test_t, test_y, test_v = read_2d_output(mypath + '/' + test_file, v_temp)

        times, wavgs = [], []
        for t in range(0, len(ref_t)):

            ref_slice = ref_v[:,t]
            ref_slice[ref_slice == 0.0] = np.nan
            indices = ~np.isnan(ref_slice)

            wavg = 0.0
            if True in indices:
                wavg = np.average(test_v[indices,t], weights = ref_v[indices,t])

            times.append(ref_t[t])
            wavgs.append(wavg)

        mean_ax.plot(times, np.convolve(wavgs, np.ones(28) / 28, mode='same'), color=colors[m], linewidth=3.0, alpha=0.5)

    mean_ax.legend(modes, frameon = False)

    plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/fancy/' + reaction + '.png',bbox_inches='tight')
