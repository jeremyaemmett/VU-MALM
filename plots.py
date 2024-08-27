from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatter
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from matplotlib import cm
from datetime import date
import matplotlib as mpl
from copy import copy
from numpy import ma
import pandas as pd
import numpy as np
import transport
import pathways
import params
import init
import csv
import os

def make_legend(clabels,plabels,axis,fls):

    xmin = (axis.get_xlim())[0]
    axis.fill_between([xmin, 0.2*xmin], [-0.11, -0.11], [-0.14, -0.14], facecolor='red', alpha=0.3,edgecolor='none', clip_on=False)
    axis.annotate('Net -', xy=(0.8*xmin, -0.123), color='black',fontsize=fls, ha='left', va='center', annotation_clip=False)
    axis.fill_between([0.2 * abs(xmin),abs(xmin)], [-0.11, -0.11], [-0.14, -0.14], facecolor='green', alpha=0.3, edgecolor='none',clip_on=False)
    axis.annotate('Net +', xy=(0.32 * abs(xmin), -0.123), color='black', fontsize=fls, ha='left', va='center',annotation_clip=False)
    styles = ['solid','dashed','dotted']
    for p in range(1,len(clabels)+1):
        axis.plot([xmin,0.7*xmin],[-0.13-p*0.05,-0.13-p*0.05],clip_on=False,color='red',linestyle=styles[p-1],linewidth=2.0)
        axis.annotate(clabels[p-1], xy=(0.6*xmin,-0.13-p*0.05), color='black', fontsize=0.9*fls, ha='left',va='center',annotation_clip=False)
    for p in range(1,len(plabels)+1):
        axis.plot([xmin,0.7*xmin],[-0.13-p*0.05-len(clabels)*0.05,-0.13-p*0.05-len(clabels)*0.05],clip_on=False,color='green',linestyle=styles[p-1],linewidth=2.0)
        axis.annotate(plabels[p-1], xy=(0.6*xmin,-0.13-p*0.05-len(clabels)*0.05), color='black', fontsize=0.9*fls, ha='left',va='center',annotation_clip=False)

    return()

def make_legend2(clabels,plabels,axis,fls,shift,show_con,show_prod):

    xmin = (axis.get_xlim())[0]
    xmax = (axis.get_xlim())[-1]
    ymin = (axis.get_ylim())[0]
    ymax = (axis.get_ylim())[-1]
    #axis.fill_between([xmin, 1.3*xmin], [-0.11, -0.11], [-0.14, -0.14], facecolor='red', alpha=0.3,edgecolor='none', clip_on=False)
    #axis.annotate('Net -', xy=(0.8*xmin, -0.123), color='black',fontsize=fls, ha='left', va='center', annotation_clip=False)
    #axis.fill_between([0.2 * abs(xmin),1.3*abs(xmin)], [-0.11, -0.11], [-0.14, -0.14], facecolor='green', alpha=0.3, edgecolor='none',clip_on=False)
    #axis.annotate('Net +', xy=(0.32 * abs(xmin), -0.123), color='black', fontsize=fls, ha='left', va='center',annotation_clip=False)
    styles = ['solid','dashed','dotted']
    for p in range(1,len(clabels)+1):
        if show_con == 1:
            axis.plot([xmin,-7*xmin],np.array([35.0*ymax-p*0.05,35.0*ymax-p*0.05])-0.9+shift,clip_on=False,color='red',linestyle=styles[p-1],linewidth=4.0)
            axis.annotate(clabels[p-1], xy=(-12*xmin,35.0*ymax-p*0.05-0.9+shift), color='black', fontsize=1.2*fls, ha='left',va='center',annotation_clip=False)
    for p in range(1,len(plabels)+1):
        if show_prod == 1:
            axis.plot([xmin,-7*xmin],np.array([35.0*ymax-p*0.05-len(clabels)*0.05,35.0*ymax-p*0.05-len(clabels)*0.05])-0.9+shift,clip_on=False,color='green',linestyle=styles[p-1],linewidth=4.0)
            axis.annotate(plabels[p-1], xy=(-12*xmin,35.0*ymax-p*0.05-len(clabels)*0.05-0.9+shift), color='black', fontsize=1.2*fls, ha='left',va='center',annotation_clip=False)

    return()

def make_legend3(clabels,plabels,axis,fls):

    xmin = (axis.get_xlim())[0]
    xmax = (axis.get_xlim())[-1]
    ymin = (axis.get_ylim())[0]
    ymax = (axis.get_ylim())[-1]
    #axis.fill_between([xmin, 1.3*xmin], [-0.11, -0.11], [-0.14, -0.14], facecolor='red', alpha=0.3,edgecolor='none', clip_on=False)
    #axis.annotate('Net -', xy=(0.8*xmin, -0.123), color='black',fontsize=fls, ha='left', va='center', annotation_clip=False)
    #axis.fill_between([0.2 * abs(xmin),1.3*abs(xmin)], [-0.11, -0.11], [-0.14, -0.14], facecolor='green', alpha=0.3, edgecolor='none',clip_on=False)
    #axis.annotate('Net +', xy=(0.32 * abs(xmin), -0.123), color='black', fontsize=fls, ha='left', va='center',annotation_clip=False)
    styles = ['solid','dashed','dotted']
    for p in range(1,len(clabels)+1):
        axis.plot([xmin,0.25*xmax],[-0.5*ymax,-0.5*ymax],clip_on=False,color='red',linestyle=styles[p-1],linewidth=4.0)
        axis.annotate(clabels[p-1], xy=(0.35*xmax,-0.5*ymax), color='black', fontsize=1.2*fls, ha='left',va='center',annotation_clip=False)
    for p in range(1,len(plabels)+1):
        axis.plot([xmin,0.25*xmax],[-0.5*ymax-len(clabels)*0.03,-0.5*ymax-len(clabels)*0.03],clip_on=False,color='green',linestyle=styles[p-1],linewidth=4.0)
        axis.annotate(plabels[p-1], xy=(0.35*xmax,-0.5*ymax-len(clabels)*0.03), color='black', fontsize=1.2*fls, ha='left',va='center',annotation_clip=False)

    return()

def make_label_lines(axis,colors,styles,lthick):

    xmin = (axis.get_xlim())[0]
    ymax = (axis.get_ylim())[0]
    axis.plot([xmin, 0.7*xmin], [1.12*ymax,1.12*ymax],clip_on=False, color=colors[0], linestyle=styles[0], linewidth=lthick)
    axis.plot([xmin, 0.7*xmin], [-0.12*ymax,-0.12*ymax],clip_on=False, color=colors[1], linestyle=styles[1],linewidth=lthick)


def fill_net_rates(depths,net_rate_profile,axis):
    #net_rate_profile=gaussian_filter(net_rate_profile,2.0)
    net_rate_profile2 = abs(net_rate_profile)
    for d in range(1,len(depths)):
        col='red'
        if net_rate_profile[d] > 0.0:
            col='green'
        if (net_rate_profile[d-1] > 0.0 and net_rate_profile[d] > 0.0) or (net_rate_profile[d-1] < 0.0 and net_rate_profile[d] < 0.0):
            axis.fill_betweenx([depths[d-1],depths[d]], [0.0,0.0], [net_rate_profile2[d-1],net_rate_profile2[d]],color=col, alpha=0.25,lw=0.0)

    return()

def fill_net_rates2(depths,net_rate_profile,axis):
    #net_rate_profile=gaussian_filter(net_rate_profile,2.0)
    net_rate_profile2 = abs(net_rate_profile)
    for d in range(1,len(depths)):
        col='red'
        if net_rate_profile[d] > 0.0:
            col='green'
        if (net_rate_profile[d-1] >= 0.0 and net_rate_profile[d] >= 0.0) or (net_rate_profile[d-1] <= 0.0 and net_rate_profile[d] <= 0.0):
            axis.fill_betweenx([depths[d-1],depths[d]], [0.0,0.0], [net_rate_profile2[d-1],net_rate_profile2[d]],color=col, alpha=0.25,lw=0.0)

    return()

def plot_site_genes(site_genes):

    # Configure plot panels
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10.0, 5.0, forward=True)
    ax1 = axes[0]
    ax2 = axes[1]
    print('gene plot test')

    ax1.set_title('June', fontsize=15)
    ax1.set_xlim([1e3, 1e7])
    ax1.set_ylim([0.45, 0.0])
    ax1.set_xscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_ylabel('Depth (m)', fontsize=15)
    ax1.set_xlabel('Gene Copies (copies/g)', fontsize=15)

    ax2.set_title('August', fontsize=15)
    ax2.set_xlim([1e3, 1e7])
    ax2.set_ylim([0.45, 0.0])
    ax2.set_xscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_ylabel(' ', fontsize=15)
    ax2.set_xlabel('Gene Copies (copies/g)', fontsize=15.0)

    l1 = ax1.plot(site_genes[site + '_jun_mcras'], site_genes[site + '_jun_depths'], color='red', marker='o',
                  linewidth=8.0, linestyle='-')
    ax2.plot(site_genes[site + '_aug_mcras'], site_genes[site + '_aug_depths'], color='red', marker='o', linewidth=8.0,
             linestyle='-')
    l2 = ax1.plot(site_genes[site + '_jun_pmoa1s'], site_genes[site + '_jun_depths'], color='forestgreen', marker='o',
                  linewidth=8.0, linestyle='-')
    ax2.plot(site_genes[site + '_aug_pmoa1s'], site_genes[site + '_aug_depths'], color='forestgreen', marker='o',
             linewidth=8.0, linestyle='-')
    l3 = ax1.plot(site_genes[site + '_jun_pmoa2s'], site_genes[site + '_jun_depths'], color='limegreen', marker='o',
                  linewidth=8.0, linestyle='-')
    ax2.plot(site_genes[site + '_aug_pmoa2s'], site_genes[site + '_aug_depths'], color='limegreen', marker='o',
             linewidth=8.0, linestyle='-')

    ax1.legend(['mcrA', 'pmoA1', 'pmoA2'], frameon=False, loc='lower left', fontsize=15)

    ### Interpolation

    l1 = ax1.plot(site_genes[site + '_jun_mcras'], model_depths, color='red', marker='o', markersize=10.0,
                  linewidth=8.0, linestyle='None')
    ax2.plot(site_genes[site + '_aug_mcras'], model_depths, color='red', marker='o', linewidth=8.0, markersize=10.0,
             linestyle='None')
    l2 = ax1.plot(site_genes[site + '_jun_pmoa1s'], model_depths, color='forestgreen', marker='o', markersize=10.0,
                  linewidth=8.0, linestyle='None')
    ax2.plot(site_genes[site + '_aug_pmoa1s'], model_depths, color='forestgreen', marker='o', markersize=10.0,
             linewidth=8.0, linestyle='None')
    l3 = ax1.plot(site_genes[site + '_jun_pmoa2s'], model_depths, color='limegreen', marker='o', markersize=10.0,
                  linewidth=8.0, linestyle='None')
    ax2.plot(site_genes[site + '_aug_pmoa2s'], model_depths, color='limegreen', marker='o', markersize=10.0,
             linewidth=8.0, linestyle='None')

    plt.savefig('C:/Users/Jeremy/Desktop/Churchill_Data/plots/' + site + '_genes.png', bbox_inches='tight')

def light_profile(axis,profile,depths,shift_c,temps):

    for i in range(0, len(depths) - 1):
        if temps[i] < 0.0:
            axis.fill_between([min(profile),max(profile)], [depths[i + 1], depths[i + 1]] - shift_c / 2.0,[depths[i], depths[i]] - shift_c / 2.0, color='lightsteelblue', alpha=0.20)
        else:
            axis.fill_between([min(profile),max(profile)], [depths[i + 1], depths[i + 1]] - shift_c / 2.0,[depths[i], depths[i]] - shift_c / 2.0, color='peru', alpha=0.15)

    return()

def rate_plot_lims(consumption,production):

    both = np.append(consumption,production)
    low = -1.2*max([abs(min(both)), abs(max(both))])
    low = 0.0
    high = 1.2*max([abs(min(both)), abs(max(both))])

    return(low,high)

def read_environment_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        temp_profiles = np.zeros((n_records, n_layers))
        moist_profiles = np.zeros((n_records, n_layers))
        ph_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            temp_profiles = read_row(row,r,1,n_layers,temp_profiles)
            moist_profiles = read_row(row,r,2,n_layers,moist_profiles)
            ph_profiles = read_row(row,r,3,n_layers,ph_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,temp_profiles,moist_profiles,ph_profiles

def read_substrate_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        ch4_profiles = np.zeros((n_records, n_layers))
        co2_profiles = np.zeros((n_records, n_layers))
        o2_profiles = np.zeros((n_records, n_layers))
        h2_profiles = np.zeros((n_records, n_layers))
        ace_profiles = np.zeros((n_records, n_layers))
        c_profiles = np.zeros((n_records, n_layers))
        doc_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            ch4_profiles = read_row(row,r,1,n_layers,ch4_profiles)
            co2_profiles = read_row(row,r,2,n_layers,co2_profiles)
            o2_profiles = read_row(row,r,3,n_layers,o2_profiles)
            h2_profiles = read_row(row,r,4,n_layers,h2_profiles)
            ace_profiles = read_row(row,r,5,n_layers,ace_profiles)
            c_profiles = read_row(row,r,6,n_layers,c_profiles)
            doc_profiles = read_row(row,r,7,n_layers,doc_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,ch4_profiles,co2_profiles,o2_profiles,h2_profiles,ace_profiles,c_profiles,doc_profiles

def read_microbe_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        acemethanogen_gene_profiles = np.zeros((n_records, n_layers))
        h2_methanogen_gene_profiles = np.zeros((n_records, n_layers))
        homoacetogen_gene_profiles = np.zeros((n_records, n_layers))
        methanotroph_gene_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            acemethanogen_gene_profiles = read_row(row,r,1,n_layers,acemethanogen_gene_profiles)
            h2_methanogen_gene_profiles = read_row(row,r,2,n_layers,h2_methanogen_gene_profiles)
            homoacetogen_gene_profiles = read_row(row,r,3,n_layers,homoacetogen_gene_profiles)
            methanotroph_gene_profiles = read_row(row,r,4,n_layers,methanotroph_gene_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,acemethanogen_gene_profiles,h2_methanogen_gene_profiles,homoacetogen_gene_profiles,methanotroph_gene_profiles

def read_flux_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = 1
        l_thicks = float(header[1])
        f_diff_ch4s = np.zeros((n_records))
        f_diff_co2s = np.zeros((n_records))
        f_diff_h2s = np.zeros((n_records))
        f_diff_o2s = np.zeros((n_records))
        plant_flux_ch4s = np.zeros((n_records))
        plant_flux_o2s = np.zeros((n_records))
        plant_flux_co2s = np.zeros((n_records))
        plant_flux_h2s = np.zeros((n_records))
        plant_flux_aces = np.zeros((n_records))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            f_diff_ch4s = read_row2(row,r,1,f_diff_ch4s)
            f_diff_co2s = read_row2(row,r,2,f_diff_co2s)
            f_diff_h2s = read_row2(row,r,3,f_diff_h2s)
            f_diff_o2s = read_row2(row,r,4,f_diff_o2s)
            plant_flux_ch4s = read_row2(row, r, 5, plant_flux_ch4s)
            plant_flux_o2s = read_row2(row, r, 6, plant_flux_o2s)
            plant_flux_co2s = read_row2(row, r, 7, plant_flux_co2s)
            plant_flux_h2s = read_row2(row, r, 8, plant_flux_h2s)
            plant_flux_aces = read_row2(row, r, 9, plant_flux_aces)
            r=r+1

    return n_layers,l_thicks,date_times,f_diff_ch4s,f_diff_co2s,f_diff_h2s,f_diff_o2s,plant_flux_ch4s,plant_flux_o2s,plant_flux_co2s,plant_flux_h2s,plant_flux_aces

def read_substrate_net_production_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        ch4_c_prod_profiles = np.zeros((n_records, n_layers))
        co2_c_prod_profiles = np.zeros((n_records, n_layers))
        o2_c_prod_profiles = np.zeros((n_records, n_layers))
        h2_c_prod_profiles = np.zeros((n_records, n_layers))
        ace_c_prod_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            ch4_c_prod_profiles = read_row(row,r,1,n_layers,ch4_c_prod_profiles)
            co2_c_prod_profiles = read_row(row,r,2,n_layers,co2_c_prod_profiles)
            o2_c_prod_profiles = read_row(row,r,3,n_layers,o2_c_prod_profiles)
            h2_c_prod_profiles = read_row(row,r,4,n_layers,h2_c_prod_profiles)
            ace_c_prod_profiles = read_row(row,r,5,n_layers,ace_c_prod_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,ch4_c_prod_profiles,co2_c_prod_profiles,o2_c_prod_profiles,h2_c_prod_profiles,ace_c_prod_profiles

def read_substrate_production_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        ch4_c_prod_acmg_profiles = np.zeros((n_records, n_layers))
        ch4_c_prod_h2mg_profiles = np.zeros((n_records, n_layers))
        ace_c_prod_aesp_profiles = np.zeros((n_records, n_layers))
        ace_c_prod_hoag_profiles = np.zeros((n_records, n_layers))
        co2_c_prod_ansp_profiles = np.zeros((n_records, n_layers))
        co2_c_prod_aesp_profiles = np.zeros((n_records, n_layers))
        co2_c_prod_acmg_profiles = np.zeros((n_records, n_layers))
        h2_c_prod_ansp_profiles = np.zeros((n_records, n_layers))
        o2_c_prod_none_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            ch4_c_prod_acmg_profiles = read_row(row,r,1,n_layers,ch4_c_prod_acmg_profiles)
            ch4_c_prod_h2mg_profiles = read_row(row,r,2,n_layers,ch4_c_prod_h2mg_profiles)
            ace_c_prod_aesp_profiles = read_row(row,r,3,n_layers,ace_c_prod_aesp_profiles)
            ace_c_prod_hoag_profiles = read_row(row,r,4,n_layers,ace_c_prod_hoag_profiles)
            co2_c_prod_ansp_profiles = read_row(row,r,5,n_layers,co2_c_prod_ansp_profiles)
            co2_c_prod_aesp_profiles = read_row(row,r,6,n_layers,co2_c_prod_aesp_profiles)
            co2_c_prod_acmg_profiles = read_row(row,r,7,n_layers,co2_c_prod_acmg_profiles)
            h2_c_prod_ansp_profiles = read_row(row,r,8,n_layers,h2_c_prod_ansp_profiles)
            o2_c_prod_none_profiles = read_row(row,r,9,n_layers,o2_c_prod_none_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,ch4_c_prod_acmg_profiles,ch4_c_prod_h2mg_profiles,ace_c_prod_aesp_profiles,ace_c_prod_hoag_profiles,co2_c_prod_ansp_profiles,co2_c_prod_aesp_profiles,co2_c_prod_acmg_profiles,h2_c_prod_ansp_profiles,o2_c_prod_none_profiles

def read_microbe_growth_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        acemethanogen_gene_prod_profiles = np.zeros((n_records, n_layers))
        h2_methanogen_gene_prod_profiles = np.zeros((n_records, n_layers))
        homoacetogen_gene_prod_profiles = np.zeros((n_records, n_layers))
        methanotroph_gene_prod_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            acemethanogen_gene_prod_profiles = read_row(row,r,1,n_layers,acemethanogen_gene_prod_profiles)
            h2_methanogen_gene_prod_profiles = read_row(row,r,2,n_layers,h2_methanogen_gene_prod_profiles)
            homoacetogen_gene_prod_profiles = read_row(row,r,3,n_layers,homoacetogen_gene_prod_profiles)
            methanotroph_gene_prod_profiles = read_row(row,r,4,n_layers,methanotroph_gene_prod_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,acemethanogen_gene_prod_profiles,h2_methanogen_gene_prod_profiles,homoacetogen_gene_prod_profiles,methanotroph_gene_prod_profiles

def read_substrate_net_consumption_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        ch4_c_cons_profiles = np.zeros((n_records, n_layers))
        co2_c_cons_profiles = np.zeros((n_records, n_layers))
        o2_c_cons_profiles = np.zeros((n_records, n_layers))
        h2_c_cons_profiles = np.zeros((n_records, n_layers))
        ace_c_cons_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            ch4_c_cons_profiles = read_row(row,r,1,n_layers,ch4_c_cons_profiles)
            co2_c_cons_profiles = read_row(row,r,2,n_layers,co2_c_cons_profiles)
            o2_c_cons_profiles = read_row(row,r,3,n_layers,o2_c_cons_profiles)
            h2_c_cons_profiles = read_row(row,r,4,n_layers,h2_c_cons_profiles)
            ace_c_cons_profiles = read_row(row,r,5,n_layers,ace_c_cons_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,ch4_c_cons_profiles,co2_c_cons_profiles,o2_c_cons_profiles,h2_c_cons_profiles,ace_c_cons_profiles

def read_microbe_mortality_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        acemethanogen_gene_cons_profiles = np.zeros((n_records, n_layers))
        h2_methanogen_gene_cons_profiles = np.zeros((n_records, n_layers))
        homoacetogen_gene_cons_profiles = np.zeros((n_records, n_layers))
        methanotroph_gene_cons_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            acemethanogen_gene_cons_profiles = read_row(row,r,1,n_layers,acemethanogen_gene_cons_profiles)
            h2_methanogen_gene_cons_profiles = read_row(row,r,2,n_layers,h2_methanogen_gene_cons_profiles)
            homoacetogen_gene_cons_profiles = read_row(row,r,3,n_layers,homoacetogen_gene_cons_profiles)
            methanotroph_gene_cons_profiles = read_row(row,r,4,n_layers,methanotroph_gene_cons_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,acemethanogen_gene_cons_profiles,h2_methanogen_gene_cons_profiles,homoacetogen_gene_cons_profiles,methanotroph_gene_cons_profiles

def read_substrate_consumption_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        ch4_c_cons_mtox_profiles = np.zeros((n_records, n_layers))
        ace_c_cons_acmg_profiles = np.zeros((n_records, n_layers))
        co2_c_cons_hoag_profiles = np.zeros((n_records, n_layers))
        co2_c_cons_h2mg_profiles = np.zeros((n_records, n_layers))
        h2_c_cons_hoag_profiles = np.zeros((n_records, n_layers))
        h2_c_cons_h2mg_profiles = np.zeros((n_records, n_layers))
        o2_c_cons_mtox_profiles = np.zeros((n_records, n_layers))
        o2_c_cons_aero_profiles = np.zeros((n_records, n_layers))
        o2_c_cons_aesp_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            ch4_c_cons_mtox_profiles = read_row(row,r,1,n_layers,ch4_c_cons_mtox_profiles)
            ace_c_cons_acmg_profiles = read_row(row,r,2,n_layers,ace_c_cons_acmg_profiles)
            co2_c_cons_hoag_profiles = read_row(row,r,3,n_layers,co2_c_cons_hoag_profiles)
            co2_c_cons_h2mg_profiles = read_row(row,r,4,n_layers,co2_c_cons_h2mg_profiles)
            h2_c_cons_hoag_profiles = read_row(row,r,5,n_layers,h2_c_cons_hoag_profiles)
            h2_c_cons_h2mg_profiles = read_row(row,r,6,n_layers,h2_c_cons_h2mg_profiles)
            o2_c_cons_mtox_profiles = read_row(row,r,7,n_layers,o2_c_cons_mtox_profiles)
            o2_c_cons_aero_profiles = read_row(row,r,8,n_layers,o2_c_cons_aero_profiles)
            o2_c_cons_aesp_profiles = read_row(row,r,9,n_layers,o2_c_cons_aesp_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,ch4_c_cons_mtox_profiles,ace_c_cons_acmg_profiles,co2_c_cons_hoag_profiles,co2_c_cons_h2mg_profiles,h2_c_cons_hoag_profiles,h2_c_cons_h2mg_profiles,o2_c_cons_mtox_profiles,o2_c_cons_aero_profiles,o2_c_cons_aesp_profiles

def read_transport_output(filename):

    lines_in_file = open(filename, 'r').readlines()
    n_records = len(lines_in_file)
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_layers = int(header[0])
        l_thicks = float(header[1])
        q_diff_ch4_profiles = np.zeros((n_records, n_layers))
        q_diff_co2_profiles = np.zeros((n_records, n_layers))
        q_diff_o2_profiles = np.zeros((n_records, n_layers))
        q_diff_h2_profiles = np.zeros((n_records, n_layers))
        q_diff_ace_profiles = np.zeros((n_records, n_layers))
        q_plant_ch4_profiles = np.zeros((n_records, n_layers))
        q_plant_co2_profiles = np.zeros((n_records, n_layers))
        q_plant_o2_profiles = np.zeros((n_records, n_layers))
        q_plant_h2_profiles = np.zeros((n_records, n_layers))
        q_plant_ace_profiles = np.zeros((n_records, n_layers))
        date_times = []
        r=0
        for row in reader:
            date_times.append(row[0])
            row = row[1:]
            q_diff_ch4_profiles = read_row(row,r,1,n_layers,q_diff_ch4_profiles)
            q_diff_co2_profiles = read_row(row,r,2,n_layers,q_diff_co2_profiles)
            q_diff_o2_profiles = read_row(row,r,3,n_layers,q_diff_o2_profiles)
            q_diff_h2_profiles = read_row(row,r,4,n_layers,q_diff_h2_profiles)
            q_diff_ace_profiles = read_row(row,r,5,n_layers,q_diff_ace_profiles)
            q_plant_ch4_profiles = read_row(row,r,6,n_layers,q_plant_ch4_profiles)
            q_plant_co2_profiles = read_row(row,r,7,n_layers,q_plant_co2_profiles)
            q_plant_o2_profiles = read_row(row,r,8,n_layers,q_plant_o2_profiles)
            q_plant_h2_profiles = read_row(row,r,9,n_layers,q_plant_h2_profiles)
            q_plant_ace_profiles = read_row(row,r,10,n_layers,q_plant_ace_profiles)
            r=r+1

    return n_layers,l_thicks,date_times,q_diff_ch4_profiles,q_diff_co2_profiles,q_diff_o2_profiles,q_diff_h2_profiles,q_diff_ace_profiles,q_plant_ch4_profiles,q_plant_co2_profiles,q_plant_o2_profiles,q_plant_h2_profiles,q_plant_ace_profiles

def read_row(row,row_num,var_num,n_layers,profiles):

    for column_idx in range((var_num-1)*n_layers, var_num*n_layers):
        val = float(row[column_idx])
        profiles[row_num, column_idx-(var_num-1)*n_layers] = val

    return(profiles)

def read_row2(row,row_num,var_num,profiles):

    for column_idx in range((var_num-1), var_num):
        val = float(row[column_idx])
        profiles[row_num] = val

    return(profiles)

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

def profiles_plot_properties(axis,profiles,profiles2,title,datetimes,depths,levels,yaxis,days,color_scheme,xlabel,temp_profiles,substance,lvls,log_flag):

    levels = np.linspace(0.0,np.nanmax(profiles),21)

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_facecolor('ghostwhite')
    max2 = np.max(profiles2)
    levels2=max2*levels
    date_format = mdates.DateFormatter('%b')
    axis.set_title(title, fontsize=26,fontweight="bold")
    axis.xaxis.set_tick_params(labelsize=26)
    axis.yaxis.set_tick_params(labelsize=26)
    axis.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axis.xaxis_date()
    axis.xaxis.set_major_formatter(date_format)
    axis.set_xticklabels(['        May','        Jun','        Jul','        Aug','        Sep','        Oct',' '])

    axis.set_xlim([datetime(2022, 5, 1), datetime(2022, 11, 15)])
    axis.set_ylim(0.5, min(depths))
    if yaxis == False:
        axis.get_yaxis().set_visible(False)
    X, Y = np.meshgrid(datetimes, depths)

    z = profiles.transpose()
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('bottom', size='5%', pad=0.5)
    l_f = LogFormatter(10, labelOnlyBase=False)
    if log_flag:
        c3 = axis.contourf(X, Y, z, levels=lvls, cmap=color_scheme, norm = LogNorm(vmin=lvls[0],vmax=lvls[-1]))
        cbar = fig.colorbar(c3,orientation='horizontal',cax=cax,format=l_f)
    else:
        lvls = levels
        c3 = axis.contourf(X, Y, z, levels=lvls, cmap=color_scheme)
        cbar = fig.colorbar(c3,orientation='horizontal',cax=cax)
    cbar.ax.tick_params(axis="both",labelsize=26,rotation=45)

    plt.minorticks_off()
    cbar.set_label(xlabel, rotation=0,fontsize=26)

    return()

def daily_mean_profile(frac_days,profile):

    # 1460 x 24
    #print(np.shape(profile))
    mean_profiles=np.zeros([365,np.shape(profile)[1]])
    for d in range(0,365):
        day_ids = np.where(np.floor(frac_days) == d)[0]
        mean_profile = np.mean(profile[day_ids[0]:day_ids[-1],:],axis=0)
        #print(day_ids[0],day_ids[-1])
        #print(profile[day_ids[0]:day_ids[-1],:])
        #print(mean_profile)
        mean_profiles[d,:] = mean_profile

    return(mean_profiles)

def cross_plots(axis,datetimes,profiles,prod_profiles,cons_profiles,temp_profiles,depths,scale):

    dm = 7
    for d in range(0,3):
        depth = depths[d*dm]
        profile_max = np.max(abs(profiles))
        profiles2 = -scale*profiles/profile_max
        profiles2[temp_profiles <= 0.0] = np.nan
        net_pos = np.copy(profiles2)
        #net_pos[profiles2 < 0.0] = 0.0
        net_pos = pd.DataFrame(net_pos[:, d * dm])
        net_pos = net_pos.rolling(20).mean()
        #axis.fill_between(range(len(datetimes)), net_pos, color='green', lw=0.0, alpha=0.2, clip_on=False)
        axis.plot(datetimes, net_pos + depth, clip_on=False, color='white', linewidth=12.0)
        axis.plot(datetimes, net_pos + depth, clip_on=False, color='black', linewidth=8.0)
        line_styles = ['solid','dashed','dotted']
        axis.plot([datetimes[0],datetimes[-1]], [0.0,0.0], clip_on=False, color='gray', linestyle=line_styles[0],linewidth=4.0)
        for c in range(0,len(prod_profiles)):
            prod_profiles2 = -scale*(prod_profiles[c])/profile_max
            prod_profiles2[temp_profiles <= 0.0] = np.nan
            prod_profiles2 = pd.DataFrame(prod_profiles2[:,d*dm])
            window_size=20
            prod_profiles2 = prod_profiles2.rolling(window_size).mean()
            axis.plot(datetimes,prod_profiles2 + depth, clip_on=False, color='white',linestyle = 'solid', linewidth=8.0)
            axis.plot(datetimes,prod_profiles2 + depth, clip_on=False, color='green',linestyle = line_styles[c], linewidth=4.0)
        for c in range(0,len(cons_profiles)):
            cons_profiles2 = -1.0*scale * (cons_profiles[c]) / profile_max
            cons_profiles2[temp_profiles <= 0.0] = np.nan
            cons_profiles2 = pd.DataFrame(cons_profiles2[:, d * dm])
            window_size = 20
            cons_profiles2 = cons_profiles2.rolling(window_size).mean()
            axis.plot(datetimes,cons_profiles2 + depth, clip_on=False, color='white',linestyle = 'solid', linewidth=8.0)
            axis.plot(datetimes,cons_profiles2 + depth, clip_on=False, color='red',linestyle = line_styles[c], linewidth=4.0)

    return()

def surface_flux_plots(axis,diff_fluxes,plant_fluxes,title,ytitle,dt,days):

    diff_fluxes = -1.0*diff_fluxes
    frac_days = params.dt * np.array(range(len(diff_fluxes)))
    axis.set_facecolor('ghostwhite')
    date_format = mdates.DateFormatter('%b')
    axis.xaxis.set_tick_params(labelsize=26)
    axis.yaxis.set_tick_params(labelsize=26)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_xlim(121,319)

    window_size=20
    diff_fluxes = pd.DataFrame(diff_fluxes)
    diff_fluxes = diff_fluxes.rolling(window_size).mean()
    net = np.squeeze(diff_fluxes) - plant_fluxes
    test = np.arange(len(net))/(1.0/params.dt)
    #axis.get_xaxis().set_visible(False)
    axis.set_ylabel(ytitle,fontsize=26)
    axis.set_title(title,fontsize=26,fontweight="bold")
    x_elements = len(diff_fluxes)
    neg_diff_fluxes = copy(diff_fluxes)
    pos_diff_fluxes = copy(diff_fluxes)
    neg_diff_fluxes[diff_fluxes > 0.0] = np.nan
    pos_diff_fluxes[diff_fluxes < 0.0] = np.nan
    neg_plant_fluxes = copy(plant_fluxes)
    pos_plant_fluxes = copy(plant_fluxes)
    neg_plant_fluxes[plant_fluxes > 0.0] = np.nan
    pos_plant_fluxes[plant_fluxes < 0.0] = np.nan

    window_size = 20

    axis.plot(test, neg_diff_fluxes, color='red', linestyle='solid', linewidth=3.0)
    axis.plot(test, pos_diff_fluxes, color='green', linestyle='solid', linewidth=3.0)
    axis.plot(test,-neg_plant_fluxes,color='green',linestyle='dashed', linewidth=3.0)
    axis.plot(test,-pos_plant_fluxes,color='red',linestyle='dashed', linewidth=3.0)
    #axis.plot(test, gaussian_filter(plant_fluxes, 2.0), color='black', linestyle='dashed')
    net_pos = np.copy(net)
    net_pos[net < 0.0] = 0.0
    axis.fill_between(test, net_pos, color='green', lw=0.0, alpha=0.2)
    net_neg = np.copy(net)
    net_neg[net > 0.0] = 0.0
    axis.fill_between(test, np.squeeze(net_neg), color='red', lw=0.0, alpha=0.2)
    ybottom = axis.get_ylim()[0]
    ytop = axis.get_ylim()[1]

    return()

ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = transport.equilibrium_concentration(15)
print(ceq_ch4,ceq_o2,ceq_co2,ceq_h2)
#stop

# Gaussian smoothing parameter
g_smooth = 0.0

data_directory = params.data_directory

### Read environmental profiles
test = read_environment_output(params.output_directory+params.site+'/envir_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
temp_profiles = gaussian_filter(test[3],g_smooth)
moist_profiles = gaussian_filter(test[4],g_smooth)
ph_profiles = gaussian_filter(test[5],g_smooth)
#
temp_profiles = temp_profiles[0:int(365/params.dt),:]
moist_profiles = moist_profiles[0:int(365/params.dt),:]
ph_profiles = ph_profiles[0:int(365/params.dt),:]

### Read flux rates
test = read_flux_output(params.output_directory+params.site+'/flux_output.txt')
date_times = test[2]
datetimes = []
for d in range(0, len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6, 0.025]
depths = []  # Calculate the layer center depths
for l in range(1, test[0] + 1):
    depths.append(l * test[1])
depths = np.array(depths)
f_diff_ch4s = gaussian_filter(test[3], 0.0)
f_diff_co2s = gaussian_filter(test[4], 0.0)
f_diff_h2s = gaussian_filter(test[5], 0.0)
f_diff_o2s = gaussian_filter(test[6], 0.0)
plant_flux_ch4s = gaussian_filter(test[7], 0.0)
plant_flux_o2s = gaussian_filter(test[8], 0.0)
plant_flux_co2s = gaussian_filter(test[9], 0.0)
plant_flux_h2s = gaussian_filter(test[10], 0.0)
plant_flux_aces = gaussian_filter(test[11], 0.0)
f_diff_ch4s = f_diff_ch4s[0:int(365/params.dt)]
f_diff_co2s = f_diff_co2s[0:int(365/params.dt)]
f_diff_h2s = f_diff_h2s[0:int(365/params.dt)]
f_diff_o2s = f_diff_o2s[0:int(365/params.dt)]
plant_flux_ch4s = plant_flux_ch4s[0:int(365/params.dt)]
plant_flux_o2s = plant_flux_o2s[0:int(365/params.dt)]
plant_flux_co2s = plant_flux_co2s[0:int(365/params.dt)]
plant_flux_h2s = plant_flux_h2s[0:int(365/params.dt)]
plant_flux_aces = plant_flux_aces[0:int(365/params.dt)]
###

### Read chemical quantities
test = read_substrate_output(params.output_directory+params.site+'/substrate_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
ch4_profiles = gaussian_filter(test[3],0.0)
co2_profiles = gaussian_filter(test[4],0.0)
o2_profiles = gaussian_filter(test[5],0.0)
h2_profiles = gaussian_filter(test[6],0.0)
ace_profiles = gaussian_filter(test[7],0.0)
c_profiles = gaussian_filter(test[8],0.0)
doc_profiles = gaussian_filter(test[9],0.0)
ch4_profiles = ch4_profiles[0:int(365/params.dt),:]
co2_profiles = co2_profiles[0:int(365/params.dt),:]
o2_profiles = o2_profiles[0:int(365/params.dt),:]
ace_profiles = ace_profiles[0:int(365/params.dt),:]
h2_profiles = h2_profiles[0:int(365/params.dt),:]
c_profiles = c_profiles[0:int(365/params.dt),:]
doc_profiles = doc_profiles[0:int(365/params.dt),:]
###

### Read net chemical production rates
test = read_substrate_net_production_output(params.output_directory+params.site+'/substrate_net_production_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
ch4_c_prod_profiles = gaussian_filter(test[3],g_smooth)
co2_c_prod_profiles = gaussian_filter(test[4],g_smooth)
o2_c_prod_profiles = gaussian_filter(test[5],g_smooth)
h2_c_prod_profiles = gaussian_filter(test[6],g_smooth)
ace_c_prod_profiles = gaussian_filter(test[7],g_smooth)
ch4_c_prod_profiles = ch4_c_prod_profiles[0:int(365/params.dt),:]
co2_c_prod_profiles = co2_c_prod_profiles[0:int(365/params.dt),:]
o2_c_prod_profiles = o2_c_prod_profiles[0:int(365/params.dt),:]
ace_c_prod_profiles = ace_c_prod_profiles[0:int(365/params.dt),:]
h2_c_prod_profiles = h2_c_prod_profiles[0:int(365/params.dt),:]
###

### Read net chemical consumption rates
test = read_substrate_net_consumption_output(params.output_directory+params.site+'/substrate_net_consumption_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
ch4_c_cons_profiles = gaussian_filter(test[3],g_smooth)
co2_c_cons_profiles = gaussian_filter(test[4],g_smooth)
o2_c_cons_profiles = gaussian_filter(test[5],g_smooth)
h2_c_cons_profiles = gaussian_filter(test[6],g_smooth)
ace_c_cons_profiles = gaussian_filter(test[7],g_smooth)
ch4_c_cons_profiles = ch4_c_cons_profiles[0:int(365/params.dt),:]
co2_c_cons_profiles = co2_c_cons_profiles[0:int(365/params.dt),:]
o2_c_cons_profiles = o2_c_cons_profiles[0:int(365/params.dt),:]
ace_c_cons_profiles = ace_c_cons_profiles[0:int(365/params.dt),:]
h2_c_cons_profiles = h2_c_cons_profiles[0:int(365/params.dt),:]
###

### Read decomposed chemical production rates
test = read_substrate_production_output(params.output_directory+params.site+'/substrate_production_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
ch4_c_prod_acmg_profiles = gaussian_filter(test[3],g_smooth)
ch4_c_prod_h2mg_profiles = gaussian_filter(test[4],g_smooth)
ace_c_prod_aesp_profiles = gaussian_filter(test[5],g_smooth)
ace_c_prod_hoag_profiles = gaussian_filter(test[6],g_smooth)
co2_c_prod_ansp_profiles = gaussian_filter(test[7],g_smooth)
co2_c_prod_aesp_profiles = gaussian_filter(test[8],g_smooth)
co2_c_prod_acmg_profiles = gaussian_filter(test[9],g_smooth)
h2_c_prod_ansp_profiles = gaussian_filter(test[10],g_smooth)
o2_c_prod_none_profiles = gaussian_filter(test[11],g_smooth)
ch4_c_prod_acmg_profiles = ch4_c_prod_acmg_profiles[0:int(365/params.dt),:]
ch4_c_prod_h2mg_profiles = ch4_c_prod_h2mg_profiles[0:int(365/params.dt),:]
ace_c_prod_aesp_profiles = ace_c_prod_aesp_profiles[0:int(365/params.dt),:]
ace_c_prod_hoag_profiles = ace_c_prod_hoag_profiles[0:int(365/params.dt),:]
co2_c_prod_ansp_profiles = co2_c_prod_ansp_profiles[0:int(365/params.dt),:]
co2_c_prod_aesp_profiles = co2_c_prod_aesp_profiles[0:int(365/params.dt),:]
co2_c_prod_acmg_profiles = co2_c_prod_acmg_profiles[0:int(365/params.dt),:]
h2_c_prod_ansp_profiles = h2_c_prod_ansp_profiles[0:int(365/params.dt),:]
o2_c_prod_none_profiles = o2_c_prod_none_profiles[0:int(365/params.dt),:]
###

### Read microbe growth rates
test = read_microbe_growth_output(params.output_directory+params.site+'/microbe_growth_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
g_smooth = 0.0
acemethanogen_gene_prod_profiles = gaussian_filter(test[3],g_smooth)
h2_methanogen_gene_prod_profiles = gaussian_filter(test[4],g_smooth)
homoacetogen_gene_prod_profiles = gaussian_filter(test[5],g_smooth)
methanotroph_gene_prod_profiles = gaussian_filter(test[6],g_smooth)
acemethanogen_gene_prod_profiles = acemethanogen_gene_prod_profiles[0:int(365/params.dt),:]
h2_methanogen_gene_prod_profiles = h2_methanogen_gene_prod_profiles[0:int(365/params.dt),:]
homoacetogen_gene_prod_profiles = homoacetogen_gene_prod_profiles[0:int(365/params.dt),:]
methanotroph_gene_prod_profiles = methanotroph_gene_prod_profiles[0:int(365/params.dt),:]
###

### Read decomposed chemistry rates
test = read_substrate_production_output(params.output_directory+params.site+'/substrate_consumption_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
ch4_c_cons_mtox_profiles = gaussian_filter(test[3],g_smooth)
ace_c_cons_acmg_profiles = gaussian_filter(test[4],g_smooth)
co2_c_cons_hoag_profiles = gaussian_filter(test[5],g_smooth)
co2_c_cons_h2mg_profiles = gaussian_filter(test[6],g_smooth)
h2_c_cons_hoag_profiles = gaussian_filter(test[7],g_smooth)
h2_c_cons_h2mg_profiles = gaussian_filter(test[8],g_smooth)
o2_c_cons_mtox_profiles = gaussian_filter(test[9],g_smooth)
o2_c_cons_aero_profiles = gaussian_filter(test[10],g_smooth)
o2_c_cons_aesp_profiles = gaussian_filter(test[11],g_smooth)
ch4_c_cons_mtox_profiles = ch4_c_cons_mtox_profiles[0:int(365/params.dt),:]
ace_c_cons_acmg_profiles = ace_c_cons_acmg_profiles[0:int(365/params.dt),:]
co2_c_cons_hoag_profiles = co2_c_cons_hoag_profiles[0:int(365/params.dt),:]
co2_c_cons_h2mg_profiles = co2_c_cons_h2mg_profiles[0:int(365/params.dt),:]
h2_c_cons_hoag_profiles = h2_c_cons_hoag_profiles[0:int(365/params.dt),:]
h2_c_cons_h2mg_profiles = h2_c_cons_h2mg_profiles[0:int(365/params.dt),:]
o2_c_cons_mtox_profiles = o2_c_cons_mtox_profiles[0:int(365/params.dt),:]
o2_c_cons_aero_profiles = o2_c_cons_aero_profiles[0:int(365/params.dt),:]
o2_c_cons_aesp_profiles = o2_c_cons_aesp_profiles[0:int(365/params.dt),:]
###

### Read microbe mortality rates
test = read_microbe_mortality_output(params.output_directory+params.site+'/microbe_mortality_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
g_smooth = 0.0
acemethanogen_gene_cons_profiles = gaussian_filter(test[3],g_smooth)
h2_methanogen_gene_cons_profiles = gaussian_filter(test[4],g_smooth)
homoacetogen_gene_cons_profiles = gaussian_filter(test[5],g_smooth)
methanotroph_gene_cons_profiles = gaussian_filter(test[6],g_smooth)
acemethanogen_gene_cons_profiles = acemethanogen_gene_cons_profiles[0:int(365/params.dt),:]
h2_methanogen_gene_cons_profiles = h2_methanogen_gene_cons_profiles[0:int(365/params.dt),:]
homoacetogen_gene_cons_profiles = homoacetogen_gene_cons_profiles[0:int(365/params.dt),:]
methanotroph_gene_cons_profiles = methanotroph_gene_cons_profiles[0:int(365/params.dt),:]
###

### Read diffusion and plant rates
test = read_transport_output(params.output_directory+params.site+'/transport_profile_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
q_diff_ch4_profiles = gaussian_filter(test[3],g_smooth)
q_diff_co2_profiles = gaussian_filter(test[4],g_smooth)
q_diff_o2_profiles = gaussian_filter(test[5],g_smooth)
q_diff_h2_profiles = gaussian_filter(test[6],g_smooth)
q_diff_ace_profiles = gaussian_filter(test[7],g_smooth)
q_plant_ch4_profiles = gaussian_filter(test[8],g_smooth)
q_plant_co2_profiles = gaussian_filter(test[9],g_smooth)
q_plant_o2_profiles = gaussian_filter(test[10],g_smooth)
q_plant_h2_profiles = gaussian_filter(test[11],g_smooth)
q_plant_ace_profiles = gaussian_filter(test[12],g_smooth)
q_diff_ch4_profiles = q_diff_ch4_profiles[0:int(365/params.dt),:]
q_diff_co2_profiles = q_diff_co2_profiles[0:int(365/params.dt),:]
q_diff_o2_profiles = q_diff_o2_profiles[0:int(365/params.dt),:]
q_diff_h2_profiles = q_diff_h2_profiles[0:int(365/params.dt),:]
q_diff_ace_profiles = q_diff_ace_profiles[0:int(365/params.dt),:]
q_plant_ch4_profiles = q_plant_ch4_profiles[0:int(365/params.dt),:]
q_plant_co2_profiles = q_plant_co2_profiles[0:int(365/params.dt),:]
q_plant_o2_profiles = q_plant_o2_profiles[0:int(365/params.dt),:]
q_plant_h2_profiles = q_plant_h2_profiles[0:int(365/params.dt),:]
q_plant_ace_profiles = q_plant_ace_profiles[0:int(365/params.dt),:]
###

ceq_ch4, ceq_o2, ceq_co2, ceq_h2 = transport.equilibrium_concentration(10.0)

### Read microbe gene quantities
test = read_microbe_output(params.output_directory+params.site+'/microbe_output.txt')
date_times = test[2]
datetimes=[]
for d in range(0,len(date_times)):
    datetimes.append(datetime.strptime(date_times[d], '%Y-%m-%d %H:%M:%S'))
x_lims = mdates.date2num(datetimes)
y_lims = [0.6,0.025]
depths = []  # Calculate the layer center depths
for l in range(1,test[0]+1):
    depths.append(l*test[1])
depths=np.array(depths)
acemethanogen_gene_profiles = gaussian_filter(test[3],0.0)
h2_methanogen_gene_profiles = gaussian_filter(test[4],0.0)
homoacetogen_gene_profiles = gaussian_filter(test[5],0.0)
methanotroph_gene_profiles = gaussian_filter(test[6],0.0)
acemethanogen_gene_profiles = acemethanogen_gene_profiles[0:int(365/params.dt),:]
h2_methanogen_gene_profiles = h2_methanogen_gene_profiles[0:int(365/params.dt),:]
homoacetogen_gene_profiles = homoacetogen_gene_profiles[0:int(365/params.dt),:]
methanotroph_gene_profiles = methanotroph_gene_profiles[0:int(365/params.dt),:]
###

f_diff_aces = plant_flux_aces
date_format = mdates.DateFormatter('%b')

#test = make_legend2(['Methanotrophy'], ['Ace. Methanogenesis', 'H$_2$ Methanogenesis'], ax21, 20, -0.05, 1, 1)
#test = make_legend2(['Methanotrophy', 'Aerobic. Respira.','Aerobic. Decomp.'],[], ax24, 20, -0.05, 1, 1)
#test = make_legend2(['          Ace. Methanogenesis'], ['          Aerobic Decomp.','          Homoacetogenesis'], ax27, 20, -0.05, 1, 1)
#test = make_legend2(['Homoacetogenesis', 'H$_2$ Methanogenesis'],['Anaero.  Decomp.', 'Aerobic. Decomp.', 'Ace. Methanogenesis'], ax30, 20, 0.05, 1, 1)
#test = make_legend2(['Homoacetogenesis', 'H$_2$ Methanogenesis'], ['Anaero. Decomp.'], ax33, 20, -0.05, 1, 1)

fig, axes = plt.subplots()
fig.patch.set_facecolor('ghostwhite')
fig.set_size_inches(40.0, 30.0, forward=True)

ax1 = plt.subplot2grid((7,15),(0,0),colspan=3,rowspan=1)
ax2 = plt.subplot2grid((7,15),(0,3),colspan=3,rowspan=1)
ax3 = plt.subplot2grid((7,15),(0,6),colspan=3,rowspan=1)
ax4 = plt.subplot2grid((7,15),(0,9),colspan=3,rowspan=1)
ax5 = plt.subplot2grid((7,15),(0,12),colspan=3,rowspan=1)
ax6 = plt.subplot2grid((7,15),(1,0),colspan=3,rowspan=2)
ax7 = plt.subplot2grid((7,15),(1,3),colspan=3,rowspan=2)
ax8 = plt.subplot2grid((7,15),(1,6),colspan=3,rowspan=2)
ax9 = plt.subplot2grid((7,15),(1,9),colspan=3,rowspan=2)
ax10 = plt.subplot2grid((7,15),(1,12),colspan=3,rowspan=2)
ax11 = plt.subplot2grid((7,15),(3,0),colspan=3,rowspan=2)
ax12 = plt.subplot2grid((7,15),(3,4),colspan=3,rowspan=2)
ax13 = plt.subplot2grid((7,15),(3,8),colspan=3,rowspan=2)
ax14 = plt.subplot2grid((7,15),(3,12),colspan=3,rowspan=2)
ax15 = plt.subplot2grid((7,15),(5,0),colspan=3,rowspan=2)
ax16 = plt.subplot2grid((7,15),(5,3),colspan=3,rowspan=2)
ax17 = plt.subplot2grid((7,15),(5,6),colspan=3,rowspan=2)
ax18 = plt.subplot2grid((7,15),(5,9),colspan=3,rowspan=2)
ax19 = plt.subplot2grid((7,15),(5,12),colspan=3,rowspan=2)

days = [0.45,0.60,0.77]

ch4_profiles[temp_profiles < 0.0] = np.nan
test = surface_flux_plots(ax1,gaussian_filter(f_diff_ch4s,0.0),gaussian_filter(plant_flux_ch4s,0.0),'CH$_4$','mol/m$^2$/day',params.dt,days)
test = profiles_plot_properties(ax6,ch4_profiles,ch4_c_prod_profiles,' ',datetimes,depths,np.linspace(0.0,0.1,21),True,days,cm.Purples,'mol/m$^3$',temp_profiles,'ch4',np.logspace(-3, 1, 9),False)

o2_profiles[temp_profiles < 0.0] = np.nan
test = surface_flux_plots(ax2,gaussian_filter(f_diff_o2s,0.0),gaussian_filter(plant_flux_o2s,0.0),'O$_2$',' ',params.dt,days)
test = profiles_plot_properties(ax7,o2_profiles,o2_c_prod_profiles,' ',datetimes,depths,np.linspace(0.0,20.0,21),False,days,cm.Greens,' ',temp_profiles,'o2',np.logspace(-3, 1.5, 10),False)

co2_profiles[temp_profiles < 0.0] = np.nan
test = surface_flux_plots(ax3,gaussian_filter(f_diff_co2s,0.0),gaussian_filter(plant_flux_co2s,0.0),'CO$_2$',' ',params.dt,days)
test = profiles_plot_properties(ax8,co2_profiles,co2_c_prod_profiles,' ',datetimes,depths,np.linspace(0.0,75.0,21),False,days,cm.Blues,' ',temp_profiles,'co2',np.logspace(-3, 2.5, 12),False)
test = cross_plots(ax8,datetimes,co2_c_prod_profiles-co2_c_cons_profiles,[co2_c_prod_ansp_profiles,co2_c_prod_aesp_profiles,co2_c_prod_acmg_profiles],[co2_c_cons_hoag_profiles,co2_c_cons_h2mg_profiles],temp_profiles,depths,0.16)

h2_profiles[temp_profiles < 0.0] = np.nan
test = surface_flux_plots(ax4,gaussian_filter(f_diff_h2s,0.0),gaussian_filter(plant_flux_h2s,0.0),'H$_2$',' ',params.dt,days)
test = profiles_plot_properties(ax9,h2_profiles,h2_c_prod_profiles,' ',datetimes,depths,np.linspace(0.0,0.10,21),False,days,cm.Oranges,' ',temp_profiles,'h2',np.logspace(0, 1, 7),False)
test = cross_plots(ax9,datetimes,h2_c_prod_profiles-h2_c_cons_profiles,[h2_c_prod_ansp_profiles],[h2_c_cons_hoag_profiles,h2_c_cons_h2mg_profiles],temp_profiles,depths,0.16)

ace_profiles[temp_profiles < 0.0] = np.nan
test = surface_flux_plots(ax5,gaussian_filter(f_diff_aces,0.0),gaussian_filter(plant_flux_aces,0.0),'Acetate',' ',params.dt,days)
test = profiles_plot_properties(ax10,ace_profiles,ace_c_prod_profiles,' ',datetimes,depths,np.linspace(0.0,0.05,21),False,days,cm.Reds,' ',temp_profiles,'acetate',np.logspace(-3, 1, 9),True)
test = cross_plots(ax10,datetimes,ace_c_prod_profiles-ace_c_cons_profiles,[ace_c_prod_aesp_profiles,ace_c_prod_hoag_profiles],[ace_c_cons_acmg_profiles],temp_profiles,depths,0.05)

acemethanogen_gene_profiles[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax11,acemethanogen_gene_profiles,acemethanogen_gene_profiles,' ',datetimes,depths,np.linspace(0,10,21),True,days,cm.Reds,'mol C/m$^3$',temp_profiles,' ',np.logspace(13,15, 13),False)
test = cross_plots(ax11,datetimes,acemethanogen_gene_prod_profiles-acemethanogen_gene_cons_profiles,[acemethanogen_gene_prod_profiles],[acemethanogen_gene_cons_profiles],temp_profiles,depths,0.02)

methanotroph_gene_profiles[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax12,methanotroph_gene_profiles,methanotroph_gene_profiles,' ',datetimes,depths,np.linspace(0,7,21),False,days,cm.Greens,' ',temp_profiles,' ',np.logspace(15, 17, 13),False)
test = cross_plots(ax12,datetimes,methanotroph_gene_prod_profiles-methanotroph_gene_cons_profiles,[methanotroph_gene_prod_profiles],[methanotroph_gene_cons_profiles],temp_profiles,depths,0.02)

homoacetogen_gene_profiles[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax13,homoacetogen_gene_profiles,homoacetogen_gene_profiles,' ',datetimes,depths,np.linspace(0,0.025,21),False,days,cm.Oranges,' ',temp_profiles,' ',np.logspace(11, 13, 13),False)
test = cross_plots(ax13,datetimes,homoacetogen_gene_prod_profiles-homoacetogen_gene_cons_profiles,[homoacetogen_gene_prod_profiles],[homoacetogen_gene_cons_profiles],temp_profiles,depths,0.02)

h2_methanogen_gene_profiles[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax14,h2_methanogen_gene_profiles,h2_methanogen_gene_profiles,' ',datetimes,depths,np.linspace(0,0.05,21),False,days,cm.Oranges,' ',temp_profiles,' ',np.logspace(15, 17, 13),False)
test = cross_plots(ax14,datetimes,h2_methanogen_gene_prod_profiles-h2_methanogen_gene_cons_profiles,[h2_methanogen_gene_prod_profiles],[h2_methanogen_gene_cons_profiles],temp_profiles,depths,0.02)

temp_profiles2 = np.copy(temp_profiles)
temp_profiles2[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax15,temp_profiles2,temp_profiles2,' ',datetimes,depths,np.linspace(-5,15,21),False,days,cm.rainbow,'Temp $^o$C',temp_profiles,' ',np.logspace(15, 17, 13),False)

moist_profiles = moist_profiles
moist_profiles[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax16,moist_profiles,moist_profiles,' ',datetimes,depths,np.linspace(0,100.0,21),False,days,cm.rainbow,'VWC %',temp_profiles,' ',np.logspace(15, 17, 13),False)

doc_profiles[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax17,doc_profiles,doc_profiles,' ',datetimes,depths,np.linspace(0,4.0,21),False,days,cm.rainbow,'DOC mol/m$^3$',temp_profiles,' ',np.logspace(15, 17, 13),False)

c_profiles[temp_profiles <= 0.0] = np.nan
print(np.nanmean(c_profiles))
test = profiles_plot_properties(ax18,c_profiles,c_profiles,' ',datetimes,depths,np.linspace(0,10000.0,21),False,days,cm.rainbow,'SOC mol/m$^3$',temp_profiles,' ',np.logspace(15, 17, 13),False)

ph_profiles[temp_profiles <= 0.0] = np.nan
test = profiles_plot_properties(ax19,ph_profiles,ph_profiles,' ',datetimes,depths,np.linspace(3.0,8.0,21),False,days,cm.rainbow,'pH',temp_profiles,' ',np.logspace(15, 17, 13),False)

#[o2_c_prod_none_profiles],[o2_c_cons_mtox_profiles,o2_c_cons_aero_profiles,o2_c_cons_aesp_profiles]
#plt.subplots_adjust(wspace=0.1)
fig.subplots_adjust(hspace=1.5)
fig.subplots_adjust(wspace=0.5)

if not os.path.exists(params.main_directory + 'modelPlots/' + params.site):
    os.mkdir(params.main_directory + 'modelPlots/' + params.site)

plt.savefig(params.main_directory + 'modelPlots/' + params.site + '/profiles.png',bbox_inches='tight')

stop