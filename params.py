import os
import csv
import random
import numpy as np
#import cg_autotuner
from os import listdir
from datetime import datetime
from datetime import timedelta
from os.path import isfile, join

def rr(mean):

    min_val = 0.5 * mean
    max_val = 1.5 * mean
    rand_val = random.uniform(min_val,max_val)

    return rand_val

def rr2(min_val, max_val):

    rand_val = random.uniform(min_val, max_val)
    rand_formatted_val = float(format(rand_val, ".4f"))

    return rand_formatted_val

# Directory paths
data_directory = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/'
cg_directory = 'C:/Users/Jeremy/Desktop/Churchill_Data/CryoGrid/'
output_directory = 'C:/Users/Jeremy/Desktop/Churchill_Data/modelOutput/'
main_directory = 'C:/Users/Jeremy/Desktop/Churchill_Data/'

# Misc
subplots = {}
subplots['palsa_low_plots'], subplots['palsa_high_plots'] = ['P1','P2','P3','P4'], ['P5','P6','P7','P8','P9','P10']
subplots['fen_low_plots'], subplots['fen_high_plots'] = ['F2','F3','F4','F5'], ['F1','F6','F7','F8','F9','F10']
subplots['tundra_low_plots'], subplots['tundra_high_plots'] = ['T1','T2','T3','T10'], ['T4','T5','T6','T7','T8','T9']

# Data standardization
#
standardize_flag = True
# geography
std_geography = True
# roots
std_roots = True
# genes
std_mcra = True
std_pmoa1 = True
std_pmoa2 = True
# temps
std_temps = True
# fluxes
std_fluxes = True
# composition
std_waterpct = True
std_mineralpct = True
std_organicpct = True
std_airpct = True
std_watertable = True

site = 'palsa_low'

type = site.split('_')[1]
water_table_flag = True
if type == 'low': water_table_flag = True
if type == 'high': water_table_flag = False

test_datetime = datetime(2022, 8, 1, 0, 0, 0)
years = 3 # Number of simulated years: for individual simulations, or for autotuner test simulations

adaptive_dt_flag = True

# Major processes
microbe_flag = True
diffusion_flag = True
plant_flag = True

# Operation modes
surface_diffusion = 'transfer' # 'transfer' for water-air interface flow, or 'fixed' for atmospheric layer convergence
genomic_flag = False # Set 'True' to express microbe abundances in [genes/m^3], 'False' to express them in [mol_MBC/m^3]
sataero_flag = True # Set 'True' to scale aerobic and anaerobic chemistry rates with saturation %
diff_mode = 'mixed' # 'water' or 'air' = water or air diffusivity through the entire soil column. 'Mixed' = weighted avg

# Extras
write_flag = False
plot_data_flag = False
process_test = 'none'
diagnostic_flag = False

# CryoGrid
cryogrid_name = 'High_Fixed_WT_og'
#cryogrid_name = 'Drying0p0010_z00p05_rootDepth0p1_fieldcap0p35'
cg_depths = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 10.0])  # meters
cg_thicks = 0.0 * cg_depths + 0.1 # dummy thicknesses, not needed
save_date = '01.01.'  # Month, Day
start_time, end_time = [2000, 9, 1], [2005, 1, 2]  # Year, Month, Day
water_table_depth = 0.05
lateral, saturated = True, True
rain_fraction, snow_fraction = 1.5, 1.8
z0 = 0.05
rootDepth = 0.1
evaporationDepth = rootDepth
distance_reservoir = 1.0
reservoir_contact_length = 1.0
begin_drying, drying_rate = 180.0, 0.0010

# Autotune parameters
autotune_flag = True
autotune_mode, autotune_name, sims_per_first_generation, sims_per_generation, n_generations = \
    'run', cryogrid_name, 3, 3, 3

saturation_threshold = 50.0 # %

dt = 0.05/24.0 # main computational time step days
nz = 12 # 16 number of layers
dz = 0.05 # 0.03 layer thickness in meters
diff_n_dt = 2 # number of diffusion sub-time-steps within dt
write_dt = 0.25 # output write interval in days

k_cpool = 0.02 # 0.0001, 0.02, 0.02 Ratio of dissolved organic carbon to soil organic carbon
v_doc_prod_ace_max = 0.5 # 0.3, 0.5, 0.7 Maximum acetate production rate from fermentation
k_doc_prod_ace = 10.0 # 5, 10, 15 Half saturation coefficient
k_ace_prod_o2 = 0.04 # 0.01, 0.04, 0.1 Half saturation coefficient
v_h2_prod_ace_max = 0.15 # 0.01, 0.15, 0.3  Maximum acetate production rate from homoacetogenesis
v_h2_prod_ch4_max = 0.15 # 0.01, 0.15, 0.3 Maximum CH4 production rate from hydrogenotrophic methanogenesis
v_ace_cons_max = 0.5 # 0.3, 0.5, 0.7 Maximum acetate consumption rate by aceclastic methanogens
v_ch4_oxid_max = 0.5 # 0.3, 0.5, 0.7 Maximum CH4 oxidation rate
k_h2_prod_ace = 0.01 # 0.01, 0.01, 0.1 Half saturation coefficient
k_co2_prod_ace = 0.05 # 0.01, 0.05, 0.1 Half saturation coefficient
k_h2_prod_ch4 = 0.01 # 0.01, 0.01, 0.1, Half saturation coefficient
k_co2_prod_ch4 = 0.05 # 0.01, 0.05, 0.1 Half saturation coeffcient
k_ace_prod_ch4 = 0.05 # 0.01, 0.05, 0.1 Half saturation coefficient
k_ch4_prod = 0.5 # 0.3, 0.5, 0.7 CH4 production ratio
k_ch4_oxid_ch4 = 0.05 # 0.01, 0.05, 0.1 Half saturation coefficient
k_ch4_oxid_o2 = 0.02 # 0.01, 0.02, 0.1 Half saturation coefficient
f_d_w = 0.8 # 0.7, 0.8, 0.9 Reduction factor for diffusion in water-filled peat
f_d_a = 0.8 # 0.7, 0.8, 0.9 Reduction factor for diffusion in air-filled peat
tau = 1.5 # 1, 1.5, 2 Root tortuosity
a_m_a = 1.85 # 0.01, 0.085, 0.1 Root ending area per root dry biomass
s_l_a = 20.0 # 15, 20, 25 Specific leaf area
sigma = 0.8 # 0.7, 0.8, 0.9 Peat porosity
k = 0.0001 # 0.0001, 0.0001, 0.0003 Time constant
# t of ebullition
k_aer = 0.1 # 0.1, 0.1, 0.2 O2 consume rate from aerobic respiration
k_aer_doc = 10.0 # 5, 10, 15 Half saturation coefficient
k_aer_o2 = 0.22 # 0.01, 0.22, 0.3 Half saturation coefficient
doc_prod_q10 = 2.5 # 1, 2.5, 5 Temperature sensitivity of DOC production
ace_prod_q10 = 2.5 # 1, 2.5, 5 Temperature senstivity of acetate production
ch4_prod_q10 = 2.5 # 1, 2.5, 5 Temperature sensitivity of CH4 production
ch4_oxid_q10 = 2.5 # 1, 2.5, 5 Temperature sensitivity of CH4 oxidation

# Adjusted params
#k_aer_o2, k_aer_doc, k_aer, a_m_a, v_ch4_oxid_max, k_ch4_oxid_o2, k_ch4_oxid_ch4,
#ch4_prod_q10, ace_prod_q10, doc_prod_q10, k_ch4_prod, k_ace_prod_ch4, v_ace_cons_max, k_cpool = \


# Homoacetogenesis-related
t_min1 = 0.0 # 0.0 Minimum soil temperature for homoacetogenesis processes
t_max1 = 20.0 # 20.0 Maximum...
t_opt1 = 10.0 # 10.0 Optimum...

# H2 Methanogenesis-related
t_min2 = 0.0 # 20.0 Minimum soil temperature for H2 methanogenesis processes
t_max2 = 50.0 # 50.0 Maximum...
t_opt2 = 35.0 # 35.0 Optimum...

ft_max_temp = 30.0 # Maximum soil temperature for acetoclastic methanogenesis, methanotrophy, and decomposition of DOC

ph_min = 3.0 # 3.0 Minimum soil pH for pH factor
ph_max = 9.0 # 9.0 Maximum...
ph_opt = 6.2 # 6.2 Optimum...

grow_homoacetogens = 0.2 # Growth efficiency of homoacetogens
grow_h2methanogens = 0.2 # ... of H2 methanogens
grow_acemethanogens = 0.3 # ... of acetoclastic methanogens
grow_methanotrophs = 0.4 # ... of methanotrophs

dead_homoacetogens = 0.06 # Death rate of homoacetogens
dead_h2methanogens = 0.06 # ... of H2 methanogens
dead_acemethanogens = 0.06 # ... of acetoclastic methanogens
dead_methanotrophs = 0.06 # ... of methanotrophs

eff_ch4 = 0.5
eff_co2 = 0.5

# Constants
gc_per_molc = 12.01
gc_per_copy_homoacetogens = 1e-13
gc_per_copy_acemethanogens = 1e-13
gc_per_copy_h2_methanogens = 1e-13
gc_per_copy_methanotrophs = 1e-13

# Autotune parameters
if autotune_flag == True:

    nz = 12     # Number of model layers
    dz = 0.05   # Layer thicknesses

    # Find the current and next generation
    mypath = 'C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/'

    # Make a site folder if it doesn't already exist
    if not os.path.exists('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/'):
        os.mkdir('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/')

    # Make an autotune name folder, in the site folder, if it doesn't already exist
    if not os.path.exists('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/'):
        os.mkdir('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/')

    # Determine the current generation
    if len(next(os.walk('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/'))[1]) == 0:
        gen = 0
    else:
        gen = int(np.array(next(os.walk('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/'))[1])[-1])
    next_gen = gen + 1

    # Make a generation 0 folder if it doesn't already exist
    if not os.path.exists('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/0'):
        os.mkdir('C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/0')

    # Find all files in the current generation
    genpath = 'C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + site + '/' + autotune_name + '/' + str(gen) + '/'
    files = [f for f in listdir(genpath) if isfile(join(genpath, f))]

    # Find the first file with a configuration that hasn't been run yet
    for file in files:
        filepath = genpath + file
        with open(filepath, newline='') as csvfile: data = list(csv.reader(csvfile))
        if len(data) == 2: # Not-run file found

            # Extract the parameter values from the file header
            param_list = data[0]
            #print('data 0: ', data[0])
            k_cpool, v_ace_cons_max, k_ace_prod_ch4, k_ch4_prod, doc_prod_q10, ace_prod_q10, ch4_prod_q10, \
            k_ch4_oxid_ch4, k_ch4_oxid_o2, v_ch4_oxid_max, a_m_a, k_aer, k_aer_doc, k_aer_o2 = \
            float(param_list[0]), float(param_list[1]), float(param_list[2]), float(param_list[3]), \
            float(param_list[4]), float(param_list[5]), float(param_list[6]), float(param_list[7]), \
            float(param_list[8]), float(param_list[9]), float(param_list[10]), float(param_list[11]), \
            float(param_list[12]), float(param_list[13])

            #print(k_aer, k_aer_doc, k_aer_o2)

            #print(a_m_a)

            # Record the file path for writing purposes
            autotune_file = filepath
            print('autotune file: ', autotune_file)

    #print(autotune_file)