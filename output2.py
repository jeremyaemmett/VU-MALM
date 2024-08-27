import os
import numpy as np
import params
import csv

def header_info(filename):

    type1 = filename.split('/')[1][0]
    type2 = filename.split('/')[1][1]
    type3 = filename.split('/')[-1].split('.')[0]

    unit = '-'
    if type1+type2 == 'ca': unit = 'mol/m3'
    if type1+type2 == 'cr': unit = 'mol/m3/day'
    if type1+type2 == 'ga': unit = 'copies/m3'
    if type1+type2 == 'gr': unit = 'copies/m3/day'
    if type2 == 'f': unit = 'mol/m2/day'

    return(unit)

def prepare_output_files(layer_thicknesses, depths, ca, ga, df, pf, cr, gr, dr, pr, crd2, en, eq):

    main_directory = params.main_directory

    # Make sure there's an output directory for the site, and a subdirectory for each dictionary
    if not os.path.exists(main_directory + 'modelOutput/' + params.site):
        os.mkdir(main_directory + 'modelOutput/' + params.site)
    main_paths = ['modelOutput/'+params.site+'/ca', 'modelOutput/'+params.site+'/ga', 'modelOutput/'+params.site+'/df',
                  'modelOutput/'+params.site+'/pf', 'modelOutput/'+params.site+'/cr', 'modelOutput/'+params.site+'/gr',
                  'modelOutput/'+params.site+'/dr', 'modelOutput/'+params.site+'/pr', 'modelOutput/'+params.site+'/crd2',
                  'modelOutput/'+params.site+'/en', 'modelOutput/'+params.site+'/eq']
    for m in main_paths:
        if not os.path.exists(main_directory+m):
            os.mkdir(main_directory+m)

    # Every dictionary key gets an output file within its respective subdirectory
    ca_keys, ga_keys, df_keys, pf_keys, cr_keys, gr_keys, dr_keys, pr_keys, crd2_keys, en_keys, eq_keys = \
        list(ca.keys()), list(ga.keys()),list(df.keys()), list(pf.keys()), list(cr.keys()), \
        list(gr.keys()), list(dr.keys()), list(pr.keys()), list(crd2.keys()), list(en.keys()), list(eq.keys())
    ca_files, ga_files, df_files, pf_files, cr_files, gr_files, dr_files, pr_files, crd2_files, en_files, eq_files = \
        [], [], [], [], [], [], [], [], [], [], []
    for j in range(0,len(ca_keys)): ca_files.append('/ca/'+ca_keys[j]+'.csv')
    for j in range(0,len(ga_keys)): ga_files.append('/ga/'+ga_keys[j]+'.csv')
    for j in range(0, len(df_keys)): df_files.append('/df/' + df_keys[j] + '.csv')
    for j in range(0, len(pf_keys)): pf_files.append('/pf/' + pf_keys[j] + '.csv')
    for j in range(0, len(cr_keys)): cr_files.append('/cr/' + cr_keys[j] + '.csv')
    for j in range(0, len(gr_keys)): gr_files.append('/gr/' + gr_keys[j] + '.csv')
    for j in range(0, len(dr_keys)): dr_files.append('/dr/' + dr_keys[j] + '.csv')
    for j in range(0, len(pr_keys)): pr_files.append('/pr/' + pr_keys[j] + '.csv')
    for j in range(0, len(crd2_keys)): crd2_files.append('/crd2/' + crd2_keys[j] + '.csv')
    for j in range(0, len(en_keys)): en_files.append('/en/' + en_keys[j] + '.csv')
    for j in range(0, len(eq_keys)): eq_files.append('/eq/' + eq_keys[j] + '.csv')
    files = ca_files + ga_files + df_files + pf_files + cr_files + gr_files + \
            dr_files + pr_files + crd2_files + en_files + eq_files

    # Create the output files and assign them to callable objects (object name = file path)
    output_files = {}
    for f in range(0,len(files)):
        output_path = main_directory+'modelOutput/'+params.site+files[f]
        unit = header_info(files[f])
        if 'time_data' not in files[f] and 'depth_data' not in files[f]:  # Exclude time and depth files
            with open(output_path, "w") as output_files[files[f]]:
                #output_files[files[f]].write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
                output_files[files[f]].write(f"{unit},{' '},{','.join(map(str, range(len(depths))))}\n")
                output_files[files[f]].write(f"{' '},{' '},{','.join(map(str, depths))}\n")
            output_files[files[f]].close()

    return(output_files)


def make_autotune_file(gen, params_dict):

    param_keys = list(params_dict.keys())
    fname = ''
    for p in range(0, len(param_keys)):
        param_val = str(params_dict[param_keys[p]])
        param_val = param_val.replace('.', 'p')
        if p < len(param_keys)-1: fname += param_val + '_'
        if p == len(param_keys)-1: fname += param_val
    fname = 'C:/Users/Jeremy/Desktop/Churchill_Data/modelTune/' + str(gen) + '/' + fname + '.csv'
    params_dict['file'] = fname
    with open(fname, "a") as current_file:
        writer = csv.writer(current_file)
        writer.writerow(list(params_dict.values()))

    return(fname)


def write_autotune_data(autotune_file, data, write_count, test_datetime):

    with open(autotune_file, "a") as current_file:
        current_file.write(f"{write_count},{test_datetime},{str(data)}\n")


def write_output(output_files, test_datetime,
                 ca, ga, df, pf,
                 forcing_t, equilibrium_t, ph_profile,
                 cr, gr, dr, pr, crd2,
                 depths, write_count):

    main_directory = params.main_directory

    output_file_list = list(output_files.keys())
    for o in range(0,len(output_file_list)):
        test = output_file_list[o]
        folder, file = test.split('/')[1], test.split('/')[2]
        if folder == 'ca': dict = ca
        if folder == 'ga': dict = ga
        if folder == 'df': dict = df
        if folder == 'pf': dict = pf
        if folder == 'cr': dict = cr
        if folder == 'gr': dict = gr
        if folder == 'dr': dict = dr
        if folder == 'pr': dict = pr
        if folder == 'crd2': dict = crd2
        if folder == 'en': dict = forcing_t
        if folder == 'eq': dict = equilibrium_t

        #print(folder, test_datetime)

        variable, extension = file.split('.')[0], file.split('.')[1]
        #print(variable, extension)
        with open(main_directory + 'modelOutput/' + params.site + '/' + folder + '/' + file, "a") as current_file:
            if np.size(dict[variable]) == 1: # 1D data
                current_file.write(f"{write_count},{test_datetime},{str(dict[variable])}\n")
            else: # 2D data
                current_file.write(f"{write_count},{test_datetime},{','.join(map(str, dict[variable]))}\n")

    return()