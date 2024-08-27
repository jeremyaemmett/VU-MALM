import params
import transport
import numpy as np
import conversions


def prepare_output_files(layer_thicknesses, depths):

    with open(r'C:\Users\Jeremy\Desktop\output\substrate_output.txt', "w") as myfile1:
        myfile1.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile1.close()
    with open(r'C:\Users\Jeremy\Desktop\output\microbe_output.txt', "w") as myfile2:
        myfile2.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile2.close()
    with open(r'C:\Users\Jeremy\Desktop\output\flux_output.txt', "w") as myfile3:
        myfile3.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile3.close()
    with open(r'C:\Users\Jeremy\Desktop\output\envir_output.txt', "w") as myfile4:
        myfile4.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile4.close()
    with open(r'C:\Users\Jeremy\Desktop\output\substrate_net_production_output.txt', "w") as myfile5:
        myfile5.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile5.close()
    with open(r'C:\Users\Jeremy\Desktop\output\substrate_net_consumption_output.txt', "w") as myfile6:
        myfile6.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile6.close()
    with open(r'C:\Users\Jeremy\Desktop\output\substrate_production_output.txt', "w") as myfile7:
        myfile7.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile7.close()
    with open(r'C:\Users\Jeremy\Desktop\output\substrate_consumption_output.txt', "w") as myfile8:
        myfile8.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile8.close()
    with open(r'C:\Users\Jeremy\Desktop\output\microbe_net_output.txt', "w") as myfile9:
        myfile9.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile9.close()
    with open(r'C:\Users\Jeremy\Desktop\output\microbe_growth_output.txt', "w") as myfile10:
        myfile10.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile10.close()
    with open(r'C:\Users\Jeremy\Desktop\output\microbe_mortality_output.txt', "w") as myfile11:
        myfile11.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile11.close()
    with open(r'C:\Users\Jeremy\Desktop\output\transport_profile_output.txt', "w") as myfile12:
        myfile12.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile12.close()

    return()


def write_output(test_datetime,
                 ch4_profile, co2_profile, o2_profile, h2_profile, ace_profile, c_profile, doc_profile,
                 acemethanogen_gene_profile, h2_methanogen_gene_profile, homoacetogen_gene_profile, methanotroph_gene_profile,
                 f_diff_ch4, f_diff_co2, f_diff_h2, f_diff_o2, plant_flux_ch4, plant_flux_o2, plant_flux_co2, plant_flux_h2, plant_flux_ace,
                 temps, moists, ph_profile,
                 ch4_c_prod_profile, co2_c_prod_profile, o2_c_prod_profile, h2_c_prod_profile, ace_c_prod_profile,
                 ch4_c_cons_profile, co2_c_cons_profile, o2_c_cons_profile, h2_c_cons_profile, ace_c_cons_profile,
                 ch4_c_prod_acmg_profile,ch4_c_prod_h2mg_profile,
                 ace_c_prod_aesp_profile,ace_c_prod_hoag_profile,
                 co2_c_prod_ansp_profile,co2_c_prod_aesp_profile,co2_c_prod_acmg_profile,
                 h2_c_prod_ansp_profile,
                 o2_c_prod_none_profile,
                 ch4_c_cons_mtox_profile,
                 ace_c_cons_acmg_profile,
                 co2_c_cons_hoag_profile,co2_c_cons_h2mg_profile,
                 h2_c_cons_hoag_profile,h2_c_cons_h2mg_profile,
                 o2_c_cons_mtox_profile,o2_c_cons_aero_profile,o2_c_cons_aesp_profile,
                 acemethanogen_gene_rate_profile,h2_methanogen_gene_rate_profile,homoacetogen_gene_rate_profile,methanotroph_gene_rate_profile,
                 acemethanogen_gene_prod_profile,h2_methanogen_gene_prod_profile,homoacetogen_gene_prod_profile,methanotroph_gene_prod_profile,
                 acemethanogen_gene_mort_profile,h2_methanogen_gene_mort_profile,homoacetogen_gene_mort_profile,methanotroph_gene_mort_profile,
                 q_diff_ch4_profile,q_diff_co2_profile,q_diff_o2_profile,q_diff_h2_profile,q_diff_ace_profile,
                 q_plant_ch4_profile,q_plant_co2_profile,q_plant_o2_profile,q_plant_h2_profile,q_plant_ace_profile):

    with open(r'C:\Users\Jeremy\Desktop\output\substrate_output.txt', "a") as myfile1:
        myfile1.write(
            f"{test_datetime},{','.join(map(str, ch4_profile))},{','.join(map(str, co2_profile))},{','.join(map(str, o2_profile))},{','.join(map(str, h2_profile))},{','.join(map(str, ace_profile))},{','.join(map(str, c_profile))},{','.join(map(str, doc_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\microbe_output.txt', "a") as myfile2:
        myfile2.write(
            f"{test_datetime},{','.join(map(str, acemethanogen_gene_profile))},{','.join(map(str, h2_methanogen_gene_profile))},{','.join(map(str, homoacetogen_gene_profile))},{','.join(map(str, methanotroph_gene_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\flux_output.txt', "a") as myfile3:
        myfile3.write(str(test_datetime) + ','
                      + str(f_diff_ch4) + ',' + str(f_diff_co2) + ',' + str(f_diff_h2) + ',' + str(f_diff_o2) + ','
                      + str(plant_flux_ch4) + ',' + str(plant_flux_o2) + ',' + str(plant_flux_co2) + ','
                      + str(plant_flux_h2) + ',' + str(plant_flux_ace))
        myfile3.write('\n')

    with open(r'C:\Users\Jeremy\Desktop\output\envir_output.txt', "a") as myfile4:
        myfile4.write(
            f"{test_datetime},{','.join(map(str, temps))},{','.join(map(str, moists))},{','.join(map(str, ph_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\substrate_net_production_output.txt', "a") as myfile5:
        myfile5.write(
            f"{test_datetime},{','.join(map(str, ch4_c_prod_profile))},{','.join(map(str, co2_c_prod_profile))},{','.join(map(str, o2_c_prod_profile))},{','.join(map(str, h2_c_prod_profile))},{','.join(map(str, ace_c_prod_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\substrate_net_consumption_output.txt', "a") as myfile6:
        myfile6.write(
            f"{test_datetime},{','.join(map(str, ch4_c_cons_profile))},{','.join(map(str, co2_c_cons_profile))},{','.join(map(str, o2_c_cons_profile))},{','.join(map(str, h2_c_cons_profile))},{','.join(map(str, ace_c_cons_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\substrate_production_output.txt', "a") as myfile7:
        myfile7.write(
            f"{test_datetime},{','.join(map(str, ch4_c_prod_acmg_profile))},{','.join(map(str, ch4_c_prod_h2mg_profile))},{','.join(map(str, ace_c_prod_aesp_profile))},{','.join(map(str, ace_c_prod_hoag_profile))},{','.join(map(str, co2_c_prod_ansp_profile))},{','.join(map(str, co2_c_prod_aesp_profile))},{','.join(map(str, co2_c_prod_acmg_profile))},{','.join(map(str, h2_c_prod_ansp_profile))},{','.join(map(str, o2_c_prod_none_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\substrate_consumption_output.txt', "a") as myfile8:
        myfile8.write(
            f"{test_datetime},{','.join(map(str, ch4_c_cons_mtox_profile))},{','.join(map(str, ace_c_cons_acmg_profile))},{','.join(map(str, co2_c_cons_hoag_profile))},{','.join(map(str, co2_c_cons_h2mg_profile))},{','.join(map(str, h2_c_cons_hoag_profile))},{','.join(map(str, h2_c_cons_h2mg_profile))},{','.join(map(str, o2_c_cons_mtox_profile))},{','.join(map(str, o2_c_cons_aero_profile))},{','.join(map(str, o2_c_cons_aesp_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\microbe_net_output.txt', "a") as myfile9:
        myfile9.write(
            f"{test_datetime},{','.join(map(str, acemethanogen_gene_rate_profile))},{','.join(map(str, h2_methanogen_gene_rate_profile))},{','.join(map(str, homoacetogen_gene_rate_profile))},{','.join(map(str, methanotroph_gene_rate_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\microbe_growth_output.txt', "a") as myfile10:
        myfile10.write(
            f"{test_datetime},{','.join(map(str, acemethanogen_gene_prod_profile))},{','.join(map(str, h2_methanogen_gene_prod_profile))},{','.join(map(str, homoacetogen_gene_prod_profile))},{','.join(map(str, methanotroph_gene_prod_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\microbe_mortality_output.txt', "a") as myfile11:
        myfile11.write(
            f"{test_datetime},{','.join(map(str, acemethanogen_gene_mort_profile))},{','.join(map(str, h2_methanogen_gene_mort_profile))},{','.join(map(str, homoacetogen_gene_mort_profile))},{','.join(map(str, methanotroph_gene_mort_profile))}\n")

    with open(r'C:\Users\Jeremy\Desktop\output\transport_profile_output.txt', "a") as myfile12:
        myfile12.write(
            f"{test_datetime},{','.join(map(str, q_diff_ch4_profile))},{','.join(map(str, q_diff_co2_profile))},{','.join(map(str, q_diff_o2_profile))},{','.join(map(str, q_diff_h2_profile))},{','.join(map(str, q_diff_ace_profile))},{','.join(map(str, q_plant_ch4_profile))},{','.join(map(str, q_plant_co2_profile))},{','.join(map(str, q_plant_o2_profile))},{','.join(map(str, q_plant_h2_profile))},{','.join(map(str, q_plant_ace_profile))}\n")

    return()


def adaptive_timestep(quantities,rates,elapsed,temps,dt):

    approx_dt = 0.5*np.min(quantities/abs(rates))

    dl = [1]
    start = 1
    for i in range(5):
        dl.append(start*2)
        start = start*2
    dl = np.array(dl)
    dts = dt/dl

    if len((dts[dts < approx_dt])) == 0:
        clean_dt = dts[-1]
    else:
        clean_dt = dts[dts < approx_dt][0]
    if np.max(rates) == 0.0:
        clean_dt = dt

    if elapsed % dt + clean_dt > dt:
        clean_dt = dt - (elapsed % dt)

    return(clean_dt)


def define_layers():

    nlayers = params.nz # Number of vertical layers
    layer_thicknesses = params.dz + np.zeros(nlayers) # All layers have 2.5 cm thickness
    depths = [] # Calculate the layer center depths
    for l in range(nlayers):
        depths.append(np.sum(layer_thicknesses[0:l]))
    depths = depths + layer_thicknesses[0]

    return(layer_thicknesses,depths)


def define_microbes(depths):

#    methanotrophs = conversions.kg2mol(1300.0 * (1.0 * 1.0e-6),'c') # kg/m3 soil * mg/kg biomass * kg/mg * mol/kg = mol/m3
#    methanotroph_profile = methanotrophs + np.zeros(len(depths))
#    acemethanogens = conversions.kg2mol(1300.0 * (1.0 * 1.0e-6),'ch4') # kg/m3 soil * mg/kg biomass * kg/mg * mol/kg = mol/m3
#    acemethanogen_profile = acemethanogens + np.zeros(len(depths))
#    h2_methanogens = conversions.kg2mol(1300.0 * (1.0 * 1.0e-6),'ch4') # kg/m3 soil * mg/kg biomass * kg/mg * mol/kg = mol/m3
#    h2_methanogen_profile = h2_methanogens + np.zeros(len(depths))
#    homoacetogens = conversions.kg2mol(1300.0 * (1.0 * 1.0e-6),'ch4') # kg/m3 soil * mg/kg biomass * kg/mg * mol/kg = mol/m3
#    homoacetogen_profile = homoacetogens + np.zeros(len(depths))

    methanotroph_genes = 9.608e11
    methanotroph_genes = 10.0
    methanotroph_genes = methanotroph_genes + np.zeros(len(depths))
    methanotroph_gene_rate = np.zeros(len(depths))
    methanotroph_gene_prod = np.zeros(len(depths))
    methanotroph_gene_mort = np.zeros(len(depths))
    acemethanogen_genes = 9.608e11
    acemethanogen_genes = 1.0
    acemethanogen_genes = acemethanogen_genes + np.zeros(len(depths))
    acemethanogen_gene_rate = np.zeros(len(depths))
    acemethanogen_gene_prod = np.zeros(len(depths))
    acemethanogen_gene_mort = np.zeros(len(depths))
    h2_methanogen_genes = 9.608e11
    h2_methanogen_genes = 1.0
    h2_methanogen_genes = h2_methanogen_genes + np.zeros(len(depths))
    h2_methanogen_gene_rate = np.zeros(len(depths))
    h2_methanogen_gene_prod = np.zeros(len(depths))
    h2_methanogen_gene_mort = np.zeros(len(depths))
    homoacetogen_genes = 9.608e11
    homoacetogen_genes = 1.0
    homoacetogen_genes = homoacetogen_genes + np.zeros(len(depths))
    homoacetogen_gene_rate = np.zeros(len(depths))
    homoacetogen_gene_prod = np.zeros(len(depths))
    homoacetogen_gene_mort = np.zeros(len(depths))

    # Gene abundances
    ga = dict([
        ('acemethanogen_genes', acemethanogen_genes), ('h2_methanogen_genes', h2_methanogen_genes),
        ('methanotroph_genes', methanotroph_genes), ('homoacetogen_genes', homoacetogen_genes),
        ('totmethanogen_genes', acemethanogen_genes + h2_methanogen_genes)
    ])

    # Net gene rates
    gr = dict([
        ('acemethanogen_gene_rate', acemethanogen_gene_rate), ('h2_methanogen_gene_rate', h2_methanogen_gene_rate),
        ('methanotroph_gene_rate', methanotroph_gene_rate), ('homoacetogen_gene_rate', homoacetogen_gene_rate),
        # Gene growth rates
        ('acemethanogen_gene_prod', acemethanogen_gene_prod), ('h2_methanogen_gene_prod', h2_methanogen_gene_prod),
        ('methanotroph_gene_prod', methanotroph_gene_prod), ('homoacetogen_gene_prod', homoacetogen_gene_prod),
        # Gene mortality rates
        ('acemethanogen_gene_mort', acemethanogen_gene_mort), ('h2_methanogen_gene_mort', h2_methanogen_gene_mort),
        ('methanotroph_gene_mort', methanotroph_gene_mort), ('homoacetogen_gene_mort', homoacetogen_gene_mort),
    ])

    #ga['unit'] = 'genes/m3'
    #gr['unit'] = 'genes/m3/day'

    return(ga,gr)


def most_complete_date(dict2search, str2search):

    data_keys = np.array(list(dict2search.keys()))
    output_array = np.zeros_like(data_keys, dtype=bool)
    test_indices = np.char.find(data_keys, str2search)
    output_array[np.where(test_indices != -1)] = True
    found_keys = data_keys[output_array]
    found_keys.sort()

    print(str2search, found_keys)
    #stop

    completenesses = []
    for key_i in found_keys:
        completenesses.append(len(set(dict2search[key_i])))

    best_key = found_keys[np.argmax(completenesses)]
    if len(set(completenesses)) == 1: # If there's a tie in completeness scores, pick the first key
        best_key = found_keys[0]
    best_date = best_key.split('_')[-3]

    return(best_date)


def define_chemistry(depths, site_data):

    ch4, o2, co2, h2, ace, c, doc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # Default initial concentrations
    ch4, o2, co2, h2, ace, c, doc = 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10  # Default initial concentrations

    ch4_eq, o2_eq, co2_eq, h2_eq = transport.equilibrium_concentration(20.0)

    ## Initial concentrations for process checks (specified in 'params')

    if params.process_test == 'microbes': # Microbes only
        ch4, o2, co2, h2 = transport.equilibrium_concentration(20.0)
        ace = 0.1

    null = np.zeros(len(depths)) # Array of zeros, used repeatedly in profile definitions

    co2_profile = co2 + null
    co2_profile[0] = co2_eq
    co2_c_rate, co2_c_prod, co2_c_cons = null, null, null
    co2_c_prod_ansp, co2_c_prod_aesp, co2_c_prod_acmg, co2_c_prod_mtox, co2_c_cons_hoag, co2_c_cons_h2mg = \
        null, null, null, null, null, null

    ch4_profile = ch4 + null
    ch4_profile[0] = ch4_eq/2.0
    ch4_c_rate, ch4_c_prod, ch4_c_cons = null, null, null
    ch4_c_prod_acmg, ch4_c_prod_h2mg, ch4_c_cons_mtox = null, null, null

    o2_profile = o2 + null
    o2_profile[0] = o2_eq
    o2_c_rate, o2_c_prod, o2_c_cons = null, null, null
    o2_c_prod_none, o2_c_cons_mtox, o2_c_cons_aero, o2_c_cons_aesp = null, null, null, null

    h2_profile = h2 + null
    h2_profile[0] = h2_eq
    h2_c_rate, h2_c_prod, h2_c_cons = null, null, null
    h2_c_prod_ansp, h2_c_cons_hoag, h2_c_cons_h2mg = null, null, null

    ace_profile = ace + null
    ace_c_rate, ace_c_prod, ace_c_cons = null, null, null
    ace_c_prod_aesp, ace_c_prod_ansp, ace_c_prod_hoag, ace_c_cons_acmg = null, null, null, null

    # Decomposition 2 profiles
    ansp_ref_rate, aesp_ref_rate, acmg_ref_rate, h2mg_ref_rate, hoag_ref_rate, mtox_ref_rate, aero_ref_rate = \
        null, null, null, null, null, null, null
    ansp_chem_term, aesp_chem_term, acmg_chem_term, h2mg_chem_term, hoag_chem_term, mtox_chem_term, aero_chem_term = \
        null, null, null, null, null, null, null
    ansp_temp_term, aesp_temp_term, acmg_temp_term, h2mg_temp_term, hoag_temp_term, mtox_temp_term, aero_temp_term = \
        null, null, null, null, null, null, null
    ansp_micb_term, aesp_micb_term, acmg_micb_term, h2mg_micb_term, hoag_micb_term, mtox_micb_term, aero_micb_term = \
        null, null, null, null, null, null, null
    ansp_ph_term, aesp_ph_term, acmg_ph_term, h2mg_ph_term, hoag_ph_term, mtox_ph_term, aero_ph_term = \
        null, null, null, null, null, null, null
    ansp_vmax_term, aesp_vmax_term, acmg_vmax_term, h2mg_vmax_term, hoag_vmax_term, mtox_vmax_term, aero_vmax_term = \
        null+params.v_doc_prod_ace_max, null+params.v_doc_prod_ace_max, null+params.v_ace_cons_max, \
        null+params.v_h2_prod_ch4_max, null+params.v_h2_prod_ace_max, null+params.v_ch4_oxid_max, null

    cr = dict([
        # Net chem rates
        ('ch4_c_rate', ch4_c_rate),('o2_c_rate', o2_c_rate),('co2_c_rate', co2_c_rate),
        ('h2_c_rate', h2_c_rate),('ace_c_rate', ace_c_rate),
        # Total chem production rates
        ('ch4_c_prod', ch4_c_prod),('o2_c_prod', o2_c_prod),('co2_c_prod', co2_c_prod),
        ('h2_c_prod', h2_c_prod),('ace_c_prod', ace_c_prod),
        # Total chem consumption rates
        ('ch4_c_cons', ch4_c_cons),('o2_c_cons', o2_c_cons),('co2_c_cons', co2_c_cons),
        ('h2_c_cons', h2_c_cons),('ace_c_cons', ace_c_cons),
        # Decomposed chem production rates
        ('ch4_c_prod_acmg', ch4_c_prod_acmg),('ch4_c_prod_h2mg', ch4_c_prod_h2mg),
        ('o2_c_prod_none', o2_c_prod_none),
        ('co2_c_prod_ansp', co2_c_prod_ansp),('co2_c_prod_aesp', co2_c_prod_aesp),
        ('co2_c_prod_acmg', co2_c_prod_acmg),('co2_c_prod_mtox', co2_c_prod_mtox),
        ('h2_c_prod_ansp', h2_c_prod_ansp),
        ('ace_c_prod_aesp', ace_c_prod_aesp),('ace_c_prod_ansp', ace_c_prod_ansp),('ace_c_prod_hoag',ace_c_prod_hoag),
        # Decomposed chem consumption rates
        ('ch4_c_cons_mtox',ch4_c_cons_mtox),
        ('o2_c_cons_mtox',o2_c_cons_mtox),('o2_c_cons_aero',o2_c_cons_aero),('o2_c_cons_aesp',o2_c_cons_aesp),
        ('co2_c_cons_hoag',co2_c_cons_hoag),('co2_c_cons_h2mg',co2_c_cons_h2mg),
        ('h2_c_cons_hoag',h2_c_cons_hoag),('h2_c_cons_h2mg',h2_c_cons_h2mg),
        ('ace_c_cons_acmg',ace_c_cons_acmg)
    ])

    crd2 = dict([
        ('activity_level', 1.0 + 0.0 * ansp_ref_rate),
        ('ansp_ref_rate', ansp_ref_rate), ('aesp_ref_rate', aesp_ref_rate), ('acmg_ref_rate', acmg_ref_rate),
        ('h2mg_ref_rate', h2mg_ref_rate), ('hoag_ref_rate', hoag_ref_rate), ('mtox_ref_rate', mtox_ref_rate),
        ('aero_ref_rate', aero_ref_rate),
        ('ansp_chem_term', ansp_chem_term),('aesp_chem_term', aesp_chem_term),('acmg_chem_term', acmg_chem_term),
        ('h2mg_chem_term', h2mg_chem_term),('hoag_chem_term', hoag_chem_term),('mtox_chem_term', mtox_chem_term),
        ('aero_chem_term', aero_chem_term),
        ('ansp_temp_term', ansp_temp_term), ('aesp_temp_term', aesp_temp_term), ('acmg_temp_term', acmg_temp_term),
        ('h2mg_temp_term', h2mg_temp_term), ('hoag_temp_term', hoag_temp_term), ('mtox_temp_term', mtox_temp_term),
        ('aero_temp_term', aero_temp_term),
        ('ansp_micb_term', ansp_micb_term), ('aesp_micb_term', aesp_micb_term), ('acmg_micb_term', acmg_micb_term),
        ('h2mg_micb_term', h2mg_micb_term), ('hoag_micb_term', hoag_micb_term), ('mtox_micb_term', mtox_micb_term),
        ('aero_micb_term', aero_micb_term),
        ('ansp_vmax_term', ansp_vmax_term), ('aesp_vmax_term', aesp_vmax_term), ('acmg_vmax_term', acmg_vmax_term),
        ('h2mg_vmax_term', h2mg_vmax_term), ('hoag_vmax_term', hoag_vmax_term), ('mtox_vmax_term', mtox_vmax_term),
        ('aero_vmax_term', aero_vmax_term),
        ('ansp_ph_term', ansp_ph_term), ('aesp_ph_term', aesp_ph_term), ('acmg_ph_term', acmg_ph_term),
        ('h2mg_ph_term', h2mg_ph_term), ('hoag_ph_term', hoag_ph_term), ('mtox_ph_term', mtox_ph_term),
        ('aero_ph_term', aero_ph_term)
    ])

    c_profile = c + np.zeros(len(depths))
    doc_profile = doc + np.zeros(len(depths))

    # Chemical concentrations

    # Find the measured SOC profile with the most completeness relative to the model grid
    best_date = most_complete_date(site_data['site_comp'], '_socmolm3')

    ca = dict([
        ('ch4', ch4_profile), ('o2', o2_profile), ('co2', co2_profile), ('h2', h2_profile), ('ace', ace_profile),
        ('c', site_data['site_comp'][params.site + '_' + best_date + '_socmolm3_VvZ']), ('doc', doc_profile)
    ])

    #ca['unit'] = 'mol/m3'
    #cr['unit'] = 'mol/m3/day'

    return(ca,cr,crd2)


def define_plants(depths):

    q_plant_ch4_profile = np.zeros(len(depths))
    q_plant_o2_profile = np.zeros(len(depths))
    q_plant_co2_profile = np.zeros(len(depths))
    q_plant_h2_profile = np.zeros(len(depths))
    q_plant_ace_profile = np.zeros(len(depths))

    plant_flux_ch4 = 0.0
    plant_flux_o2 = 0.0
    plant_flux_co2 = 0.0
    plant_flux_h2 = 0.0
    plant_flux_ace = 0.0

    ## Store arrays in dictionaries with readable keys

    # Plant transport rates [n_layers, 1]
    pr = dict([
        ('q_plant_ch4_profile', q_plant_ch4_profile),('q_plant_o2_profile', q_plant_o2_profile),
        ('q_plant_co2_profile', q_plant_co2_profile),('q_plant_h2_profile', q_plant_h2_profile),
        ('q_plant_ace_profile', q_plant_ace_profile)
    ])
    # Plant surface fluxes [1]
    pf = dict([
        ('plant_flux_ch4', plant_flux_ch4),('plant_flux_o2', plant_flux_o2),
        ('plant_flux_co2', plant_flux_co2),('plant_flux_h2', plant_flux_h2),
        ('plant_flux_ace', plant_flux_ace)
    ])

    #pr['unit'] = 'mol/m3/day'
    #pf['unit'] = 'mol/m2/day'

    return(pr,pf)


def define_diffusion(depths):

    q_diff_ch4_profile = np.zeros(len(depths))
    q_diff_o2_profile = np.zeros(len(depths))
    q_diff_co2_profile = np.zeros(len(depths))
    q_diff_h2_profile = np.zeros(len(depths))
    q_diff_ace_profile = np.zeros(len(depths))

    f_diff_ch4 = 0.0
    f_diff_o2= 0.0
    f_diff_co2 = 0.0
    f_diff_h2 = 0.0
    f_diff_ace = 0.0

    ## Store arrays in dictionaries with readable keys

    # Diffusion rates [n_layers,1]
    dr = dict([
        ('q_diff_ch4_profile', q_diff_ch4_profile),('q_diff_o2_profile', q_diff_o2_profile),
        ('q_diff_co2_profile', q_diff_co2_profile),('q_diff_h2_profile', q_diff_h2_profile)
    ])
    # Diffusive surface fluxes [1]
    df = dict([
        ('f_diff_ch4', f_diff_ch4),('f_diff_o2', f_diff_o2),('f_diff_co2', f_diff_co2),('f_diff_h2', f_diff_h2),
        ('f_diff_ace', f_diff_ace)
    ])

    #dr['unit'] = 'mol/m3/day'
    #df['unit'] = 'mol/m2/day'

    return(dr,df)


def define_physical(depths):

    null = np.zeros(len(depths))

    ph = 5.0
    ph_profile = ph + null

    temp = 10.0
    temp_profile = temp + null

    return(ph_profile, temp_profile)