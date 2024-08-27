import params
import pathways
import numpy as np
import conversions

def methanogen_temp2mort(temp):

    t0 = -273.15
    q2 = 4.3
    kd = 0.003
    a2 = q2**( (0.1*temp) / (1.0 - (temp/t0) ) )
    mort = kd * a2
    mort = 0.0 * temp + 0.06 # [1/day]

    return(mort)

def h2_methanogen_temp2mort(temp):

    t0 = -273.15
    q2 = 4.3
    kd = 0.00007
    a2 = q2**( (0.1*temp) / (1.0 - (temp/t0) ) )
    mort = kd * a2
    mort = 0.0 * temp + 0.12

    return(mort)

def methanotroph_temp2mort(temp):

    t0 = -273.15
    q2 = 0.3
    kd = 0.0007
    a2 = q2**( (0.1*temp) / (1.0 - (temp/t0) ) )
    mort = kd * a2
    mort = 0.0*temp + 0.06

    return(mort)


def population(rate,dt,mort,mics,gc_per_copy,grow,ft):

    gc_per_molc = 12.01 # g/mol conversion factor for carbon

    if params.genomic_flag == True:

        mic_prod = grow * rate * (1.0 / gc_per_copy) * gc_per_molc # [] x [mol_MBC/m^3/day] x [copies/g_MBC] x [g_MBC/mol_MBC] = [copies/m^3/day]
        mic_mort = mort * mics * ft # [1/day] * [copies/m^3] * [] = [copies/m^3/day]

    else:

        mic_prod = grow * rate # [] x [mol_MBC/m^3/day] = [mol_MBC/m^3/day]
        mic_mort = mort * mics * ft # [1/day] x [mol_MBC/m^3] = [mol_MBC/m^3/day]

    mic_rate = mic_prod - mic_mort # [copies/m^3/day] or [mol_MBC/m^3/day]
    mics = mics + dt * mic_rate # [copies/m^3] or [mol_MBC/m^3]

    return(mics,mic_rate,mic_prod,mic_mort)


def aerobic_factor(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb, forcingt):

    # Aerobic factor (0 = anoxic, 1 = oxic)
    aef = 1.0 - 0.01 * forcingt['satpct_data']

    #aef[forcingt['satpct_data'] == 100.0] = 0.0

    ## Reduce rates and decomposed terms by the anaerobic factor

    # Production rates and terms
    for sub in list((prod_rates.keys())): prod_rates[sub] = aef * prod_rates[sub]

    # Consumption rates and terms
    for sub in list((cons_rates.keys())): cons_rates[sub] = aef * cons_rates[sub]

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def anaerobic_factor(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb, forcingt):

    # Anaerobic factor (0 = oxic, 1 = anoxic)
    anf = 0.01 * forcingt['satpct_data']

    anf[forcingt['satpct_data'] < params.saturation_threshold] = 0.0

    #print('sat: ', forcingt['satpct_data'])
    #print('anf: ', anf)

    #print('sat: ', forcingt['satpct_data'])

    ## Reduce rates and decomposed terms by the anaerobic factor

    # Production rates and terms
    for sub in list((prod_rates.keys())): prod_rates[sub] = anf * prod_rates[sub]

    # Consumption rates and terms
    for sub in list((cons_rates.keys())): cons_rates[sub] = anf * cons_rates[sub]

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def microbes2(dt, forcingt, lthick, ga, ph, cr, gr, ca, crd2):

    al = crd2['activity_level']

    # Unpack relevant forcing profiles
    temp = forcingt['temp_data']

    gc_per_copy = 1e-13 # g_MBC/gene_copy ('c' in Louca)

    # Calculate temperature-dependent mortality rate for each population
    methanogen_mort = methanogen_temp2mort(temp)
    h2_methanogen_mort = h2_methanogen_temp2mort(temp)
    methanotroph_mort = methanotroph_temp2mort(temp)

    # No mortality if the layer is frozen
    methanogen_mort[temp < 0.0] = 0.0
    h2_methanogen_mort[temp < 0.0] = 0.0
    methanotroph_mort[temp < 0.0] = 0.0

    # Anaerobic substrate production (ANSP)
    prod_rates_ansp, cons_rates_ansp, ref_rate_ansp, ref_vmax_ansp, ref_chem_ansp, ref_temp_ansp, ref_ph_ansp, ref_micb_ansp = pathways.ansp(ca, temp, ph, al)
    #
    if params.sataero_flag:
        prod_rates_ansp, cons_rates_ansp, ref_rate_ansp, ref_vmax_ansp, ref_chem_ansp, ref_temp_ansp, ref_ph_ansp, ref_micb_ansp = \
        anaerobic_factor(prod_rates_ansp, cons_rates_ansp, ref_rate_ansp, ref_vmax_ansp, ref_chem_ansp, ref_temp_ansp, ref_ph_ansp, ref_micb_ansp, forcingt)

    # Aerobic substrate production (AESP)
    prod_rates_aesp, cons_rates_aesp, ref_rate_aesp, ref_vmax_aesp, ref_chem_aesp, ref_temp_aesp, ref_ph_aesp, ref_micb_aesp = pathways.aesp(ca, temp, ph, al)
    #
    if params.sataero_flag:
        prod_rates_aesp, cons_rates_aesp, ref_rate_aesp, ref_vmax_aesp, ref_chem_aesp, ref_temp_aesp, ref_ph_aesp, ref_micb_aesp = \
        aerobic_factor(prod_rates_aesp, cons_rates_aesp, ref_rate_aesp, ref_vmax_aesp, ref_chem_aesp, ref_temp_aesp, ref_ph_aesp, ref_micb_aesp, forcingt)

    # Homoacetogenesis (HOAG)
    prod_rates_hoag, cons_rates_hoag, ref_rate_hoag, ref_vmax_hoag, ref_chem_hoag, ref_temp_hoag, ref_ph_hoag, ref_micb_hoag = pathways.hoag(ca, temp, ph, ga, al)

    # Hydrogenotrophic methanogenesis (H2MG)
    prod_rates_h2mg, cons_rates_h2mg, ref_rate_h2mg, ref_vmax_h2mg, ref_chem_h2mg, ref_temp_h2mg, ref_ph_h2mg, ref_micb_h2mg = pathways.h2mg(ca, temp, ph, ga, al)
    if params.sataero_flag:
        prod_rates_h2mg, cons_rates_h2mg, ref_rate_h2mg, ref_vmax_h2mg, ref_chem_h2mg, ref_temp_h2mg, ref_ph_h2mg, ref_micb_h2mg = \
        anaerobic_factor(prod_rates_h2mg, cons_rates_h2mg, ref_rate_h2mg, ref_vmax_h2mg, ref_chem_h2mg, ref_temp_h2mg, ref_ph_h2mg, ref_micb_h2mg, forcingt)

    # Acetoclastic methanogenesis (ACMG)
    prod_rates_acmg, cons_rates_acmg, ref_rate_acmg, ref_vmax_acmg, ref_chem_acmg, ref_temp_acmg, ref_ph_acmg, ref_micb_acmg = pathways.acmg(ca, temp, ph, ga, al)
    #print('test1: ', prod_rates_acmg['ch4'][0:3])
    #print(prod_rates_acmg['ch4'])
    if params.sataero_flag:
        prod_rates_acmg, cons_rates_acmg, ref_rate_acmg, ref_vmax_acmg, ref_chem_acmg, ref_temp_acmg, ref_ph_acmg, ref_micb_acmg = \
        anaerobic_factor(prod_rates_acmg, cons_rates_acmg, ref_rate_acmg, ref_vmax_acmg, ref_chem_acmg, ref_temp_acmg, ref_ph_acmg, ref_micb_acmg, forcingt)
    #print('test2: ', prod_rates_acmg['ch4'][0:3])
    #print('anaerobic applied')
    #print(np.max(abs(prod_rates_acmg['ch4'])), np.max(abs(prod_rates_h2mg['ch4'])))

    # Aerobic methanotrophy (MTOX)
    prod_rates_mtox, cons_rates_mtox, ref_rate_mtox, ref_vmax_mtox, ref_chem_mtox, ref_temp_mtox, ref_ph_mtox, ref_micb_mtox = pathways.mtox(ca, temp, ph, ga, al)
    #if params.sataero_flag:
    #    prod_rates_mtox, cons_rates_mtox, ref_rate_mtox, ref_vmax_mtox, ref_chem_mtox, ref_temp_mtox, ref_ph_mtox, ref_micb_mtox = \
    #    aerobic_factor(prod_rates_mtox, cons_rates_mtox, ref_rate_mtox, ref_vmax_mtox, ref_chem_mtox, ref_temp_mtox, ref_ph_mtox, ref_micb_mtox, forcingt)

    # Aerobic respiration (AERO)
    prod_rates_aero, cons_rates_aero, ref_rate_aero, ref_vmax_aero, ref_chem_aero, ref_temp_aero, ref_ph_aero, ref_micb_aero = pathways.aero(ca, temp, ph, al)

    ### Record chemistry rates

    # CH4
    cr['ch4_c_prod_acmg'], cr['ch4_c_prod_h2mg'] = np.copy(prod_rates_acmg['ch4']), np.copy(prod_rates_h2mg['ch4'])
    cr['ch4_c_prod'] = cr['ch4_c_prod_acmg'] + cr['ch4_c_prod_h2mg'] # Total production
    cr['ch4_c_cons_mtox'] = np.copy(cons_rates_mtox['ch4'])
    cr['ch4_c_cons'] = np.copy(cr['ch4_c_cons_mtox']) # Total consumption
    cr['ch4_c_rate'] = cr['ch4_c_prod'] - cr['ch4_c_cons'] # Net

    # Acetate
    cr['ace_c_prod_aesp'], cr['ace_c_prod_ansp'], cr['ace_c_prod_hoag'] = \
        np.copy(prod_rates_aesp['ace']), np.copy(prod_rates_ansp['ace']), np.copy(prod_rates_hoag['ace'])
    # Total production
    cr['ace_c_prod'] = cr['ace_c_prod_aesp'] + cr['ace_c_prod_ansp'] + cr['ace_c_prod_hoag']
    cr['ace_c_cons_acmg'] = np.copy(cons_rates_acmg['ace'])
    cr['ace_c_cons'] = np.copy(cr['ace_c_cons_acmg']) # Total consumption
    cr['ace_c_rate'] = cr['ace_c_prod'] - cr['ace_c_cons'] # Net

    # CO2
    cr['co2_c_prod_ansp'], cr['co2_c_prod_aesp'], cr['co2_c_prod_acmg'], cr['co2_c_prod_mtox'] = \
        np.copy(prod_rates_ansp['co2']), np.copy(prod_rates_aesp['co2']), \
        np.copy(prod_rates_acmg['co2']), np.copy(prod_rates_mtox['co2'])
    # Total production
    cr['co2_c_prod'] = cr['co2_c_prod_ansp'] + cr['co2_c_prod_aesp'] + cr['co2_c_prod_acmg'] + cr['co2_c_prod_mtox']
    cr['co2_c_cons_hoag'], cr['co2_c_cons_h2mg'] = np.copy(cons_rates_hoag['co2']), np.copy(cons_rates_h2mg['co2'])
    cr['co2_c_cons'] = cr['co2_c_cons_hoag'] + cr['co2_c_cons_h2mg'] # Total consumption
    cr['co2_c_rate'] = cr['co2_c_prod'] - cr['co2_c_cons'] # Net

    # H2
    cr['h2_c_prod_ansp'] = np.copy(prod_rates_ansp['h2'])
    cr['h2_c_prod'] = np.copy(cr['h2_c_prod_ansp']) # Total production
    cr['h2_c_cons_hoag'], cr['h2_c_cons_h2mg'] = np.copy(cons_rates_hoag['h2']), np.copy(cons_rates_h2mg['h2'])
    cr['h2_c_cons'] = cr['h2_c_cons_hoag'] + cr['h2_c_cons_h2mg'] # Total consumption
    cr['h2_c_rate'] = cr['h2_c_prod'] - cr['h2_c_cons'] # Net

    # O2
    cr['o2_c_prod_none'] = 0.0*np.copy(cons_rates_mtox['o2'])
    cr['o2_c_prod'] = np.copy(cr['o2_c_prod_none']) # Total production
    cr['o2_c_cons_mtox'], cr['o2_c_cons_aero'], cr['o2_c_cons_aesp'] = np.copy(cons_rates_mtox['o2']), np.copy(cons_rates_aero['o2']), np.copy(cons_rates_aesp['o2'])
    cr['o2_c_cons'] = cr['o2_c_cons_mtox'] + cr['o2_c_cons_aero'] # + o2_c_cons_aesp? # Total consumption
    cr['o2_c_rate'] = cr['o2_c_prod'] - cr['o2_c_cons'] # Net

    ## Decomposition 2 (Michaelis-Menten terms)

    # Activity level
    crd2['activity_level'] = al

    # ref rate
    crd2['ansp_ref_rate'], crd2['aesp_ref_rate'], crd2['acmg_ref_rate'], crd2['h2mg_ref_rate'], \
    crd2['hoag_ref_rate'], crd2['mtox_ref_rate'], crd2['aero_ref_rate'] = \
    ref_rate_ansp, ref_rate_aesp, ref_rate_acmg, ref_rate_h2mg, ref_rate_hoag, ref_rate_mtox, ref_rate_aero

    # vmax
    crd2['ansp_vmax_term'], crd2['aesp_vmax_term'], crd2['acmg_vmax_term'], crd2['h2mg_vmax_term'], \
    crd2['hoag_vmax_term'], crd2['mtox_vmax_term'], crd2['aero_vmax_term'] = \
    ref_vmax_ansp, ref_vmax_aesp, ref_vmax_acmg, ref_vmax_h2mg, ref_vmax_hoag, ref_vmax_mtox, ref_vmax_aero

    # chem
    crd2['ansp_chem_term'], crd2['aesp_chem_term'], crd2['acmg_chem_term'], crd2['h2mg_chem_term'], \
    crd2['hoag_chem_term'], crd2['mtox_chem_term'], crd2['aero_chem_term'] = \
    ref_chem_ansp, ref_chem_aesp, ref_chem_acmg, ref_chem_h2mg, ref_chem_hoag, ref_chem_mtox, ref_chem_aero

    # temp
    crd2['ansp_temp_term'], crd2['aesp_temp_term'], crd2['acmg_temp_term'], crd2['h2mg_temp_term'], \
    crd2['hoag_temp_term'], crd2['mtox_temp_term'], crd2['aero_temp_term'] = \
    ref_temp_ansp, ref_temp_aesp, ref_temp_acmg, ref_temp_h2mg, ref_temp_hoag, ref_temp_mtox, ref_temp_aero

    # ph
    crd2['ansp_ph_term'], crd2['aesp_ph_term'], crd2['acmg_ph_term'], crd2['h2mg_ph_term'], \
    crd2['hoag_ph_term'], crd2['mtox_ph_term'], crd2['aero_ph_term'] = \
    ref_ph_ansp, ref_ph_aesp, ref_ph_acmg, ref_ph_h2mg, ref_ph_hoag, ref_ph_mtox, ref_ph_aero

    # micb
    crd2['ansp_micb_term'], crd2['aesp_micb_term'], crd2['acmg_micb_term'], crd2['h2mg_micb_term'], \
    crd2['hoag_micb_term'], crd2['mtox_micb_term'], crd2['aero_micb_term'], = \
    0.0*ga['acemethanogen_genes'], 0.0*ga['acemethanogen_genes'], ga['acemethanogen_genes'], ga['h2_methanogen_genes'], \
    ga['homoacetogen_genes'], ga['methanotroph_genes'], 0.0*ga['acemethanogen_genes']

    ## Update the microbe populations and record their growth and mortality rates

    #print(temp)
    # Methanotrophs
    ga['methanotroph_genes'], gr['methanotroph_gene_rate'], gr['methanotroph_gene_prod'], gr['methanotroph_gene_mort'] = \
        population(cons_rates_mtox['o2'],dt,params.dead_methanotrophs, ga['methanotroph_genes'],gc_per_copy, params.grow_methanotrophs, (pathways.temp_facs(temp, params.ch4_prod_q10))[0])
    #print((pathways.temp_facs(temp, params.ch4_prod_q10))[0])
    # Acetoclastic methanogens
    ga['acemethanogen_genes'], gr['acemethanogen_gene_rate'], gr['acemethanogen_gene_prod'], gr['acemethanogen_gene_mort'] = \
        population(cons_rates_acmg['ace'], dt, params.dead_acemethanogens, ga['acemethanogen_genes'], gc_per_copy, params.grow_acemethanogens, (pathways.temp_facs(temp, params.ch4_prod_q10))[0])
    #print((pathways.temp_facs(temp, params.ch4_prod_q10))[0])
    # Hydrogenotrophic methanogens
    ga['h2_methanogen_genes'], gr['h2_methanogen_gene_rate'], gr['h2_methanogen_gene_prod'], gr['h2_methanogen_gene_mort'] = \
        population(prod_rates_h2mg['ch4'], dt, params.dead_h2methanogens, ga['h2_methanogen_genes'], gc_per_copy, params.grow_h2methanogens ,(pathways.temp_facs(temp, params.ch4_prod_q10))[2])
    #print((pathways.temp_facs(temp, params.ch4_prod_q10))[2])

    # Homoacetogens
    ga['homoacetogen_genes'], gr['homoacetogen_gene_rate'], gr['homoacetogen_gene_prod'], gr['homoacetogen_gene_mort'] = \
        population(prod_rates_hoag['ace'], dt, params.dead_homoacetogens, ga['homoacetogen_genes'], gc_per_copy, params.grow_homoacetogens, (pathways.temp_facs(temp, params.ch4_prod_q10))[1])

    ga['totmethanogen_genes'] = ga['acemethanogen_genes'] + ga['h2_methanogen_genes']
    #print((pathways.temp_facs(temp, params.ch4_prod_q10))[1])

    #stop

    return(ga, gr, cr, crd2)