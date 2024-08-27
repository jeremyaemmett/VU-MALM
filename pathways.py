import numpy as np
import params

# Equations for substrate production, methanogenesis, and CH4 oxidation after Song et al. (2020);
# Based on Michaelis-Menten kinetics with temperature, moisture, acidity, and substrate concentration dependencies;
# Currently neglects CH4 transport by diffusion, plant transport, and ebullition.

def ph_fac(ph):
    # Output: 'f_ph' = soil ph factor ()
    # Input: soil acidity 'ph' (ph)

    ph_min, ph_max, ph_opt = params.ph_min, params.ph_max, params.ph_opt
    f_ph = ((ph - ph_min) * (ph - ph_max)) / ((ph - ph_min) * (ph - ph_max) * ((ph - ph_opt) ** 2.0))
    #print('f_ph: ',f_ph)
    return f_ph


def temp_facs(t, q10):
    # Output: 'f_tX' = soil temperature factors ()
    # Input: temperature 't' (C); temperature sensitivity 'q10' (1/10K)

    t_min1, t_max1, t_opt1 = params.t_min1, params.t_max1, params.t_opt1
    t_min2, t_max2, t_opt2 = params.t_min2, params.t_max2, params.t_max2

    f_t = q10 ** ((t - params.ft_max_temp) / 10.0) # t - 30.0
    #print('f_t: ',f_t)
    f_t[t < 0.0] = 0.0
    f_t[t > params.ft_max_temp] = 1.0 # > 30.0
    f_t1 = ((t - t_min1) * (t - t_max1)) / ((t - t_min1) * (t - t_max1) - ((t - t_opt1) ** 2.0))
    f_t1[t < t_min1] = 0.0
    f_t1[t > t_max1] = 0.0
    f_t2 = ((t - t_min2) * (t - t_max2)) / ((t - t_min2) * (t - t_max2) - ((t - t_opt2) ** 2.0))
    f_t2[t < t_min2] = 0.0
    f_t2[t > t_max2] = 0.0
    #print('temp facts: ',t, f_t1, f_t2, q10, f_t)
    return f_t, f_t1, f_t2


def mm_ref_rate(vmax, q10, subs, ks, t, ph, al, **kwargs):

    ### Michaelis-Menten Kinetics for reactions independent of microbe abundance (no 'micb_term')

    ## Reference terms

    # Max theoretical reaction rate, temperature factor, and pH factor
    vmax_term, temp_term, ph_term = 0.0 * subs[0] + vmax, (temp_facs(t, q10))[0], ph_fac(ph)
    micb_term = kwargs.get('micb', None)

    # Substrate limitation
    chem_term = (subs[0] / (ks[0] + subs[0]))
    if len(subs) > 0: # Product sum of substrate ratios if n_subs > 1
        for sub in range(1,len(subs)):
            chem_term = np.multiply(chem_term,(subs[sub] / (ks[sub] + subs[sub])))

    ## Reference rate is the product of the reference terms. Include microbe density when applicable.

    if micb_term is None:
        mm_rate = vmax * chem_term * temp_term * ph_term
    else:
        mm_rate = vmax * chem_term * temp_term * ph_term * micb_term

    #mm_rate[np.where(t < 0.5)[0]] = 0.0 # Microbes inactive below 0.5 C

    ## Modify terms according to the activity level (0-1) in the layer.

    #mm_rate, vmax_term, chem_term, temp_term, ph_term = \
    #    al * mm_rate, al * vmax_term, al * chem_term, al * temp_term, al * ph_term
    #if micb_term is not None: micb_term = al * micb_term

    return(mm_rate, vmax_term, chem_term, temp_term, ph_term, micb_term)


def mm_decomp(ref_rate, prod_balance, cons_balance):

    # Decompose the reference MM rate into the reaction rates of each reactant and product involved in the reaction.

    prod_rates, cons_rates = {}, {}

    prod_substrate_list, cons_substrate_list, prod_values_list, cons_values_list = \
        list(prod_balance.keys()), list(cons_balance.keys()), list(prod_balance.values()), list(cons_balance.values())

    # Production rates and decomposition
    for s in range(0, len(prod_values_list)):
        prod_rates[prod_substrate_list[s]] = prod_values_list[s] * ref_rate

    # Consumption rates and decomposition
    for s in range(0, len(cons_values_list)):
        cons_rates[cons_substrate_list[s]] = cons_values_list[s] * ref_rate

    return(prod_rates, cons_rates)


def ansp(ca, t, ph, al):

    # {DOC} + H2O -> 6{Ace} + 3CO2 + H2
    cons_balance, prod_balance = {}, dict([('ace', 1.0), ('co2', 0.5), ('h2', 1.0 / 6.0)])

    # MM-Kinetics
    ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
        mm_ref_rate(params.v_doc_prod_ace_max,          # vmax
                     params.ace_prod_q10,               # q10
                     [ca['doc']],                       # substrate species
                     [params.k_doc_prod_ace],           # substrate k factors
                     t, ph, al)                             # temperature and pH

    prod_rates, cons_rates = mm_decomp(ref_rate, prod_balance, cons_balance)

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def aesp(ca, t, ph, al):

    # {DOC} + H2O + O2 -> 2{Ace} + CO2
    cons_balance, prod_balance = dict([('o2', 1.0)]), dict([('ace', 1.0), ('co2', 0.5)])

    # MM-Kinetics
    ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
        mm_ref_rate(params.v_doc_prod_ace_max,                                # vmax
                     params.ace_prod_q10,                                     # q10
                     [ca['doc'], ca['o2']],                                   # substrate species
                     [params.k_doc_prod_ace, params.k_ace_prod_o2],           # substrate k factors
                     t, ph, al)                                                   # temperature and pH

    prod_rates, cons_rates = mm_decomp(ref_rate, prod_balance, cons_balance)

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def hoag(ca, t, ph, ga, al):

    # 4H2 + 2CO2 -> {Ace} + 2H2O
    cons_balance, prod_balance = dict([('co2', 2.0), ('h2', 4.0)]), dict([('ace', 1.0)])

    # MM-Kinetics
    ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
        mm_ref_rate(params.v_h2_prod_ace_max,                                 # vmax
                     params.ace_prod_q10,                                     # q10
                     [ca['co2'], ca['h2']],                                   # substrate species
                     [params.k_co2_prod_ace, params.k_h2_prod_ace],           # substrate k factors
                     t, ph, al,                                                  # temperature and pH
                     micb = ga['homoacetogen_genes'])                         # microbe density

    prod_rates, cons_rates = mm_decomp(ref_rate, prod_balance, cons_balance)

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def h2mg(ca, t, ph, ga, al):

    # CO2 + 4H2 -> CH4 + 2H2O
    cons_balance, prod_balance = dict([('co2', 1.0), ('h2', 4.0)]), dict([('ch4', 1.0)])

    # MM-Kinetics
    ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
        mm_ref_rate(params.v_h2_prod_ch4_max,                                 # vmax
                     params.ch4_prod_q10,                                     # q10
                     [ca['co2'], ca['h2']],                                   # substrate species
                     [params.k_co2_prod_ch4, params.k_h2_prod_ch4],           # substrate k factors
                     t, ph, al,                                                  # temperature and pH
                     micb = ga['h2_methanogen_genes'])                        # microbe density
    #ref_rate = 0.0 * ref_rate
    prod_rates, cons_rates = mm_decomp(ref_rate, prod_balance, cons_balance)

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def acmg(ca, t, ph, ga, al):

    # {Ace} -> CH4 + CO2
    cons_balance, prod_balance = dict([('ace', 1.0)]), dict([('ch4', 1.0), ('co2', 1.0)])

    # MM-Kinetics
    ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
        mm_ref_rate(params.v_ace_cons_max,                                    # vmax
                     params.ch4_prod_q10,                                     # q10
                     [ca['ace']],                                             # substrate species
                     [params.k_ace_prod_ch4],                                 # substrate k factors
                     t, ph, al,                                                  # temperature and pH
                     micb = ga['acemethanogen_genes'])                        # microbe density
    #ref_rate = 0.0 * ref_rate
    prod_rates, cons_rates = mm_decomp(ref_rate, prod_balance, cons_balance)

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def mtox(ca, t, ph, ga, al):

    # CH4 + 2O2 -> CO2 + 2H2O
    cons_balance, prod_balance = dict([('ch4', 1.0), ('o2', 2.0)]), dict([('co2', 1.0)])

    # MM-Kinetics
    ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
        mm_ref_rate(params.v_ch4_oxid_max,                                    # vmax
                     params.ch4_oxid_q10,                                     # q10
                     [ca['ch4'], ca['o2']],                                   # substrate species
                     [params.k_ch4_oxid_ch4, params.k_ch4_oxid_o2],           # substrate k factors
                     t, ph, al,                                                   # temperature and pH
                     micb = ga['methanotroph_genes'])                         # microbe density

    prod_rates, cons_rates = mm_decomp(ref_rate, prod_balance, cons_balance)

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


def aero(ca, t, ph, al):

    # DOC + O2 -> 'aerobic microbe carbon'
    cons_balance, prod_balance = dict([('doc', 1.0), ('o2', 2.0)]), {}

    # MM-Kinetics
    ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = \
        mm_ref_rate(params.k_aer,                                             # vmax
                     params.doc_prod_q10,                                     # q10
                     [ca['doc'], ca['o2']],                                   # substrate species
                     [params.k_aer_doc, params.k_aer_o2],                     # substrate k factors
                     t, ph, al)                                                   # temperature and pH

    prod_rates, cons_rates = mm_decomp(ref_rate, prod_balance, cons_balance)

    return(prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb)


    #o2_cons2 = k_aer * (doc / (k_aer_doc + doc)) * (o2 / (k_aer_o2 + o2)) * (temp_facs(t, doc_prod_q10))[0] * ph_fac(ph)




