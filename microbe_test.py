import params
import pathways
import numpy as np
import conversions
import microbes
import pathways

temp = np.array([7.908,7.908])
methanotrophs = np.array([1e11,1e11])
methanotroph_genes = np.array([1e11,1e11])
ch4 = np.array([0.00015,0.00015])
o2 = np.array([19.0,19.0])
ph = np.array([5.0,5.0])
doc = np.array([1.0,1.0])
dt = 0.25
methanotroph_mort = 0.06
gc_per_copy = 1e-13 # g_c/copy

mtox_rate = pathways.methanotrophic_oxidation(v_ch4_oxid_max=params.v_ch4_oxid_max, methanotrophs=methanotrophs,
                                              ch4=ch4, k_ch4_oxid_ch4=params.k_ch4_oxid_ch4, o2=o2,
                                              k_ch4_oxid_o2=params.k_ch4_oxid_o2, ch4_oxid_q10=params.ch4_oxid_q10, t=temp,
                                              ph=ph, doc=doc, k_aer=params.k_aer, k_aer_doc=params.k_aer_doc,
                                              k_aer_o2=params.k_aer_o2, doc_prod_q10=params.doc_prod_q10)
print('roxidch4 total: ',mtox_rate[0])
methanotroph_genes, methanotroph_gene_rate, methanotroph_gene_prod, methanotroph_gene_mort, methanotrophs = microbes.population(mtox_rate[0],dt,methanotroph_mort,methanotroph_genes,gc_per_copy,0.4,(pathways.temp_facs(temp, params.ch4_prod_q10))[0])