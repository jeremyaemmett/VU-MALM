import pathways
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ternary import TernaryDiagram

fig, ax = plt.subplots()

ax.clear()

d = {'temp': [1, 2], 'chem': [3, 4], 'micb': [5, 6]}
df2 = pd.DataFrame(data=d)

min, max, nvals = 0.0, 1.0, 50
int = (max-min)/(nvals+1)
temp_vals, chem_vals, micb_vals = list(np.arange(min,max,int)), list(np.arange(min,max,int)), list(np.arange(min,max,int))
#temp_vals, chem_vals, micb_vals = list(range(51)), list(range(51)), list(range(51))
temp_vals_master = []
for i in range(0,nvals):
    nreps = nvals + 1 - i
    for r in range(0,nreps):
        #print(i)
        temp_vals_master.append(temp_vals[i])
temp_vals_master.append(temp_vals[-1])

chem_vals_master = []
for i in range(1,nvals):
    nreps = nvals + 1 - i
    for r in range(0,nreps):
        #print(range(nreps)[r])
        chem_vals_master.append(chem_vals[range(nreps)[r]])
chem_vals_master = chem_vals + chem_vals_master
chem_vals_master.append(chem_vals[0])

micb_vals_master = []
for i in range(1,nvals+1):
    nreps = nvals + 1 - i
    for r in range(0,nreps+1):
        micb_vals_master.append(micb_vals[nreps - r])

micb_vals_master.append(micb_vals[0])

z = []
ca = {'ace': 0.0}
ga = {'acemethanogen_genes': 0.0}
temp = np.array([0.0])
ph = np.array([7.0])
for i in range(0,len(temp_vals_master)):
    # Acetoclastic methanogenesis (ACMG)
    ca['ace'], temp[0], ph[0], ga['acemethanogen_genes'] = chem_vals_master[i], temp_vals_master[i], 7.0, micb_vals_master[i]
    prod_rates_acmg, cons_rates_acmg, ref_rate_acmg, ref_vmax_acmg, ref_chem_acmg, ref_temp_acmg, ref_ph_acmg, ref_micb_acmg = pathways.acmg(ca, temp, ph, ga)
    z.append(ref_rate_acmg[0])

d = {'temp': temp_vals_master, 'chem': chem_vals_master, 'micb': micb_vals_master, 'z': z}
df = pd.DataFrame(data=d)

#df = pd.read_csv("example_contour.csv")

df.head()

materials = df.columns[0:3].tolist()

materials = ['temp', 'chem', 'micb']

td = TernaryDiagram(materials=materials)
mappable = td.contour(df[materials], z=df["z"], flag_cbar = False, z_min = 0.0, z_max = 0.04)
td.colorbar(mappable=mappable, label="Rate [mol/m3/day]")

fig.savefig("C:/Users/Jeremy/Desktop/ternary/figure.png", dpi=144)
