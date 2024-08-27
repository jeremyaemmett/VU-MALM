import numpy as np
import conversions
import csv

file = 'C:/Users/Jeremy/Desktop/VUA/the_files/postprocess.csv'

with open(file, newline='') as csvfile:
    data = list(csv.reader(csvfile))
n_lines = np.shape(data)[0]

print(data[0][19])

labels, ch4_ppms, co2_ppms, V_a = [], [], [], []
for i in range(532, 654):
    labels.append(data[i][8].split()[0])
    ch4_ppms.append(float(data[i][19]))
    co2_ppms.append(float(data[i][20]))
    V_a.append(float(data[i][9]))

ch4_a, ch4_a_0 = 0.0409 * np.array(ch4_ppms) * 16.04, 0.0409 * 1.750 * 16.04  # ug/L
co2_a, co2_a_0 = 0.0409 * np.array(co2_ppms) * 44.01, 0.0409 * 383.0 * 44.01 # ug/L
V_a = np.array(V_a) # exetainer air volume [mL]
V_L = 12.0 - V_a # # exetainer liquid volume [mL]

print(ch4_a_0)
print(ch4_a)
stop

ch4_L_0 = (ch4_a * (V_a + 0.03 * V_L) - ch4_a_0 * V_a) / V_L # ug/mL
# Negative if ch4_a * (V_a + 0.03 * V_L) < ch4_a_0 * V_a
ch4_kg_m3 = 0.001 * ch4_L_0 # kg/m^3

ch4_mol_m3 = conversions.kg2mol(ch4_kg_m3, 'ch4') # mol/m^3


print(ch4_mol_m3)
