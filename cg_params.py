import numpy as np
import matplotlib.pyplot as plt
import csv

def microbe_soc():

    soc_filename = 'C:/Users/Jeremy/Desktop/Data/churchill_data/SOC/Palsa/High/Aug_2022/SOC_palsa_top_aug_2022.txt'
    density_filename = 'C:/Users/Jeremy/Desktop/Data/churchill_data/dry_density/Palsa/High/Aug_2022/dry_density_palsa_top_aug_2022.txt'

    depth1 = np.flip(np.array(read_csv(soc_filename, 0, 'float')))
    before = np.array(read_csv(soc_filename, 1, 'float'))
    after = np.array(read_csv(soc_filename, 2, 'float'))
    foil = np.array(read_csv(soc_filename, 3, 'float'))
    depth2 = np.flip(np.array(read_csv(density_filename, 0, 'float')))
    wet = np.flip(np.array(read_csv(density_filename, 1, 'float')))
    tray = np.flip(np.array(read_csv(density_filename, 2, 'float')))
    dry = np.flip(np.array(read_csv(density_filename, 3, 'float')))
    diameter = np.flip(np.array(read_csv(density_filename, 4, 'float')))
    length = np.flip(np.array(read_csv(density_filename, 5, 'float')))
    volume = length * np.pi * (diameter / 2.0) ** 2

    depths = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,
              0.525,0.55,0.575,0.6]

    new_grid, before = interp2grid(depth1, before, np.array(depths))
    new_grid, after = interp2grid(depth1, after, np.array(depths))
    new_grid, foil = interp2grid(depth1, foil, np.array(depths))
    new_grid, wet = interp2grid(depth2, wet, np.array(depths))
    new_grid, dry = interp2grid(depth2, dry, np.array(depths))
    new_grid, tray = interp2grid(depth2, tray, np.array(depths))
    new_grid, volume = interp2grid(depth2, volume, np.array(depths))

    before = before - foil
    after = after - foil
    wet = wet - tray
    dry = dry - tray

    soc_dry_g_g = (before - after) / before  # g C/g dry soil
    min_dry_g_g = after / before

    v_ratio = (min_dry_g_g * 1.4) / (soc_dry_g_g * 2.6)
    v_frac_min = v_ratio / (v_ratio + 1.0)
    v_frac_soc = 1.0 - v_frac_min

    rho_soil = dry / volume  # g-dry / cm3

    porosity = 1.0 - (rho_soil / 2.6)

    gwc = (wet - dry) / dry  # g/g
    vwc = gwc * (rho_soil / 1.0)

    soc_g_cm3 = soc_dry_g_g * rho_soil

    return(soc_g_cm3)

def read_csv(filename, column_idx, var_type):
    with open(filename) as f:
        reader = csv.reader(f)
        header1 = next(reader)
#        for index, column_header in enumerate(header1):
#            print(index, column_header)
        vals = []
        for row in reader:
            if var_type == 'string':
                val = row[column_idx]
            if var_type == 'integer':
                val = int(row[column_idx])
            if var_type == 'float':
                if row[column_idx] == '':
                    val = -9999.0
                else:
                    val = float(row[column_idx])
                    #print(val)
            vals.append(val)
    return vals


def interp2grid(depths, values, newgrid):
    x = depths
    y = values

    xnew = newgrid
    ynew = np.interp(xnew, x, y)

    return (xnew, ynew)

soc_filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/SOC/SOC_palsa_low_aug_2022.txt'
density_filename = 'C:/Users/Jeremy/Desktop/Churchill_Data/siteData/dry_density/dry_density_palsa_low_aug_2022.txt'

depth1 = np.flip(np.array(read_csv(soc_filename, 0, 'float')))
before = np.array(read_csv(soc_filename, 1, 'float'))
after = np.array(read_csv(soc_filename, 2, 'float'))
foil = np.array(read_csv(soc_filename, 3, 'float'))
depth2 = np.flip(np.array(read_csv(density_filename, 0, 'float')))
wet = np.flip(np.array(read_csv(density_filename, 1, 'float')))
tray = np.flip(np.array(read_csv(density_filename, 2, 'float')))
dry = np.flip(np.array(read_csv(density_filename, 3, 'float')))
diameter = np.flip(np.array(read_csv(density_filename, 4, 'float')))
length = np.flip(np.array(read_csv(density_filename, 5, 'float')))
volume = length*np.pi*(diameter/2.0)**2

new_grid, before = interp2grid(depth1,before,np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6]))
new_grid, after = interp2grid(depth1,after,np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6]))
new_grid, foil = interp2grid(depth1,foil,np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6]))
new_grid, wet = interp2grid(depth2,wet,np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6]))
new_grid, dry = interp2grid(depth2,dry,np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6]))
new_grid, tray = interp2grid(depth2,tray,np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6]))
new_grid, volume = interp2grid(depth2,volume,np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6]))

before = before - foil
after = after - foil
wet = wet - tray
dry = dry - tray

soc_dry_g_g = (before - after) / before # g C/g dry soil
min_dry_g_g = after / before

v_ratio = (min_dry_g_g * 1.4) / (soc_dry_g_g * 2.6)
v_frac_min = v_ratio / (v_ratio + 1.0)
v_frac_soc = 1.0 - v_frac_min

rho_soil = dry / volume # g-dry / cm3

porosity = 1.0 - (rho_soil / 2.6)

gwc = (wet - dry) / dry # g/g
vwc = gwc * (rho_soil / 1.0)

soc_g_cm3 = soc_dry_g_g * rho_soil
min_g_cm3 = min_dry_g_g * rho_soil

v_solid = (1.0 - porosity)
v_min = np.round(v_frac_min * v_solid,3)
v_soc = np.round(v_frac_soc * v_solid,3)
v_ice = np.round(vwc,3)

#test = microbe_soc(new_grid,soc_g_cm3)

print(v_soc + v_min + porosity)

print(' ')

print('Depths [m]: ', new_grid)
print('waterIce: ', 0.90*v_ice)
print('mineral: ', v_min)
print('organic: ', v_soc)
print('sat waterIce: ', 1.0 - (v_min+v_soc))
