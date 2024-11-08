from datetime import datetime, date
import numpy as np

# test 8/21/2024

def molecular_weight(sub):
    # Output: 'mw_g_mol' = molecular weight of a substance (g/mol)
    # Input: 'sub' lower-case substance name e.g. 'co2'
    mw_g_mol = None
    if sub == 'co2':
        mw_g_mol = 44.01
    if sub == 'c':
        mw_g_mol = 12.0107
    if sub == 'o2':
        mw_g_mol = 31.9988
    if sub == 'ch4':
        mw_g_mol = 16.043
    if sub == 'h2':
        mw_g_mol = 2.016
    if sub == 'ace':
        mw_g_mol = 59.044
    return mw_g_mol


def mol2kg(mol, sub):
    # Output: 'kg' kilograms of a substance (kg)
    # Input: 'mol' moles of a substance (mol); 'sub' lower-case substance name e.g. 'co2'
    mw_g_mol = molecular_weight(sub)
    mw_kg_mol = mw_g_mol / 1000.0
    kg = mol * mw_kg_mol
    return kg


def mol2g(mol, sub):
    # Output: 'g' kilograms of a substance (kg)
    # Input: 'mol' moles of a substance (mol); 'sub' lower-case substance name e.g. 'co2'
    mw_g_mol = molecular_weight(sub)
    g = mol * mw_g_mol
    return g

def ug2umol(ug, sub):
    # Output: 'umol' micromoles of a substance (umol)
    # Input: 'ug' micrograms of a substance (ug); 'sub' lower-case substance name e.g. 'co2'
    mw_g_mol = molecular_weight(sub)
    mw_mol_ug = 1.0 / ((1e6) * mw_g_mol)
    mol = ug * mw_mol_ug
    umol = (1e6) * mol
    return umol


def umol2g(umol, sub):
    # Output: 'g' grams of a substance (kg)
    # Input: 'umol' micromoles of a substance (umol); 'sub' lower-case substance name e.g. 'co2'
    mw_g_mol = molecular_weight(sub)
    mw_g_umol = mw_g_mol/(1e6)
    g = umol * mw_g_umol
    return g


def kg2mol(kg, sub):
    # Output: 'mol' moles of a substance (mol)
    # Input: 'kg' kilograms of a substance (kg); 'sub' lower-case substance name e.g. 'co2'
    mw_g_mol = molecular_weight(sub)
    mw_mol_kg = 1000.0 / mw_g_mol
    mol = kg * mw_mol_kg
    return mol


def ppm2molm3(ppm, sub):
    # Output: 'molm3' moles of a gas per cubic meter of air at 1 atm and 25 degC (m3)
    # Input: 'ppm' parts per million of a gas in air at 1 atm and 25 degC (ppm); 'sub' lower-case substance name e.g. 'co2'
    mgm3 = 0.0409 * ppm * molecular_weight(sub)
    gm3 = 0.001 * mgm3
    molm3 = gm3 / molecular_weight(sub)
    return molm3


def chamber_flux(ppm1,ppm2,time1,time2):
    time1 = datetime.strptime(time1, '    %d/%m/%y %H:%M:%S.%f')
    time2 = datetime.strptime(time2, '    %d/%m/%y %H:%M:%S.%f')
    dt = (time2 - time1).total_seconds()
    area = np.pi * 0.15**2.0
    volume = area * 0.40
    air_temp = 273.0
    standard_temp = 273.0
    dcdt = (ppm2-ppm1) / (dt)
    molm2s = dcdt * ((volume * 1.0e-6) / (22.4e-3)) * (1.0 / area)
    molm2d = 86400.0 * molm2s
    return molm2d


def molm3d2kgm3hr(molm3d, sub):
    # Output: 'kgm3hr' reaction rate in (kg/m3/hr)
    # Input: 'molm3d' reaction rate in (mol/m3/d); 'sub' lower-case substance name e.g. 'co2'
    kgm3d = mol2kg(molm3d, sub)
    kgm3hr = kgm3d / 24.0
    return kgm3hr


def kgm3hr2molm3d(kgm3hr, sub):
    # Output: 'molm3d' reaction rate in (mol/m3/d); 'sub' lower-case substance name e.g. 'co2'
    # Input: 'kgm3hr' reaction rate in (kg/m3/hr)
    molm3hr = kg2mol(kgm3hr, sub)
    molm3d = 24.0 * molm3hr
    return molm3d


def molm2d2kgm2d(molm2d, sub):
    # Output: 'kgm2hr' reaction rate in (kg/m2/hr)
    # Input: 'molm2d' reaction rate in (mol/m2/d); 'sub' lower-case substance name e.g. 'co2'
    kgm2d = mol2kg(molm2d, sub)
    return kgm2d


def nmolm2s2mgm2hr(nmolm2s, sub):
    # Output: 'mgm2hr' flux in (mg/m2/hr); 'sub' lower-case substance name e.g. 'co2'
    # Input: 'nmolm2s2' flux in (nano moles/m2/s)
    molm2s = (1.0e-9) * nmolm2s
    kgm2s = mol2kg(molm2s, sub)
    mgm2s = (1.0e6) * kgm2s
    mgm2hr = mgm2s * 60.0 * 60.0
    return mgm2hr


def datetime2doy(date_str, time_str):
    date_split = date_str.split('/')
    time_split = time_str.split(':')
    day = float(date(int(date_split[2]), int(date_split[0]), int(date_split[1])).timetuple().tm_yday)
    time = float(time_split[0])/24.0 + float(time_split[1])/(60.0*24.0)
    day_of_year = str(day+time)
    return day_of_year


def yyyymmddhhmm2doy(yyyymmddhhmm):
    yyyymmdd = float(date(int(yyyymmddhhmm[0:4]), int(yyyymmddhhmm[4:6]), int(yyyymmddhhmm[6:8])).timetuple().tm_yday)
    time = float(yyyymmddhhmm[8:10])/24.0 + float(yyyymmddhhmm[10:12])/(60.0*24.0)
    doy = str(yyyymmdd+time)
    return doy


def ddmmyyyy2doy(ddmmyyyy):
    doy = datetime(int(ddmmyyyy[4:]), int(ddmmyyyy[2:4]), int(ddmmyyyy[0:2])).timetuple().tm_yday
    return(doy)


def activation2q10(activation_energy):
    t0_2 = (-273.15) ** 2.0
    k = 8.617e-5
    q10 = np.exp(activation_energy / (0.1 * t0_2 * k))
    return(q10)


def sort_xylist_by_xlist(xlist, ylist):
    ylist = [x for _, x in sorted(zip(xlist, ylist))]
    xlist = sorted(xlist)
    return(xlist, ylist)

