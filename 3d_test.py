from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyrolite.util.plot.style import ternary_color
from matplotlib.collections import PatchCollection
from pyrolite.util.synthetic import normal_frame
from matplotlib.ticker import LogFormatter
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib import ticker, cm
from os.path import isfile, join
from data import read_site_data
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from matplotlib import cm
from datetime import date
import matplotlib as mpl
from os import listdir
from copy import copy
from numpy import ma
import pandas as pd
import numpy as np
import transport
import diffusion
import pathways
import params
import math
import init
import csv
import os

ph = 6.0
ga= {}
ga['h2_methanogen_genes'] = 1.0
ga['acemethanogen_genes'] = 1.0
ga['homoacetogen_genes'] = 1.0
ga['methanotroph_genes'] = 1.0
al = 1.0
lvls = np.logspace(-2, 1, 100)

fig, ax = plt.subplots()

ax = fig.add_subplot(111,projection='3d')
ax.axes.set_xlim3d(left=0, right=1)
ax.axes.set_ylim3d(bottom=0, top=1)
ax.axes.set_zlim3d(bottom=5, top=25)

nx = 101
ny = 101

# Set up survey vectors
xvec = np.linspace(0.001, 1.0, nx)
yvec = np.linspace(0.001, 1.0, ny)
x1, x2 = np.meshgrid(xvec, yvec)
ca = {}
ca['ch4'] = x1
ca['o2'] = x2
for temp in range(5,26,5):
    t = 0.0 * x1 + temp
    #prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = pathways.ansp(ca, t, ph, ga, al)
    prod_rates, cons_rates, ref_rate, ref_vmax, ref_chem, ref_temp, ref_ph, ref_micb = pathways.mtox(ca, t, ph, ga, al)
    ax.contourf(x1, x2, ref_rate, zdir='z', offset=temp, cmap='rainbow', alpha=0.5, levels=lvls, antialiased=True)

plt.show()