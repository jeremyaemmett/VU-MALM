import numpy as np
from PIL import Image
from matplotlib.collections import PolyCollection
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from datetime import timedelta
from datetime import datetime
from datetime import date
from matplotlib import cm
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema
import cv2 as cv
import math
import init
import os
import csv
import params
import data
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib as mpl
from osgeo import gdal, ogr
import rasterio
from rasterio.plot import show
from rasterio.enums import ColorInterp
import cv2
from matplotlib.cbook import get_sample_data
import imageio as iio


def profile_plot(axis, box_aspect, bg_resize, bg_zoom, bg_image_path, data, depths, lthick, cut_idx):

    axis.set_box_aspect(box_aspect)
    axis.set_ylim((-depths[-1], 0.0))
    axis.set_xlabel('[kg/m3]')
    format_axis(axis)
    axis.get_yaxis().set_visible(False)
    logo = Image.open(bg_image_path)
    logo = logo.resize(bg_resize)
    imagebox = OffsetImage(logo, zoom=bg_zoom)
    if np.sum(data) > 0.0: axis.set_xlim([0.0, np.max(data[0:cut_idx])])
    axis.fill_betweenx(-depths[0:cut_idx] + lthick / 2, 0.0 * data[0:cut_idx], data[0:cut_idx], color='white', alpha=0.2, step='mid')
    xctr, yctr = np.mean(axis.get_xlim()), np.mean(axis.get_ylim())
    ab = AnnotationBbox(imagebox, (xctr, yctr), frameon=False, zorder=0, alpha=0.5)
    if np.sum(data) > 0: axis.add_artist(ab)
    for i in range(0, len(depths[0:cut_idx])):
        axis.plot([0.0, np.max(data)], [-depths[i], -depths[i]], linestyle='-', color='white', linewidth=2.0)
    axis.fill_betweenx(-depths[0:cut_idx] + lthick / 2, data[0:cut_idx],
                      np.max(rootkgm3) + 0.0 * data[0:cut_idx], color='white', alpha=1.0, step='mid')
    axis.fill_betweenx(-depths[0:cut_idx] + lthick / 2, 0.0 * data[0:cut_idx], data[0:cut_idx], color='white', alpha=0.2, step='mid')
    if np.sum(data) == 0.0:
        axis.fill_between([-0.5, 0.5, 0.5, -0.5, -0.5], [-depths[cut_idx-1], -depths[cut_idx-1], 0.0, 0.0, -depths[cut_idx-1]], facecolor="none", hatch = '//', edgecolor = 'grey')
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    plt.xticks(rotation=30.0)

    return()


def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0.1, 0.1 + height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid


def interpolate(p_from, p_to, num):
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    distance = np.linalg.norm(p_to - p_from) / (num - 1)

    ret_vec = []

    for i in range(0, num):
        ret_vec.append(p_from + direction * distance * i)

    return np.array(ret_vec)


"""
    plot image (plane) in 3D with given Pose (R|t) of corner point

    ax      : matplotlib axes to plot on
    R       : Rotation as roation matrix
    t       : translation as np.array (1, 3), left down corner of image in real world coord
    size    : Size as np.array (1, 2), size of image plane in real world
    img_scale: Scale to bring down image, since this solution needs 1 face for every pixel it will become very slow on big images 
"""

def plotImage(ax, img, R, t, size=np.array((1, 1)), img_scale=8):
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))

    corners = np.array(([0., 0, 0], [0, size[0], 0],
                        [size[1], 0, 0], [size[1], size[0], 0]))

    corners += t
    corners = corners @ R
    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    l1 = interpolate(corners[0], corners[2], img_size[0])
    xx[:, 0] = l1[:, 0]
    yy[:, 0] = l1[:, 1]
    zz[:, 0] = l1[:, 2]
    l1 = interpolate(corners[1], corners[3], img_size[0])
    xx[:, img_size[1] - 1] = l1[:, 0]
    yy[:, img_size[1] - 1] = l1[:, 1]
    zz[:, img_size[1] - 1] = l1[:, 2]

    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]

    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img / 255, shade=False)
    return None


def format_axis(ax):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(' ')

    return()


def find_unique_dates(site, data_name, extra):

    # Find all files
    file_list = os.listdir(params.main_directory+'siteData/standard/'+site+'/'+data_name+'/')

    # Find all files belonging to the site
    file_list2 = []
    for f in file_list:
        if site in f:
            file_list2.append(f)

    # Find all unique dates among the files belonging to the site
    if len(extra) == 0:
        unique_dates = np.sort(list(set([i.split('.', 1)[0][-8:] for i in file_list2])))
    else:
        unique_dates = np.sort(list(set([i.split('.', 1)[0][-(8+len(extra)):-len(extra)] for i in file_list2])))

    return(unique_dates)


def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


def find_keys_with_substring(dict, substr):

    data_dict_keys = np.array(list(dict.keys()))
    output_array = np.zeros_like(data_dict_keys, dtype=bool)
    val_indices = np.char.find(data_dict_keys, substr)
    output_array[np.where(val_indices != -1)] = True
    VvT_vals = data_dict_keys[output_array]

    return(VvT_vals)


def draw_core(axis, color):

    axis.plot([0.0, 0.75, 0.75, 0.0, 0.0], [0.0, 0.0, 0.8, 0.8, 0.0], linestyle='-', linewidth=5.0, color=color, alpha = 0.2)
    axis.plot([0.0, 0.75, 1.0, 0.25, 0.0], [0.8, 0.8, 0.9, 0.9, 0.8], linestyle='-', linewidth=5.0, color=color, alpha = 0.2)
    axis.plot([0.75, 1.0, 1.0, 0.75, 0.75], [0.0, 0.1, 0.9, 0.8, 0.0], linestyle='-', linewidth=5.0, color=color, alpha = 0.2)

    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.spines[['left', 'right', 'top', 'bottom']].set_visible(False)

    return()

def rot3d(x_points,y_points,z_points,x_theta,y_theta,z_theta):

    x_theta = math.radians(x_theta)
    y_theta = math.radians(y_theta)
    z_theta = math.radians(z_theta)

    # X-axis rotation
    x_points = x_points
    y_points = y_points*np.cos(x_theta) - z_points*np.sin(x_theta)
    z_points = y_points*np.sin(x_theta) + z_points*np.cos(x_theta)

    # Y-axis rotation
    x_points = x_points*np.cos(y_theta) + z_points*np.sin(y_theta)
    y_points = y_points
    z_points = z_points*np.cos(y_theta) - x_points*np.sin(y_theta)

    # Z-axis rotation
    x_points = x_points*np.cos(z_theta) - y_points*np.sin(z_theta)
    y_points = x_points*np.sin(z_theta) + y_points*np.cos(z_theta)
    z_points = z_points

    return(x_points,y_points,z_points)

def get_length(flux,lfactor):

    if flux <= 0.0: direction = -1.0
    if flux > 0.0: direction = 1.0

    af = abs(flux)

    sign = -1.0
    if flux < 0.0:
        sign = -1.0
    if flux > 0.0:
        sign = 1.0

    length = 0.0
    if 1e-4 <= af < 2e-4:
        length = 0.03
    if 2e-4 <= af < 5e-4:
        length = 0.06
    if 5e-4 <= af < 1e-3:
        length = 0.09
    if 1e-3 <= af < 2e-3:
        length = 0.12
    if 2e-3 <= af < 5e-3:
        length = 0.15
    if 5e-3 <= af < 1e-2:
        length = 0.18
    if 1e-2 <= af < 2e-2:
        length = 0.21
    if 2e-2 <= af < 5e-2:
        length = 0.24
    if 5e-2 <= af < 1e-1:
        length = 0.27
    if 1e-1 <= af < 2e-1:
        length = 0.30
    if 2e-1 <= af < 5e-1:
        length = 0.33
    if 5e-1 <= af < 1:
        length = 0.36
    if 1 <= af < 2:
        length = 0.39
    if 2 <= af < 5:
        length = 0.42
    if 5 <= af < 10:
        length = 0.45
    if 10 <= af < 20:
        length = 0.48

    length = 0.05 * direction * rad1 + lfactor * sign * length

    return(length, direction)


def tstring_to_fracday(tstring):
    split_date_time = np.array(tstring.split(' '))
    date=split_date_time[0]
    time=split_date_time[1]
    split_time=np.array(time.split(':'))
    hour = float(split_time[0])
    minute = float(split_time[1])
    second = float(split_time[2])
    frac_day = (hour / 24.0) + (minute / (60.0*24.0)) + (second / (60.0*60.0*24.0))
    return frac_day

def read_csv(filename, column_idx, var_type):
    with open(filename) as f:
        reader = csv.reader(f)
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

def map_plot(origin,angle,labels,cols,distances,radii,site_sites,site_fluxes,site_dates_times,axis,lfactor,north_offset,marker_size,rad1,rad2):
    flux_array = np.array(site_fluxes)
    length_avgs = np.zeros(10)
    length_counts = np.zeros(10)
    directions = []
    for t in range(len(site_dates_times)):
        length, direction = get_length(flux_array[t], l_factor)
        site = site_sites[t]
        mode = site[len(site)-1]
        plot_num = int(site[1:-1])
        length_avgs[plot_num-1] += length
        length_counts[plot_num-1] += 1
    length_avgs = length_avgs / length_counts
    for l in range(0, 10):
        if length_avgs[l] <= 0.0: directions.append('negative')
        if length_avgs[l] > 0.0: directions.append('positive')
    for t in range(len(site_dates_times)):
        test_date_time = site_dates_times[t]
        test_label = site_sites[t]
        if len(site_sites[t]) == 3:
            test_site = (site_sites[t])[0:2]
        if len(site_sites[t]) == 4:
            test_site = (site_sites[t])[0:3]
        col = cols[(np.where(np.array(labels) == test_site)[0])[0]]
        if flux_array[t] < 0.0: col2 = 'red'
        if flux_array[t] > 0.0: col2 = 'green'
        distance = distances[(np.where(np.array(labels) == test_site)[0])[0]]
        radius = radii[(np.where(np.array(labels) == test_site)[0])[0]]
        frac_day = tstring_to_fracday(test_date_time)
        angle2 = 360.0 * frac_day
        length = get_length(flux_array[t],l_factor)
        print(t,test_site[1:],labels[(np.where(np.array(labels) == test_site)[0])[0]],flux_array[t],length)
        #if abs(length) > 1.0:
        #    length = 0.1 * length / abs(length)
        zcoord = 0.0
        xcoord = distance * (np.cos(math.radians(angle))) + origin[0] + radius * (np.cos(math.radians(angle - 90.0)))
        ycoord = distance * (np.sin(math.radians(angle))) + origin[1] + radius * (np.sin(math.radians(angle - 90.0)))
        xcoord,ycoord,zcoord = rot3d(xcoord,ycoord,0.0,0.0,0.0,north_offset)
        plot_idx = int(test_site[1:])
        if length_avgs[plot_idx-1] <= 0.0: col2 = 'red'
        if length_avgs[plot_idx-1] > 0.0: col2 = 'green'
        if test_site != 'P1':
            axis.plot([xcoord], [ycoord], marker='o', markersize=marker_size + 100.0 * abs(length_avgs[plot_idx-1]), color=col2, alpha=0.1)
            axis.plot([xcoord + 0.05], [ycoord - 0.05], marker='o', markersize=marker_size, color='black', alpha=0.3)
            axis.plot([xcoord], [ycoord], marker='o', markersize=marker_size, color=col, alpha=1.0)
            axis.annotate(test_site[1:], xy=(xcoord, ycoord), color='black', fontsize=20, ha='center',
                     va='center')
        theta = np.linspace(0, 2 * np.pi, 150)
        #xt = [rad1 * np.cos(np.radians(angle2)) + xcoord, length * np.cos(np.radians(angle2)) + xcoord]
        #yt = [rad1 * np.sin(np.radians(angle2)) + ycoord, length * np.sin(np.radians(angle2)) + ycoord]
        #xc = rad1 * np.cos(theta) + xcoord
        #yc = rad1 * np.sin(theta) + ycoord
        #col = 'black'
        #if test_label[-1] == 'L':
        #    col = 'white'
        #    thick = 3.0
        #if test_label[-1] == 'D':
        #    col = 'black'
        #    thick = 2.0
        #axis.plot(xc, yc, linestyle='dashed', color=col, linewidth=2.0)
        #axis.plot([origin[0]],[origin[1]],marker='o')
        #axis.plot(xt, yt, color=col, linewidth=thick)

def normalize(x, lower, upper):

    x_max = np.max(x)
    x_min = np.min(x)

    m = (upper - lower) / (x_max - x_min)
    x_norm = (m * (x - x_min)) + lower

    return(x_norm)

# Rotation matrix function
def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

bg_resize, bg_zoom = [1400,2000], 0.127

img1 = rasterio.open(r'C:/Users/Jeremy/Desktop/map_data/22SEP05180342-S2AS_R2C4-050132005010_01_P001.TIF')

full_img1 = img1.read([3,2,1])

full_img1 = full_img1[:, 5200:6300, 300:1200]

data_norm = np.array([normalize(full_img1[i, :, :], 0, 255) for i in range(full_img1.shape[0])])

data_rgb = data_norm.astype("uint8")

mpl.rcParams.update({'font.size':20})
fig = plt.figure(figsize=(35, 12.5))
gs0 = fig.add_gridspec(1, 3, width_ratios=[1,1,1], height_ratios=[1],
                      left=0.1, right=0.9, bottom=0.05, top=0.95,
                      wspace=0.05, hspace=0.0)

ax0 = fig.add_subplot(gs0[0,0])
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax0.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
rasterio.plot.show(data_rgb, ax = ax0)
x_offset, y_offset, angle, side = 400, 400, 30.0, 30.0
sx, sy = np.array([-side, side, side, -side, -side]), np.array([-side, -side, side, side, -side])

xr, yr = rotate_matrix(sx, sy, angle)
xr, yr = xr + x_offset, yr + y_offset
for i in range(0, 4):
    ax0.plot([xr[i], xr[i+1]], [yr[i], yr[i+1]], linestyle='-', linewidth = 7.0, color='salmon', alpha = 0.5)

#im = plt.imread(r'C:/Users/Jeremy/Desktop/the_files/fen_photo.jpg') # insert local path of the image.
#implot = plt.imshow(im)
file = r'C:\Users\Jeremy\Desktop\VUA\the_files\maps\Aug5,6-53AM-poly_2\point_cloud_rgb.txt'
photo = r'C:\Users\Jeremy\Desktop\VUA\the_files\palsa_2m_photo.jpg'
im = plt.imread(photo) # insert local path of the image.
ax1 = fig.add_subplot(gs0[0,1])
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
node1, node2, node3, node4 = [500, 3250], [2000, 2100], [4500, 2500], [4250, 4000]
ax1.plot([node1[0], node2[0], node3[0], node4[0], node1[0]], [node1[1], node2[1], node3[1], node4[1], node1[1]], linestyle = '-', linewidth = 8.0, color = 'salmon')
implot = ax1.imshow(im)
text_array = np.loadtxt(file)
x = text_array[0::100, 0]
y = text_array[0::100, 1]
z = text_array[0::100, 2]
z = z + 0.9
#z[z < 0.05] = 0.0
#z[z > 0.5] = 0.5
ax2 = fig.add_subplot(gs0[0,2])
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
x, y = rotate_matrix(x, y, -39.0)
ax2.plot([min(x), max(x), max(x), min(x), min(x)], [min(y), min(y), max(y), max(y), min(y)], linestyle = '-', linewidth = 12.0, color = 'salmon')
cb = ax2.tricontourf(x,y,z, linewidths=0.5, cmap='terrain', levels=np.arange(0.0,1.0,0.01)) # choose 20 contour levels, just to show how good its interpolation is
fig.colorbar(cb, shrink = 0.6, label = 'Relief (m)')
plt.gca().set_aspect('equal', adjustable='box')
#plt.savefig('C:/Users/Jeremy/Desktop/map.pdf', bbox_inches='tight', transparent=True)

###

dates_times = np.array(read_csv(r'C:\Users\Jeremy\Desktop\data\august_fluxes2.txt', 1, 'string'))
fluxes = np.array(read_csv(r'C:\Users\Jeremy\Desktop\data\august_fluxes2.txt', 2, 'float'))
site_labels = np.array(read_csv(r'C:\Users\Jeremy\Desktop\data\august_fluxes2.txt', 0, 'string'))

site_letters = np.array([x[0] for x in site_labels])

palsa_a_sites = site_labels[site_letters == 'P']
palsa_a_dates_times = dates_times[site_letters == 'P']
palsa_a_fluxes = fluxes[site_letters == 'P']

c_x = 5.3
c_y = 4.3
c_diam = 0.3
rad1 = 0.6
rad2 = 0.5
max_flux = 0.05
flux_array = palsa_a_fluxes
l_factor = 1.0
label1='-  0.05'
label2='+ 0.05'
north_offset = 30.0

test = map_plot([3.0, -5.5],115.0,['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
                ['lightgreen','lightgreen','lightgreen','lightgreen','orange','orange','orange','orange','orange','orange'],
                [1.12, 2.85, 3.77, 2.22, 6.41, 8.14, 7.68, 7.56, 8.48, 8.16],
                [1.10, 1.74, 0.24, 1.30, -1.25, 4.76, -2.31, -0.14, 1.57, -0.39],
                palsa_a_sites,palsa_a_fluxes,palsa_a_dates_times,ax2,l_factor,north_offset,30,0.3,0.25)

plt.savefig('C:/Users/Jeremy/Desktop/VUA/the_files/'+params.site+'_map.png')

fig = plt.figure(figsize=(35.0, 55.0))
gs1 = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
                      left=0.0, right=1.0, bottom=0.0, top=1.0,
                      wspace=0.0, hspace=0.0)

ax4 = fig.add_subplot(gs1[:,0], projection='3d')
ax4.set_xlim((0.,1.))
ax4.set_ylim((0.,1.))
ax4.set_zlim((-0.45,0.15))
ax4.set_box_aspect((1, 1, 3))
ax4.set_axis_off()
ax4.view_init(elev=30, azim=-30, roll=0)
ax4.dist = 6

layer_thicknesses, depths = init.define_layers()
site_data = data.read_site_data(params.site, depths)

o2umolL_dict = site_data['site_o2umolL']
unique_dates = np.flip(np.sort(find_unique_dates('palsa_high', 'o2umolL', extra = '')))
o2wat_1 = o2umolL_dict[find_keys_with_substring(o2umolL_dict, unique_dates[0] + '_o2molm3')[0]]

o2pct_dict = site_data['site_o2pct']
unique_dates = np.flip(np.sort(find_unique_dates('palsa_high', 'o2pct', extra = '')))
o2air_1 = 18.0 * o2pct_dict[find_keys_with_substring(o2pct_dict, unique_dates[0] + '_o2pct')[0]] / 21.29

o2molm3_1 = o2wat_1 + o2air_1

# Temps at 5, 15, 25 cm
temps_dict = site_data['site_temps']
unique_dates = np.flip(np.sort(find_unique_dates('palsa_high', 'temps', extra = '_05cm')))
#
temps05_1 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[0] + '_temps05')[0]]
temps15_1 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[0] + '_temps15')[0]]
temps25_1 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[0] + '_temps25')[0]]
dtimes_1 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[0] + '_dtimes')[0]]
temps05_1_hourly, temps05_1_hourly_count = [0.0]*24, [0.0]*24
temps15_1_hourly, temps15_1_hourly_count = [0.0]*24, [0.0]*24
temps25_1_hourly, temps25_1_hourly_count = [0.0]*24, [0.0]*24
for i in range(0, len(dtimes_1)):
    temps05_1_hourly[dtimes_1[i].hour] += temps05_1[i]
    temps05_1_hourly_count[dtimes_1[i].hour] += 1
    temps15_1_hourly[dtimes_1[i].hour] += temps15_1[i]
    temps15_1_hourly_count[dtimes_1[i].hour] += 1
    temps25_1_hourly[dtimes_1[i].hour] += temps25_1[i]
    temps25_1_hourly_count[dtimes_1[i].hour] += 1
temps05_1_hourly = np.array(temps05_1_hourly) / np.array(temps05_1_hourly_count)
temps15_1_hourly = np.array(temps15_1_hourly) / np.array(temps15_1_hourly_count)
temps25_1_hourly = np.array(temps25_1_hourly) / np.array(temps25_1_hourly_count)

new_grid = layer_thicknesses[0] * range(13)

hourly_temp_profiles1 = []
for i in range(0, len(temps05_1_hourly)):
    x = [0.05, 0.15, 0.25]
    y = [temps05_1_hourly[i], temps15_1_hourly[i], temps25_1_hourly[i]]
    xs, cs = new_grid, CubicSpline(x, y)
    xs = xs[:-2]
    hourly_temp_profiles1.append(cs(xs))
hourly_temp_depths1 = new_grid[:-2]

temps05_2 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[1] + '_temps05')[0]]
temps15_2 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[1] + '_temps15')[0]]
temps25_2 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[1] + '_temps25')[0]]
dtimes_2 = temps_dict[find_keys_with_substring(temps_dict, unique_dates[1] + '_dtimes')[0]]
temps05_2_hourly, temps05_2_hourly_count = [0.0]*24, [0.0]*24
temps15_2_hourly, temps15_2_hourly_count = [0.0]*24, [0.0]*24
temps25_2_hourly, temps25_2_hourly_count = [0.0]*24, [0.0]*24
for i in range(0, len(dtimes_2)):
    temps05_2_hourly[dtimes_2[i].hour] += temps05_2[i]
    temps05_2_hourly_count[dtimes_2[i].hour] += 1
    temps15_2_hourly[dtimes_2[i].hour] += temps15_2[i]
    temps15_2_hourly_count[dtimes_2[i].hour] += 1
    temps25_2_hourly[dtimes_2[i].hour] += temps25_2[i]
    temps25_2_hourly_count[dtimes_2[i].hour] += 1
temps05_2_hourly = np.array(temps05_2_hourly) / np.array(temps05_2_hourly_count)
temps15_2_hourly = np.array(temps15_2_hourly) / np.array(temps15_2_hourly_count)
temps25_2_hourly = np.array(temps25_2_hourly) / np.array(temps25_2_hourly_count)

hourly_temp_profiles2 = []
for i in range(0, len(temps05_2_hourly)):
    x = [0.05, 0.15, 0.25]
    y = [temps05_2_hourly[i], temps15_2_hourly[i], temps25_2_hourly[i]]
    xs, cs = new_grid, CubicSpline(x, y)
    xs = xs[:-2]
    hourly_temp_profiles2.append(cs(xs))
hourly_temp_depths2 = new_grid[:-2]

doc_dict = site_data['site_doc']
unique_dates = np.sort(find_unique_dates('palsa_high', 'doc', extra = ''))
doc_1 = doc_dict[find_keys_with_substring(doc_dict, unique_dates[0] + '_docmolm3')[0]]
doc_2 = doc_dict[find_keys_with_substring(doc_dict, unique_dates[1] + '_docmolm3')[0]]

flux_dict = site_data['site_fluxes']
unique_dates = np.flip(np.sort(find_unique_dates('palsa_high', 'fluxes', extra = '')))
fluxes_1 = flux_dict[find_keys_with_substring(flux_dict, unique_dates[0] + '_fluxesmolm2day')[0]]
flux_times_1 = flux_dict[find_keys_with_substring(flux_dict, unique_dates[0] + '_T')[0]]
print(len(flux_times_1))
stop
fluxes_2 = flux_dict[find_keys_with_substring(flux_dict, unique_dates[1] + '_fluxesmolm2day')[0]]
flux_times_2 = flux_dict[find_keys_with_substring(flux_dict, unique_dates[1] + '_T')[0]]
fluxtimes_1, fluxtimes_2, fluxtemps_1, fluxtemps_2 = [], [], [], []
for i in range(0, len(fluxes_1)):
    closest_time_idx = np.where(abs(dtimes_1 - flux_times_1[i]) == np.min(abs(dtimes_1 - flux_times_1[i])))[0][0]
    fluxtimes_1.append(dtimes_1[closest_time_idx])
    fluxtemps_1.append(temps05_1[closest_time_idx])
for i in range(0, len(fluxes_2)):
    closest_time_idx = np.where(abs(dtimes_2 - flux_times_2[i]) == np.min(abs(dtimes_2 - flux_times_2[i])))[0][0]
    fluxtimes_2.append(dtimes_2[closest_time_idx])
    fluxtemps_2.append(temps05_2[closest_time_idx])

site_data2 = data.read_site_data('palsa_low', depths)

mcra_dict = site_data['site_mcra']
pmoa_a_dict = site_data['site_pmoa1']
pmoa_b_dict = site_data['site_pmoa2']
#
mcracopm3_1 = mcra_dict[find_keys_with_substring(mcra_dict, 'copm3')[0]]
pmoa1copm3_1 = pmoa_a_dict[find_keys_with_substring(pmoa_a_dict, 'copm3')[0]]
pmoa1copm3_1[np.isnan(pmoa1copm3_1)] = 0.0
pmoa2copm3_1 = pmoa_b_dict[find_keys_with_substring(pmoa_b_dict, 'copm3')[0]]
pmoa2copm3_1[np.isnan(pmoa2copm3_1)] = 0.0
#
mcracopm3_2 = mcra_dict[find_keys_with_substring(mcra_dict, 'copm3')[1]]
pmoa1copm3_2 = pmoa_a_dict[find_keys_with_substring(pmoa_a_dict, 'copm3')[1]]
pmoa1copm3_2[np.isnan(pmoa1copm3_2)] = 0.0
pmoa2copm3_2 = pmoa_b_dict[find_keys_with_substring(pmoa_b_dict, 'copm3')[1]]
pmoa2copm3_2[np.isnan(pmoa2copm3_2)] = 0.0

root_dict = site_data['site_roots']
rootkgm3 = root_dict[find_keys_with_substring(root_dict, 'roots')[0]]

comp_dict = site_data['site_comp']
h2om3m3 = comp_dict[find_keys_with_substring(comp_dict, 'h2om3m3')[0]]
socm3m3 = comp_dict[find_keys_with_substring(comp_dict, 'socm3m3')[0]]
minm3m3 = comp_dict[find_keys_with_substring(comp_dict, 'minm3m3')[0]]
airm3m3 = comp_dict[find_keys_with_substring(comp_dict, 'airm3m3')[0]]

cut_idx = np.where(np.diff(h2om3m3) == 0)[0][0]
#ax4.plot([-0.2, -0.2], [0.0, 0.0], [-depths[cut_idx] + layer_thicknesses[0], 0.0], linewidth = 15.0, color = 'black')
for i in range(0, 7):
    d = -0.1 * (i * 0.5)
    if i%2 == 0:
        ax4.plot([0.0, 0.0], [-0.05, 0.0], [d, d], linewidth = 15.0, color = 'gray')
        ax4.text(0.0, -0.15, d, str(abs(round(100.0 * d))), ha='center', va='center', fontsize=72, weight='bold', color='gray')

h2om3m3, socm3m3, minm3m3, airm3m3, depths = h2om3m3[0:cut_idx], socm3m3[0:cut_idx], minm3m3[0:cut_idx], airm3m3[0:cut_idx], depths[0:cut_idx]

h2om3m3_verts = []
airm3m3_verts = []
socm3m3_verts = []
minm3m3_verts = []
xs = np.linspace(0., 1., 26)
zs = np.ones(len(h2om3m3))
zoffset = layer_thicknesses[0]/2.0
for i in range(0,len(zs)):
    ys = -1.0 * depths[i] + np.zeros(len(xs))
    x1, x2 = 0.0, h2om3m3[i]
    depth2 = 0.4 * layer_thicknesses[0]
    h2om3m3_vert = [(x1, -depths[i] + depth2 + zoffset), (x2, -depths[i] + depth2 + zoffset), (x2, -depths[i] - depth2 + zoffset), (x1, -depths[i] -depth2 + zoffset), (x1, -depths[i] + depth2 + zoffset)]
    h2om3m3_verts.append(h2om3m3_vert)
    x1, x2 = h2om3m3[i], h2om3m3[i] + airm3m3[i]
    airm3m3_vert = [(x1, -depths[i] + depth2 + zoffset), (x2, -depths[i] + depth2 + zoffset), (x2, -depths[i] - depth2 + zoffset), (x1, -depths[i] -depth2 + zoffset), (x1, -depths[i] + depth2 + zoffset)]
    airm3m3_verts.append(airm3m3_vert)
    x1, x2 = h2om3m3[i] + airm3m3[i], h2om3m3[i] + airm3m3[i] + socm3m3[i]
    socm3m3_vert = [(x1, -depths[i] + depth2 + zoffset), (x2, -depths[i] + depth2 + zoffset), (x2, -depths[i] - depth2 + zoffset), (x1, -depths[i] -depth2 + zoffset), (x1, -depths[i] + depth2 + zoffset)]
    socm3m3_verts.append(socm3m3_vert)
    x1, x2 = h2om3m3[i] + airm3m3[i] + socm3m3[i], h2om3m3[i] + airm3m3[i] + socm3m3[i] + minm3m3[i]
    minm3m3_vert = [(x1, -depths[i] + depth2 + zoffset), (x2, -depths[i] + depth2 + zoffset), (x2, -depths[i] - depth2 + zoffset), (x1, -depths[i] -depth2 + zoffset), (x1, -depths[i] + depth2 + zoffset)]
    minm3m3_verts.append(minm3m3_vert)
    ax4.plot([1.0, 1.0], [0.0, 1.0], [-depths[i], -depths[i]], linewidth = 15.0, color = 'gray', alpha = 1.0)
ax4.add_collection3d(PolyCollection(h2om3m3_verts, facecolors=['cornflowerblue'], alpha=1.0, edgecolors=['None']), zs=zs, zdir='x')
ax4.add_collection3d(PolyCollection(airm3m3_verts, facecolors=['lavender'], alpha=1.0), zs=zs, zdir='x')
ax4.add_collection3d(PolyCollection(socm3m3_verts, facecolors=['burlywood'], alpha=1.0), zs=zs, zdir='x')
ax4.add_collection3d(PolyCollection(minm3m3_verts, facecolors=['grey'], alpha=1.0), zs=zs, zdir='x')

x, y, z = np.indices((1, 1, 1))
cube1 = (x <= 1)
voxelarray = cube1
colors = np.empty(voxelarray.shape, dtype=object)
colors[cube1] = 'slategrey'
x,y,z = np.indices(np.array(voxelarray.shape)+1)
###ax4.voxels(x, y, -(0.05) * z - depths[-1] - 0.05, voxelarray, facecolors='slategrey', edgecolor=None, alpha = 1.0, clip_on = False)

x, y, z = np.indices((1, 1, 1))
cube1 = (x <= 1)
voxelarray = cube1
colors = np.empty(voxelarray.shape, dtype=object)
colors[cube1] = 'peru'
x,y,z = np.indices(np.array(voxelarray.shape)+1)
###ax4.voxels(x, y, -(depths[-1]) * z, voxelarray, facecolors=colors, edgecolor='peru', alpha = 1.0, clip_on = False)

x, y, z = np.indices((1, 1, 1))
cube1 = (x <= 1)
voxelarray = cube1
colors = np.empty(voxelarray.shape, dtype=object)
colors[cube1] = 'palegreen'
x,y,z = np.indices(np.array(voxelarray.shape)+1)
###ax4.voxels(x, y, 0.05 * z + 0.05, voxelarray, facecolors=colors, edgecolor='palegreen', alpha = 1.0, clip_on = False)

arr = iio.imread('C:/Users/Jeremy/Desktop/VUA/the_files/veg_pic_1.png')
# 10 is equal length of x and y axises of your surface
stepX, stepY = 1. / arr.shape[0], 1. / arr.shape[1]
X1 = np.arange(0, 1, stepX)
Y1 = np.arange(0, 1, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
arr = arr / np.max(arr)
xdim, ydim = arr.shape[0], arr.shape[1]
z = np.zeros((xdim, ydim)) + 2 * 0.05
ax4.plot_surface(X1, Y1, z, rstride=100, cstride=100, facecolors=arr, clip_on = False)

arr = arr[0:round(arr.shape[1]), :]
xdim, ydim = arr.shape[0], arr.shape[1]
X1 = np.linspace(0, 1, xdim+1)#np.arange(0, 1, stepX)
Y1 = np.linspace(0.05, 0.05 + 0.05, ydim+1)#, np.arange(0, 0.1, stepY)#1 - depths[cut_idx - 1], stepY)
X1, Y1 = np.meshgrid(X1, Y1)
z = np.zeros_like(X1)
arr = arr.reshape((ydim, xdim, 3))#np.array([[arr[:,:,0].T], arr[:,:,1].T, arr[:,:,2].T])
ax4.plot_surface(X1, z, Y1, rstride=100, cstride=100, facecolors=arr, clip_on = False)
R_rad = np.array((0.0, 0.0, 90.0)) * math.pi / 180
R = cv.Rodrigues(R_rad)[0]
t = np.array(([0.5, 0.5, -0.1]))

arr = arr[:, 0:round(arr.shape[1])]
xdim, ydim = arr.shape[0], arr.shape[1]
X1 = np.linspace(0.05, 0.05 + 0.05, xdim+1)#np.arange(0, 1, stepX)
Y1 = np.linspace(0, 1, ydim+1)#, np.arange(0, 0.1, stepY)#1 - depths[cut_idx - 1], stepY)
X1, Y1 = np.meshgrid(X1, Y1)
z = np.zeros_like(X1)
arr = arr.reshape((ydim, xdim, 3))#np.array([[arr[:,:,0].T], arr[:,:,1].T, arr[:,:,2].T])
ax4.plot_surface(z + 1.0, Y1, X1, rstride=100, cstride=100, facecolors=arr, clip_on = False)
R_rad = np.array((0.0, 0.0, 90.0)) * math.pi / 180
R = cv.Rodrigues(R_rad)[0]
t = np.array(([0.5, 0.5, -0.1]))

al_ratio = depths[cut_idx-1] / 1.0

arr = iio.imread('C:/Users/Jeremy/Desktop/peat_texture.png')
xdim, ydim = np.shape(arr)[0], np.shape(arr)[1]
stepX, stepY = 1. / arr.shape[0], 1. / arr.shape[1]
X1 = np.arange(0, 1, stepX)
Y1 = np.arange(0, 1, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
arr = arr / np.max(arr)
z = np.zeros((xdim, ydim))
ax4.plot_surface(X1, Y1, z, rstride=10, cstride=10, facecolors=arr, clip_on = False, zorder = 0)

arr = arr[0:round(arr.shape[1]), :]
xdim, ydim = arr.shape[0], arr.shape[1]
X1 = np.linspace(0, 1, xdim+1)#np.arange(0, 1, stepX)
Y1 = np.linspace(-depths[cut_idx-1], 0.0, ydim+1)#, np.arange(0, 0.1, stepY)#1 - depths[cut_idx - 1], stepY)
X1, Y1 = np.meshgrid(X1, Y1)
z = np.zeros_like(X1)
arr = arr.reshape((ydim, xdim, 3))#np.array([[arr[:,:,0].T], arr[:,:,1].T, arr[:,:,2].T])
ax4.plot_surface(X1, z, Y1, rstride=10, cstride=10, facecolors=arr, clip_on = False)
R_rad = np.array((0.0, 0.0, 90.0)) * math.pi / 180
R = cv.Rodrigues(R_rad)[0]
t = np.array(([0.5, 0.5, -0.1]))

arr = arr[:, 0:round(arr.shape[1])]
xdim, ydim = arr.shape[0], arr.shape[1]
X1 = np.linspace(-depths[cut_idx-1], 0.0, xdim+1)#np.arange(0, 1, stepX)
Y1 = np.linspace(0, 1, ydim+1)#, np.arange(0, 0.1, stepY)#1 - depths[cut_idx - 1], stepY)
X1, Y1 = np.meshgrid(X1, Y1)
z = np.zeros_like(X1)
arr = arr.reshape((ydim, xdim, 3))#np.array([[arr[:,:,0].T], arr[:,:,1].T, arr[:,:,2].T])
ax4.plot_surface(z + 1.0, Y1, X1, rstride=10, cstride=10, facecolors=arr, clip_on = False)
R_rad = np.array((0.0, 0.0, 90.0)) * math.pi / 180
R = cv.Rodrigues(R_rad)[0]
t = np.array(([0.5, 0.5, -0.1]))

arr = iio.imread('C:/Users/Jeremy/Desktop/permafrost_texture.png')
stepX, stepY = 1. / arr.shape[0], 1. / arr.shape[1]
X1 = np.arange(0, 1, stepX)
Y1 = np.arange(0, 1, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
arr = arr / np.max(arr)
xdim, ydim = arr.shape[0], arr.shape[1]
z = np.zeros((xdim, ydim))
ax4.plot_surface(X1, Y1, z - depths[cut_idx-1] - 0.05, rstride=10, cstride=10, facecolors=arr, clip_on = False)

arr = arr[0:round(arr.shape[1]), :]
xdim, ydim = arr.shape[0], arr.shape[1]
X1 = np.linspace(0, 1, xdim+1)#np.arange(0, 1, stepX)
Y1 = np.linspace(-depths[-1] - 0.05 - 0.1, -depths[-1] - 0.05, ydim+1)#, np.arange(0, 0.1, stepY)#1 - depths[cut_idx - 1], stepY)
X1, Y1 = np.meshgrid(X1, Y1)
z = np.zeros_like(X1)
arr = arr.reshape((ydim, xdim, 3))#np.array([[arr[:,:,0].T], arr[:,:,1].T, arr[:,:,2].T])
ax4.plot_surface(X1, z, Y1, rstride=10, cstride=10, facecolors=arr, clip_on = False)
R_rad = np.array((0.0, 0.0, 90.0)) * math.pi / 180
R = cv.Rodrigues(R_rad)[0]
t = np.array(([0.5, 0.5, -0.1]))

arr = arr[:, 0:round(arr.shape[1])]
xdim, ydim = arr.shape[0], arr.shape[1]
X1 = np.linspace(-depths[-1] - 0.05 - 0.1, -depths[-1] - 0.05, xdim+1)#np.arange(0, 1, stepX)
Y1 = np.linspace(0, 1, ydim+1)#, np.arange(0, 0.1, stepY)#1 - depths[cut_idx - 1], stepY)
X1, Y1 = np.meshgrid(X1, Y1)
z = np.zeros_like(X1)
arr = arr.reshape((ydim, xdim, 3))#np.array([[arr[:,:,0].T], arr[:,:,1].T, arr[:,:,2].T])
ax4.plot_surface(z + 1.0, Y1, X1, rstride=10, cstride=10, facecolors=arr, clip_on = False)
R_rad = np.array((0.0, 0.0, 90.0)) * math.pi / 180
R = cv.Rodrigues(R_rad)[0]
t = np.array(([0.5, 0.5, -0.1]))

#Xc,Yc,Zc = data_for_cylinder_along_z(0.5,0.5,0.3,0.1)
#ax4.plot_surface(Xc, Yc, Zc, alpha=0.5, clip_on = False, zorder = -1, color = 'orange')

plt.savefig('C:/Users/Jeremy/Desktop/VUA/the_files/'+params.site+'_cores.png')

fig = plt.figure(figsize=(24.0, 12.0))
gs2 = fig.add_gridspec(2, 10, width_ratios=[3, 1, 3, 3, 3, 1, 3, 1, 3, 3], height_ratios=[1, 1],
                      left=0.05, right=0.95, bottom=0.05, top=0.95,
                      wspace=0.05, hspace=0.0)

lthick = layer_thicknesses[0]

ax6 = fig.add_subplot(gs2[0,3])
profile_plot(ax6, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/pmoa_texture.jpg', (8e-14) * pmoa1copm3_1, depths, lthick, cut_idx)

ax7 = fig.add_subplot(gs2[0,4])
profile_plot(ax7, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/pmoa_texture.jpg', (8e-14) * pmoa2copm3_1, depths, lthick, cut_idx)

ax13 = fig.add_subplot(gs2[0,2])
profile_plot(ax13, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/mcra_texture.jpg', (8e-14) * mcracopm3_1, depths, lthick, cut_idx)

ax14 = fig.add_subplot(gs2[1,2])
profile_plot(ax14, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/mcra_texture.jpg', (8e-14) * mcracopm3_2, depths, lthick, cut_idx)

ax15 = fig.add_subplot(gs2[1,3])
profile_plot(ax15, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/pmoa_texture.jpg', (8e-14) * pmoa1copm3_2, depths, lthick, cut_idx)

ax16 = fig.add_subplot(gs2[1,4])
profile_plot(ax16, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/pmoa_texture.jpg', (8e-14) * pmoa2copm3_2, depths, lthick, cut_idx)

ax7 = fig.add_subplot(gs2[0,6])
profile_plot(ax7, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/root_texture.jpg', 0.0*rootkgm3, depths, lthick, cut_idx)

ax17 = fig.add_subplot(gs2[1,6])
profile_plot(ax17, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/root_texture.jpg', rootkgm3, depths, lthick, cut_idx)

ax8 = fig.add_subplot(gs2[0,0])
ax8.set_box_aspect(1.5)
ax8.set_ylim((-depths[-1], 0.0))
ax8.set_xlim([0.0, 24.0])
ax8.set_xlabel('%')
format_axis(ax8)
#for i in range(0, 24):
#    ax8.fill_between([i, i + 1, i + 1, i, i], [-0.1, -0.1, -0.0, -0.0, -0.1], color=cm.rainbow(temps05_1_hourly[i] / 15.0), edgecolor = None)
#    ax8.fill_between([i, i + 1, i + 1, i, i], [-0.2, -0.2, -0.1, -0.1, -0.2], color=cm.rainbow(temps15_1_hourly[i] / 15.0), edgecolor = None)
#    ax8.fill_between([i, i + 1, i + 1, i, i], [-0.3, -0.3, -0.2, -0.2, -0.3], color=cm.rainbow(temps25_1_hourly[i] / 15.0), edgecolor = None)

for i in range(0, 24):
    for j in range(1, len(hourly_temp_depths1)):
        ax8.fill_between([i, i + 1, i + 1, i, i], [-depths[j], -depths[j], -depths[j-1], -depths[j-1], -depths[j]], color=cm.rainbow(hourly_temp_profiles1[i][j] / 20.0), edgecolor = None)

ax9 = fig.add_subplot(gs2[1,0])
ax9.set_box_aspect(1.5)
ax9.set_ylim((-depths[-1], 0.0))
ax9.set_xlim([0.0, 24.0])
ax9.set_xlabel('%')
format_axis(ax9)
#for i in range(0, 24):
#    ax9.fill_between([i, i + 1, i + 1, i], [-0.1, -0.1, -0.0, -0.0], color=cm.rainbow(temps05_2_hourly[i] / 15.0), edgecolor = None)
#    ax9.fill_between([i, i + 1, i + 1, i], [-0.2, -0.2, -0.1, -0.1], color=cm.rainbow(temps15_2_hourly[i] / 15.0), edgecolor = None)
#    ax9.fill_between([i, i + 1, i + 1, i], [-0.3, -0.3, -0.2, -0.2], color=cm.rainbow(temps25_2_hourly[i] / 15.0), edgecolor = None)
#ax9.get_yaxis().set_visible(False)

for i in range(0, 24):
    for j in range(1, len(hourly_temp_depths2)):
        ax9.fill_between([i, i + 1, i + 1, i, i], [-depths[j], -depths[j], -depths[j-1], -depths[j-1], -depths[j]], color=cm.rainbow(hourly_temp_profiles2[i][j] / 20.0), edgecolor = None)

ax10 = fig.add_subplot(gs2[0,8])
profile_plot(ax10, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/doc_texture.jpg', doc_1, depths, lthick, cut_idx)

ax18 = fig.add_subplot(gs2[1,8])
profile_plot(ax18, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/doc_texture.jpg', doc_2, depths, lthick, cut_idx)

ax12 = fig.add_subplot(gs2[0,9])
profile_plot(ax12, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/o2_texture.jpg', 0.0*o2molm3_1, depths, lthick, cut_idx)

ax19 = fig.add_subplot(gs2[1,9])
profile_plot(ax19, 1.5, bg_resize, bg_zoom, 'C:/Users/Jeremy/Desktop/o2_texture.jpg', o2molm3_1, depths, lthick, cut_idx)

#ax6.step(pmoacopm3[0:cut_idx], -depths[0:cut_idx] + lthick, color = 'orange', alpha = 0.5)
plt.savefig('C:/Users/Jeremy/Desktop/VUA/the_files/'+params.site+'_profiles.png', bbox_inches='tight')

fig = plt.figure(figsize=(18.0, 12.0))
gs3 = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.1)

ax11 = fig.add_subplot(gs3[0,0])
ax11.set_box_aspect(1.5)
ax11.set_xlabel('%')
format_axis(ax11)
print(len(fluxes_1))
stop
for f in range(0, len(fluxtemps_1)):
    ax11.plot([fluxtemps_1[f]], [fluxes_1[f]], marker = 'o', color = 'blue')
for f in range(0, len(fluxtemps_2)):
    ax11.plot([fluxtemps_2[f]], [fluxes_2[f]], marker = 'o', color = 'red')

#palsa_a_sites,palsa_a_fluxes,palsa_a_dates_times
print(palsa_a_dates_times)

plt.savefig('C:/Users/Jeremy/Desktop/VUA/the_files/'+params.site+'_flux_temps.png')