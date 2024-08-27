# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:38:55 2018

@author: Meli
"""
from __future__ import with_statement, print_function, division
import rasterio
from rasterio import warp
from rasterio.crs import CRS
from rasterio.mask import mask
from shapely.geometry import box
import rasterio.plot
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from os import remove
from osgeo import gdal, ogr
#import gdal
from osgeo import ogr, gdal_array, gdalconst
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from collections import Counter
import os
from osgeo import gdal, ogr, osr
import sys

import skimage
from skimage.filters.rank import modal
from skimage.morphology import disk

from scipy import signal
from scipy.ndimage import label, binary_dilation
from collections import Counter

from pylab import *


# ---------------------------------------------------------------------------------------------------
## To do

# Tidy up code - make functions
# Email Ko for 2017 + atm correction
# Optimize number of trees used in classifier
# Find accuracy measurement
# Preprocess check data


# function to parse feature from GeoDataFrame in such a manner that rasterio wants them
def getFeatures(gdf):
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def f(row):
    if 'P' in row:
        val = 'Pond'
    elif 'L' in row:
        val = 'Lake'
    elif 'S' in row:
        val = 'Stream'
    else:
        val = 'Tundra'
    return val


# Function to read in xy data from raster
def ix2xy(r, c, gt):
    '''Gets x,y from row and column'''
    x = gt[0] + r * gt[1]
    y = gt[3] + c * gt[5]
    return (x, y)


def coord_conversion(inputEPSG, outputEPSG, coords):
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    x_list = []
    y_list = []

    for index, row in coords.iterrows():
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(row['x'], row['y'])
        point.Transform(coordTransform)
        x_list.append(point.GetX())
        y_list.append(point.GetY())
        newcoords = pd.DataFrame({'x': x_list, 'y': y_list})
    # newcoords['Site'] = coords['Site']

    return newcoords


def shapefile_conversion(shapefile, outputEPSG, outputname):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()

    inSpatialRef = layer.GetSpatialRef()

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    if os.path.exists(outputname):
        driver.DeleteDataSource(outputname)
    ds = driver.CreateDataSource(outputname)
    outlayer = ds.CreateLayer('name', outSpatialRef, ogr.wkbPolygon)
    outlayer.CreateField(ogr.FieldDefn('Classname', ogr.OFTString))
    outlayer.CreateField(ogr.FieldDefn('Classid', ogr.OFTInteger))

    # apply transformation
    i = 0

    for feature in layer:
        transformed = feature.GetGeometryRef()
        transformed.Transform(coordTransform)

        geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
        defn = outlayer.GetLayerDefn()
        feat = ogr.Feature(defn)

        feat.SetField('Classname', feature.GetField('Classname'))
        feat.SetField('Classid', feature.GetField('ClassID'))
        feat.SetGeometry(geom)
        outlayer.CreateFeature(feat)
        i += 1
        feat = None

    ds = None

    return outlayer


def reproj(in_raster, dst_crs, out_ras):
    driver = gdal.GetDriverByName('GTiff')
    in_ras = rasterio.open(in_raster)
    transform, width, height, = rasterio.warp.calculate_default_transform(in_ras.crs, \
                                                                          dst_crs, in_ras.width, \
                                                                          in_ras.height, *in_ras.bounds)
    kwargs = in_ras.meta.copy()
    kwargs.update({'crs': dst_crs,
                   'transform': transform,
                   'width': width,
                   'height': height})

    out_file_name = out_ras

    with rasterio.open(out_file_name, 'w', **kwargs) as dst:
        for i in range(1, in_ras.count + 1):
            rasterio.warp.reproject(
                source=rasterio.band(in_ras, i),
                destination=rasterio.band(dst, i),
                src_transform=in_ras.transform,
                src_crs=in_ras.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=rasterio.warp.Resampling.bilinear)

    return out_file_name


def reproj_resampl(in_raster, ref_raster, out_ras):
    driver = gdal.GetDriverByName('GTiff')

    in_ras = gdal.Open(in_raster)
    input_Proj = in_ras.GetProjection()

    reference = gdal.Open(ref_raster)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    bandreference = reference.GetRasterBand(1)
    x = reference.RasterXSize
    y = reference.RasterYSize

    output = driver.Create(out_ras, x, y, 1, bandreference.DataType)
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)

    gdal.ReprojectImage(in_ras, output, input_Proj, referenceProj, gdalconst.GRA_Bilinear)

    output = None
    return output


def clipping(in_raster, out_raster, mixx_, miny_, maxx_, maxy_):
    driver = gdal.GetDriverByName('GTiff')
    in_ras = rasterio.open(in_raster)
    bbox = box(mixx_, miny_, maxx_, maxy_)

    # insert bbox into GeoDataFrame
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=in_ras.crs)

    # coordinates in format accepted by rasterio
    coords = getFeatures(geo)
    print(coords)

    # clip using mask
    out_img, out_transform = mask(dataset=in_ras, shapes=coords, crop=True)

    # copy metadata
    out_meta = in_ras.meta.copy()

    # update  metadata with new dimensions, transform (affine), CRS
    out_meta.update({'driver': 'GTiff',
                     'height': out_img.shape[1],
                     'width': out_img.shape[2],
                     'transform': out_transform,
                     'crs': in_ras.crs})

    with rasterio.open(out_raster, 'w', **out_meta) as dest:
        dest.write(out_img)

    return out_raster


# Calculate Earth-Sun distance
def Earth_Sun_distance(year, month, day, hh, mm, ss):
    # Universal time
    UT = hh + (mm / 60) + (ss / 3600)

    # Julian Day (JD)
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25 * (year + 4716) + int(30.6001 * (month + 1)) + day + (UT / 24) + B - 1524.5)

    # Earth-Sun distance (dES)
    D = JD - 2451545
    g = 357.529 + 0.98560028 * D  # (in degrees)
    dES = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * np.cos(np.deg2rad(2 * g))

    return dES


def refl(in_raster, absFactors, effBandWidths, ESUN, dES, theta, out_raster_TOA, out_raster_surf):
    drv = gdal.GetDriverByName('GTiff')
    raster = gdal.Open(in_raster)

    referenceProj = raster.GetProjection()
    referenceTrans = raster.GetGeoTransform()
    x = raster.RasterXSize
    y = raster.RasterYSize
    n = raster.RasterCount

    raster_array = raster.ReadAsArray()
    data = np.single(raster_array)
    data = data.T
    data[data == 0] = np.nan

    # Plot 4 bans of multi raster
    fig = plt.subplots(figsize=(13, 7))

    plt.subplot(221)
    plt.imshow(data[:, :, 0] / 255)
    title('Blue')
    plt.colorbar

    plt.subplot(222)
    plt.imshow(data[:, :, 1] / 255)
    title('Green')
    plt.colorbar

    plt.subplot(223)
    plt.imshow(data[:, :, 2] / 255)
    title('Red')
    plt.colorbar

    plt.subplot(224)
    plt.imshow(data[:, :, 3] / 255)
    title('NIR')
    plt.colorbar

    plt.show()

    # Emtpy matrices that will store the TOA and surface reflectance data
    refl_TOA = np.zeros(data.shape)
    refl_surf = np.zeros(data.shape)

    for i in range(4):
        im = data[:, :, i]
        # Calculate DN to radiance
        L = (im * absFactors[i]) / effBandWidths[i]
        L = np.squeeze(L)
        # Calculates the theoretical radiance of a dark object as 1% of the max possible radiance
        L1percent = (0.01 * ESUN[i] * np.cos(np.deg2rad(theta))) / (dES ** 2 * math.pi)
        # Find darkest pixel in image
        Lmini = np.nanmin(L)
        # The difference between the theoretical 1% radiance of a dark object and the radiance of the darkest image pixel is due to the atm (empirical)
        Lhaze = Lmini - L1percent
        # TOA reflectance
        refl_TOA[:, :, i] = (L * math.pi * dES ** 2) / (ESUN[i] * np.cos(np.deg2rad(theta)))
        # Surface reflectance
        refl_surf[:, :, i] = (math.pi * (L - Lhaze) * dES ** 2) / (ESUN[i] * np.cos(np.deg2rad(theta)))

    # Save to rasters
    refl_TOA = np.float32(refl_TOA)
    output_TOA = drv.Create(out_raster_TOA, x, y, n, gdal.GDT_Float32)
    if output_TOA is None:
        print('The output raster could not be created')
        sys.exit(-1)
    output_TOA.SetGeoTransform(referenceTrans)
    output_TOA.SetProjection(referenceProj)

    refl_TOA = refl_TOA.T
    for i, image in enumerate(refl_TOA, 1):
        output_TOA.GetRasterBand(i).WriteArray(image)
        output_TOA.GetRasterBand(i).SetNoDataValue(0)

    output_TOA = None

    refl_surf = np.float32(refl_surf)
    output_surf = drv.Create(out_raster_surf, x, y, n, gdal.GDT_Float32)
    if output_surf is None:
        print('The output raster could not be created')
        sys.exit(-1)
    output_surf.SetGeoTransform(referenceTrans)
    output_surf.SetProjection(referenceProj)

    refl_surf = refl_surf.T
    for i, image in enumerate(refl_surf, 1):
        output_surf.GetRasterBand(i).WriteArray(image)
        output_surf.GetRasterBand(i).SetNoDataValue(0)

    output_surf = None

    return output_TOA, output_surf


# Calculate NDWI
def NDWI(filePath, outFilePath):
    # Load multispectral raster
    in_ras = gdal.Open(filePath, gdal.GA_ReadOnly)

    # Fetch resolution
    ncol = in_ras.RasterXSize
    nrow = in_ras.RasterYSize

    # Fetch projection and extent
    proj = in_ras.GetProjectionRef()
    ext = in_ras.GetGeoTransform()

    # Check raster was opened
    if in_ras is None:
        print('The raster could not be opened')
        sys.exit(-1)

    # Create new raster
    memory_driver = gdal.GetDriverByName('GTiff')
    out_ras = memory_driver.Create(outFilePath, ncol, nrow, 1, gdal.GDT_Float32)

    # Check output raster was created
    if out_ras is None:
        print('The output raster could not be created')
        sys.exit(-1)

    # Set projection and extent
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(ext)

    # Get hold of RED and NIR image bands
    green_band = in_ras.GetRasterBand(2).ReadAsArray()
    nir_band = in_ras.GetRasterBand(4).ReadAsArray()

    # Can't divide by 0, so they can't add up to 0
    mask = np.not_equal(green_band + nir_band, 0)

    # Request exceptions to be ignored or it does not work
    with np.errstate(invalid='ignore', divide='ignore'):
        NDWI = np.choose(mask, (-99, (green_band - nir_band) / (green_band + nir_band)))

    out_ras.GetRasterBand(1).WriteArray(NDWI)

    out_ras = None

    return out_ras


# Calculate NDVI
def NDVI(filePath, outFilePath):
    # Load multispectral raster
    in_ras = gdal.Open(filePath, gdal.GA_ReadOnly)

    # Fetch resolution
    ncol = in_ras.RasterXSize
    nrow = in_ras.RasterYSize

    # Fetch projection and extent
    proj = in_ras.GetProjectionRef()
    ext = in_ras.GetGeoTransform()

    # Check raster was opened
    if in_ras is None:
        print('The raster could not be opened')
        sys.exit(-1)

    # Create new raster
    memory_driver = gdal.GetDriverByName('GTiff')
    out_ras = memory_driver.Create(outFilePath, ncol, nrow, 1, gdal.GDT_Float32)

    # Check output raster was created
    if out_ras is None:
        print('The output raster could not be created')
        sys.exit(-1)

    # Set projection and extent
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(ext)

    # Get hold of RED and NIR image bands
    red_band = in_ras.GetRasterBand(3).ReadAsArray()
    nir_band = in_ras.GetRasterBand(4).ReadAsArray()

    # Can't divide by 0, so they can't add up to 0
    mask = np.not_equal(red_band + nir_band, 0)

    # Request exceptions to be ignored or it does not work
    with np.errstate(invalid='ignore', divide='ignore'):
        NDVI = np.choose(mask, (-99, (nir_band - red_band) / (nir_band + red_band)))

    out_ras.GetRasterBand(1).WriteArray(NDVI)

    out_ras = None

    return out_ras


## Stack all rasters to be used in classification
def stack_rasters(raster_list, names_list, out_raster):
    i = 0
    for raster in raster_list:
        names_list[i] = gdal.Open(raster)
        i += 1

    ncol = names_list[0].RasterXSize
    nrow = names_list[0].RasterYSize

    # Fetch projection and extent
    proj = names_list[0].GetProjectionRef()
    ext = names_list[0].GetGeoTransform()

    # Get all the bands
    bands = []
    for ras in names_list:
        for band in range(ras.RasterCount):
            band += 1
            bands.append(ras.GetRasterBand(band).ReadAsArray())

    # Change rgb raster data type to float
    for band in bands:
        band = np.float32(band)

    names_list[0].GetRasterBand(1).GetNoDataValue()

    # Create new raster
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_raster, ncol, nrow, len(bands), gdal.GDT_Float32)

    # Set projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Write bands
    i = 1
    for band in bands:
        out_raster_ds.GetRasterBand(i).WriteArray(band)
        out_raster_ds.GetRasterBand(i).SetNoDataValue(0)
        i += 1

    # Save raster
    out_raster_ds = None

    return out_raster_ds


## Creating new filed in shapefile (integer for classification)
def new_field(shapefile):
    dataset = ogr.Open(shapefile, update=1)
    field_defn = ogr.FieldDefn('ClassID', ogr.OFTInteger)
    layer.CreateField(field_defn)

    for feature in layer:
        wb = feature.GetField('Classname')
        if 'L' in wb:
            feature.SetField('ClassID', 1)
            layer.SetFeature(feature)
        elif 'P' in wb:
            feature.SetField('ClassID', 2)
            layer.SetFeature(feature)
        elif 'S' in wb:
            feature.SetField('ClassID', 3)
            layer.SetFeature(feature)
        else:
            feature.SetField('ClassID', 4)
            layer.SetFeature(feature)
        feature = layer.GetNextFeature()

    # Close shapefile
    dataset = None
    return dataset


## Create training raster
def training_raster(shapefile, ref_raster, out_raster):
    # Open training dataset from ArcGIS shapfile
    driver = gdal.GetDriverByName('ESRI Shapefile')
    dataset = ogr.Open(shapefile, update=1)

    # Make sure the dataset exists -- it would be None if we couildn't open it
    if not dataset:
        print('Error: could not open dataset')

    # Get driver from dataset file
    driver = dataset.GetDriver()
    print('Dataset driver is: {n}\n'.format(n=driver.name))

    # Check how many layers are contained in dataset Shapefile
    layer_count = dataset.GetLayerCount()
    print('The shapfeile has {n} layer(s)\n'.format(n=layer_count))

    # Check name of 1st layer
    layer = dataset.GetLayerByIndex(0)
    print('The layer is named: {n}\n'.format(n=layer.GetName()))

    # What is the layer's geoemtry?
    geometry = layer.GetGeomType()
    geometry_name = ogr.GeometryTypeToName(geometry)
    print("The layer's geometry is: {geom}\n".format(geom=geometry_name))

    # Get the spatial reference
    spatial_ref = layer.GetSpatialRef()

    # Export spatial reference to something that can be read
    # proj4 = spatial_ref.ExportProj4()
    # print('Layer projection is: {proj4}\n'.format(proj4=proj4))

    # How many features are in the layer
    feature_count = layer.GetFeatureCount()
    print('Layer has {n} features\n'.format(n=feature_count))

    ## How many fields are in the shapefile and what are their names?
    # Capture layer definition
    defn = layer.GetLayerDefn()

    # How many fields
    field_count = defn.GetFieldCount()
    print('Layer has {n} fields'.format(n=field_count))

    # Field names
    print('Their names are: ')
    for i in range(field_count):
        field_defn = defn.GetFieldDefn(i)
        print('\t{name} - {datatype}'.format(name=field_defn.GetName(), datatype=field_defn.GetTypeName()))

    # Open raster
    raster_ds = gdal.Open(ref_raster)

    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    raster_ds = None

    # Create the training raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_raster, ncol, nrow, 1, gdal.GDT_Byte)

    # Set the training image's projection and extent to our input raster's proejction and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Fill our output with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    b.SetNoDataValue(0)

    layer = dataset.GetLayerByIndex(0)

    # Rasterize the shapefile layer to our new dataset
    status = gdal.RasterizeLayer(out_raster_ds, [1], layer, burn_values=[0],
                                 options=['ALL_TOUCHED=TRUE', 'ATTRIBUTE=ClassID'])

    # Close dataset
    out_raster_ds = None

    if status != 0:
        a = "I don't think it worked..."
    else:
        a = "Success"

    return print(a)


def classification(img, trai, out):
    gdal.UseExceptions()
    gdal.AllRegister()

    # Read in raster and training raster
    img_ds = gdal.Open(img)
    trai_ds = gdal.Open(trai)

    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), \
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    # Read each band into an array
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    trai = trai_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    # Check NoData value = 0
    print(trai_ds.GetRasterBand(1).GetNoDataValue())

    # Find how many non-zero entries we have -- i.e how many trainign data samples?
    n_samples = (trai > 0).sum()
    print('We have {n} samples'.format(n=n_samples))

    # What are the classficiation labels?
    labels = np.unique(trai[trai > 0])
    print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))

    # We will need a 'X' matrix containing out features and a 'y' array containing our labels, these will have n sample rows
    X = img[trai > 0, :]
    y = trai[trai > 0]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    print('Our X matrix is sized: {sz}'.format(sz=X.shape))
    print('Our y matrix is sized: {sz}'.format(sz=y.shape))

    # Mask out clouds, cloud shadows and snow using Fmask

    # Initialise our model with 250 trees - optimize
    rf = RandomForestClassifier(n_estimators=75, oob_score=True)

    # Fit our model to training data
    rf = rf.fit(X, y)

    for b, imp in zip(range(img_ds.RasterCount), rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))

    # Take our full image and reshape it into a long 2d array (nrow * ncol, nband) for classification
    # Change to img.shape[2]-1 when using 4 bands (stacked NDWI + RGB)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])

    img_as_array = img[:, :, :img.shape[2]].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

    # Now predict for each pixel
    class_prediction = rf.predict(img_as_array)

    # F1 score
    # score = metrics.f1_score(y_test, class_prediction, pos_label=list(set(y_test)))
    # Overall accuracy
    # pscore = metrics.accuracy_score(y_test, )

    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)

    print('Our OOB prediction of accuracy is: {oob}'.format(oob=rf.oob_score_ * 100))

    trai_ds = None

    geo = img_ds.GetGeoTransform()
    proj = img_ds.GetProjectionRef()

    ncol = img_ds.RasterXSize
    nrow = img_ds.RasterYSize

    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(out, ncol, nrow, 1, gdal.GDT_Byte)

    ds.SetGeoTransform(geo)
    ds.SetProjection(proj)

    ds.GetRasterBand(1).WriteArray(class_prediction)

    ds = None


## Clean up salt and pepper from classification
def cleanup(in_raster, out_raster, disk_size):
    drv = gdal.GetDriverByName('GTiff')
    raster = gdal.Open(in_raster)
    ncol = raster.RasterXSize
    nrow = raster.RasterYSize
    # Fetch projection and extent
    proj = raster.GetProjectionRef()
    ext = raster.GetGeoTransform()

    raster_array = raster.ReadAsArray()

    # Create mask so that ponds are not included in filter
    mask_ = np.zeros(raster_array.shape, dtype=np.uint8)
    boolean = raster_array != 2
    mask_[boolean] = 1

    # Apply filter while ignoring ponds
    cleaned = skimage.filters.rank.modal(raster_array, skimage.morphology.disk(disk_size), mask=mask_)

    # If pond has been changed to tundra, change it back
    cleaned_with_ponds = np.where(((raster_array == 2) & (cleaned == 4)), 2, cleaned)

    # Fill nodata (0) with the most common surrounding value
    mask2 = cleaned_with_ponds == 0
    labels, count = label(mask2)
    arr_out = cleaned_with_ponds.copy()
    for idx in range(1, count + 1):
        hole = labels == idx
        surrounding_values = cleaned_with_ponds[binary_dilation(hole) & ~hole]
        most_frequent = Counter(surrounding_values).most_common(1)[0][0]
        arr_out[hole] = most_frequent

    out_ras = drv.Create(out_raster, ncol, nrow, 1, gdal.GDT_Byte)
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(ext)
    out_ras.GetRasterBand(1).WriteArray(arr_out)
    out_ras = None

    return out_ras


## Calculate area covered by each water body based on class pixels
def calc_wb_areas(in_raster, wbs_dict):
    # Open reprojected raster
    raster = gdal.Open(in_raster)
    raster_array = raster.GetRasterBand(1).ReadAsArray()

    # Sum pixel amount of each class
    count = Counter(raster_array.flatten())

    # Extract total of each class into list
    wbs = []
    wb_pixel_sum = []
    for class_, i in count.items():
        wb_pixel_sum.append(i)
        wbs.append(list(wbs_dict)[class_ - 1])

    # Get pixel size and calculate area
    gt = raster.GetGeoTransform()
    pixel_area = gt[1] * (-gt[5])

    total_area = sum(wb_pixel_sum) * pixel_area

    # Multiply each total in list with pixel_area to get area covered by wb in km2
    # Create list of names matching numbers
    wb_areas = {}
    for i, j in zip(wb_pixel_sum, wbs):
        wb_areas[j] = round(i * pixel_area * 10 ** -6, 2)

    return wb_areas, wb_pixel_sum, total_area


# Extract reflectance data in terms of class
def reflect_classified(class_ras, refl_ras, classes, in_bands, names):
    drv = gdal.GetDriverByName('GTiff')
    classif = gdal.Open(class_ras)
    reflectance = gdal.Open(refl_ras)

    clas_ = classif.GetRasterBand(1).ReadAsArray()

    i = 0
    for c in range(1, classes + 1):
        print(c)
        for b in range(1, reflectance.RasterCount + 1):
            print(b)
            in_bands[b] = reflectance.GetRasterBand(b).ReadAsArray()
            names[i] = in_bands[b][(clas_ == c)]
            i += 1

    return names


def final_cleanup(inras, polyras, outras):
    drv = gdal.GetDriverByName('GTiff')
    inraster = gdal.Open(inras)
    polygon = gdal.Open(polyras)

    ncol = inraster.RasterXSize
    nrow = inraster.RasterYSize
    # Fetch projection and extent
    proj = inraster.GetProjectionRef()
    ext = inraster.GetGeoTransform()

    in_array = inraster.GetRasterBand(1).ReadAsArray()
    pol_ras = polygon.GetRasterBand(1).ReadAsArray()

    new_array = in_array + pol_ras

    out = drv.Create(outras, ncol, nrow, 1, gdal.GDT_Byte)

    out.SetGeoTransform(ext)
    out.SetProjection(proj)
    out.GetRasterBand(1).WriteArray(new_array)
    out = None

    return out






