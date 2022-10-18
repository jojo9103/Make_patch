import os
import numpy as np
import openslide
import scipy
import time
import matplotlib.pyplot as plt
from PIL import Image
import concurrent.futures
from itertools import repeat
from tqdm import tqdm
import pandas as pd
from skimage import transform

rela_path='../../'
import sys
#sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
from torchstain import MacenkoNormalizer

def wsi_coarse_level(Slide,Magnification,Stride,tol=0.02):
    # get slide dimensions, zoom levels, and objective information
    Factors = Slide.level_downsamples
    Objective = float(Slide.properties[
                          openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    Available = tuple(Objective / x for x in Factors)
    Mismatch = tuple(x - Magnification for x in Available)
    AbsMismatch = tuple(abs(x) for x in Mismatch)
    if min(AbsMismatch) <= tol:
        Level = int(AbsMismatch.index(min(AbsMismatch)))
        Factor = 1
    else:
        if min(Mismatch) < 0:  # determine is there is magnifications below 2.5x
            # pick next lower level, upsample
            Level = int(min([i for (i, val) in enumerate(Mismatch) if val < 0]))
        else:
            # pick next higher level, downsample
            Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))

        Factor = Magnification / Available[Level]

    Tout = [round(Stride[0]*Magnification/Objective), round(Stride[0]*Magnification/Objective)]


    return Level,Tout,Factor

def parallel_tiling(i,X,Y,dest_imagePath,img_name,Stride,File,color_norm):
    Slide = openslide.OpenSlide(File)

    for j in range(X.shape[1] - 1):
        Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
        Tile = np.asarray(Tile)
        Tile = Tile[:, :, :3]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[1] * 0.3: <- 원래 값
                    tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'

            if color_norm == True:
                try:
                    Tile,_,_ = macenko_norm(Tile)
                except:
                    print('i=%d,j=%d' % (i, j))
                    continue

            img = Image.fromarray(Tile)
            img.save(dest_imagePath + tile_name)

            # for debug
            # if debug_g==True:
            #     pred_gg[i,j]=255

def parallel_tiling_roi(i,X,Y,dest_imagePath,img_name,Stride,File,color_norm,roi_mask):
    Slide = openslide.OpenSlide(File)

    for j in range(X.shape[1] - 1):
        Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
        Tile = np.asarray(Tile)
        Tile = Tile[:, :, :3]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 1 and bn < Stride[0] * Stride[
            1] * 0.3 and roi_mask[i,j]==1:
            #print(X[i,j],Y[i,j])
            tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'

            if color_norm == True:
                try:
                    Tile,_,_ = macenko_norm(Tile)
                except:
                    print('i=%d,j=%d' % (i, j))
                    continue

            img = Image.fromarray(Tile)
            img.save(dest_imagePath + tile_name)


def wsi_tiling(File,dest_imagePath,img_name,Tile_size,color_norm=False, tumor_mask=None, debug=False,parallel_running=True):
    since = time.time()
    # open image
    Slide = openslide.OpenSlide(File)

    xr = float(Slide.properties['openslide.mpp-x'])  # pixel resolution at x direction
    yr = float(Slide.properties['openslide.mpp-y'])  # pixel resolution at y direction
    # generate X, Y coordinates for tiling
    Stride = [round(Tile_size[0] / xr), round(Tile_size[1] / yr)]
    Dims = Slide.level_dimensions
    X = np.arange(0, Dims[0][0] + 1, Stride[0])
    Y = np.arange(0, Dims[0][1] + 1, Stride[1])
    X, Y = np.meshgrid(X, Y)

    if debug==True:
        pred_g = np.zeros((X.shape[0] - 1, X.shape[1] - 1, 3), 'uint8')
        global pred_gg
        pred_gg=pred_g

        global debug_g
        debug_g=debug

    if parallel_running==True and tumor_mask==None:
        for i in range(X.shape[0]-1):
            parallel_tiling(i,X,Y,dest_imagePath,img_name,Stride, File, color_norm)
    elif parallel_running==True and tumor_mask!=None:
        tumor_mask=plt.imread(tumor_mask+img_name[:-5]+'.png')
        tumor_mask=transform.resize(tumor_mask,(X.shape[0]-1,X.shape[1]-1),order=0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
             for _ in executor.map(parallel_tiling_roi, list(range(X.shape[0]-1)), repeat(X), repeat(Y), repeat(dest_imagePath),repeat(img_name),
                                   repeat(Stride),repeat(File),repeat(color_norm),repeat(tumor_mask)):
                 pass
    else:
        for i in range(150,X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                    Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
                    Tile = np.asarray(Tile)
                    Tile = Tile[:, :, :3]
                    bn=np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
                    if (np.std(Tile[:,:,0])+np.std(Tile[:,:,1])+np.std(Tile[:,:,2]))/3>18 and bn<Stride[0]*Stride[1]*0.3:
                        tile_name=img_name.split('.')[0]+'_'+str(X[i,j])+'_'+str(Y[i,j])+'_'+str(Stride[0])+'_'+str(Stride[1])+'_'+'.png'
                        img = Image.fromarray(Tile)
                        img.save(dest_imagePath+tile_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
 
