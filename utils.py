import os
import csv
import glob as _glob
import numpy as np
from PIL import Image
import random
import math
import cv2 as cv

from torch.utils.data import Dataset
from torchvision import transforms
import torch

import skimage.io as iio

csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

def writecsv(csvname,contents):
    f = open(csvname, 'a', newline='',encoding='UTF-8')
    wr = csv.writer(f)
    wr.writerow(contents)
    f.close()

def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def save_paths(img_pathcs, dir_path, shuffle):
    if len(img_pathcs) < 400:
        idx = np.arange(len(img_pathcs))
        idx = np.random.choice(idx, 400, True)
        img_pathcs = np.array(img_pathcs)[idx].tolist()

    if shuffle==True:
        random.shuffle(img_pathcs)
    num_save_pathcs = math.ceil(len(img_pathcs)/400)


    for i in range(num_save_pathcs):
        if (i+1)*400> len(img_pathcs):
            save_targets = img_pathcs[len(img_pathcs)-400 : len(img_pathcs)]
        else:
            save_targets = img_pathcs[i*400:(i+1)*400]
        save_img = np.zeros((540, 540, 3))
        iter_idx = 0
        for sav_hidx in range(20):
            for sav_widx in range(20):
                save_img[(sav_hidx*27):((sav_hidx+1)*27),(sav_widx*27):((sav_widx+1)*27) ] = save_targets[iter_idx]
                iter_idx += 1
        cv.imwrite(dir_path+dir_path.split('/')[-2]+'_'+str(i)+'.png', save_img)


def save_paths_MIL(img_pathcs, dir_path, shuffle):
    if len(img_pathcs) < 400:
        idx = np.arange(len(img_pathcs))
        idx = np.random.choice(idx, 400, True)
        img_pathcs = np.array(img_pathcs)[idx].tolist()

    if shuffle==True:
        random.shuffle(img_pathcs)
    num_save_pathcs = math.ceil(len(img_pathcs)/400)


    for i in range(num_save_pathcs):
        if (i+1)*400> len(img_pathcs):
            save_targets = img_pathcs[len(img_pathcs)-400 : len(img_pathcs)]
        else:
            save_targets = img_pathcs[i*400:(i+1)*400]
        save_img = np.zeros((108, 108, 3))
        iter_idx = 0
        for iidx in range(25):
            for sav_hidx in range(4):
                for sav_widx in range(4):
                    save_img[(sav_hidx*27):((sav_hidx+1)*27),(sav_widx*27):((sav_widx+1)*27) ] = save_targets[iter_idx]
                    iter_idx += 1
            cv.imwrite(dir_path+dir_path.split('/')[-2]+'_'+str(i)+'_'+str(iidx)+'.png', save_img)

### 128x128 용
### 27x27 로 잘라서 합친것이 성능이 좋지 않아서 크게 자르면 좋아지나 공부하려고 진행
def save_paths_MIL_128x128(img_pathcs, dir_path, shuffle):
    for idx, img in enumerate(img_pathcs):
        cv.imwrite(dir_path+dir_path.split('/')[-2]+'_'+str(idx)+'.png', img)
