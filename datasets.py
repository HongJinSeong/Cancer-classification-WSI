import numpy as np
from PIL import Image
import cv2 as cv
from torch.utils.data import Dataset
import random
import torch
import math
from utils import *
import skimage.io as iio
from itertools import product


class custom_ds_classification(Dataset):
    def __init__(self, img_ls, lbl_ls,train_tf ,transform=None):
        self.img_ls = img_ls
        self.lbl_ls = lbl_ls
        self.transform = transform
        self.train_tf = train_tf


    def __len__(self):
        return len(self.img_ls)


    def __getitem__(self, idx):
        img_path = self.img_ls[idx].replace('\\','/')

        img = iio.imread(img_path)

        img = Image.fromarray(img)

        img = self.transform(img)

        if self.train_tf == True:
            return img,  self.lbl_ls[idx]
        else:
            if self.lbl_ls==None:
                return img
            else:
                return img, self.lbl_ls[idx], img_path



class custom_ds_classification_MIL_TRAIN(Dataset):
    def __init__(self, data_id, tabular, lbl_ls, train_tf, transform=None):
        self.data_id = data_id
        self.tabular = tabular
        self.lbl_ls = lbl_ls
        self.transform = transform
        self.train_tf = train_tf

        ## standard scailing
        self.mean = [0.43, 0.21, 2.13, 1.89, 2.59, 2.13, 1.41, 1.38, 1.25, 1.15, 0.75, 0.82, 0.63]
        self.std = [0.17, 0.13, 0.66, 0.71, 0.65, 0.66, 0.69, 0.55, 0.93, 0.36, 0.96, 0.38, 0.48]


    def __len__(self):
            return len(self.data_id)

    #### bag size == 20
    #### 데이터 조합의 수가 많기 때문에 random하게 instance들을 선택하고 데이터 순회를 늘림 (train set 갯수 * 5로 epoch 진행)
    def __getitem__(self, idx):
        if self.train_tf==True:
            iidx = idx % len(self.data_id)
        else:
            iidx = idx
        dataid = self.data_id[iidx]
        if self.lbl_ls != None:
            img_paths = glob('datas/train_imgs_patchs_MIL_128x128/' + dataid, '*')
        else:
            img_paths = glob('datas/test_imgs_patchs_MIL_128x128/' + dataid, '*')

        img_paths.sort(key=len)

        if self.train_tf == True:
            if len(img_paths)<40:
                img_paths += [random.choice(img_paths) for i in range(40 - len(img_paths))]
        else:
            if len(img_paths) < 40:
                img_paths += [img_paths[i % len(img_paths)] for i in range(40 - len(img_paths))]

        tarb_data = self.tabular[iidx].copy()

        ms_idx=0
        for tidx in range(2,tarb_data.shape[0]):
            tarb_data[tidx] = (tarb_data[tidx] - self.mean[ms_idx]) / self.std[ms_idx]
            ms_idx += 1

        tarb_data = tarb_data.astype(float)


        if self.train_tf == True:
            ### intance를 random하게 20개 choice
            inst_set = random.sample(img_paths,40)

            bag = []

            for path in inst_set:
                img = iio.imread(path)
                img = Image.fromarray(img)
                bag.append(self.transform(img).numpy())

            return np.array(bag), self.lbl_ls[iidx], tarb_data
        else:
            ### intance를 일단 bag에 다담아 두고 순차적으로 25개씩 꺼내먹음
            valid_bag = []

            for path in img_paths[0:40]:
                img = iio.imread(path)
                img = Image.fromarray(img)
                valid_bag.append(self.transform(img).numpy())
            if self.lbl_ls != None:
                return np.array(valid_bag), self.lbl_ls[iidx], tarb_data
            else:
                return np.array(valid_bag), tarb_data, dataid

class custom_ds_classification_MIL_test(Dataset):
    def __init__(self, img_ls, lbl_ls,transform=None):
        self.img_ls = img_ls
        self.lbl_ls = lbl_ls
        self.transform = transform

    def __len__(self):
        return len(self.img_ls)

    def __getitem__(self, idx):
        img_path = self.img_ls[idx].replace('\\','/')

        img = iio.imread(img_path)

        h, w, _ = img.shape

        d = 108

        img = Image.fromarray(img)

        grid = product(range(0, h - h % d, d), range(0, w - w % d, d))

        crop_ls = []
        for i, j in grid:
            box = (j, i, j + d, i + d)
            crop_ls.append(img.crop(box))

        for c_idx in range(len(crop_ls)):
            crop_ls[c_idx] = self.transform(crop_ls[c_idx]).numpy()

        if self.train_tf == True:
            return np.array(crop_ls), self.lbl_ls[idx]
        else:
            return np.array(crop_ls), self.lbl_ls[idx], img_path


class custom_ds_classification_multimodal_MIL(Dataset):
    def __init__(self, img_ls,tarb_ls, lbl_ls,train_tf ,transform=None):
        self.img_ls = img_ls
        self.tarb_ls = tarb_ls
        self.lbl_ls = lbl_ls
        self.transform = transform
        self.train_tf = train_tf

        ## standard scailing
        self.mean = [0.43, 0.21, 2.13, 1.89, 2.59, 2.13, 1.41, 1.38, 1.25, 1.15, 0.75, 0.82, 0.63]
        self.std = [0.17, 0.13, 0.66, 0.71, 0.65, 0.66, 0.69, 0.55, 0.93, 0.36, 0.96, 0.38, 0.48]


    def __len__(self):
        return len(self.img_ls)


    def __getitem__(self, idx):
        img_path = self.img_ls[idx].replace('\\','/')

        img = iio.imread(img_path)

        img = Image.fromarray(img)

        img = self.transform(img)

        tarb_data = self.tarb_ls[idx].copy()

        ms_idx=0
        for tidx in range(2,tarb_data.shape[0]):
            tarb_data[tidx] = (tarb_data[tidx] - self.mean[ms_idx]) / self.std[ms_idx]
            ms_idx += 1

        tarb_data = tarb_data.astype(float)

        if self.train_tf == True:
            return img, tarb_data, self.lbl_ls[idx], img_path
        else:
            return img, tarb_data, self.lbl_ls[idx], img_path


#region segmentation 용
class madison_seg_DS_real(Dataset):
    def __init__(self, lbl_ls, size, aug, train, transform=None):
        self.lbl_ls = lbl_ls
        self.size = size
        self.aug = aug
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.lbl_ls)

    def random_float(self,shape=[], minval=0.0, maxval=1.0):
        rnd = np.random.uniform(low=minval, high=maxval, size=shape)

        return rnd


    # 16bit로 표현되어 있기 때문에 16비트에 맞게 clipping 적용
    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 2**16-1)

        return img.astype(np.uint16)

    def _get_shear_matrix(self,x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]])

        return shear_matrix

    def shear(self,img,lbl,max_shear_degree,size):
        x_degree = np.random.uniform(-max_shear_degree,
                                  max_shear_degree)

        y_degree = np.random.uniform(-max_shear_degree,
                                  max_shear_degree)

        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        warp_matrix = (shear_matrix)

        img = cv.warpPerspective(
            img,
            warp_matrix,
            dsize=size,
            borderValue=(0))

        lbl = cv.warpPerspective(
            lbl,
            warp_matrix,
            dsize=size,
            borderValue=(255),
            flags=cv.INTER_NEAREST)

        return img, lbl

    def _get_translation_matrix(self, x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]])
        return translation_matrix

    def translation(self,img,lbl,max_translate_ratio,width,height,size):
        trans_x = np.random.uniform(0.1 - max_translate_ratio,
                                 0.1 + max_translate_ratio) * width
        trans_y = np.random.uniform(0.1 - max_translate_ratio,
                                 0.1 + max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (translate_matrix)

        img = cv.warpPerspective(
            img,
            warp_matrix,
            dsize=size,
            borderValue=(0))

        lbl = cv.warpPerspective(
            lbl,
            warp_matrix,
            dsize=size,
            borderValue=(255),
            flags=cv.INTER_NEAREST)

        return img ,lbl

    def resize_by_ratio(self, img, zoom_factor=2,inter=cv.INTER_CUBIC):

        return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor,interpolation=inter)

    def randomcrop(self, img, lbl):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.size[0], 0)
        margin_w = max(img.shape[1] - self.size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.size[1]

        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        lbl = lbl[crop_y1:crop_y2, crop_x1:crop_x2, ...]

        return img, lbl

    def cutout(self,img, n_holes, maxratio):
        if len(img.shape)==3:
            h, w, c = img.shape
        else:
            h, w= img.shape
        n_holes = np.random.randint(n_holes[0], n_holes[1] + 1)

        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)

            xratio = np.random.uniform(0.05, maxratio)
            yratio = np.random.uniform(0.05, maxratio)

            cutout_w = int(xratio * w)
            cutout_h = int(yratio * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            if len(img.shape)==3:
                img[y1:y2, x1:x2,:] = 0
            else:
                img[y1:y2, x1:x2] = 0

        return img

    # 0 channel ==> large_bowel  /  1 channel ==> small_bowel  /  2 channel ==> stomach

    def __getitem__(self, idx):
        if self.train==True:

            lbl_path = self.lbl_ls[idx].replace('\\','/')
            img_path = lbl_path.replace('train_masks','train_imgs')

            img = cv.imread(img_path)
            lbl = cv.imread(lbl_path, cv.IMREAD_GRAYSCALE)

            img = cv.resize(img, self.size, interpolation=cv.INTER_CUBIC)
            lbl = cv.resize(lbl, self.size, interpolation=cv.INTER_NEAREST)

            # cv.imshow('img',img)
            # cv.imshow('lbl',lbl)
            # cv.waitKey(0)

            if self.aug == True:
                # horizontal flip( img , lbl )
                if self.random_float() > 0.5:
                    img = cv.flip(img, 1)
                    lbl = cv.flip(lbl, 1)

                # translate ( img , lbl )
                if self.random_float() > 0.5:
                    max_translate_ratio = 0.05
                    img, lbl = self.translation(img, lbl, max_translate_ratio, self.size[0], self.size[1], self.size)

                # rotate( img , lbl )
                if self.random_float() > 0.5:
                    rotate_angle=np.random.randint(-30,30)
                    img_mat = cv.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), rotate_angle, 1)
                    lbl_mat = cv.getRotationMatrix2D((lbl.shape[0] / 2, lbl.shape[1] / 2), rotate_angle, 1)
                    img = cv.warpAffine(img, img_mat, (0, 0), borderValue=(0))
                    lbl = cv.warpAffine(lbl, lbl_mat, (0, 0), borderValue=(255), flags=cv.INTER_NEAREST) ## label은 값의 변화가 없도록 INTER_NEAREST

                # shear ( img , lbl )
                if self.random_float() > 0.5:
                    max_shear_degree = 2.0
                    img, lbl = self.shear(img,lbl, max_shear_degree, self.size)

                # zoom ( img , lbl )
                if self.random_float() > 0.5:
                    random_zoom_factor = np.random.uniform(1,1.3)
                    img = self.resize_by_ratio(img, random_zoom_factor)
                    lbl = self.resize_by_ratio(lbl, random_zoom_factor, cv.INTER_NEAREST)
                    img, lbl = self.randomcrop(img,lbl)

                # cutout ( img )
                if self.random_float() > 0.5:
                    max_holes = np.random.randint(2,5)
                    n_holes = (1, max_holes)
                    img = self.cutout(img,n_holes,0.3)

        else:
            lbl_path = self.lbl_ls[idx].replace('\\', '/')
            img_path = lbl_path.replace('train_masks', 'train_imgs')

            img = cv.imread(img_path)
            lbl = cv.imread(lbl_path, cv.IMREAD_GRAYSCALE)

            img = cv.resize(img, self.size, interpolation=cv.INTER_CUBIC)
            lbl = cv.resize(lbl, self.size, interpolation=cv.INTER_NEAREST)

        # img = img.astype(np.float32)
        # 16 bit라서 값이 pytorch 함수만으로 정규화 제대로 안되서 minmax scaling (0 ~ 1) 때려줌
        # img = (img - np.min(img)) / (np.max(img)-np.min(img))
        img = self.transform(img)
        lbl[np.where(lbl != 255)] = 1
        lbl[np.where(lbl != 1)] = 0

        return img , lbl

class madison_seg_DS_real_test(Dataset):
    def __init__(self, img_ls, size, transform=None):
        self.img_ls = img_ls
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.img_ls)



    def __getitem__(self, idx):
        img_path = self.img_ls[idx].replace('\\','/')

        img_origin = cv.imread(img_path)

        img = cv.resize(img_origin, self.size, interpolation=cv.INTER_CUBIC)

        img = self.transform(img)

        return img, img_origin, img_path.split('/')[-1]

#endregion