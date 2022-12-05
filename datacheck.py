from utils import *
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd

### seg train 통계치(전체이미지)
# img_paths = glob('datas/train_imgs','*')
#
# r=[]
# g=[]
# b=[]
# h=[]
# w=[]
# for pat in img_paths:
#     img1 = cv.imread(pat)
#     r.append(np.mean(img1[:, :, 0])/255)
#     g.append(np.mean(img1[:, :, 1])/255)
#     b.append(np.mean(img1[:, :, 2])/255)
#     h.append(img1.shape[0])
#     w.append(img1.shape[1])
# print(np.mean(np.array(r)))
# print(np.std(np.array(r)))
#
# print(np.mean(np.array(g)))
# print(np.std(np.array(g)))
#
# print(np.mean(np.array(b)))
# print(np.std(np.array(b)))
#
# print(np.min(np.array(h)))
# print(np.max(np.array(h)))
#
# print(np.min(np.array(w)))
# print(np.max(np.array(w)))


### classifier train 통계치(patch 이미지)
# img_paths = glob('datas/train_imgs_patchs','*/*')
#
# r=[]
# g=[]
# b=[]
#
# for pat in img_paths:
#     img1 = cv.imread(pat)
#     r.append(np.mean(img1[:, :, 0])/255)
#     g.append(np.mean(img1[:, :, 1])/255)
#     b.append(np.mean(img1[:, :, 2])/255)
#
# print(np.mean(np.array(r)))
# print(np.std(np.array(r)))
#
# print(np.mean(np.array(g)))
# print(np.std(np.array(g)))
#
# print(np.mean(np.array(b)))
# print(np.std(np.array(b)))
#
#
# print('aaa')

## 나이 5살 단위로 categorical 하게 처리해줌
## 수술 년월일 제외
## 진단명 categorical
## 암의 위치 categorical
## 암의 갯수 categorical
## 암의 장경(사이즈) numerical ==> nan값은 평균으로 처리
categorical=['나이','진단명','암의 위치','NG','HG','HG_score_1','HG_score_2','HG_score_3','DCIS_or_LCIS_여부','T_category','ER','PR','HER2_IHC']
numerical=['암의 장경','ER_Allred_score','PR_Allred_score','KI-67_LI_percent','암의 개수']

csv = pd.read_csv('datas/train.csv')

age_div = np.arange(25,95,5)

for i in range(0,age_div.shape[0]):
    if i != 13:
        csv['나이'][(csv['나이'] >= age_div[i]) & (csv['나이'] < age_div[i+1])] = i
    else:
        csv['나이'][(csv['나이'] >= age_div[i])] = i

csv['진단명'] = csv['진단명'] - 1
csv['암의 위치'] = csv['암의 위치'] - 1

### 평균값으로 nan 채움(18.375563909774435)
csv['암의 장경'][csv['암의 장경'].isna()] = csv['암의 장경'].mean()

### 가장 많이 등장하는 category기준으로 채움
csv['NG'][csv['NG'].isna()] = 2
csv['NG']=csv['NG']-1

csv['HG'][csv['HG'].isna()] = 2
csv['HG'] = csv['HG'] - 1

csv['HG_score_1'][csv['HG_score_1'].isna()] = 3
csv['HG_score_1'] = csv['HG_score_1'] - 1
csv['HG_score_2'][csv['HG_score_2'].isna()] = 2
csv['HG_score_2'] = csv['HG_score_2'] - 1
csv['HG_score_3'][csv['HG_score_3'].isna()] = 1
csv['HG_score_3'] = csv['HG_score_3'] - 1

csv['T_category'][csv['T_category'].isna()] = 1

csv['ER'][csv['ER'].isna()] = 1
csv['ER_Allred_score'][(csv['ER']==0) & (csv['ER_Allred_score'].isna())] = 3
csv['ER_Allred_score'][(csv['ER']==1) & (csv['ER_Allred_score'].isna())] = 7

csv['PR'][csv['PR'].isna()] = 1
csv['PR_Allred_score'][(csv['PR'] == 0) & (csv['PR_Allred_score'].isna())] = 2
csv['PR_Allred_score'][(csv['PR'] == 1) & (csv['PR_Allred_score'].isna()) | (csv['PR_Allred_score'][(csv['PR'] == 1)]>8)] = 7

### 평균값으로 nan 채움 (18.037228758169935)
csv['KI-67_LI_percent'][csv['KI-67_LI_percent'].isna()] = csv['KI-67_LI_percent'].mean()


csv['HER2'][csv['HER2'].isna()] = 0

### HER2==0 이면 1
### HER2==1 이면 2
csv['HER2_IHC'][(csv['HER2']==0) & (csv['HER2_IHC'].isna())] = 1
csv['HER2_IHC'][(csv['HER2']==1) & (csv['HER2_IHC'].isna())] = 2

csv['BRCA_mutation'][csv['BRCA_mutation'].isna()] = 4


csv = csv[['ID', 'img_path', 'mask_path']+categorical+numerical+['N_category']]
csv.to_csv('datas/train_new.csv', index=False)
#

#######################
####### test ##########
csv = pd.read_csv('datas/test.csv')


age_div = np.arange(25,95,5)

for i in range(0,age_div.shape[0]):
    if i != 13:
        csv['나이'][(csv['나이'] >= age_div[i]) & (csv['나이'] < age_div[i+1])] = i
    else:
        csv['나이'][(csv['나이'] >= age_div[i])] = i

csv['진단명'] = csv['진단명'] - 1
csv['암의 위치'] = csv['암의 위치'] - 1

### 평균값으로 nan 채움(18.375563909774435)
csv['암의 장경'][csv['암의 장경'].isna()] = 18.3755

### 가장 많이 등장하는 category기준으로 채움
csv['NG'][csv['NG'].isna()] = 2
csv['NG']=csv['NG']-1

csv['HG'][csv['HG'].isna()] = 2
csv['HG'] = csv['HG'] - 1

csv['HG_score_1'][csv['HG_score_1'].isna()] = 3
csv['HG_score_1'] = csv['HG_score_1'] - 1
csv['HG_score_2'][csv['HG_score_2'].isna()] = 2
csv['HG_score_2'] = csv['HG_score_2'] - 1
csv['HG_score_3'][csv['HG_score_3'].isna()] = 1
csv['HG_score_3'] = csv['HG_score_3'] - 1

csv['T_category'][csv['T_category'].isna()] = 1

csv['ER'][csv['ER'].isna()] = 1
csv['ER_Allred_score'][(csv['ER']==0) & (csv['ER_Allred_score'].isna())] = 3
csv['ER_Allred_score'][(csv['ER']==1) & (csv['ER_Allred_score'].isna())] = 7

csv['PR'][csv['PR'].isna()] = 1
csv['PR_Allred_score'][(csv['PR'] == 0) & (csv['PR_Allred_score'].isna())] = 2
csv['PR_Allred_score'][(csv['PR'] == 1) & (csv['PR_Allred_score'].isna()) | (csv['PR_Allred_score'][(csv['PR'] == 1)]>8)] = 7

### 평균값으로 nan 채움 (18.037228758169935)
csv['KI-67_LI_percent'][csv['KI-67_LI_percent'].isna()] = 18.03722


csv['HER2'][csv['HER2'].isna()] = 0

### HER2==0 이면 1
### HER2==1 이면 2
csv['HER2_IHC'][(csv['HER2']==0) & (csv['HER2_IHC'].isna())] = 1
csv['HER2_IHC'][(csv['HER2']==1) & (csv['HER2_IHC'].isna())] = 2

csv['BRCA_mutation'][csv['BRCA_mutation'].isna()] = 4

csv = csv[['ID', 'img_path']+categorical+numerical]
csv.to_csv('datas/test_new.csv', index=False)
print('ccc')