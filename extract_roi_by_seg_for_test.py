import cv2 as cv
import numpy as np
import math
import os

from utils import *


# 시드(seed) 설정
RANDOM_SEED = 2022
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


seg_paths = glob('outputs/segments/test_img/', '*')

dir_path = 'datas/test_imgs_patchs/'


for seg_pth in seg_paths:
    patchs_images = []
    seg_pth = seg_pth.replace('\\','/')
    origin_pth = seg_pth.replace('outputs/segments/test_img','datas/test_imgs')

    origin_img = cv.imread(origin_pth)
    seg = cv.imread(seg_pth)

    seg[np.where(seg == [255, 0, 0])] = 255
    seg[np.where(seg != [255, 255, 255])] = 0

    component_target = seg[:, :, 0]

    cnt, labels, stats, centroids = cv.connectedComponentsWithStats(component_target)
    patchs = 0
    for i in range(1,cnt):

        ##너비 높이가 최소 27은 넘는 것들로 patch get
        if stats[i][2]>=27 and stats[i][2]>=27:
            (xcen, ycen) = centroids[i]
            (x, y, w, h, area) = stats[i]

            # cv.imwrite('oriori.png', component_target[y:y+h, x:x+w])

            #center 좌표 기준 제일 첫 시작점
            new_start_x = int(xcen-13.5)
            new_start_y = int(ycen-13.5)
            if w > 100 and h > 100 :
                numboxx = int(w // 33.3)
                if numboxx % 2 == 0 : numboxx -= 1
                numboxy = int(h // 33.3)
                if numboxy % 2 == 0 : numboxy -= 1

                new_start_xs = []
                new_start_ys = []

                centerx = math.trunc(numboxx / 2)
                centery = math.trunc(numboxy / 2)
                for xidx in range(numboxx):
                    for yidx in range(numboxy):
                        new_start_xs.append(new_start_x - (27 * (centerx - xidx)))
                        new_start_ys.append(new_start_y - (27 * (centery - yidx)))
                for idx_box in range(numboxx * numboxy):
                    patch = component_target[new_start_ys[idx_box]:new_start_ys[idx_box] + 27, new_start_xs[idx_box]:new_start_xs[idx_box] + 27]

                    # cv.imwrite( str(idx_box)+'patch.png', origin_img[new_start_ys[idx_box]:new_start_ys[idx_box] + 27, new_start_xs[idx_box]:new_start_xs[idx_box] + 27])
                    chk_idx = np.where(patch==255)
                    if chk_idx[0].shape[0]/(27*27)>0.7:
                        if np.mean(origin_img[new_start_ys[idx_box]:new_start_ys[idx_box] + 27,new_start_xs[idx_box]:new_start_xs[idx_box] + 27]) < 230:
                            patchs_images.append(origin_img[new_start_ys[idx_box]:new_start_ys[idx_box] + 27, new_start_xs[idx_box]:new_start_xs[idx_box] + 27])
            else:
                patch = component_target[new_start_y:new_start_y+27, new_start_x:new_start_x+27]
                chk_idx = np.where(patch == 255)
                if chk_idx[0].shape[0] / (27 * 27) > 0.7:
                    if np.mean(origin_img[new_start_y:new_start_y+27, new_start_x:new_start_x+27])<230:
                        patchs_images.append(origin_img[new_start_y:new_start_y+27, new_start_x:new_start_x+27])
    save_dir = dir_path + seg_pth.split('/')[-1].split('.')[0] + '/'
    createFolder(save_dir)
    save_paths(patchs_images,save_dir,False)

