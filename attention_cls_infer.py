### 이미지가 너무너무 커서 ROI 영역 찾은 이후에

from utils import *
from models import *
from datasets import *

import pandas as pd
from tqdm.auto import tqdm
import random
from sklearn.model_selection import StratifiedKFold

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import timm

# device='cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 시드(seed) 설정
RANDOM_SEED = 199002
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

csv_sol = pd.read_csv('datas/sample_submission.csv')

### segmentation 할 때와 다르게 scikit image 로 부를것이기 때문에  bgr ==> rgb 순으로 해주고 normalize
tfrom_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.811, 0.661, 0.78 ), std=(0.0353, 0.067, 0.04))
])

weight_path = 'outputs/MIL_ATTN_128X128/'

test_dirs = glob('datas/test_imgs_patchs_shuffle/', '*')
test_dirs.sort()
print(test_dirs)
tarb_csv = pd.read_csv('datas/test_tabular_v1.csv').values

# data_id, tabular, lbl_ls, train_tf, transform=None

test_ds = custom_ds_classification_MIL_TRAIN(tarb_csv[:,0], tarb_csv[:,2:], None, False ,tfrom_test)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

model = MIL_Attention()

weights =['0fold_7epoch_classifier.pt']
en_M = ensemble_M(model, weights, weight_path)
en_M = en_M.to(device)

idx = 0
model.eval()
with torch.no_grad():
    for imgs, tarb_data, dataid in tqdm(iter(test_loader)):
        imgs = imgs.to(device)
        tarb_data = tarb_data.to(device)

        pred_label = 0

        scls1=[]
        vots1=[]
        if idx==26:
            print('aa')

        for v_idx in range(int(imgs.shape[1] / 20)):
            score_ls, vote_ls = en_M(imgs[:, v_idx * 20 : (v_idx+1) * 20, :, :, :], tarb_data)

            scls1 += score_ls
            vots1 += vote_ls

        ## scroe 기반 평균으로 ensemble(not vote)
        # if np.mean(np.array(scls1))>=0.5:
        #     pred_label = 1


        ## voting 기반으로 ensemble( 한개라도 암이 있다고 판단하면 암으로 ver)
        val, counts = np.unique(np.array(vots1),return_counts=True)
        if val.shape[0] == 2:
            if counts[1] > 0:
                pred_label = 1
        else:
            pred_label = val[0]

        csv_sol['N_category'][idx] = pred_label

        idx += 1

csv_sol.to_csv('128128new.csv', index=False)
