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

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

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

BATCH_SIZE = 1
EPOCHs = 200
outputpath='outputs/MIL_ATTN_128X128/'

### segmentation 할 때와 다르게 scikit image 로 부를것이기 때문에  bgr ==> rgb 순으로 해주고 normalize
tfrom_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine((0, 270),(0, 0.1)),
    transforms.RandomEqualize(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.811, 0.661, 0.78 ), std=(0.0353, 0.067, 0.04)),
    transforms.RandomErasing()
])

tfrom_test = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize(mean=(0.811, 0.661, 0.78 ), std=(0.0353, 0.067, 0.04))
])

train_ids = glob('datas/train_imgs_patchs_MIL_128x128','*')
for i in  range(len(train_ids)):
    train_ids[i] = train_ids[i].replace('\\','/').split('/')[-1]

train_ids = np.array(train_ids)
train_csv = pd.read_csv('datas/train.csv')
train_tabular_csv = pd.read_csv('datas/train_tabular_v1.csv')

for idx,values in enumerate(train_csv.values):
    if np.where(train_ids == values[0])[0].shape[0]==0:
        print(values[0])
        train_csv = train_csv.drop(idx)
        train_tabular_csv = train_tabular_csv.drop(idx)


# 10-fold 로 setting
# folder 기준으로 10fold 나누고 나눈 10-fold에서 폴더 별로 풀어서 train set / valid set 정의
kf = StratifiedKFold(n_splits=5, shuffle=True)

lbls = train_csv['N_category'].values


for kfold_idx, (index_kf_train, index_kf_validation) in enumerate(kf.split(lbls,lbls)):
    train_data = train_csv.iloc[index_kf_train]
    train_tabular = train_tabular_csv.iloc[index_kf_train]
    valid_data = train_csv.iloc[index_kf_validation]
    valid_tabular = train_tabular_csv.iloc[index_kf_validation]

    train_ds = custom_ds_classification_MIL_TRAIN(train_data.values[:, 0].tolist(), train_tabular.values[:, 3:-1], train_data.values[:, -1].tolist(), True, tfrom_train)
    valid_ds = custom_ds_classification_MIL_TRAIN(valid_data.values[:, 0].tolist(), valid_tabular.values[:, 3:-1], valid_data.values[:, -1].tolist(), False, tfrom_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0)

    model_ft = timm.create_model('convnext_tiny_in22ft1k', pretrained=True)
    train_nodes, eval_nodes = get_graph_node_names(model_ft)

    return_nodes = {
        train_nodes[277]: 'x1',
    }

    feature_extract = create_feature_extractor(model_ft, return_nodes)

    model = MIL_pretrained_CNN(feature_extract)
    #
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.000003)

    criterion = torch.nn.BCELoss()

    for ep in range(EPOCHs):
        model.train()
        train_loss = []
        train_acc = []

        train_idx=0
        for imgs, labels, tabular in tqdm(iter(train_loader)):
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(imgs,tabular)

            loss = criterion(outputs,labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            acc = torch.sum(torch.where(outputs >= 0.5, 1, 0)==labels.reshape(-1,1)) / labels.shape[0]

            train_acc.append(acc.item())

            train_idx+=1

        model.eval()
        valid_loss = []
        valid_acc = []
        with torch.no_grad():
            for imgs, labels, tabular in tqdm(iter(valid_loader)):
                imgs = imgs.to(device)
                tabular = tabular.to(device)
                labels = labels.float().to(device)

                pred_label=0

                for v_idx in range(int(imgs.shape[1]/40)):
                    outputs = model(imgs[:, v_idx * 40 : (v_idx+1) * 40, :, :, :], tabular)

                    loss = criterion(outputs, labels.unsqueeze(1))

                    valid_loss.append(loss.item())

                    if outputs >= 0.5:
                        pred_label=1
                acc = torch.sum(pred_label==labels.reshape(-1,1)) / labels.shape[0]

                valid_acc.append(acc.item())

        writecsv(outputpath + str(kfold_idx) + 'fold_acc_loss.csv',
                 [ep, np.mean(np.array(train_loss)), np.mean(np.array(train_acc)), np.mean(np.array(valid_loss)),
                  np.mean(np.array(valid_acc))])

        print(
            f'Epoch : [{ep}] Train Loss : [{np.mean(np.array(train_loss)):.5f}] Train acc : [{np.mean(np.array(train_acc)):.5f}] Val loss : [{np.mean(np.array(valid_loss)):.5f}] Val ACC : [{np.mean(np.array(valid_acc)):.5f}]')

        torch.save(model.state_dict(), outputpath + str(kfold_idx) + 'fold_' + str(ep) + 'epoch_classifier.pt')
