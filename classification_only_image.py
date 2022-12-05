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

BATCH_SIZE = 8
EPOCHs = 300
outputpath='outputs/simple_540/'

### segmentation 할 때와 다르게 scikit image 로 부를것이기 때문에  bgr ==> rgb 순으로 해주고 normalize
tfrom_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomAffine((0, 270),(0, 0.1)),
    transforms.RandomEqualize(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.811, 0.661, 0.78 ), std=(0.0353, 0.067, 0.04)),
    transforms.RandomErasing()
])

tfrom_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.811, 0.661, 0.78 ), std=(0.0353, 0.067, 0.04))
])

##BC_01_1205 제거
train_csv = pd.read_csv('datas/train.csv')


# 10-fold 로 setting
# folder 기준으로 10fold 나누고 나눈 10-fold에서 폴더 별로 풀어서 train set / valid set 정의
kf = StratifiedKFold(n_splits=10, shuffle=True)

lbls = train_csv['N_category'].values


for kfold_idx, (index_kf_train, index_kf_validation) in enumerate(kf.split(lbls,lbls)):
    train_data = train_csv.iloc[index_kf_train]
    valid_data = train_csv.iloc[index_kf_validation]

    train_img_ls = []
    train_labels = []

    for datas in train_data.values:
        tls = glob('datas/train_imgs_patchs_shuffle/'+datas[0], '*')
        cls = (np.ones(shape=(len(tls))) * datas[-1]).tolist()
        train_img_ls += tls
        train_labels += cls

    valid_img_ls = []
    valid_labels = []

    for datas in valid_data.values:
        tls = glob('datas/train_imgs_patchs_shuffle/' + datas[0], '*')
        cls = (np.ones(shape=(len(tls))) * datas[-1]).tolist()
        valid_img_ls += tls
        valid_labels += cls

    train_ds = custom_ds_classification(train_img_ls, train_labels, True, tfrom_train)
    valid_ds = custom_ds_classification(valid_img_ls, valid_labels, False, tfrom_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0)

    # 현재 BEST RESNET-50d 68%
    # model = custom_transformer_v2()
    # model = timm.create_model('convnext_tiny_in22ft1k', pretrained=True, drop_rate=0.1, num_classes=1)
    model = SimpleCNN()
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00001)
    criterion = nn.BCEWithLogitsLoss().to(device)


    for ep in range(EPOCHs):
        model.train()
        train_loss = []
        train_acc = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss_cls = criterion(outputs, labels.reshape(-1,1))
            loss_cls.backward()
            optimizer.step()
            train_loss.append(loss_cls.item())
            acc = torch.sum(torch.where(nn.Sigmoid()(outputs) >= 0.5, 1, 0)==labels.reshape(-1,1)) / labels.shape[0]

            train_acc.append(acc.item())

        model.eval()
        valid_loss = []
        valid_acc = []
        with torch.no_grad():
            for imgs, labels,img_path in tqdm(iter(valid_loader)):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)

                loss_cls = criterion(outputs, labels.reshape(-1,1))

                valid_loss.append(loss_cls.item())
                acc = torch.sum(torch.where(nn.Sigmoid()(outputs) >= 0.5, 1, 0)==labels.reshape(-1,1)) / labels.shape[0]

                valid_acc.append(acc.item())

                writecsv(outputpath + str(kfold_idx) + 'fold_'+str(ep)+'EP_preds.csv',
                         [img_path,nn.Sigmoid()(outputs).item(),labels.item()])

        writecsv(outputpath + str(kfold_idx) + 'fold_acc_loss.csv',
                 [ep, np.mean(np.array(train_loss)), np.mean(np.array(train_acc)), np.mean(np.array(valid_loss)),
                  np.mean(np.array(valid_acc))])
        print(
            f'Epoch : [{ep}] Train Loss : [{np.mean(np.array(train_loss)):.5f}] Train acc : [{np.mean(np.array(train_acc)):.5f}] Val loss : [{np.mean(np.array(valid_loss)):.5f}] Val ACC : [{np.mean(np.array(valid_acc)):.5f}]')

        torch.save(model.state_dict(), outputpath + str(kfold_idx) + 'fold_' + str(ep) + 'epoch_classifier.pt')
