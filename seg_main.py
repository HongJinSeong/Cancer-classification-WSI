### 이미지가 너무너무 커서 ROI 영역 찾은 이후에

from utils import *
from models import *
from datasets import *

from tqdm.auto import tqdm
import random
from torchmetrics.classification import BinaryJaccardIndex

import torch
import torch.optim as optim

# device='cpu'
device = 'cuda' if torch.cuda.is_available else 'cpu'


size=(512,512)
Epoch=300
outputpath='outputs/segments/'

# 시드(seed) 설정
RANDOM_SEED = 2022
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

tfrom=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.925, 0.91, 0.93), std=(0.01, 0.02, 0.01))
])

label_paths = glob('datas/train_masks/','*')

random.shuffle(label_paths)

train_data = label_paths[:50]
test_data = label_paths[50:]


train_ds = madison_seg_DS_real(train_data, size, True, True, transform=tfrom)
valid_ds = madison_seg_DS_real(test_data, size, False, False, transform=tfrom)



train_loader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=4,
    pin_memory=True,
    shuffle=True,
    drop_last=False,
)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_ds,
    batch_size=1,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
)

model = UNet(3,1)
model = model.to(device)

criterion = DiceBCELoss()
criterion = criterion.to(device)

optimizer = optim.AdamW(model.parameters(),lr=0.0005)

metric = BinaryJaccardIndex()

for ep in range(Epoch):
    model.train()
    tloss_ls=[]
    for i, (img, lbl) in enumerate(tqdm(iter(train_loader))):
        optimizer.zero_grad()

        img = img.to(device)
        lbl = lbl.to(device)

        outputs = model(img)
        loss = criterion(outputs, lbl.unsqueeze(dim=1).float())

        loss.backward()
        optimizer.step()
        tloss_ls.append(np.mean(loss.detach().cpu().numpy()))

    print(str(ep + 1) + ' EPOCH train LOSS :' + str(np.mean(np.array(tloss_ls))))
    model.eval()
    vloss=[]
    miou=[]
    best_miou=0
    with torch.no_grad():
        for i, (img, lbl) in enumerate(tqdm(iter(valid_loader))):
            img = img.to(device)
            lbl = lbl.to(device)

            # crossentropy 쓸 때
            # lbl_cross = lbl_cross.to(device)

            outputs = model(img)

            loss = criterion(outputs, lbl.float())

            # crossentropy 쓸 때
            # loss = criterion(outputs, lbl_cross.long())

            vloss.append(np.mean(loss.detach().cpu().numpy()))
            outputs = nn.Sigmoid()(outputs.detach().cpu())

            outputs = torch.where(outputs <= 0.5, 0, 1)
            lbl = lbl.detach().cpu().type(torch.int)

            iou = metric(outputs[0], lbl)
            vloss.append(np.mean(loss.detach().cpu().numpy()))
            miou.append(float((iou).numpy()))
    print(str(ep) + ' EPOCH END !!')
    print('-------------------------------------------------')
    print('Valid LOSS : ' + str(np.mean(np.array(vloss))))
    print('mIOU : ' + str(np.mean(np.array(miou))))
    print('-------------------------------------------------')
    writecsv(outputpath + 'logs/validloss.csv',
             [ep, np.mean(np.array(tloss_ls)), np.mean(np.array(vloss)), np.mean(np.array(miou))])
    if np.mean(np.array(miou)) > best_miou:
        torch.save(model.state_dict(), outputpath+'ckpts/' + 'best_iout.pt')
        best_miou = np.mean(np.array(miou))
