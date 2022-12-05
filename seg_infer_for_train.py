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

label_paths = glob('datas/train_masks/', '*')

test_paths = glob('datas/train_imgs/', '*')
test_paths = np.array(test_paths)
test_paths = np.char.replace(test_paths,'\\','/')
for lblpth in label_paths:
    lblpth = lblpth.replace('\\','/')
    lblpth = lblpth.replace('train_masks','train_imgs')
    test_paths = np.delete(test_paths, np.where(test_paths==lblpth), axis = 0)


test_ds = madison_seg_DS_real_test(test_paths, size, transform=tfrom)



test_loader = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=1,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
)

model = UNet(3,1)
model = model.to(device)

model.load_state_dict(torch.load(outputpath+'ckpts/best_iout.pt'))

with torch.no_grad():
    for i, (img, img_origin, path) in enumerate(tqdm(iter(test_loader))):
        img = img.to(device)
        outputs = model(img)
        outputs = nn.Sigmoid()(outputs.detach().cpu())

        outputs = torch.where(outputs <= 0.5, 0, 1).numpy()[0,0]

        outputs = cv.resize(outputs, (img_origin.numpy()[0].shape[1], img_origin.numpy()[0].shape[0]), interpolation=cv.INTER_NEAREST)

        img_origin = img_origin[0].numpy()

        img_origin[np.where(outputs == 1)] = [255,0, 0]

        cv.imwrite(outputpath+'output_img/' + path[0], img_origin)

        print('ccc')
