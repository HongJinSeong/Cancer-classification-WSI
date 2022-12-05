from utils import *
from models import *
from datasets import *


outputpath='outputs/segments/'
label_paths = glob('datas/train_masks/','*')

for pth in label_paths:
    lbl_path = pth.replace('\\', '/')
    lbl = cv.imread(lbl_path, cv.IMREAD_GRAYSCALE)

    img_path = lbl_path.replace('train_masks', 'train_imgs')
    img = cv.imread(img_path)

    lbl[np.where(lbl != 255)] = 1
    lbl[np.where(lbl != 1)] = 0

    img[np.where(lbl == 1)] = [255, 0, 0]

    cv.imwrite(outputpath + 'output_img/' + img_path.split('/')[-1], img)

print('sss')