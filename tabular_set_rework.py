from utils import *
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### categorical 하니 데이터가 제한적인데 아주 많은 embedding layer 사이에 조합간의 충분한 결합이 안일어남
### 차라리 categorical datas를 one-hot 형태로 표현하면 조합과 무관하게 모든 파라미터간에 데이터가 영향이 가게 됨으로 좀더 좋은 학습이 되지 않을까?
### 여러 embedding layer를 엮고 싶다면 데이터의 조합이 충분히 모두 반영될수 있다고 생각될 때 의미 있는 행위를 할수 있을 것으로 보임
### 여기서 데이터가 학습되는 형태의 최적의 조합 찾고 nan값 채워서 또 경향 만들어 줘보는 것으로...

#### category 가운데 모든 데이터가 다 있고 모든 조합이 충분히 만들어지며 수치적인 특징이 없기 때문에 category로 둠
categorical=['진단명','암의 위치']

#### categorical 하게 정의 되어있으나 score 형태거나 특정 수치의 기준으로 나누어진 case면 numerical로 간주 (데이터가 적어서 모두 categorical 했다간 embedding layer 못쓰고 
#### one-hot 형태로 Linear 연산만 혀야함
numerical = ['나이','암의 장경','NG','HG','HG_score_1','HG_score_2','HG_score_3','T_category','HER2_IHC','암의 개수','DCIS_or_LCIS_여부','ER','PR']

## 나이 min max scailing 이후  std
## 암의 장경 정규 분포인데 튀어나간 값있음  min max scailing 이후에 std
## 나머지는 전원 std scailing

csv_re = pd.read_csv('datas/train.csv')


# sns.distplot(csv['DCIS_or_LCIS_여부'], kde=True, rug=True)
# plt.show()
#

csv_re['나이'] = (csv_re['나이'] - 25) / (93-25)

csv_re['암의 장경'][csv_re['암의 장경'].isna()] = 18.48
csv_re['암의 장경'] = (csv_re['암의 장경']) / (90)

csv_re['진단명'] = csv_re['진단명'] - 1
csv_re['암의 위치'] = csv_re['암의 위치'] - 1

csv_re['NG'][csv_re['NG'].isna()] = 2.07
csv_re['HG'][csv_re['HG'].isna()] = 1.9
csv_re['HG_score_1'][csv_re['HG_score_1'].isna()] = 2.59
csv_re['HG_score_2'][csv_re['HG_score_2'].isna()] = 2.14
csv_re['HG_score_3'][csv_re['HG_score_3'].isna()] = 1.43
csv_re['T_category'][csv_re['T_category'].isna()] = 1.28
csv_re['HER2_IHC'][csv_re['HER2_IHC'].isna()] = 1.25
csv_re['ER'][csv_re['ER'].isna()] = 0.82
csv_re['PR'][csv_re['PR'].isna()] = 0.64

csv_re = csv_re[['ID', 'img_path', 'mask_path']+categorical+numerical+['N_category']]
csv_re.to_csv('datas/train_tabular_v1.csv', index=False)

#######################
####### test ##########
####### train에서는 categorical 하게 데이터를 줘도 없으면 빼면 그만이지만 test에서는 nan을 뺄 수가 없음으로 모델에서 nan값을 없는 제일 마지막 category로 만들어서 쓰는걸로...
####### embedding layer를 실제 category+1로 구성
csv_test = pd.read_csv('datas/test.csv')


csv_test['진단명'] = csv_test['진단명'] - 1
csv_test['암의 위치'] = csv_test['암의 위치'] - 1


csv_test['NG'][csv_test['NG'].isna()] = 2.13
csv_test['HG'][csv_test['HG'].isna()] = 1.89

csv_test['HG_score_1'][csv_test['HG_score_1'].isna()] = 2.59
csv_test['HG_score_2'][csv_test['HG_score_2'].isna()] = 2.14
csv_test['HG_score_3'][csv_test['HG_score_3'].isna()] = 1.43

csv_test['T_category'][csv_test['T_category'].isna()] = 1.28

csv_test['HER2_IHC'][csv_test['HER2_IHC'].isna()] = 1.25

csv_test['암의 장경'][csv_test['암의 장경'].isna()] = 18.48

csv_test['나이'] = (csv_test['나이'] - 25) / (93-25)
csv_test['암의 장경'] = (csv_test['암의 장경']) / (90)

csv_test = csv_test[['ID', 'img_path']+categorical+numerical]

csv_test.to_csv('datas/test_tabular_v1.csv', index=False)

print('nananan')