# Cancer-classification-WSI

<유방암의 임파선 전이 예측 AI 경진대회>

개수가 작은 데이터셋 (1000장) very very large 이미지 classification ( maximum 약 7000 x 7000 )  

실제 성능이 좋지 못했음 (단순 CNN으로 이미지만 돌려서 public 기준 70%)

Multi instance learing 을 처음으로 해보았으나 성능은 좋지 못했음 (데이터가 매우매우 작아서 overfitting 매우매우 심했음)

Whole Slide image 생성과 관련하여 opencv로 구현해봄 ==> 종종 ROI 영역 잘라낼 때 사용해봄직 함(extract roi part)
