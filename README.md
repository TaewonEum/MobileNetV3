# MobileNetV3 ~12.31 ing

Google Colab

tensorflow사용

intel_image_classifiaction 이미지 데이터 사용

이미지 분류 모델 MobileNetV3사용

# 사용한 패키지

import os

import numpy as np

import random

from datetime import datetime

import time

import math

import gdown

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.applications import *

from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D, Softmax

AUTOTUNE = tf.data.AUTOTUNE

# Model parameter

Res=224 #이미지 사이즈

N_class=6 #총 클래스 개수

batch_size=64 #배치 사이즈

epoch=5 #반복횟수

LR=0.01 #학습률

# Google Drive Mount 및 데이터 압축해제

![image](https://user-images.githubusercontent.com/104436260/203930367-94bc2444-2471-4702-a00d-c48b692738a3.png)

# 경로 설정 및 라벨값 카테고리 확인

![image](https://user-images.githubusercontent.com/104436260/203930725-989ba318-fcdc-48e5-809d-46ac266f4f24.png)

# Dataset 만들기

![image](https://user-images.githubusercontent.com/104436260/201830219-90e9af40-089d-4b20-a5d3-c893d6c8d090.png)

# prefetch 적용

![image](https://user-images.githubusercontent.com/104436260/201830360-19acfd4a-4889-420c-acb5-006e1f213af8.png)

# Pretrained Model load

![image](https://user-images.githubusercontent.com/104436260/201830479-326b10ef-6a12-4bf8-b246-71b076cf228a.png)

# Model 학습시키기

![image](https://user-images.githubusercontent.com/104436260/201830581-c28a2bcc-3520-4b15-9082-aa2491c73e4e.png)

optimizer는 처음에는 SGD사용

다음으로는 현재 가장 많이 사용하는 optimizer인 Adam으로 학습 진행

![image](https://user-images.githubusercontent.com/104436260/201831396-45cb096b-1375-4e84-beac-a02c3a61c94b.png)

optimizer변경 후 학습

![image](https://user-images.githubusercontent.com/104436260/203933848-31a322c8-f747-432e-b65e-0d4e9ed411bb.png)

SGD보다는 트레인 셋에 과적합된 모습이 보임 파라미터들을 수정한 후 다시 학습

# Random crop을 통한 Data Augmantation 진행

트레인 이미지를 256으로 늘려주고 랜덤으로 224사이즈로 랜덤 crop 해주어 데이터 변형

![image](https://user-images.githubusercontent.com/104436260/201831917-40d96a76-757b-4ad3-9727-7b15ed0bc842.png)

batch를 풀어주어 각각의 사진이 random crop 될 수 있도록 함.

drop_ramainder=True를 통해 배치사이즈를 만족시키지 못하는 마지막 데이터들을 버려줌

![image](https://user-images.githubusercontent.com/104436260/201832731-2b9d5a4c-ddb2-41aa-92e1-bd6b792a5538.png)

data augmentation을 해준후 epoch 횟수를 1회 증가시켜준 후 학습 진행

# Cutmix 알고리즘 구현하여 dataset 생성하기

Cutmix알고리즘을 사용하기 위해서는 기존의 라벨링값을 one hot encoding으로 바꾸어주어야 함

![image](https://user-images.githubusercontent.com/104436260/201834015-efdef40a-8bd2-42f9-8f9c-fdc6ce237654.png)

현재는 라벨링 값이 0~5사이 값으로 입력되어 있는값을 onehot encoding형태로 바꾸어주어야 함.

![image](https://user-images.githubusercontent.com/104436260/201834994-d85abbc7-895c-4076-9183-5a08df806a23.png)

one-hot encoding으로 라벨값 바꿔줌

![image](https://user-images.githubusercontent.com/104436260/201858941-47774c25-795e-4c24-8430-92c7e551671d.png)

cutmix함수 만들어줌

cutmix는 균등분포에서 0.5보다 큰값이 나오면 해당 이미지에 cutmix를 해줌

![image](https://user-images.githubusercontent.com/104436260/201859582-a7ffad6b-288d-4a9e-b237-1a9cd8ddfeb4.png)

cutmix data 확인

![image](https://user-images.githubusercontent.com/104436260/204228817-03554dc3-9663-4aa1-877d-44e3e2739be0.png)

cutmix한 데이터로 train 및 Validation 진행

이때 손실함수는 categorical crossentropy로 바꿈 정수값이던 label값이 onehot encoding으로 바뀌었기 때문
# Data size 늘려서 Model 성능 높이기

![image](https://user-images.githubusercontent.com/104436260/207208516-447e0cdf-ffea-41f1-a2dc-c807757430a2.png)

