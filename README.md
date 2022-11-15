# MobileNetV3

Google Colab

tensorflow사용

intel_image_classifiaction 이미지 데이터 사용

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

LR=0.0001 #학습률

# Dataset 만들기

![image](https://user-images.githubusercontent.com/104436260/201830219-90e9af40-089d-4b20-a5d3-c893d6c8d090.png)

# prefetch 적용

![image](https://user-images.githubusercontent.com/104436260/201830360-19acfd4a-4889-420c-acb5-006e1f213af8.png)

# Pretrained Model load

![image](https://user-images.githubusercontent.com/104436260/201830479-326b10ef-6a12-4bf8-b246-71b076cf228a.png)

#Model 학습시키기

![image](https://user-images.githubusercontent.com/104436260/201830581-c28a2bcc-3520-4b15-9082-aa2491c73e4e.png)
