```python
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential
```

```
os.listdir("../input/humpback-whale-identification/")
```

```python
train_df = pd.read_csv("../input/humpback-whale-identification/train.csv")
train_df.head()
```

```python
def prepareImages(data, m):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/humpback-whale-identification/train/"+fig, target_size =(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
```

```python
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
```

```py
X = prepareImages(train_df, train_df.shape[0])
X /= 255
```

```py
model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
```

## 전처리

### def prepareImages(data, m): 

prepareImages 라는 함수 생성 매개변수로는 data, m을 받는다.

### X_train = np.zeros((m, 100, 100, 3))

X_train 에 np.zeros()함수를 써서 배열을 만들어준다. 

np.zeros(10,10) = 10행 10렬짜리 행렬을 만든다. 

np.zeros(m, 10,10) = 10행 10렬짜리 행렬을 만들고 m번 반복한다. m대신 2라면 10 x 10 행렬을 두번 만든다.

 np.zeros(m, 10,10, 3) = 10행 10렬짜리 행렬을 만들고 m번 반복한다. m대신 2라면 10 x 10 행렬을 두번 만든걸 3열로 자른다. 3대신 4라면 4열로 자른다.

###     for fig in data['Image']:

입력 받을 데이터 안에 ['Image'] 라는 컬럼명을 가진 갯수 만큼 fig라는 변수를 통해 for loop을 돌려준다.

###   img = image.load_img("../input/humpback-whale-identification/train/"+fig, target_size=(100, 100, 3))

image.load_img 를 사용하려고 위에서 `from keras.preprocessing import image` 를 import 해주었다. 

이걸 통해 이미지를 불러고는데 target_size = (100, 100, 3) 을 지정해 100 x 100 으로 설정하고 3차원 구조를 갖는 배열로 생성한다.

`"../input/humpback-whale-identification/train/"+fig` 경로를 지정해주고 뒤에 fig라는 for문 안에 있는 변수?이고 data['Image'] 값을 for loop돌리니까 index 0번의 데이터가 fig자리에 대체 되어 img를 불러오게 되는 원리.

여기서 data 는 Image, Id 이고 m 은 행의 갯수, dataset 은 train이다 

### x = image.img_to_array(img)

###         x = preprocess_input(x)

 image.img_to_array 이 함수를 사용하면 ()안에 있는 img를 numpy 배열로 변환해준다.

preprocess_input()  기능은 이미지가 모델이 요구하는 형식에 적합하도록하기위한 것.

### X_train[count] = x

불러온 img를 numpy 배열로 변환 후, 형식에 맞게 전처리 한 값을 X_train[count]값으로 설정 

```
[[[-1.49390030e+01 -4.97789993e+01 -8.16800003e+01]
  [-6.93900299e+00 -4.37789993e+01 -7.46800003e+01]
  [ 1.20609970e+01 -2.77789993e+01 -5.56800003e+01]
  ...
  [ 3.06099701e+00 -4.07789993e+01 -7.06800003e+01]
  [-4.79390030e+01 -7.87789993e+01 -1.08680000e+02]
  [ 1.00609970e+01 -2.77789993e+01 -6.06800003e+01]]
```

### print(X_train[count]) 통해 출력해보니 위 값 나옴. 느낌상 RGB 값 아닌가 싶다. 

###  if (count%500 == 0):

###             print("Processing image: ", count+1, ", ", fig)

###         count += 1

count 증가값이 500으로 나눠서 나머지 값이 0이면 즉 500의 배수이면 현 상황을 알려주는   print("Processing image: ", count+1, ", ", fig) 출력하고 count 값 증가 시켜라.

```
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
```

### def prepare_labels(y): 함수 생성하고 y를 매개변수로 받는다. 

###     values = np.array(y) 

y를 np.array , array라는 함수에 리스트를 넣으면 배열로 변환해준다. 

###     label_encoder = LabelEncoder()

LabelEncoder() 를 통해 라벨 인코더 생성, 문자를 0부터 시작하는 정수형 숫자로 바꿔주는 기능 반대로 코드 숫자를 이용해서 원본 값 구하기도 가능 

###     integer_encoded = label_encoder.fit_transform(values)

fit은 정규화(통계에서 정규분포 만들려고 x - 평균 한 값을 다시 표준편차로 나누는 작업,이 작업을 하려고 평균과 표준편차를 계산하는 작업이 fit()이다. ) transform()은 정규화 작업 하는거. (x - 평균) / 표준편차 해서 새로운 x프라임 값 생성하는 것. 

###     onehot_encoder = OneHotEncoder(sparse=False)

원핫 인코딩 이란. 단 하나의 값만 True, 나머지는 False인 인코딩. 주로 여러개 분류하는 문제에서 사용

OneHotEncoder : 숫자로 표현된 범주형 데이터를 인코딩한다. sparse=False 이건 Sparse matrix 형태로 출력하려고 한듯

###     integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

integer_encoded 열벡터로 reshape 한걸 interger_encoded 변수에 담는다. 행을 interger_encoded 열은 1 즉 interger_encoded x 1열

```
onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
```

OneHotEncoder로 틀을 잡고 그 안에 integer_encoded 를 사용해 실제 들어올 데이터 값으로 행렬을 만들고  onehot_encoder.fit_transform(integer_encoded) 이건 값을 실제로 넣는거 같다. 

###     return y, label_encoder

```py
X = prepareImages(train_df, train_df.shape[0])
X /= 255
```

이제 위에 만들어 둔 함수에 값을 넣자. prepareImages는 우리가 위에서 만든 함수 호출하고 매개변수로 설정한 data, m 값에 train_df, train_df.shape[0] 이 값을 넣어준다. train_df는 csv파일 불러온 데이터고 train_df.shape[0] 는 행 갯수다. 행 index 0부터 넣어준다로 이해

## ===============================================================

## model

```
model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
```

### model = Sequential() :

 딥러닝의 구조를 짜고 층을 설정하는 부분

### model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

 model.add를 통해 hidden layer 층 추가 하는데 conv2D 클래스 사용 

첫 번째 인자 : convolution의 filter 수, 코드에선 32

두 번째 인자 : convolution의 행열, (7, 7) = 7 x 7

strides는 크기 4인 1차원 리스트. [0], [3]은 반드시 1. 일반적으로 [1], [2]는 같은 값 사용. 보통 [1,2 ,2, 1] 이런식으로 쓰는거 같음. [1, 2, 2, 1] = 2칸씩 움직이려면 저렇게 

### input_shape : 

샘플 수를 제외한 입력 형태를 정의, model에서 첫 레이어일 때만 정의하면 된다. (첫 층일때만)

(행, 열, 채널 수)로 정의. 흑백영상인 경우에는 채널이 1, 컬러(RGB)영상인 경우에는 채널을 3으로 설정.

### model.add(BatchNormalization(axis = 3, name = 'bn0'))

케라스에서 배치 정규화는 하나의 레이어(`BatchNormalization()`)처럼 작동하며, 보통 `Dense` 혹은 `Convolution` 레이어와 활성함수(`Activation`) 레이어 사이에 들어간다. axis=3 줘서 3차원 유지하는거 같음

### model.add(Activation('relu'))

activation(활성함수) relu 쓰겠다는거고 

### model.add(MaxPooling2D((2, 2), name='max_pool'))

컨볼루션 레이어의 출력 이미지에서 주요값만 뽑아 크기가 작은 출력 영상을 만든다. 

pool_size : 수직, 수평 축소 비율을 지정, (2, 2)이면 출력 영상 크기는 입력 영상 크기의 반으로 줄어든다.

### model.add(Flatten())

CNN에서 convolution 레이어나 maxpooling 레이어를 반복적으로 거치면 주요 특징만 추출되고, 학습. 컨볼루션 레이어나 맥스풀링 레이어는 주로 2차원 자료를 다루지만 학습하려면 1차원 자료로 바꿔줘 함. 이 때 사용되는 것이 flatten 레이어