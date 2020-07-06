```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

df = np.loadtxt("../dataset/ThoraricSurgery.csv", delimiter = ",")

x = df[:, 0:17]
y = df[:, 17]

model = Sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ['accuracy'])
model.fit(x, y, epochs = 100, batch_size = 10)
```

model 이라는 함수를 선언한 부분은 모두 딥러닝의 모델을 설정하고 구동하는 부분이다.

`model = Sequential()` : 딥러닝의 구조를 짜고 층을 설정하는 부분

`model.compile()` : 위에서 정해진 model을 컴퓨터가 알아들을 수 있게 compile

`model.fit()` : model을 실제로 수행하는 부분 

model.add()로 시작되는 라인이 두개 있다. = 두개의 층을 가진 model 생성

맨 마지막 층은 결과를 출력하는 **출력층(output layer)**이다. 나머지는 모두 **은닉층(hidden layer)** 

`model.add(Dense(30, input_dim = 17, activation = 'relu'))` 30은 이 층에 30개의 node를 생성,  input_dim은 입력 데이터에서 몇 개의 값을 가져올지 설정

keras 는 입력층을 따로 만들지 않고 첫 번째 은닉층(hidden layer)에 input_dim을 적어 첫 번째 Dense가 은닉층(hidden layer) + 입력층(input_layer)의 역할 , 데이터에서 17개의 값을 받아 은닉층의 30개 노드로 보낸다.

은닉층의 각 노드는 17개의 입력 값에서 임의의 가중치(w)를 가지고 각 노드로 전송되어 활성화 함수(activation)를 만나고 활성화 함수를 거친 결과값이 출력층으로 전달된다. 