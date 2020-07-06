## 케라스 라이브러리 및 딥러닝의 큰 틀

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

Sequential() 함수는 딥러닝의 구조를 한 층 한 층 쉽게 쌓아올릴 수 있게 해준다. 

Sequential() 함수를 선언하고나서 model.add() 함수를 사용해 필요한 층을 차례로 추가해준다. 

```python
model.add(3층 모델 옵션)
model.add(2층 모델 옵션)
model.add(1층 모델 옵션)
```

케라스의 가장 큰 장점 중 하나는 model.add() 함수를 이용해 필요한 만큼의 층을 빠르고 쉽게 쌓아 올리는 것. 

`model.add(Dense(30, input_dim = 17, activation = 'relu'))`

model.add() 함수 안에 Dense()함수는 각 층이 제각각 어떤 특성을 가질지 옵션을 설정하는 역할

딥러닝의 구조와 층별 옵션을 정하고 나면 compile() 함수를 이용해 이를 실행 시킨다. 

activation : 다음 층으로 어떻게 값을 넘길지 결정하는 부분, 가장 많이 사용되는 함수는 relu(), sigmoid()이다.

loss : 한번 신경망이 시행될 때마다 오차 값을 추적하는 함수

optimizer : 오차를 어떻게 줄여 나갈지 정하는 함수

