## CNN(Convolution N)

<hr>

직접 이미지, 비디오, 텍스트 또는 사운드를 분류하는 머신 러닝의 한 유형인 딥러닝에 가장 많이 사용되는 알고리즘

![cnn](https://user-images.githubusercontent.com/50917797/86808799-2a087680-c0b6-11ea-9958-c735d5d3fbc9.jpg)



## convolutional layer 

입력 데이터(input)로부터 특징 추출하는 역할(filter), 이 filter 값을 비선형 값으로 바꿔주는 activation(활성함수)로 이루어진다.

## activation map

convolution layer의 입력 데이터를 필터가 순회하며 convolution(합성곱)을 통해서 만든 출력을 feature map 또는 activation map이라고 한다. feature map은 convolution으로 만들어진 행렬. activation map은 feature fap 행렬에 activation(활성 함수)를 적용한 결과. convolution 레이어의 최종 출력 결과가 activation map입니다.

## Sub sampling, Pooling

위 convolutional layer를 거쳐 추출된 특징들을 줄이는 작업 max pooling, average pooling이 있는데 주로 max pooling을 많이 사용

## Max pooling

맥스 풀링은 activation map을 행열의 크기로 자르고, 그 안에서 가장 큰 값을 뽑아내는 방법.

<img width="444" alt="maxpooling" src="https://user-images.githubusercontent.com/50917797/86808920-46a4ae80-c0b6-11ea-9975-7cf706c57ce2.png">



