---
layout: post
title:  "Basic NLP_Part.2"
date:   2018-07-12
excerpt: "Basic things about NLP"
tag:
- markdown 
comments: false
---
# **Basic NLP**_Part.2

이 포스트는 Tacademy의 [토크ON세미나] 자연어처리-자연어처리에 대한 이해를 듣고 정리한 것임을 밝힌다.

### **딥러닝을 이용한 자연어처리 기법 (Part.1에 이어서)**

2) Recurrent Neural Networks (RNN)의 이해 

* Intro

  input layer - hidden layer - output layer로 구성되어있다. 이 때 RNN은 시계열 데이터를 시간의 순서대로  확실히 하기 위해 hidden layer 사이에 recurrent connection을 둔다. 시간이 지남에 따라 hidden layer의 matrics가 어떻게 변하는지를 보는 파라미터를 쓰는 것이다. 하지만 recurrent weight 학습이 어려워서 컴퓨팅 파워가 부족했던 과거에는 많이 쓰이지 않았다. 또한 층이 너무 많으면 아래 층은 error가 거의 없다는 문제가 있었다. (시간축이 너무 많아도 마찬가지) 하지만 최근에 activation function이 바뀌고 긴 시퀀스 학습 가능해지면서 널리 쓰이고 있다. 문서에 있는 텍스트 데이터 역시 시계열 데이터로 보기 때문에 기계학습을 이용한 자연어 처리는 대부분 RNN을 사용한다.

  ​


* 활용분야

  * Sequence Labeling : Classification 문서에 있는 단어 시퀀스를 보고 감정 예측도 가능 / 필기체 인식 (단어마다 이미지 끊어서 CNN으로 풀 수 있고, 필체 시퀀스 저장하는 시계열 데이터로 하는 RNN으로 풀 수 있다) / 음성 인식 (파형을 모델링 하는 방식으로 coefficient 뽑아서 classification했었다, 스펙트로그램을 그려서 이미지로 만들고 그 위에 CNN, 다음에 RNN 사용)

  * 시계열 예측 : 주식 그래프를 하나하나의 값을 가지는 데이터로 가지고 RNN, 심박수 모델링

  * 파형 생성 : 주기적인 모양을 가지는 그래프 학습 시키면 그 다음의 모양 예측할 수 있음

  * Text Generation : 셰익스피어 글들을 학습하고 단어를 던져주면 셰익스피어 스타일로 글 쓰기 / 리눅스 코드 제너레이션

  * Image Caption Generation : 이미지에 맞는 문장을 생성한다, 이미지 입력하고 캡션 다는건 CNN, 그렇게 달린 캡션으로 문장 생성하는건 RNN

  * Question & Answer dataset

    ​

* RNN의 원리

  ![RNN](http://www.popit.kr/wp-content/uploads/2016/12/hello_charseq.jpeg)

  * 정리 : 위 그림은 타임마다 각 캐릭터가 들어갈 때 다음 캐릭터를 맞추는 문제를 RNN으로 표현한 것이다. 이 때 맨 왼쪽 한개의 column만 있으면 Neural Network -> recurrenct connection을 주면 RNN이 된다. 이 그림에서는 4차원짜리 벡터로 각 캐릭터 표현하였고 'h' 다음에는 'o'가 나와야 한다고 학습시키는 것이다. 그 hidden layer를 연결해서 다음 타입에는 이전 hidden layer를 어느 정도 계산해서 현재의 hidden layer에 반영한다. 이러한 방식을 계속적으로 거쳐서 그 다음의 캐릭터 예측하는 것이다. RNN은 기존 Neural Network와 달리 weight가 이전 hidden layer의 것을 반영하는 것 하나가 더 추가된다. 또한 RNN의 출력 시점은 본인이 가지고 있는 데이터에 따라 달라질 수 있다. 한 캐릭터씩 바로 error 계산해서 back propagation 할 수도 있지만, 문장 전체 다 보고 맞추는 것이라면 마지막에만 에러 계산하면 된다.

    ![Architecture](https://i.stack.imgur.com/hzZ4m.png)

  * Architecture : U, V, W는 같은 파라미터를 공유한다.

  * Process : 매 타임 스텝마다 error를 계산해서 back propagation 한다. 문장 단위로 classification 하는거면 그 때의 hidden, output layer를 가지고 error 계산해서 back propagation을 하는 것이다. 따라서 모델링 하는 관점에 따라 인풋 사이즈, 데이터 인스턴스의 사이즈가 모두 다를 수 있다.

  * Model : 전 스텝의 hidden layer와 weight를 추가적으로 더해준다는 것이 RNN의 가장 큰 특징이다. 마지막에 소프트맥스나 시그모이드를 통해 classification을 하게 된다. 이 때 엔트로피 낮추는 방향으로 학습을 진행하고, 다 입력받아서 error를 계산한다.

  * Learning : error가 나오면 U, V, W 각각에 대해 편미분을 진행한다. 이 때 맨 마지막의 W와 U의 weight를 계산하려면 처음 시퀀스까지 다 돌아가서 weight를 계산해야 한다. 반면에 맨 마지막 V의 weight를 업데이트 할 때에는 이전의 정보들이 필요가 없다. 그런데 weight가 계속 1보다 작으면 W와 U 처음의 weight가 너무 작아지고, 계속 1보다 크면 너무 커지는 문제가 발생해서 gradient를 어느 정도 범위 안에 두겠다고 clipping을 하거나, 필요한 길이만큼만 가서 계산을 하는 방식을 하기도 한다. 혹은 forget gate와 remember gate를 두고 시퀀스가 길어졌을 때 어디까지 기억할지를 결정한다.(LSTM, long short term memory)





#### **Reference** ####
 Tacademy [토크ON세미나] 자연어처리-자연어처리에 대한 이해

http://www.popit.kr/wp-content/uploads/2016/12/hello_charseq.jpeg

https://i.stack.imgur.com/hzZ4m.png