---
layout: post
title:  "Basic NLP_Part.1"
date:   2018-07-11
excerpt: "Basic things about NLP"
tag:
- markdown 
comments: false
---
# **Basic NLP**_Part.1

이 포스트는 Tacademy의 [토크ON세미나] 자연어처리-자연어처리에 대한 이해를 듣고 정리한 것임을 밝힌다.

### **규칙 기반 대화 모델링**

* Terry Winograd's Dissertation, SHARDLU 프로그램 개발

  ​	3차원 공간에 색깔과 모양을 가지는 물체들을 만들어 놓았다. 박스모양의 물체 안에 물건을 넣기도 하고 옮기기도 하며 프로그램을 실행시킨다. 이는 컴퓨터가 대답하는 프로그램이며 자연어 챗봇의 시초라고 할 수 있다. NLP 기반(동사/명사 태깅, 모든 속성을 디자인)으로 만들어진 프로그램으로, 사람이 자연어로 질문했을 때 컴퓨터가 이해할 수 있는지에 초점을 맞추어 설계되었다고 한다. 하지만 이 프로그램은 50개의 단어만 이해할 수 있었다.  프로그래머가 초기에 지정한 50개의 단어가 손으로 직접 처리되었기 때문에 그 단어들만 기억하는 것이다. 이 때 가상환경에서 나올 법한 지식 체계를 'Ontology'라고 한다.이 프로그램은 또한 메모리도 가지고 있으며, 시퀀스를 기억할 수 있다. 

![SHARDLU](https://github.com/jiwon12-31/jiwon12-31.github.io/blob/master/assets/img/shardlu1.JPG)



​	위에 보이는 그림이 SHARDLU 프로그램에 의해 3차원 공간에 그려진 물체들이다.

​	이전에도 언급했듯이 이 프로그램은 모든 것들을 손으로 다 태깅해두었다. 가상 월드에서 있을 수 있는 지식을 모두 저장하여 질문의 종류에 따라 파싱하고, 파싱한 것들에 대한 행동 시퀀스 모두 저장해 주는 방식으로 프로그램을 설계한 것이다. 따라서 이는 정해진 일들만 할 수 있고 확장성 없다는 단점이 있다.

​	금융 챗봇도 이와 비슷한 상황으로 돌아간다고 생각하면 쉽다. 어느 정도의 룰을 사람이 미리 다 해 두고, 중요한 것들은 사람들이 미리 다 태깅하며 새로운 질문의 경우에는 딥러닝을 사용해서 유연하게 대처하는 것이다. 

### **확률 모델 기반 대화 모델링**

규칙 기반의 대화 모델링은 deterministic이라고 하고, 확률 기반의 대화 모델링은 randomness라고 한다.

* Naive Bayes

  ![베이즈](https://s3.ap-south-1.amazonaws.com/techleer/204.png)

  ​	Naive Bayes는 단어들의 연속인 문장이 발생될 확률을 모델링한다. 각 Class가 주어지면 각각의 단어가 나타날 확률은 독립이라고 가정하고 모델링을 시작하게 된다.

  ​	스팸 메일 분류기를 예시로 Naive Bayes에 대해 알아보자.우선 스팸 메일에서 자주 나오는 단어와 스팸이 아닌 메일에서 자주 나오는 단어들의 빈도수 찾는다. 그리고 그 빈도수에 따라서 스팸인지 아닌지 판단한다. 이메일이 왔을 때 스팸인 확률과 스팸이 아닌 확률을 비교하여 스팸 여부를 판단한다.

  ​	파이썬에서 Naive Bayes를 활용할 때에는 sklearn의 GaussianNB, BernoulliNB, MultinomialNB를 import하여 fitting하는 형태로 이용할 수 있다.

  ​

* 자동 커피 주문을 위한 확률 모델

  ​	사용자가 하는 말의 의도 태깅하여 발화 의도 예측을 통해 대화를 모델링한다. 이를 좀 더 자세히 풀어서 설명하자면, 우선 representation을 컴퓨터가 이해하게 바꾸어 representation이 속하는 의도를 예측하고, 손님이 한 의도 다음에 나올 점원이 할 의도를 예측한다.  그리고 그 의도에 따라 점원이 할 비슷할 말을 가지고 와서 답변하게 되는 것이다. 따라서 이 경우에는 topic classification이 매우 중요해진다.

  ​

* Markov Decision Process 기반 대화 모델링

  ​	reward를 학습해서 policy를 찾는다.

  ​     

### **딥러닝을 이용한 자연어처리 기법**

1) 단어 임베딩의 이해

​	이미지 안에 데이터 값들은 연속성이 있다. 하지만 언어 데이터는 연속성을 찾아볼 수 없고, 모든 단어에는 표현 상의 일정한 흐름이 없다. 이러한 단어들을 컴퓨터가 이해할 수 있는 방식으로 바꾸는 것을 임베딩이라고 한다.

* One-hot encoding

  ​	one-hot encoding은 단어를 가지고 딕셔너리를 만든다. 하지만 너무 sparse하기 때문에 메모리 문제가 생길 수 있다. 너무 용량이 크면 연산량이 많아지기 때문이다. 또한 단어 사이의 거리가 매우 중요하지만 이 방식으로는 많은 것들을 얻을 없다. 요즘엔 one-hot encoding을 많이 안쓰고 벡터 자체가 의미를 가지도록 하는 것을 많이 쓴다.

  ​

* Neural Word Embeddings

  ​	조금 더 작은 차원의 벡터로 단어의 의미 표현한다. 앞에 나온 단어들을 보고 지정한 단어를 잘 맞출 수 있는지를 학습시키는데, 이 때 지정한 단어는 앞에 나온 것들과 비슷할 것이라고 가정하고 뉴럴 네트워크를 학습시키는 것이다. 예를 들어, 50차원*4 = 200차원의 hidden representation을 만들고 나를 표현하는 50차원의 벡터를 잘 맞추는지를 찾아가는 것이다. 이 때 그 차이를 줄여가는 것을 back-propagation이라고 한다. 

  ​	비슷한 의미를 가지는 단어는 비슷한 거리의 벡터로 표현하게 된다. 하지만 rm 값과 딱 맞는 벡터가 없으면 무슨 단어인지 정확히 알 수 없다는 단점이 있다. 이를 해결하기 위해 classification을 추가하기도 한다

  ​

* Word2Vec

  ​	distributed semantics

  ​	CBOW( 내 주변을 보고 가운데에 있는 단어를 맞출 수 있도록, cost 줄임, 단어 채우는 문제도 쉽게 풀 수 있음), skip-gram(반대, 한 단어 가지고 주변을 맞춤)

  ​

* GloVe

  얼마나 자주 같이 발생하는지를 나타내는 co-occurence matrix(object, 진리값) 구하고 이를 분해하여 단어에 대한 representation을 찾는다.





#### **Reference** ####
 Tacademy [토크ON세미나] 자연어처리-자연어처리에 대한 이해

https://s3.ap-south-1.amazonaws.com/techleer/204.png