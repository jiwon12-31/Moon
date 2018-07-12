---
layout: post
title:  "Basic NLP_Part.3"
date:   2018-07-12
excerpt: "Basic things about NLP"
tag:
- markdown 
comments: false
---
# **Basic NLP**_Part.3

이 포스트는 Tacademy의 [토크ON세미나] 자연어처리-자연어처리에 대한 이해를 듣고 정리한 것임을 밝힌다.

### **Neural Conversation**

1. Dialog Modeling Problem

   ​	과거 대부분의 챗봇은 rule-based, ontology-based였기 때문에 heuristic한 성질을 가지고 있다. 사람이 기계에 직접 rule을 넣어주면 그 rule에 따라 대화가 진행되지만, 실재 대화를 모두 커버하기엔 rule이 너무 많다. 또한 수동적인 매뉴얼 라벨로 인해 에러가 많고 비싸다는 단점이 있다.

   ​	현재에는 neural network를 사용하여 챗봇이 직접 문장을 생성한다. 챗봇은 문장의 단어 하나하나씩 생성해가면 결국 문장을 만들게 된다. 이 방법은 매뉴얼 라벨링이 필요 없다는 장점이 있고 데이터가 많기 때문에 주목 받고 있다. 하지만 여전히 문법적 오류는 존재한다. 

   ​

2. Recurrent Language Model

   ​	$$X$$를 하나의 영어 문장이라고 하자. 그러면 $$X$$ 안에 있는 각 단어 $$x_1, x_2, ..., x_T$$의 시퀀스가 생성된다.  문장 $$X$$가 등장할 확률 $$p(X)$$ 계산하는 것이 이 모델의 궁극적인 목적이다.  이 때의 $$p(X)$$는 각각의 단어들이 동시에 등장할 확률과 같다. 이를 수식으로 정리하여 표현해보자.

   $$p(X) = p(x_1, x_2, x_3,..., x_T)$$ (joint probability)

   $$p(X) = p(x_T|x_1, x_2, ...x~T-1~)*p(x_1, x_2, ..., x~T-1~)$$ (conditional prob * marginal prob)

   marginal prob $$= p(x~T-1~|x_1, ...,x~T-2~)*p(x_1, ..., x~T-2~) $$ 

   이런 방식으로 계속 쪼갤 수 있다. 따라서 이전 문장의 히스토리가 필요하고, 이전 단어들이 나왔을 때 한 단어가 나올 확률을 계산하는데, 조건부 확률에 들어있는 조건이 hidden layer에 담겨져 있다.

   ​	결국 RNN의 hidden layer는 이전 입력들의 정보를 가지고 있고, 그 때 다음 단어를 예측한다. 이러한 방식을 통해 각각의 probability term도 계산할 수 있다. o = softmax(W*h+b) (선형 결합한 다음에 소프트맥스에 대입)하여 o를 구하면 softmax를 통한 확률이 계산된다.

   ​

3. Neural Machine Translation

   ​	이 모델에서는 encoder RNN, decoder RNN으로 구성된 총 2개의 RNN이 사용된다. 처음에 입력된 문장은 encoding되고 encoder RNN은 입력 데이터들의 정보를 가지고 있다. 또한 decoder RNN의 hidden layer는 encoder RNN이 저장한 정보와 생성한 문장 안에서의 information을 가지고 있다.

   ​	대화이건 번역이건 입력 문장과 출력 문장은 pair로 이루어진다. encoder RNN은 단어를 계속 받아서 hidden vector를 update하고 dexoder RNN에 전달한다. 정보를 전달받은 decoder RNN은 hidden vector를 계산할 때 추가로 인코더 관련해서 계산하는 항을 더한다. 따라서 decoder RNN에서 계산한 hidden vector는 입력한 정보도 갖고 있고 그 정보를 통해 새로운 단어를 생성한다.

   ​	여기서 memory attention이라는 개념이 추가된다. hidden vector는 이제 하나가 아니다. 단어마다 hidden vector를 가지고 있고, 그것을 업데이트 시키는 히스토리 과정을 각각 단어에 대한 hidden vector에 저장한다. 이렇게 여러 개의 hidden vector로 쪼개서 가중치가 적용된 합으로 hidden vector를 계산하는 것이 바로 memory attention이다. 이 방식은 기존의 방식보다 메모리를 좀 더 많이 쓴다는 특징을 갖는다.

   ​

4. Beyond Neural Machine Translation

   ​	image caption generation이나 memory networks, neural turing machine에서 많이 사용된다.

   ​

5. Naive Seq2seq Model

   ​	attention이 추가되지 않은 모델의 성능이 가장 낮았고, attention이 추가된 모델의 성능은 이전의 것보다 조금 높았다. 여기서 한 개의 모델이 더 추가되었는데, attention 없는 모델의 hidden vector와 있는 모델의 hidden vector를 연결하여 새로운 hidden vector를 가지는 모델을 만들면 가장 성능이 좋았다. 이렇게 만들어진 딥러닝 모델은 rule-based 기법이나 통계적인 기법들로 만들어진 모델보다 훨씬 성능이 뛰어나다.

   ​	하지만 이러한 모델에도 문제점은 존재하는데, 그것은 바로 바로 전의 입력 문장만 고려하고 이전의 문장들은 전혀 고려하지 않는다는 것이다. 이 모델은 maximum likelihood 기법에 따라 데이터 안에 가장 많이 있는 상황에 따라 답변한다. 하지만 학습할 경우의 수가 너무 많기 때문에 이를 다 커버하기 힘들고 문맥을 많이 고려하지 못한다.

6. Context-sensitive Model

   ​	기존의 챗봇은 입력->출력하는 하나의 문장만 고려하지만, 여기에 이전 문장까지 고려하는 방법이 바로 context-sensitive model이다. 문장 2개를 embedding 해서 벡터를 만들고 이 벡터를 이어 붙여 디코더의 히든 벡터에 바로 넣어주는 방식으로 모델이 작동한다.

7. Mutual Information-based Objective Function

   ​	이전의 모델은 maximum likelihood estimation을 활용하였지만 이 모델은 maximum mutual information으로 object function을 바꾸었다.

8. Personal-based Model

   ​	이전 챗봇들은 페르소나, 즉 성격이 전혀 없었다. 같은 내용을 다른 문장으로 물어보면 같은 챗봇이라도 다른 대답이 나온다는 것이다. 그래서 user를 고려하여 모델에 페르소나를 준다. hidden vector를 계산할 때 입력 단어와 이전 시간 단위의 히든 벡터를 사용하고, 이에 추가로 user information까지 고려하는 방식을 활용한다.




#### **Reference** ####
 Tacademy [토크ON세미나] 자연어처리-자연어처리에 대한 이해
