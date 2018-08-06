---
layout: post
title:  "Online Machine Learning"
date:   2018-08-06
excerpt: "Basic things about Online Machine Learning"
tag:
- markdown 
comments: false
---
# Online Machine Learning

![online vs. offline](C://Users//jeje9//Desktop//자연어스터디//online vs. batch.JPG)

### **Offline / Batch Learning**

​	오프라인 러닝 혹은 배치 러닝이란, 우리가 일반적으로 알고 있는 머신러닝의 한 부류이다. 이 알고리즘은 모든 데이터를 한꺼번에 모델에 넣고, 그렇게 train시킨 모델을 활용하여 테스트 샘플을 예측하게 된다.



### **Online Learning**

​	하지만 온라인 러닝은 한 번에 하나의 데이터만 관찰한다. 처음에 모델에 관하여 추측을 한 다음, 하나씩 데이터를 넣으며 파라미터에 대한 가중치를 수정해 나간다. 

​	이 때 모델이 수정되는 방향은 regret을 최소로 줄이는 것이다. regret이란 온라인 알고리즘과 이상적인 알고리즘의 성능 차이를 이야기한다. 쉽게 말해 loss를 줄이는 방향으로 결과값을 예측하게 된다고 이해하면 쉽다.



​	온라인 러닝의 특징은 다음과 같다.

* 컴퓨팅적으로 더 빠르고 공간 효율성이 높다

  ​	온라인 모델에서는 한 번에 하나의 데이터만 사용하고,  이미 사용한 데이터는 이후에 사용할 필요가 없으므로 그것을 따로 저장할 필요가 없어져 공간을 효율적으로 사용할 수 있다.

* 실행이 쉽다

  ​	순차적으로 하나씩 데이터를 넣기 때문에 알고리즘을 비교적 간단하게 만들 수 있다.

* 평가가 어렵다

  ​	온라인 러닝은 분포에 가정을 하지 않기 때문에 테스트셋이 없어 결과에 대한 평가가 어렵고, 그에 따라 알고리즘이 맞는지 틀린지 정확히 알기 힘들다.



​	이러한 특징을 갖는 온라인 러닝은 큰 데이터를 다루거나, 새로운 데이터가 시간에 의존적일 때 유용하다고 할 수 있다.

### **Algorithms of Online Learning**

​	우선 구체적인 알고리즘 3개를 설명하기에 앞서, 이들을 이해할 수 있는 간단한 예시를 보려고 한다.

![예시](C://Users//jeje9//Desktop//자연어스터디//ex1.JPG)

​	우리는 내일 비가 올지 안올지를 예측하려고 한다. 11명의 개인은 서로 다른 파라미터에 따라 내일 비가 내릴지 여부를 결정하여 각각의 결과를 가지고 있다. 각각의 결과에 대해 우리는 가중치를 부여하여 결정하게 되는데, 최초의 가중치는 모두 1이다.

![예시2](C://Users//jeje9//Desktop//자연어스터디//ex2.JPG)

​	결과적으로 우리는 가중치를 모두 합한 결과가 큰 쪽으로 결정을 내리게 된다. 다음 날의 결과를 보고 우리는 가중치를 변경하게 되는데, 결과와 예측이 일치하면 가중치를 그대로 두고, 결과가 틀리면 본래 가중치에서 1.2를 나눈다. 그렇게 1000번을 반복하면 우리는 정확도가 거의 100프로인 결정을 내릴 수 있다.

![예시3](C://Users//jeje9//Desktop//자연어스터디//ex3.JPG)

​	이제 온라인 러닝이 가지고 있는 구체적인 3개의 알고리즘에 대해 알아보자

1) Randomized Weighted Majority Algorithm

![알고리즘1](C://Users//jeje9//Desktop//자연어스터디//al1.JPG)

​	우선 모두가 초기 가중치 1을 가지고 시작한다. 랜덤하게 한 사람을 골라서 그 사람의 결과를 예측으로 한다. 맞는 답을 찾고, 그것이 예측과 다를 경우 그 예측의 가중치를 줄이는 방향으로 알고리즘이 진행된다.



2) Winnow Algorithm

![알고리즘2](C://Users//jeje9//Desktop//자연어스터디//al2.JPG)

​	모두의 초기 가중치가 1인 것은 Randomized Weighted Majority Algorithm과 동일하다. 하지만 임의의 한 사람만 선택했던 것과는 다르게 가중치 행렬의 전치행렬과 예측 행렬을 곱하여 특정 수 n보다 크면 1이라고 예측하고, 아닌 경우 0이라고 예측한다. 

​	1이라고 예측했지만 실제 결과가 0이라면 가중치를 모두 2배 늘리고, 0이라고 예측했지만 실제 결과가 1이라면 가중치를 모두 절반으로 줄이는 방식으로 모델을 수정한다.



3) Stochastic Gradient Descent

​	Stochastic Gradient Descent(SGD)는 전체 데이터 대신 일부 데이터의 모음(mini-batch)에 대해서만 loss function을 계산하여 훨씬 빠르게 결과를 낼 수 있다.

​	SGD는 Momentum, NAG, Adagrad, AdaDelta, RMSprop 등의 여러 변형 알고리즘을 가지고 있다. 하지만 이 페이지에서는 가장 기본적인 SGD의 파이썬 코드만 보려 한다. 각 변형 알고리즘에 대한 추가적인 설명은 아래의 Reference 마지막 링크를 참고하길 바란다.



```{python}
# import the necessary libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
```
```{python}
# load the data
boston = load_boston()
```
```{python}
# load the data
boston = load_boston()
```
```{python}
# split the data into training and tests sets
X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)
```
```{python}
# instantiate the model
robust = SGDRegressor(loss='huber',
                      penalty='l2', 
                      alpha=0.0001, 
                      fit_intercept=False, 
                      n_iter=5, 
                      shuffle=True, 
                      verbose=1, 
                      epsilon=0.1, 
                      random_state=42, 
                      learning_rate='invscaling', 
                      eta0=0.01, 
                      power_t=0.5)
```
```{python}
sc_boston = StandardScaler()
X_train_boston = sc_boston.fit_transform(X_train_boston)
X_test_boston = sc_boston.transform(X_test_boston)
```
```{python}
def sum(a, b):
    return a+b
```
```
-- Epoch 1
Norm: 0.01, NNZs: 13, Bias: 0.000000, T: 404, Avg. loss: 2.274738
Total training time: 0.00 seconds.
-- Epoch 2
Norm: 0.01, NNZs: 13, Bias: 0.000000, T: 808, Avg. loss: 2.274682
Total training time: 0.00 seconds.
-- Epoch 3
Norm: 0.01, NNZs: 13, Bias: 0.000000, T: 1212, Avg. loss: 2.274675
Total training time: 0.00 seconds.
-- Epoch 4
Norm: 0.01, NNZs: 13, Bias: 0.000000, T: 1616, Avg. loss: 2.274671
Total training time: 0.00 seconds.
-- Epoch 5
Norm: 0.01, NNZs: 13, Bias: 0.000000, T: 2020, Avg. loss: 2.274669
Total training time: 0.00 seconds.
```

```
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
  
SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
       fit_intercept=False, l1_ratio=0.15, learning_rate='invscaling',
       loss='huber', max_iter=None, n_iter=5, penalty='l2', power_t=0.5,
       random_state=42, shuffle=True, tol=None, verbose=1,
       warm_start=False)
```

```{python}
# check the RMSE on the test set
mean_squared_error(y_test_boston, robust.predict(X_test_boston)) ** 0.5
```
```
23.132572138391186
```





#### **Reference** ####

<https://en.wikipedia.org/wiki/Online_machine_learning>

<https://dziganto.github.io/data%20science/online%20learning/python/scikit-learn/An-Introduction-To-Online-Machine-Learning/>

<https://www.analyticsvidhya.com/blog/2015/01/introduction-online-machine-learning-simplified-2/>

<https://www.slideshare.net/queirozfcom/online-machine-learning-introduction-and-examples>

<http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html>