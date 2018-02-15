---
layout: post
title:  "Practice-Logistic Regression"
date:   2018-02-15
excerpt: "Simple codes to use logistic regression"
tag:
- markdown 
- practice
- python
comments: false
---


## 로지스틱 회귀 Practice
로지스틱 회귀가 실제로 어떻게 쓰이는지 실습을 통해 알아보자. 데이터는 Kaggle의 Bike Sharing Demand에서 주어진 데이터이다. (https://www.kaggle.com/c/bike-sharing-demand)

로지스틱 회귀 사용은 간단하게 sklearn.linear_model이라는 라이브러리의 LogisticRegression을 import하면 할 수 있다.


```python
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

train data를 'train'이라는 이름으로 불러온다.


```python
train = pd.read_csv("train.csv")
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.isnull().sum()
```




    datetime      0
    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    casual        0
    registered    0
    count         0
    dtype: int64



이 데이터는 시간, 계절, 휴일 여부, 날씨, 온도, 습도, 바람 등의 여러가지 변수에 따른 자전거 대여 횟수를 나타내고 있다.
다행히도 NA 값은 없다.

로지스틱 회귀는 종속변수가 범주형일 때 더 잘 작동하기 때문에 'count'라는 연속형 종속변수를 변형 시키려고 한다. 평균적 대여 횟수 이상인지 여부를 알려주는 'above average'라는 새로운 변수를 추가하여 종속변수로 삼기로 하자.


```python
train['count']
```




    0         16
    1         40
    2         32
    3         13
    4          1
    5          1
    6          2
    7          3
    8          8
    9         14
    10        36
    11        56
    12        84
    13        94
    14       106
    15       110
    16        93
    17        67
    18        35
    19        37
    20        36
    21        34
    22        28
    23        39
    24        17
    25        17
    26         9
    27         6
    28         3
    29         2
            ... 
    10856    525
    10857    353
    10858    268
    10859    168
    10860    132
    10861     81
    10862     41
    10863     15
    10864      3
    10865      5
    10866      7
    10867     31
    10868    112
    10869    363
    10870    678
    10871    317
    10872    164
    10873    200
    10874    236
    10875    213
    10876    218
    10877    237
    10878    334
    10879    562
    10880    569
    10881    336
    10882    241
    10883    168
    10884    129
    10885     88
    Name: count, Length: 10886, dtype: int64




```python
train['count'].mean()
```




    191.57413191254824




```python
a = []
for i in train['count']:
    if i >= train['count'].mean():
        a.append(1)
    else:
        a.append(0)
```


```python
train['above_average'] = a
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>above_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



01.독립변수가 연속형 변수(temp)일 때

독립변수를 'temp', 종속변수를 'above_average'로 하는 train set과 test set을 나눈다.


```python
feature_cols = ['temp']
X = train[feature_cols] 
y = train['above_average']
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

train set을 로지스틱 회귀식에 접합시킨다.


```python
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



접합시킨 회귀식을 다시 test set에 적용시켜 보았을 때 정확도는 아래와 같이 약 0.67정도가 나온다.


```python
print('Accuracy of Logistic Regression: {:.2f}'.format(logreg.score(X_test, y_test)))
```

    Accuracy of Logistic Regression: 0.67
    

02.독립변수가 범주형 변수(season)일 때

독립변수가 'season'과 같이 범주형 변수이면 거쳐야 할 단계가 한가지 더 있다. 그것은 바로 그 독립변수를 더미화 시켜주는 것이다. 더미화는 get_dummies로 간단하게 할 수 있다.


```python
train2 = pd.get_dummies(train, columns=['season'])
```


```python
train2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>above_average</th>
      <th>season_1</th>
      <th>season_2</th>
      <th>season_3</th>
      <th>season_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



여기서 주의해야 할 점이 있다. 보통 Random Forest와 같은 기법을 사용하면 더미화한 범주형 변수를 그냥 두고 모델을 돌려도 상관 없다. 하지만 회귀 모델에서는 더미화한 뒤에 꼭 하나를 빼주어야한다.


```python
train2.drop(train2.columns[15], axis=1, inplace=True)
```


```python
train2.columns
```




    Index(['datetime', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
           'humidity', 'windspeed', 'casual', 'registered', 'count',
           'above_average', 'season_1', 'season_2', 'season_3'],
          dtype='object')



독립변수를 'seaon'을 더미화 한 3개의 columns, 종속변수를 'above_average'로 하는 train set과 test set을 나눈다. 


```python
feature_cols = ['season_1', 'season_2', 'season_3']
X = train2[feature_cols]
y = train2['above_average']
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

train set을 로지스틱 회귀식에 접합시킨다.


```python
logreg.fit(X_train, y_train) 
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



접합시킨 회귀식을 다시 test set에 적용시켜 보았을 때 정확도는 아래와 같이 약 0.59정도가 나온다.


```python
print('Accuracy of Logistic Regression: {:.2f}'.format(logreg.score(X_test, y_test)))
```

    Accuracy of Logistic Regression: 0.59

#### **Reference** ####
https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/
