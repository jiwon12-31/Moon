---
layout: post
title:  "Summary of RDD Paper"
date:   2018-10-02
excerpt: "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing"
tag:
- markdown 
- Spark
comments: false
---
# Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing


## Abstract

RDD(Resilient Distributed Datasets) : 기존 computing framework의 비효율성을 보완하여 in-memory computation + fault-tolerant한 방식으로 Spark에서 구동

RDD는 in-memory computation을 통해 규모에 따라 순서를 정하여 수행함으로써 효율성을 높일 수 있고, coarse-grained transformation한 방식을 통해 fault-tolerant한 시스템을 가진다. 



## 1. Introduction

기존에 있던 frameworks에서는 중간 결과를 reuse하거나  같은 데이터에 대하여 여러개의 ad-hoc queries를 돌리는 interactive data mining에서 문제를 겪는다. 왜냐하면 결과를 reuse하기 위해서는 external stable storage system에 데이터를 저장해야 하는데, 이와 같은 과정을 거치면 실행 시간이 과도하게 길어지기 때문이다.

이를 해결하기 위해 Pregel, HaLoop과 같은 framework가 개발되었지만, 이들은 특정한 computation pattern만 지원하기 때문에 효율이 높지 않다.

RDD는 이러한 문제를 크게 해결한 새로운 framework라고 할 수 있다. RDD의 '효율적으로' 데이터를 저장하는 것에 가장 큰 목적을 두고 있다. 기존의 framework는 데이터를 복사하고 로그를 업데이트 하는 것을 across machines으로 수행하여 과도한 workload를 감당해야했지만, RDD는 많은 데이터에 같은 변형을 가하는 coarse-grained transformation의 방식에 기반을 두어 최대한의 효율을 달성하였다. 이는 데이터 자체에 변형을 바로 가하는 것이 아니라 계보(lineage)를 저장해둠으로써 일부 데이터가 사라져도 저장해놓은 계보를 통해 다시 기존과 같은 데이터를 생성할 수 있다는 장점을 가진다.



## 2. Resilient Distributed Datasets(RDDs)

### 2.1 RDD Abstraction

RDD는 read-only, partitioned collection of records이다. 이는 다른 RDD에서 만들거나 안정적인 data storage에서 불러와야만 만들 수 있고, 이 과정을 우리는 'transformation'이라고 부른다. RDD의 transformation은 매번 데이터를 변형하여 저장하는 것이 아니라 lineage만을 저장하여 효율성을 달성하였다. transformation 연산에는 map(), filter(), join() 등이 있다.

또한 RDD는 최적화를 위한 persistence와 partitioning이라는 특징도 갖는다. persistence는 user가 어떤 RDD를 재사용할지를 결정하고 memory에 저장하는 것이며, partitioning은 각 기록의 key에 근거하여 RDD의 요소가 분배되는 것을 의미한다.



### 2.2 Spark Programming Interface

프로그래머들은 1개 이상의 RDD를 transformation을 통해 정의하고, actions로 RDD를 사용할 수 있다. action을 통해 비로소 테이터가 변형되기 시작한다. action 연산에는 count(), collect(), save()가 있다. 이 때 Spark는 lazy execution을 활용하여 RDD에 저장된 lineage를 기록된 순서대로 처리하는 것이 아니라 효율을 높일 수 있는 순서대로 처리하게 된다.



**2.2.1 Example: Console Log Mining**

웹에서 error가 생겼을 때 HDFS에 저장된 매우 큰 로그에서 Spark를 활용하여 그 원인을 어떻게 찾을 수 있을까?

![1](https://user-images.githubusercontent.com/31986977/46329863-157a8d00-c64b-11e8-95e3-9dfd5c0adeab.PNG)

![2](https://user-images.githubusercontent.com/31986977/46329870-21fee580-c64b-11e8-9e9c-2128c45dd4b1.PNG)

![3](https://user-images.githubusercontent.com/31986977/46329875-2c20e400-c64b-11e8-884b-9af3322063be.PNG)



### 2.3 Advantages of the RDD Model

RDD와 DSM(distributed shared memory)을 비교함으로써 RDD의 장점을 알아보자.

RDD는 only writing만, DSM은 read&write 둘 다 가능하다. 이와 같은 특징으로 DSM은 효율적이고 fault-tolerant한 방식으로 테이터를 활용하기 어렵지만, RDD는 그렇지 않다. 과도한 checkpointing이 필요없기 때문에 lineage를 통해 쉽게 복구가 가능하고, 일부가 사라져도 전체 프로그램을 돌릴 필요 없이 그 부분만 다시 계산하면 된다.

또한 RDD는 느린 작업을 복사하여 다른 node에서 처리하도록 하는 방식으로 작업을 빠르게 수행하도록 도와준다. 이와 비슷한 맥락으로 RDD는 task를 직접 schedule하여 성능으로 높인다. 게다가 RAM에서 용량이 부족하다면 disk에 일부를 저장하여 기존의 data-parallel system과 비슷한 성능을 제공한다.



### 2.4 Applications Not Suitable for RDDs

RDD는 모든 데이터에 같은 과정을 적용하는 batch application에서 가장 큰 성능을 보인다. 하지만 asynchronous fine-grained updates에는 잘 맞지 않을 수 있다. 이러한 경우에는 오히려 기존의 시스템을 사용하는 것이 더 효율적이다.



## 3. Spark Programming Interface

Scala 언어를 사용한다.

![4](https://user-images.githubusercontent.com/31986977/46329888-3cd15a00-c64b-11e8-92b8-b3ae933fac56.PNG)

Spark를 사용하기 위해 workers 클러스터와 연결하는 driver program을 활용해야 한다. driver는 1개 이상의 RDD를 정의하고 action을 일으킨다. driver의 코드는 또한 RDD의 lineage를 track하기도 한다. worker는 여러 과정을 통해 RAM에 RDD를 저장하는 long-lived process를 말한다.



### 3.1 RDD Operations in Spark

![5](https://user-images.githubusercontent.com/31986977/46329898-4955b280-c64b-11e8-810e-61c8cd051094.PNG)

이전에 언급한 바와 같이 transformation은 새로운 RDD를 정의하고, action은 프로그램에 값을 반환하기 위해 연산을 수행하고 외부 저장소에 데이터를 작성하는 것이다.



### 3.2 Example Applications

**3.2.1 Logistic Regression**

![6](https://user-images.githubusercontent.com/31986977/46329907-5b375580-c64b-11e8-8a8d-3b77935a6fb6.PNG)

gradient descent를 활용하여 구분되는 지점 찾기



**3.2.2 PageRank**

![7](https://user-images.githubusercontent.com/31986977/46329913-62f6fa00-c64b-11e8-9efe-21885f21db33.PNG)



## 4. Representing RDDs

RDD를 제공함에 있어서 많은 transformation을 거친 lineage를 추적할 수 있는 표현(representation)을 고르는 것이 중요하다. -> simple graph-based representation

![8](https://user-images.githubusercontent.com/31986977/46329919-6c806200-c64b-11e8-8921-6c6dfb239b97.PNG)

그 표현은 5가지 정보(a set of partition, a set of dependencies of parent RDD, a function, metadata about its partitioning scheme and data placement)를 노출하는 공통적인 interface를 통해 이루어진다. 

** Dependencies

![9](https://user-images.githubusercontent.com/31986977/46329932-75713380-c64b-11e8-875d-b169af1cce47.PNG)

parent RDD의 각 partition이 child RDD의 한 partition에 의해 사용되는 narrow dependencies, 여러개의 child partitions가 의존하는 wide dependencies 두 개로 분류된다. 이러한 narrow dependencies의 경우 하나의 클러스터 노드 안에서 pipeline이 그려진다. 하지만 wide dependencies에서는 모든 parent partition에 있는 데이터를 필요로하고 클러스터에 구애받지 않고 pipeline이 생성된다. 또한 narrow efficiency에서 복구가 훨씬 쉽고, wide dependencies에서는 한 노드가 제 기능을 하지 못하면 완전한 재실행이 필요하다.



## 5. Implementation

### 5.1 Job Scheduling

![10](https://user-images.githubusercontent.com/31986977/46329962-7f933200-c64b-11e8-9c5c-259a56d2a865.PNG)

simple graph-based representation을 사용함. user가 action을 시행할 경우 lineage를 Directed Acyclic Graph(DAG)로 디자인해나간다. 각각의 단계는 최대한 narrow dependencies의 형태로 많은 pipeline을 포함한다. 각 단계의 경계는 wide dependencies를 위해 필요한 셔플 과정이 있는곳, 또는 parent RDD의 계산이 이미 이루어진 곳을 기준으로 나누어진다. 그 이후에 scheduler는 target RDD를 계산할 때 까지 각 단계로부터 missing partition을 계산하는 일을 수행한다.

scheduler는 delay scheduling을 시용하여 data locality에 기반해 일을 할당한다. 만약 어떤 task가 한 노드에 있는 메모리를 partition을 필요로한다면, 이 task는 그 node로 보내진다. 반대로, 만약 RDD가 HDFS file처럼 선호되는 위치에서 제공되면, partition으로 task를 보낸다.

wide dependencies에서는 fault로 인한 복구를 간단하게 하기 위해서 중간의 기록을 materialize해둔다. 

task를 수행하지 못한다면, 그 단계의 parents가 사용가능한 상황에서는 다른 노드에 그것을 다시 돌린다. stage가 사용불가능하게 된다면, missing partition을 계산하도록 task를 다시 제출한다.

Spark의 모든 계산이 driver program이 요구하는 action에 반응하여 일어나지만, lookup 과정을 통해 클러스터가 일을 수행하도록 하기도 한다. lookup이란 key에 의해 hash-partition된 RDD의 요소에 random한 접근을 제공하는 것을 뜻한다.



### 5.2 Interpreter Integration

![11](https://user-images.githubusercontent.com/31986977/46329970-87eb6d00-c64b-11e8-8929-ee97e7182053.PNG)

Scala interpreter는 JVM에 loading해서 클래스를 컴파일하는 방식으로 작동한다. Spark interpreter에 2가지 변화를 줄 수 있다. 첫번째는 Class shipping으로 HTTP를 통해서 worker node가 bytecode를 가져오도록 하는 것이다. 그리고 두번째는 Modified code generation으로 코드의 각 줄에 있는 object의 instance를 바로 참조할 수 있도록 한다.



### 5.3 Memory Management

Spark는 persistent RDD의 저장을 위해 세가지 옵션을 제공한다.

- in-memory storage as deserialized Java objects: 가장 빠른 성능을 제공한다.
- in-memory storage as serialized data: 공간이 제한되어있을 때 더 memory-efficient한 방법이다.
- on-disk storage: RAM에 담아두기 너무 큰 RDD에서 유용하다.



제한된 메모리를 이용하기 위해서 LRU eviction policy를 활용한다. 만약 새로운 RDD partition을 저장할 공간이 충분하지 않다면 우리는 가장 조금 활용된 RDD로부터 partition을 추출한다. 이는 대부분의 과정이 전체 RDD 위에서 돌아가 이미 메모리에 있는 partition이 미래에도 사용될 수 있다는 점에서 중요하다.



### 5.4 Support for Checkpointing

lineage가 failure 뒤에 RDD를 복구하기 위해서 항상 사용되지만, 이러한 복구가 시간이 오래 소모될 수 있기 때문에 안정적인 저장소에 몇몇 RDD checkpoint를 만들어 두는 것이 유용하다.

보통 checkpoint는 PageRank의 rank datasets와 같이 wide dependencies와 함께 긴 lineage graph를 가지고 있는 RDD에서 유용하다. 이 경우에 일부 데이터가 손실되면 전체를 다시 돌려야한다.  반면에 logistic regression example이나 PageRank의 link list처럼 narrow dependencies를 가지고 있는 RDD의 경우에는 checkpoint가 필요하지 않다.

Spark는 checkpointing을 위한 API를 제공하고는 있지만, 그 결정은 온전히 user에게 넘겨두었다. 



## 6. Evaluation

Spark는 Hadoop보다 몇십배 더 빠르며, 노드가 fail해도 Spark는 없어진 RDD partition만 다시 세움으로써 빠르게 복구할 수 있다.



### 6.1 Iterative Machine Learning Applications

![12](https://user-images.githubusercontent.com/31986977/46329979-92a60200-c64b-11e8-8b2a-bf431b24c388.PNG)

![13](https://user-images.githubusercontent.com/31986977/46329986-9afe3d00-c64b-11e8-943e-dc6059a6d579.PNG)



### 6.2 PageRank

![14](https://user-images.githubusercontent.com/31986977/46330003-b0736700-c64b-11e8-94dc-3cf936324e6c.PNG)



### 6.3 Fault Recovery

![15](https://user-images.githubusercontent.com/31986977/46330007-b8330b80-c64b-11e8-9de7-b2e4eb21874a.PNG)



### 6.4 Behavior with Insufficient Memory

![16](https://user-images.githubusercontent.com/31986977/46330014-bf5a1980-c64b-11e8-89a4-425010158ff8.PNG)



### 6.5  User Applications Built with Spark

![17](https://user-images.githubusercontent.com/31986977/46330022-c719be00-c64b-11e8-8d25-35dedbecd3d6.PNG)



### 6.6 Interactive Data Mining

![18](https://user-images.githubusercontent.com/31986977/46330031-ced96280-c64b-11e8-8434-e779c1bfeca2.PNG)



## 7. Discussion

RDD가 immutable한 속성과 coarse-grained trasformation 때문에 제한된 programming interface라고 여겨질 수 있지만, 이는 다양한 분야에 적용이 가능하다. 특히 RDD는 매우 많은 수의 cluster programming model을 표현할 수 있다는 장점이 있다.



### 7.1 Expressing Existing Programming Models

RDD는 매우 많은 수의 cluster programming model을 '효율적으로' 표현할 수 있다는 장점이 있다. 효율적으로 표현함으로써 RDD는 이러한 모델에 쓰여진 프로그램과 같은 결과를 생산할 수 있고, 또한 이러한 frameworks가 수행하는 optimization까지 잡을 수 있다.



### 7.2 Leveraging RDDs for Debugging

RDD의 lineage를 logging함으로써

- RDD를 나중에 다시 구축할 수 있고, user가 query를 상호적으로 사용할 수 있다.
- RDD partition을 다시 계산함으로써 single-process debugger에 어떤 task도 다시 돌릴 수 있다.



## 8. Related Work

Cluster Programming Models

Caching Systems

Lineage

Relational Databases



## 9. Conclusion

RDD is an efficient, general-purpose and fault-tolerant abstraction for sharing data in cluster applications.
