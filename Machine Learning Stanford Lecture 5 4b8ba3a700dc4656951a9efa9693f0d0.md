# Machine Learning Stanford Lecture 5

Class: CS229
Created: Jul 15, 2020 10:43 AM
Reviewed: Yes
Type: Lecture

# Generative Learning Algorithms

Discriminative Learning algorithms

[https://hwiyong.tistory.com/27](https://hwiyong.tistory.com/27)

1. 코끼리와 개를 distinguish하려고 할 때,  logistic regression이나 perception알고리즘은 직선, decision boundary로 개와 코끼리를 분리하는 것을 찾으려고 한다. 새로운 동물을 입력받았을때, 이 동물이 바운더리 안에 있는지, 밖에 있는지에 대해서 예측할 수 있다.

2. 이와는 다르게 다른 접근법이 있다.  코끼리가 어떻게 생겼는지에 대한 모델을 생성하고, 개가 어떻게 생겼는지에 대한 모델을 각각 생성한다. 새로운 동물이 등장했을때, 이 동물이 코끼리 모델과 match인지 개와 match인지 비교해볼 수 있다. 

1 에서 접근한 방식은 input X를 바로 0,1로 label하여서 바로 값을 얻는 직관적인 discriminative learning algorithm이다. 2에서 접근한 방식은 p(y|x)을 모델링하는 generative learning algorithm이다. 

2의 방식은, 개일 경우 0, 코끼리일 경우 1이라고 표현을 하면, p(x|y=0)은 개의 feature을 분포로 표현하며, p(x|y =1 )은 코끼리의 feature을 분포로 표현한다. 사전확률인 p(y)을 모델링하고, p(x|y)를 모델링하면, 우리는 Bayes Rule을 이용하여 사후확률인 posterior distribution을 얻을 수 있다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled.png)

분모인 p(x)는 결과론적으로 constant인 양의 상수이기 때문에 p(y|x)에 영향을 주지 못하기 때문에,

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%201.png)

이렇게 표현할 수 있다.

## Gaussian discriminant analysis

### the multivariate normal distribution

일반적으로 가우시안 분포는 정규분포 그 자체라고 불리울 정도로 익숙할 것이다. 그러나, 가우시안 분포에서 variable이 multi 개수일 경우에는 그림이 정규분포와는 조금 달라진다. 이 분포에서는 평균 벡터인 mu라고 불리우는 것과, covariance matrix인 시그마가 분포에 영향을 미치는 parameter이 된다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%202.png)

이러한 식이 나오는데, (n은 변수의 종류 갯수) 2개의 변수가 있는 경우의 그래프를 보면,

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%203.png)

각각 왼쪽으로 부터 시그마가

1 0             0.6 0            2 0

0 1             0 0.6            0 2 이다.

이 경우에는 뮤가 0,0으로 그래프의 정중앙에 위치하는데 이를 standard normal distribution이라고 부른다.

눈여겨 볼 점은 시그마는 항상 symmetric함을 알 수 있다. (교수님이 그 외의 경우를 보지 못했다고 함~아마?)

또다른 예제는 다음과 같다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%204.png)

위 그래프에서는 1001이 아닌 경우에는 45도 돌아간 그래프인것을 볼 수 있다.

등고선처럼 표현하면 다음과 같이 표현할수있다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%205.png)

왼쪽부터 1001 /1 0.5 0.5 1/ 1 0.8 0.8 1

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%206.png)

음수인 경우에는 compress된 방향이 -45인것을 볼 수있다.

다음은 mu의 값이 달라진 경우를 보겠다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%207.png)

각각 양수 음수에 따라 평행 이동함을 알 수 있다.

## The Gaussian Discriminant Analysis model (GDA)

분류 문제에서 input x 이 연속적인 랜덤 변수일 경우에는 GDA 모델을 이용할수 있다. 이는 모델 p(x|y)는 다중 변수 정규분포를 이용하기 떄문이다.

알아놓으면 좋은 것:

likelihood estimation → maximize p of y given x

GDA maximize p of x and y

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%208.png)

여기서 우리가 조정하는 parameter은 phi, 시그마, mu0, mu1이다. 여기서 log-likelihood of data는 

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%209.png)

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2010.png)

위 변수같은 경우에 최대 likelihood estimate of parameter이 된다.

mu0,mu1은 각각 각 traindata의 mean값이다 (평균)

cf) arg min/max란, 어떤 함수를 최소 최대로 만드는 정의역의 점들, 매개변수들이다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2011.png)

등고선으로 표현하였을 경우에 GDA는 다음과 같이 표현되게 되며, x와 o의 classification이 이루어지게 됨을 알 수 있다. (한눈에 봤을 때에도 logistic보다 좋아보임)

## discussion: GDA and Logistic regression

어떤게 더 좋을까?

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2012.png)

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2013.png)

로 표현이 되는데, (세타는 phi, sigma mu0, mu1으로 표현됨)이는 logistic function의 형태이다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2014.png)

사전확률이 다중 변수 가우시안 이라면, 사후확률은 logistic function이지만, 그 역은 성립하지 않는다.

이는 GDA가 더욱 강력한 model assumption을 가지는 것을 알 수 있다. 

결국! assumption이 옳다면, 무 조 건! GDA가 logistic regression보다 더 좋은 알고리즘임을 알수 있다

(asymptotically efficient) 심지어 작은 dataset에서도 마찬가지이다.

그러나 non 가우시안 분포에서는 assumption이 옳지 않으므로 logistic regression이 더욱 괜찮은 prediction이 됨을 알 수 있다 such as 푸아송 분포같은 경우.

따라서 어떠한 분포인지 파악을 하여서 가우시안 분포임을 안다면 GDA아니면 logistic이 더욱 좋은 방안임을 알 수 있겠다.

# Naive Bayes → 순진한? WHY?

classification에서 순진한이라는 단어가 들어감은 다소 억지스러운 가정이 들어갔을때 나오기 때문에 naive bayes라는 이름이 붙었다고 한다.

이는 바로 모든 input x가 독립적인 (independent) 한 경우인 것을 가정했을때 나온다고 할 수 있다.

such as spam 메일 판별법

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2015.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2015.png)

수많은 단어들 중에 어떤것이 valid한지에 대해서 표현하는 x → {0,1}^n ⇒2^n possible values

모든 단어들이 independent하기 떄문에 우리는 위와 같은 식을 다음과 같이 정리할 수 있다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2016.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2016.png)

억지스러운 가정이지만 실제로는 다양한 문제들에 적용이 되며 실제로 잘 실행이 된다.

![Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2017.png](Machine%20Learning%20Stanford%20Lecture%205%204b8ba3a700dc4656951a9efa9693f0d0/Untitled%2017.png)

위에서부터 phi를 설명하자면,

y=1일때 xj라는 text가 valid할 확률이며, (xj(i)가 1이면서 y(i) = 1일떄)

y=0일떄 xj라는 text가 valid할 확률이며,(xj(i)가 0이면서 y(i) = 1일떄)

phi(y)는 실제 spam메일일 확률이다.

이는 계속 변경될 때에도 식이 일관성을 가진다.