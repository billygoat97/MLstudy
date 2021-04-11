# Machine Learning Stanford Lecture 15.5

Created: Aug 19, 2020 12:04 PM
Reviewed: Yes

# PCA (Principal components analysis)

identify the subspace in which the data approximately lies.

Dataset

m different type (max speed, turn radius...)

만약에 xi ⇒ 시간당 킬로, xj ⇒ 시간당 마일 (그러면 사실 같은거임)

n subspace ⇒ n-1 subspace

우리는 이러한 겹치는걸 어케 아냐???

less contrived(인위적인) 예로,

교수님이 사랑하는 헬리콥터 x1(i)가 헬리콥터 스킬이고, x2(i)가 애정도라면, 두개는 연관되어있음~

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled.png)

noise도 적고 그렇다~

PCA 알고리즘이란? 

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%201.png)

하기 전에 데이터 전처리를 해야 함 variance랑 bias 처리해야함

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%202.png)

1. bias 평균화
2. normalize → bias 감소

3,4. variance 감소 (rescale to have unit variance → 나누기)

전처리가 끝났으니?

우리가 해야 할 것? ⇒ major axis of variation u를 compute 해야 함.

데이터가 어떤 방향으로 되어있는지

찾는 한가지 방법: (cf 다른방법은manifold learning)

unit vector u를 찾는 것 (길이 1로 조정)

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%203.png)

위의 직선은 u1으로 projection 한 경우에도 큰 variance가 있음을 알 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%204.png)

이번 선은 u2로 projection시에 variance가 매우 감소했음을 알 수 있다.

우리는 어떠한 u를 선택해야 할까?

우리는 이러한 u를 선택하기 위해(자동적으로) formalize해서 살펴보기로 한다

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/pca02.gif](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/pca02.gif)

점 x와 unit vector u에 대해 x가 u에 projection한 것은, $x^Tu$로 표현된다. x(i)가 데이터세트 위의 점이면, u로 projection한 것은 $x^Tu$ 의 거리와 같다. 따라서 projection variance을 maximize 하려면 u를 이렇게 해야한다

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%205.png)

||u|| = 1이므로

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%206.png)

이다 ⇒ empirical covariance matrix of the data (평균이 0이라는 가정 하에)

PCA는 다음과 같은 단계로 이루어진다.

1. 학습 데이터셋에서 분산이 최대인 축(axis)을 찾는다.
2. 이렇게 찾은 첫번째 축과 직교(orthogonal)하면서 분산이 최대인 두 번째 축을 찾는다.
3. 첫 번째 축과 두 번째 축에 직교하고 분산을 최대한 보존하는 세 번째 축을 찾는다.
4. `1~3`과 같은 방법으로 데이터셋의 차원(특성 수)만큼의 축을 찾는다.

정리하자면, 우리가 1차원 subspace를 이용하여 data를 approximate하려면, 우리가 u를 prinipal eigenvector of $\sum$로 설정해야함.

더 일반적으로, 우리가 k차원으로 하고싶으면, u1, .., uk를 top k eigenvector로 골라야 함

따라서 x(i)를 표현하기 위해서는 

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%207.png)

를 compute하면 된다.

PCA는 dimension감소 알고리즘이다. u1 ~ uk는 first k principal component of data 이다.

# ICA (independent Components Analysis)

칵테일 파티~

여러 voice가 있을 때, 근원적인 voice를 찾으려고 하는 것 

#of speaker ≤ #of microphone (전제조건)

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%208.png)

s가 근원적으로 우리가 찾아야 하는것,

x가 주어진 것

A가 변형 Matrix (Mixing Matrix)

우리가 구해야 하는 것은 s

즉, A의 역행렬→ W(Unmixing Matrix)를 구할 수 있다면, S를 구할 수 있게 된다.

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%209.png)

위의 식을 computing 하면서 구할 수 있다.

우리는 W를 

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2010.png)

위와 같이 표현함으로서 s를 w와 x로 표현할 수 있다.

## ICA ambiguities (모호함?)

이는 우리가 어떻게 원래의 것을 알아 낼 수 있는지!

permutational matrix → 순열? 각 행과 열에 1이 하나씩이고 나머지가 0인 matrix ⇒ P

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2011.png)

x(i)만 주어진다면, W와 PW를 구분 X

근데 다행히 대부분의 경우 별 상관 x

그리고 w(i) 의 정확한 수 (배수로만 알수 있음) 는 알 수 없다. 그러나 A ⇒ 2A라면, s(i) ⇒ 0.5*s(i) 로 scale하면 됨. 

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2012.png)

하지만 이러한 모호함 또한 칵테일 파티에서는 중요 x

왜냐? → voice 크기 차이일 뿐 (참고로 음수랑 양수는 volume에 영향 x)

s(i)가 non gaussian이면 ICA를 할 수 있다.

## Densities & linear transformation

Ps(s) ⇒ 를 구하는데, x = As 임을 이용하여, 구해야 한다.

s = Wx (W는 A의 역행렬)

Px(x) = ps(Wx)

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2013.png)

## ICA Algorithm

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2014.png)

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2015.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2015.png)

p(s) ⇒ p(x)로 변환하는 과정에서 다음과 같이 변한다. (왜냐? →실제 확인 가능한 데이터는 x)

likelihood 계산하려는데, 여기서 계싼의 편의를 위해 log likelihood 씀

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2016.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2016.png)

learning (update)를 위해 MLE를 이용하여 위와 같은 식을 도출해낼 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2017.png](Machine%20Learning%20Stanford%20Lecture%2015%205%20c4ae6147189f4356b40fbe4a49140b98/Untitled%2017.png)

learning algorithm이 converge하게 되면 이때 얻은 W를 이용하여 원래 것을 구할 수 있게 되어  recover original sources 를 compute할 수 있다.