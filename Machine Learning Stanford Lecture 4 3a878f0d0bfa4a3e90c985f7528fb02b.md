# Machine Learning Stanford Lecture 4

Class: CS229
Created: Jul 10, 2020 7:03 PM
Reviewed: Yes
Type: Lecture

# Digression: perceptron learning algorithm

과거에 이용됐던 1950년대 algorithm으로,

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled.png)

처럼 결과값을 0 혹은 1로 강제하게 하는 알고리즘이다.

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%201.png)

위와 같은 update 알고리즘을 가지는 것이 perceptron learning algorithm이다.

y(i) - h세타(x(i)) ⇒ 0 or +- 1 ⇒ scalar

따라서 세타j는 cost function이 활성화 됨에 따라서 결과값이 달라짐.

perceptron이 동작하는 방식은 다음과 같다. 각 노드의 가중치와 입력치를 곱한 것을 모두 합한 값이 활성함수에 의해 판단되는데, 그 값이 임계치(보통 0)보다 크면 뉴런이 활성화되고 결과값으로 1을 출력한다. 뉴런이 활성화되지 않으면 결과값으로 -1을 출력한다. (wiki 검색)

→ 결과값이 다르게 나오면 고친다는 의미 (문제풀다가 틀리면 다시 문제 푸는 것과 동일)

학습을 하면서 weight 조절

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%202.png)

[https://sacko.tistory.com/10](https://sacko.tistory.com/10)

단점: XOR 연산 불가능

개선점: 다중 perceptron이용하면 xor 연산 가능하다.

그러나, perceptron의 예측으로 의미 있는, 믿을만한 확률 해석이 되지 못한다.

# Generalize Linear Models

일반화 선형 모델이란?

일반화 선형 모형은 종속변수가 정규분포하지 않는 경우를 포함하는 선형모형의 확장이며 glm()함수를 사용한다.

### exponential family (지수족)

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%203.png)

지수족이란 지수함수와 연관되어 있는 특정 확률 분포를 의미. 다양한 분석을 massaging하면 위와 같은 형태로 정리가 가능해진다.

eta(n처럼 생긴것): natural parameter 혹은 canonical parameter of distribution

T(y): sufficient statistic → 여기서는 그냥 y로 통칭

a(eta): log partition function

베르누이 분포와, 가우시안 분포 또한 지수족으로 상단의 형식으로 표현이 가능하다.

베르누이 분포의 경우,

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%204.png)

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%205.png)

둘째줄부터 exponential과 log를 동시에 쓰면 같은 효과로 없어진다. 따라서 식을 정리시에 다음과 같이 표현이 된다.

이때, 

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%206.png)

라고 가정한다면, 지수족에 포함됨을 알 수 있다.

가우시안 분포의 경우, 선형 회귀에서는 시그마^2은 아무런 우리의 세타와 cost function에 결과 영향이 없기 때문에(양의 상수),  시그마^2에 아무런 값을 넣든지 간에 상관이 없기 때문에 시그마^2 = 1을 넣도록 하자, 그럴 경우

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%207.png)

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%208.png)

위의 식이 아래의 식으로 변하게 되고,

exponential은 지수의 더하기는 곧 곱셈으로 표현할수 있다.

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%209.png)

다음과 같이 정리 될 수 있다.

분류 작업을 할 때에는 다음과 같은 방식을 채택하면 된다

1. 실제 값 → 가우스 분포
2. 이진 분류 → 베르누이 분포
3. Count-data → 푸아송 분포
4. 실제 양수 값 → gamma, exponential

전부 다 지수족에 포함된다.

### Constructing GLMs

우리의 문제에 대한 good model을 어떻게 생성을 하는가?

어떻게 random variable y를 주어진 x로부터 잘 도출을 하는가?

이에 대한 해답으로 3가지 assumption을 해야 한다.

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2010.png)

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2011.png)

각각 설명을 하자면,

1. 분포가 지수족에 해당되는 경우여야 한다.
2. T(y)를 예측하는데, T(y) = y이므로, 우리가 learned 한 h의 h(x)의 결과값은 h(x) = E[y|x]를 만족해야 한다. 이는 logistic, linear 회귀분석 모두 해당한다.
3. eta와 x는 linearly 연관되어있어야 한다. (상수배)

learning update Rule

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%201.png)

### Logistic Regression

이진 분류 문제에서 가장 많이 이용되는 회귀 분석이다. 상기 언급했듯이, 이진 분류는 베르누이 분포를 이용하는 것이 가장 좋다.

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2012.png)

일단, 베르누이 분포라는 것은 지수족이고, GLM으로서의 조건을 만족한다.

cf) canonical response function = 일반화 선형모형

### Softmax Regression

항상 모든 회귀 분석이 이진 분류는 아니다. 이럴 경우에는 softmax regression을 이용하여 multi class classification을 이용한다.

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2013.png)

간단히 이야기 하자면, 각각의 x를 분석을 하여 exponential → normalize → p(y) 를 통해 이 결과값이 어느 class에 해당하는지, 가장 값이 높게 나오는지를 분석하여 one hot encoding 마냥 가장 큰 값을 가지고 있는 곳, class에 배당을 한다. 이곳에서는 one hot vector이라고 한다.

![Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%204%203a878f0d0bfa4a3e90c985f7528fb02b/Untitled%2014.png)

pdf에 나온 식 관련 연산은 수업 시간에 자세하게 다루지는 않았음.