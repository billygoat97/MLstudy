# Machine Learning Stanford Lecture 3

Class: CS229
Created: Jul 9, 2020 9:53 PM
Reviewed: Yes
Type: Lecture

# Locally weighted linear regression

not all problems are globally linear.

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled.png)

left one is linear, which is underfitted, right one is overfitted.

왼쪽 것은 선형, 즉 너무 untrained, 오른 쪽은, dataset에 너무 최적화가 되어버린 overfitting이다.

we'd need to gain one in the middle.

따라서 중간의 값을 가지도록 하는 것이 좋다.

(Why 오른쪽의 것이 좋지 않은가? → 우리는 trainingset 를 검증하는 것이 아닌 test set를 검증해 보았을때 원하는 결과가 나와야 하는데 training set ≠ test set이기 때문에 너무 trainingset 에 최적화가 되면 안된다)

So we have to find that one in linear function(locally) 

우리는 선형으로 중간 결과를 얻기 위해서는 local 함수들을 얻어야 한다. 

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%201.png)

in here, w(i) is called weight (which is non-negative) 

w(i)는 가중치로, 각각의 값과 거리에 대한 가중치에 차등을 둠을 의미한다.

so we have to make w(i) small enough so that

w(i)가 너무 크지 않아야 cost function이 fit를 영향을 주지 않도록 하여야 한다.

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%202.png)

error cannot effect (ignore) the fit.

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%203.png)

// 잘 이해 되다가 왜 weight가 정확히 이 수식이 나오는지 잘 이해가 되지 않습니다 거리에 따른 가중치를 주는 것은 이해가 되지만, 가우스 커널 function이 가장 잘 표현한다고만 설명이 되어있어 설명이 살짝 부족한거 같은데 이에 대해서 잘 아시면 이야기 해주시면 감사하겠습니다.

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%204.png)

참고: [https://beginningwithml.wordpress.com/2018/07/02/5-locally-weighted-linear-regression/](https://beginningwithml.wordpress.com/2018/07/02/5-locally-weighted-linear-regression/)

standard fit of w(i) is as shown above.

if |x(i)-x| is small, then w(i) is close to 1

만약에 |x(i)-x| 이 작다면, e^0이므로 1에 가까울 것이고, 크다면 0으로 수렴한다. 즉, 가까우면 중요, 멀다면 안중요.

else, w(i) is close to small

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%205.png)

controls how quickly the example falls off with distance of its x(i) from query point x

it is called band width parameter.

x와 x(i)와의 거리가 어느 단위로 , 어느 거리로 떨어져 있는지 알려주는 band width parameter

Locally weigthed linear regression is non-parametric algorithm

which is opposite of parametric algorithm. so it uses non-fixed parameter,

so hypothesis h grows linearly with the size of the training set.

hypothesis는 training set 가 커지면서 linearly 커진다.

# Probabilistic interpretation

(선형회귀분석 중 확률형 모형)

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%206.png)

target variables are related to above equation.

e(i) is an error term that captures unmodeled effects. (매우 연관성 떨어지는 결과) (모델과 연관성 떨어지는)

e(i) ⇒ distributed IID

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%207.png)

so we call this likelihood function L(theta) (왜냐면 e(i)가 independent, so is y(i))

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%208.png)

our goal is to make maximum likelihood so make the data high probability as possible.

maximum likelihood estimation = 미지의 θ인 확률분포에서 뽑은 표본(관측치) x들을 바탕으로 θ를 추정하는 기법

so we should choose theta to maximize L(theta).

it is hard to calculate L(theta) because it is all multiplied. so, putting Log helps calculation easy.,

로그를 L(세타)에 씌운것은 계산의 편의성을 위해서이다.

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%209.png)

so in summary, least squares regression correspond with maximum likelihood estimate of theta.

# Logistic Regression

it is easy for us to get target boundary in 0~1. but it is most likely that it is over 1 or less than 0.

so we use sigmoid function. (or logistic funciton)

시그모이드 함수를 통해 0~1 값을 도출해냄

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2010.png)

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2011.png)

picture above shows the result.

if z → infinity, g(z) goes to 1

if z→ -infinity, g(z) goes to 0

so, the result ends in [0,1]

z범위에 관계없이 항상 0~1사이의 값으로 고정된다.

derivative of sigmoid function is:

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2012.png)

시그모이드 함수의 도함수는 위와같이 표현되는데,

assume that

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2013.png)

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2014.png)

so we can change it into

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2015.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2015.png)

to maximize likelihood, we use gradient ascent. (도함수의 성질 이용)

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2016.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2016.png)

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2017.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2017.png)

# Newton's Method

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2018.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2018.png)

Much quicker to calculate. get derivative and get x coord, get derivative and so on.

so it leads to f(theta) ⇒ 0

뉴턴 method는 점진적으로 일정 값으로 변경하는 것이 아닌, 도함수를 이용하여 값을 급진적으로 바꾸는 방식이다.

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2019.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2019.png)

if theta is vector valued, we have to generalize newton's method into vector setting.

![Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2020.png](Machine%20Learning%20Stanford%20Lecture%203%20e93e10e4beaa491da2df78579d9f23bf/Untitled%2020.png)

Newton's method enjoys faster convergence than batch gradient descent and much fewer iteration, but if size gets bigger, it gets expensive because of inverting n-by-n hessian. so as long as n isn't so big, it is much likely to use newton method.

if newton's method is applied to maximize logistic regression, resulting method is called fisher scoring.

뉴턴의 기법은 batch gradient descent 보다 dataset가 적을 경우에는 훨씬 빠르지만, dataset가 커지면 n^n hessian을 변환하기 때문에 너무 비용이 커지게 된다. 그럼에도 불구하고, dataset의 사이즈가 적절하면 newton method 방식을 쓰는 것이 합리적이다.