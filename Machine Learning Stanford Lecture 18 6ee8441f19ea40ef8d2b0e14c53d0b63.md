# Machine Learning Stanford Lecture 18

Class: CS229
Created: Aug 13, 2020 3:20 PM
Reviewed: Yes

# Continuous state MDP

실제로 우리는 실생활에서 이동을 할 때나, 물체가 이동 할 때 등, continuous한 state를 보인다. 이에 따라 state를 다르게 표현을 한다 단순히 (x,y)가 아님.

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled.png)

6 dimensional State!

S = R^6 은 무한정한 state이다. why? → infinite한 position있기 때문

만약에 헬리콥터라면 더 많음!

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%201.png)

그러면 실생활에 못씀? That's No No

# Discretization

→ 쉽게 말해서, 조각조각 범위를 하나로 think할 수 있음.

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%202.png)

단점이 두개 있음

1. naive 하다.

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%203.png)

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%204.png)

위의 graph를 Discretization하면 밑의graph처럼 된다. not a good representation

2. 다차원으로 가면 안됨

curse of dimensionality

dimension이 늘어날수록 k dimension, 100 value라고 하면,

$100^k$ 의 discrete state가 됨 고로 1d나 2d로만 하세여 ^오^

# Value Function approximation

고로 다른 방안이 필요하다.  디스크리뭐시기 빼고 다른 방법이!

## Using model or simulator

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%205.png)

how to get model

한 방법은 physics simulator

다른 방법은 learn model from data → 이거 쓴답니다

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%206.png)

st+1를 st와 at로부터 배울 수 있ㄷr. (linear version) → works okay for helicopter

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%207.png)

A와 B를 배웠다? → one option is to build a deterministic model

혹은

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%208.png)

이렇게 할 수 있음.

stochastic model 을 쓰면? → 입실론을 가우시안 분포에서 구해서 항상 이용함 simulator generates random errors

et는 noise term

노이즈는 넣는것이 좋다 → 실제로 noise 비슷한것이 많을 것

근데 non -linear일 경우엔?

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%209.png)

가능

model based RL 이라고 우리는 이것을 부른다. ㅋㅋ 미국말같다

## Fitted value iteration

choose feature $\phi(s)$ of state s

V(s) = $\theta^T\phi(s)$

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2010.png)

Value iteration

1. choose state randomly, and $\theta$:= 0 (아 쓰다 보니 글 써져있네 캡처함)

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2011.png)

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2012.png)

Psa ↔q(a) (estimate of expectation) (compute)

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2013.png)

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2014.png)

fitted value iteration은 항상 converge한다고 proved 안됨. 근데 많은 문제에서 해결책이 됨 → what?

fit value iteration은 k =1로 설정함으로 simplify 가능

⇒ the value returns to same value 이기 때문에 가능하다.

Fitted VI gives approximation to V*

Implicitly defines $\pi$*

$\pi$*(s) = argmax Es'psa[V*(s')]

used s'1 ... s'k ~Psa to approximate expectation

![Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2015.png](Machine%20Learning%20Stanford%20Lecture%2018%206ee8441f19ea40ef8d2b0e14c53d0b63/Untitled%2015.png)

s' ~ Psa but with deterministic simulator