# Machine Learning Stanford Lecture 19

Class: CS229
Created: Aug 13, 2020 3:20 PM
Reviewed: Yes

## overview

State action rewards

Finite horizon MDP

Linear Dynamical systems

- model
- LQR (linear quadratic regulation)

이전에 배운건, continous 또는 discrete 배웠는데,

우리는 general을 원함 외쳐 EE!

We want to write equations that make sense for both the discrete and
the continuous case. We’ll therefore write (~따라서)

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled.png)

immediate reward depends on action+state you take

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%201.png)

지금 취하는 액션도 영향을 준다... 고로 max를 앞으로!

Finite horizon(호라이즌) MDP에서 수렴이 없어 불필요한 $\gamma$대신  Horizon time T 로 바꿈

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%202.png)

stationary는 시간 따라 달라짐

non stationary policy → changees over time

St+1 ~ Pstat

ex)

changing dynamics

weather forecast

industrial ~ 글씨 좀 사람답게 써라 진짜 ㅠ

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%203.png)

별 친 이유는 

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%204.png)

expected total payoff starting in state s at time t execute $\pi^*$

→ for dynamic programming

## Linear Quadratic Regulation (LQR)

linear 

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%205.png)

Psa : ——————————————————요 식

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%206.png)

.reward는 다음과 같고, 항상 negative

reward is equivalent to saying that we want our state to be close to the origin

Where to get A, B?

S0 (i) ... St(i)

i ~= 0~

linearize a non linear model

![Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2019%20e6513c3100f94fcaaef639c96c072853/Untitled%207.png)

if u want  MDP modeled as linear dynamical system, with quadratic cost function, V star is quadratic function so, V* can be computed

59분 03초부터 다시