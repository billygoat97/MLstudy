# Machine Learning Stanford Lecture 20

Class: CS229
Created: Aug 13, 2020 3:20 PM
Reviewed: Yes

죄송;; 이거 하려고 했는데 이제야 올립미다.. 열심히 하겠슴당

# RL Debugging and Diagnostics

Autonomous Flight

자동비행

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled.png)

어떤 파트를 improve해야 하는지에 대해서 think!

만약에, controller given by $\pi$RL 이 이상하게 된다면?

3가지 가설이 있다.

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%201.png)

1. 헬리콥터 시뮬이 정확하다
2. 강화 알고리즘이 헬리콥터를 제대로 컨트롤한다 → Maximize payoff
3. maximizing payoff이 정확하게 자동비행이 반응한다
4. 아스날

→ 강화 학습된 컨트롤러는 실제 제대로 굴러간다~

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%202.png)

실제로 문제가 생기는 경우,

1. 시뮬에 잘돌아가고 실제로 안돌아가면, 시뮬에 문제가 있는것,
2. 사람이 강화학습보다 더 잘하면, 강화학습 알고리즘에 문제 있음
3. 아니면, cost function에 문제가 있음
4. Arsenal

Robotic dog 

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%203.png)

value function approximation

Policy Search

direct policy search (좋은거 바로 계산)

V* → $\pi^*$

Try to find a good policy

we have to come up with how we'll deal with $\pi$ ~= 1/(1 + $e^(\theta^(t)s))$

a stochastic policy is a function  $\pi$ : SXA ↔ R where pi(s,a) is the probability of taking action a in state s. (확률의 합은 1)

s상태에서 a를 할 확률이다~ 이말이야

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%204.png)

이렇게 됨. reasonable policy

#pendulum (진자 ← 검색 그만 ㅠ)

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%205.png)

phi가 더 크면 likely R될 것

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%206.png)

Reinforce algorithm:

loop: 

sample s0 a0 s1 a1 ...

compute payoff : R(s0) ...

update : $\theta := \theta +\alpha$[~

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%207.png)

→ shows average update to theta

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%208.png)

우리는 maximize payoff 하고 싶으니, 도함수 구해야됨!

f($\theta$)g($\theta)$h($\theta$) 의 도함수 구하는 법을 이용하여~ 구할 수 있음

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%209.png)

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%2010.png)

이 gradient update 로 이어짐

## POMDP → partially observable ⇒ 일부분만 알수 있음

at each step get a partial (potentially noisy) measurement state by using action "a"

우리는 pi*(s)를 못씀. 왜냐 모르거든!

![Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%2020%20e0461fefac0c41589cca5e70d9996772/Untitled%2011.png)

 direct policy search good

is pi* simpler or is V* simpler?

Low level control task V

high level control task pi*