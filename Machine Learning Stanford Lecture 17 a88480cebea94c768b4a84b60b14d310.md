# Machine Learning Stanford Lecture 17

Class: CS229
Created: Aug 13, 2020 3:20 PM
Reviewed: Yes

WA! happy thanksgiving!!!

# MDPs & Value/Policy Iteration

리마인드~

s (state)

a (action)

gamma  (discount)

R (reward function)

Ps (확률)

다섯가지 tuple

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled.png)

V pi (s) → expected total payoff

→ value function for policy PI!

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%201.png)

we should choose positive ones

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%202.png)

벨만 이쿠에이셔언~

saids that you're expected to pay off  감마 영향을 받아서

R(s) ⇒ 즉각적인 reward, 뒤에거는 추후의 reward

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%203.png)

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%204.png)

## Value iteration and policy iteration

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%205.png)

가치 초기화!

이전 가치 토대로 next 가치 update

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%206.png)

## Learning a model for an MDP

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%207.png)

max likelihood 통해서 다음 식 얻을 수 있음.

![Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2017%20a88480cebea94c768b4a84b60b14d310/Untitled%208.png)