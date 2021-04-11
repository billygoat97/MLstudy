# Machine Learning Stanford Lecture 16

Class: CS229
Created: Aug 13, 2020 3:20 PM
Reviewed: Yes

# RL(Reinforcement Learning and Control)

Helicopter → where to go? → 답이 없어! (정답이!)

그러면... doesn't tell true answer 

good dog , bad dog define? → don't know!

R(s) ⇒ 

R → reward function

s → state

chess라던지 보상을 주는 형식으로 해서 방향성을 잡아가는 것.

## MDP (Markov decision processes)

![Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled.png)

S → state (체스 포지션, 헬리콥터의 위치)

A → action (헬리콥터 갈수 있는 위치, 체스에서 다음 수를 놓을 수 있는 곳)

Psa → state transition probabilities → 다음 변화될 확률 .. ⇒ 다 더하면 1됨 !

gamma → discount factor

R → reward function

MDP

R2D2가 이동하려고 할 때, 이동할 확률이 다 다름 0.8, 0.1, 0.1, 0 이런식으로

![Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled%201.png)

Reward Function

navigate well ⇒ design reward function

![Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled%202.png)

how it works

s0 에서 시작 

choose action a0

goes to state s1

get to s1 ~ Ps0a0

choose action a1

get to s2 ~ Ps1a1

g = gamma ~~ 0.99 정도?

total pay = R(s0) + g* R(s1)  + g^2*R(s2) + ..

discount factor 는 encourages 현재 state

따라서 total pay를 maximize하려고 하는 것이 목표 

Policy  

![Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2016%2024a50dc4d4cc4de3b98439f4d2f5f492/Untitled%203.png)

execute policy ⇒ state S일때, take action given by PI of S

maximize ..