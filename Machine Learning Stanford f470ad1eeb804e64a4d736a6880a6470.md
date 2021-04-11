# Machine Learning Stanford

Class: CS229
Created: Jul 8, 2020 4:37 PM
Materials: Machine%20Learning%20Stanford%20f470ad1eeb804e64a4d736a6880a6470/cs229-notes1.pdf
Reviewed: Yes
Type: Lecture

# Machine Learning

Definition of Machine Learning

→ learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

3 Taxonomy of Machine Learning

1. Supervised Learning (지도 학습)
2. Unsupervised Learning (비지도 학습)
3. Reinforcement Learning (강화 학습)

x(i) will be used to denote the input variable, or input feature. (입력 값)

y(i) will be used to denote the output or target variables that we'll try to predict. (결과 값)

we have to find relationship between x(i) and y(i). (X → Y) (입력값&결과값의 연관성 찾기)

A pair (x(i), y(i)) is called training example. 

data set (x(i), y(i); i = 1 ~ n) is called training sets.

The goal of supervised learning problem is to learn a function h: X → Y which can predict good output variable. this function h is called hypothesis ( it is not 100% correct)

(지도 학습의 목표는, h라는 함수를 찾되, 결과값에 가장 비슷하게 도출되는 결과를 출력하는 GOOD 함수 찾기 → 따라서 hypothesis라고 부름)

if target variable we're trying to predict is continuous, this problem is regression problem.

결과값이 많고 연속성을 띄면, 회귀 문제.

if target variable we're trying to predict is small number of discrete values, we call it classification problem.

결과 값이 적고, 몇개로 분류 되면, 분류 문제.