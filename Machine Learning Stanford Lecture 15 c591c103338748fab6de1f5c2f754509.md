# Machine Learning Stanford Lecture 15

Class: CS229
Created: Aug 12, 2020 12:47 AM
Reviewed: Yes

으어어어어어 ㅁㄴㅇㄹ 오늘도 어렵구만

# Factor analysis

[제4장 인자분석](https://m.blog.naver.com/PostView.nhn?blogId=chcher&logNo=70139319614&proxyReferer=https:%2F%2Fwww.google.com%2F)

n >> m일때는, 가우시안 분포일 경우, mean 과 covariance를 maximum likelihood estimator이용 구하기 가능

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled.png)

$\sum$의 역이 없다. → singular

and 1/|Σ|^(1/2) = 1/0

data의 수가 적다면? →

## Restriction of $\sum$

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%201.png)

jj ⇒ 대각 행렬

대각선에만 숫자가 들어감

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%202.png)

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%203.png)

## Marginals and conditionals of Gaussians

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%204.png)

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%205.png)

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%206.png)

![Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2015%20c591c103338748fab6de1f5c2f754509/Untitled%207.png)

## The Factor analysis model

여기서부터는 이해가 안됨.. 다시 리뷰할때 다시 봐야 할듯;; 

## EM for factor analysis