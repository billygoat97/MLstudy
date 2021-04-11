# Machine Learning Stanford Lecture 9

Class: CS229
Created: Jul 22, 2020 11:07 PM
Reviewed: Yes
Type: Lecture

- setup/assumption
- bias/variance
- approx, estimation
- empirical rsik minimizer
- uniform convergence
- vc dimension (다음 시간에 다루기로 함)

 Assumptions

1) Data distribution → Train, Test (으로 분산)

2) Independent Samples

data → learning algo → hypothesis and theta ~ sampling distibution

train기준

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled.png)

under, right, over순

parameter 기준

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%201.png)

Bias → 편향

variance → 분산

별 모양 → 실제 값!

m→ infinity 할 경우 (train data 의 수가 무한정으로 늘어날 때,)

Var[theta hat] → 0 (결국엔 하나로 수렴 (한 점으로 실제값으로)→ 오차가 없어짐)

Statistical efficiently

→ rate which variance drops to 0

train 많이 시킬수록 실제값에 매우 유사해짐

## Fighting variance (변수 정확성 높이는 방법)

1. m → infinity (검사 횟수 증강)
2. regularization

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%202.png)

regularization 이용하면 bias가 조금 될지라도 variances는 줄어듬

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%203.png)

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%204.png)

목표는 일반적인 예측을 할때 정확도를 높이는, generalization error을 줄이는 것.

### fight high Bias

make H bigger

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%205.png)

## Uniform Convergence (균등수렴)

(이 파트에서 h인지 람다인지 글씨 이상하게 되어있어서 너무 헷깔림)

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%206.png)

정확한 true error를 측정하는 것은 불가능. 그러나 test data를 i.i.d하게 뽑는다면 test error는 true error의 unbiased estimator가 된다. 그러나 training error는 unbiased estimator가 아니다. 왜냐하면 error가 hypothesis 에 의해 결정이 되는데, h는 training data D에 dependent하기 때문이다.

train error를 minimize하는 것: empirical risk minimization

결론은 일반 error과 emperical error간의 차이를 줄이는 방법을 고안 하는 것.

이를 위해 쓰는 tool → union bound, hoeffding's idea

union bound → 

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%207.png)

hoeffding's idea →

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%208.png)

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%209.png)

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2010.png)

[Machine Learning 스터디 (10) PAC Learning & Statistical Learning Theory](http://sanghyukchun.github.io/66/)

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2011.png)

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2012.png)

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2013.png)

감마와 델타가 0보다 크다면, m이 얼마나 커야 확률이 1-델타가 되며, training error이 일반error과의 오차가 감마보다 작아지게 될까? 델타를 위의 값처럼 설정을 한다면, m은 

![Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%209%20b100f858f10d43af826b0c02a5f62041/Untitled%2014.png)

이상일 경우에는 감마보다 error차가 줄어듦을 알수 있다.

수식이 복잡하기는 하지만, 결론적으로는

(증명을 수업에서 모두는 다루지 않았음 추가 정보는 상위의 링크 참조) 

m을 증가시키면 (dataset의 개수 증가,) if you sample different set, 다양한 set들을 sample 할 경우, 점선 그래프 (emp~) → 실선 그래프로 더욱 유사하게 변함

VC dimension은 다음시간에 다루기로 함.