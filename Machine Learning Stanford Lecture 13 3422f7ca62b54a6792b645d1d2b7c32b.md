# Machine Learning Stanford Lecture 13

Class: CS229
Created: Jul 28, 2020 3:55 PM
Reviewed: Yes

새로운 알고리즘 배우는 것이 아닌, 효율적으로 debugging하는 법에 대해서 잘 알려주는 강의

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled.png)

보통 알고리즘이라고 하는것이 한번에 실행이 성공적으로 되지 않는다. 이에 우리는 계속 training 및 debugging을 해야 한다. 방향성이 잘못되었을 수도, syntax 에러일 수 있지만,

debugging은 한줄씩 확인하는 것은 비효율적이다.~ 고로 효율적인 방법을 찾아야 한다.

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%201.png)

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%202.png)

(추후에 나오는 파트를 미리 땡겨서 가져옴)

각각의 시도에 따른 개선점.

1. train data 증가
2. feature 작은 set
3. feature 큰 set
4. email header feature → 제목에서 쓰인 feature처럼 접근방식 변경 (가중치)
5. gradient descent 더 많은 iteration
6. 뉴턴 메소드 쓰기
7. 다른 lambda 쓰기
8. SVm 쓰기

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%203.png)

high variance일 경우, 

test error, train error 모두 performance로 m이 커져야 천천히 변함

train error 증가 → 데이터 증가 시에 fit 절대 수는 늘어나긴 함. overfit 방지하려고 하기 때문에

test error 하락

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%204.png)

bias이기 때문에, performance에 가까워지거나 멀어지거나 하는 것이 m이 늘어난다고 해서 하지는 않는다. 

상기 적혀져 있는 해결 방안들을 시도함으로서 variance와 bias 해결 가능'

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%205.png)

2% spam 못거르는거와 2% non-spam을 spam으로 인식하는 것은 매우 다름 (가중치 부여)

logistic보다 svm이 이 경우에 더 나을 수 있다(실제로는) 근데 효율성을 위해서 그냥 logistic이용!

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%206.png)

hard to tell algorithm is converged by looking at objective

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%207.png)

함수가 제대로 최적화가 됐니?

cost function을 svm blr에 비교해 보는 것

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%208.png)

case 1:

세타(blr)이 costfunction maximize 실패 → convergence 안됨 → optimization algorithm문제

case 2: 

세타(blr)이 성공 했지만, a와 연관 x → objective function of maximize problem

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%209.png)

가정:

1. 시뮬 정확
2. 강화 알고리즘이 헬리콥터를 정확히 컨트롤 하고, cost function 최소하 한다
3. cost function min 하는 것이 자동비행에 대응한다.

헬리콥터 문제(강화학습) → 실제로 제대로 안된다면?

1. 시뮬에서 잘 되고, 실제 상황에서 잘 안되면, simulator문제
2. 사람이 하는 것 에 비해 안좋으면, 강화 learning 알고리즘 문제
3. cost function 문제일 수 있음

---

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%2010.png)

error analysis

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%2011.png)

error analysis는 component를 하나씩 추가해 나가면서 accuracy의 증가 요인을 확인해 보는 것. 이 경우에는 face detection이 가장 큰 요인이므로 concentrate해야 하는 것.

![Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%2013%203422f7ca62b54a6792b645d1d2b7c32b/Untitled%2012.png)

ablative analysis는 전체에서 하나씩 요인들을 빼가면서 accuracy에 가장 큰 영향을 끼친 것을 알아보는 것. 이 경우에는 email header features가 되겠다.