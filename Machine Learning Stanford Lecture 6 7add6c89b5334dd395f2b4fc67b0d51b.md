# Machine Learning Stanford Lecture 6

Class: CS229
Created: Jul 15, 2020 10:43 AM
Reviewed: Yes
Type: Lecture

## Laplace smoothing

만약에 한번도 받지 못한 단어(text)가 메일로 오면 이것을 스팸이라고 계산할까 아니면 스팸이 아니라고 계산을 할까?

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled.png)

스팸이라고 판단하려는 식을 이전 강의에 비추어서 실행을 해볼 시에, 0/0+0이라는 식이 결과로 나와 0/0이라는 결과값이 나온다. 이는 컴퓨터가 수식을 계산할수 없을 뿐더러 적절한 결과값을 도출할 수 없다. 이에 대한 해답을 제시하기에 앞서 다음 예시를 보자.

예) 스탠포드 미식축구팀은 현재 4개의 대학과 경기를 했고, 5번째 경기를 하려고 하는데 5번째 경기를 이길 확률은 어떻게 되는가?

1. 패
2. 패
3. 패
4. 패
5. ?

원래대로라면 간단히 표시 할 시에 (이긴 경기 수) / (전체 경기 수)로 나오겠지만, 이는 좋은 결과가 아니다.

다음 경기에 대해 Laplace smoothing인, 이긴 경기수, 진 경기수에 모두 1을 더해주는(이 경우에는 2가지 경우만 있지만 더 다양한 classification이 있으면 각각 1씩 더해준다).

이 경우에 0/4+0 → (0+1)/{(4+1)+(0+1)} = 1/6의 확률로 이길 수 있음을 알 수 있다.

이처럼 스팸인지 아닌지부터 다양한 예외를 처리하기 위해서는 (절대라는 결과는 없기 때문에) Laplace smoothing을 통해 사후확률을 조정해 준다.

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%201.png)

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%202.png)

reasonable estimate of P(y=1)이 된다.

## Event Models for text classification

1. multi variate bernoulli event model
2. multinomial event model

multi variate bernoulli event model이란,

스팸 메일 기준, 메일이 random하게 온다고 가정할 때, 단어가 포함되는지 안되는지는

출현의 유무 {0,1} 둘중 하나로 결과가 저장이 된다. 그로 인해 spam인지에 대한 결과값을 classification한다.

multinomial event model이란,

개념을 조금 다르게 생각해야 한다.

 xj 는 j번째 단어라고 생각하며, xj의 값은 dictionary 기재된 번호이다.

즉 xj →{1 ~|V|}, (x1~xn)이라고 생각하면 된다

이때, 중복된 단어를 {0,1} 로 그저 표시하는 것이 아닌 빈도수에 weight를 부과하는것 마냥

[1,23,45000, 1] 이런식으로 별도로 처리하는 것이다. 이로 인해서 결과값에 별도로 independently하게 계산하는것이다.

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%203.png)

따라서 위와 같은 확률이 나오게 된다.

식은 비슷하게 1.과 같이 보이지만, 실제로는 변수가 의미하는 바가 다르게 된다

xj|y는 다항분포가 된다.

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%204.png)

따라서 likelihood를 표현할 시 위와 같이 표시하게 되며,

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%205.png)

확률은 다음과 같이 나오게 된다 (laplace smoothing을 거친 것)

(밑의 식을 설명하자면 분모는 sum of not spam, 분자는 k란 단어가 나왔을 때의 spam이 아닐 확률의 총합이다 (중복도 더함))

# Support Vector Machines

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%206.png)

B1이 더 여유있게 나눔

Margins? → classification (선)과의 거리

클수록 confidence가 커짐 (classification의 정확도)

Road map으로 3가지가 있는데,

1. optimal margin classifier (분리 가능한 것)
2. kernels
3. inseperatable classifier

을 탐구해 볼 것이다.

보통 classifying 할 때에는 직선으로 항상 표시하기는 어렵다.(불가능에 가깝다) 그렇다면 higher dimension으로 이를 끌어올려서 classify하면 된다.

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%207.png)

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%208.png)

선이 조금 이상해 보이긴 하지만, c의 경우, classify하는 것이 조금만 변동이 있다면 바뀔 수 있다는 점을 유의해야 한다. 이를 위해, classify하는 선과의 거리(margin)를 최대한 멀리 떨어뜨려 놓는 것이 (a처럼) 가장 classification에 confident 하다고 할 수 있다. 이를 위해, margin을 최대한 confident하게 하는 방법을 강구해야 한다.

마진 최대화!

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%207.png)

의 값에 따라 logistic regression에서는 0 혹은 1로 classify했지만,

SVM에서는 부호의 이용을 위해, -1, 1로 구분하도록 하였다.

따라서 위의 식이 ≥0 일 경우 y(i) = 1 반대일 경우 y(i) = -1로 하도록 하였다.

## Functional and geometric margins

→ euclidean distance between certain x to hyperplane

training set가 주어졌을때, 

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%209.png)

r(i)이 거리를 나타내고, 이를 최대화 해야 한다.

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2010.png)

유념해야 하는것은, r(i)의 값이 양수라면 우리의 결과값은 correct한것이므로, y(i)가 <0일 경우 wTx+b 또한 <0이 되도록 해주어야 하는 조절이 가능하게 된다. (because of 1, -1로 바뀌기 때문)

또한, 부호의 중요성이 전부이기 때문에 일정 상수배로 바꾸어도 결과값은 유지가 된다.

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2011.png)

가장 가까운 값이 가장 커야 하므로, 위와 같은 식에서 ^r 표시 된값의 최대값을 구해야 한다.

우리는 r(i)을 어떻게 구하는가?

앞서 말했듯, 상수배는 결과값에 영향을 미치지 않음을 인지하고 문제에 접근해야 한다.

일단, w/(|w|) 은 w의 방향성을 나타내며, A는 x(i)를 표현한다. B는 x(i) - r(i)*w/(|w|)이다. 그리고 b는 decision boundary에 위치하므로, wTx +b = 0 이므로,

wT(x(i)-r(i)*w/(|w|)) +b =0이 된다.

이를 풀게 되면

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2012.png)

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2013.png)

가 된다.

## Optimal margin classifier

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2014.png)

r을 maximize하기 위해서는, 위와 같은 식을 이용을 하는데,

||w|| = 1이므로, functional margin과 geometric margin이 같은 것으로 이해할 수 있다.

그러면 위와 같은 식으로 변경할수 있다.

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2015.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2015.png)

우리는 상수배를 곱해도 아무런 상관이 없기 때문에, training set 를 변형하여 ^r는 1이 되도록 한다.

이는

max 1/(||w||) 이고 이는 min1/2||w||^2를 찾는것과 같다. (도함수)

![Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2016.png](Machine%20Learning%20Stanford%20Lecture%206%207add6c89b5334dd395f2b4fc67b0d51b/Untitled%2016.png)

우리는 남은 식을 쉽게 풀수 있는 형태로 변형시켰다. (convex → linear)

이 솔루션은 우리에게 optimal margin classifier을 준다.

여기서 우리는 그치지 않고 라그랑주 duality로 넘어갈 것이다. 이는 추후에 나올 커널에서 optimizaton problem에서 다차원 공간에서 efficient하게 get할수 있도록 도와준다.

- 이쪽 내용 다시 한번 이해 할 필요 있음

To be continued.. lecture 7