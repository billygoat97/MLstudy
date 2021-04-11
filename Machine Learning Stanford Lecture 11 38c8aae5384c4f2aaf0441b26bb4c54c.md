# Machine Learning Stanford Lecture 11

Class: CS229
Created: Jul 25, 2020 6:03 PM
Reviewed: Yes
Type: Lecture

# Introduction to Neural Network

Introduction 이라 계산식은 거의 안나오고 설명도 개념 위주로 설명만 해줌;;

- logistic regression
- neural network

Logistic regression (yes or no define하는 것~)

3단계로 나누어 설명함!

Goal 1: find cat in images

1 → cat 존재

0 → 없음

pixel별로 값을 가진다 (rgb)

64x64x3  (3은 rgb)

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled.png)

ㅋㅋㅋㅋㅋ <cat>

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%201.png)

logistic regression 후 sigmoid function에 넣어서 하나의 값으로 변형시킴 (0~1)

이를 위해서는 3단계의 과정이 필요하다

1. initialize parameter W,b (weight, bias)
2. Find the optimal W,b 
3. Use y hat(결과값) to predict

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%202.png)

this is quite simple because it has just one output answer (yes or no)

그 다음 단계로 넘어가면

Goal 2: find 3 types of animals

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%203.png)

여기서 이용하는 parameter의 개수는 goal 1에서의 3배인데, 이는 각각의 equation마다  parameter을 가지기 때문이다.

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%204.png)

여기서 결과값은, (1,0,0)이 될수도 있지만, (1,1,0) 처럼 결과값이 중복으로 나올 수 있다. 중복으로 결과가 나올 수 있음을 처리해주지 않았기 때문이다. 이에 대한 대책을 Goal 3에서 다루기로 하였다

Goal 3: + constraint unique animal on image → softmax regression

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%205.png)

위의 식은 goal2과 같은 방식이되, 마지막에 전체 비율로 따져서 비교함으로 인해 하나의 결과값만이 1이 될 수 있도록 하는 것이 softmax multi class regression의 아이디어이다.

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%206.png)

총 합이 1이 됨으로서 one hot encoding을 하여 하나만이 1이 되도록 하는 softmax regression 의 loss function은 다음과 같이 정리된다. goal2 에서 loss function을 전체만큼 덧셈을 통해 얻을 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%207.png)

# Neural Network

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%208.png)

뉴럴 네트워크는 다음과 같이 여러 뉴런들을 이용하여 다양한 layer들을 이용하여 결과값을 도출하는 방식이다 이는 실제 우리 신체에서 뉴런들이 작동하는 방식과 동일한 방식을 토대로 작동한다. 더 고(deeper)층 layer로 이동할수록 더욱 고등화된 개념을 얻을 수 있다.

이에 대하여 직관적인 예시를 말하지면 house pricing problem이라고 할 수 있다.

house price에 대해 예측 하는 이 문제는

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%209.png)

단순히 하나의 input으로 결정되지 않는데, 이 예시에는 

size, #of bedroom, zip code, wealth로 4가지 feature이 주어진다.

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2010.png)

4가지 feature은 3개의 또다른 feature로 정립이 되며, 결론적으로 price y로 귀결이 된다. 

cf) fully connect되려면 4개의 feature이 모두 3개의 결과값에 edge 연결되어있어야

이러한 결과가 순차적으로 결과를 전달해 주는 것은 propagation equation을 통해 이루어진다.

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2011.png)

상기 그래프를 수식으로 표현하면 위와 같이 표현이 된다. (연쇄적으로 영향이 주어짐)

하지만 하나의 feature이 아닌, batch 형식으로 한다면 결과가 어떻게 될까? → matrix를 이용하여 계산

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2012.png)

W와 X는 변경이 되는데 b[i]는 이와 matrix 크기에 맞지 않다. 우리는 이를 계산하기 위하여 어떠한 방식을 하여야 하는가? → broadcasting(같은 크기로 늘려야 함) numpy에서는 자동 지원이 됨

우리는 이와 같은 neural network를 train (optimize)해야 하는데, 이를 위해서 W,b인 parameter을 조정을 해주어야 한다! (cost function loss function 최소화 minimize)

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2013.png)

j → cost function, L → loss function

parameter update를 위해서 layer중에 어떤 layer에 있는 parameter을 조정을 선행적으로 해주는가? → 가장 y에 가까운 layer 부터! (why? → y값을 가장 적절하게 바꾸는데 조정이 쉬운 순으로 변경)

propagation의 역순이므로 → back propagation이라고 부름 (다음 강의에 더 자세히 다룸)

main idea는 chain rule을 이용하여 derivative 계산을 통해 parameter 값을 조정하는 것.

![Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%2011%2038c8aae5384c4f2aaf0441b26bb4c54c/Untitled%2014.png)

위의 식을 기준으로 볼 경우, 상위 layer의 derivate 계산값이 연쇄적으로 back 방향으로 계산이 들어가는 것을 알 수 있다. 이와 같은 back propagation의 도함수를 계산하기 위해서는 forward propagation을 학습해야.