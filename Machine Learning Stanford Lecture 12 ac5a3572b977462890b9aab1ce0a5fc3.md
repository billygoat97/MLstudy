# Machine Learning Stanford Lecture 12

Class: CS229
Created: Jul 28, 2020 3:55 PM
Reviewed: Yes

# Back Propagation

back propagation의 식을 계산하는 방식을 이해하도록 하겠다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled.png)

cost function J와 Loss function L이 있는데, L은 위와 같이 표현됨을 지난 번에 이해하였고,

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%201.png)

각 변수를 update하기 위한 방법을 알 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%202.png)

3차 layer을 가지는 경우에 다음과 같이 식을 일차적으로 정리할 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%203.png)

"back propagation"이므로, 3 → 2로 역순으로 가는 방법에 대해서 알아보아야 한다.

따라서 위에처럼 chain rule을 이용하여 하나하나 역순으로 같은 식을 유도하며

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%204.png)

마지막으로 3.26과 같은 식으로 표현이 3,2번쨰 layer의 feature과 결과값으로 도출될수 있다. 여기서 상기 배운 내용을 이용하여 정리를 하면, (g는 sigmoid함수이며, simoid 함수의 도함수는 

---

sigmoid 도함수:

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%205.png)

---

따라서 정리를 할 경우에 3.28처럼 정리가 될 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%206.png)

하지만 우리가 T 처리등 사이즈에 맞지 않게 연산을 해왔기 때문에, 연산 사이즈에 맞게 계산을 해준다면,

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%207.png)

chain rule로, 역순으로 얻은 식을 바탕으로 현 layer의 도함수를 도출해낼 수 있다.

# Improving Neural Network

A. Activation Function (활성 함수)

1) 시그모이드 함수

1/(1+e^(-z)) 의 함수로, 0~1의 함수 그래프 모양이 나옴

장점) easy to use, used for classificastion, probability

단점)high activation일때, 양극의 feature일 경우, 기울기가 0에 거의 수렴. back propagation할 때 어려움

2) ReLU

if z> 0 → z

else → 0 으로 변형

제일 자주 씀

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%208.png)

3) tanh

(e^z-e^(-z))/(e^z + e^(-z)) 의 그래프로 -1 ~ 1의 그래프

시그모이드의 단점과 동일

why do we need activation function?

만약에 아무리 다층 layer을 생성한다고 해도, 정리하면 linear function 꼴로 변하게 되는데, 이를 위해서 activation function을 이용하면 그렇지 않음

# initialization

Normalization (정규화 → 계산에 용이하게 각 값을 미리 조절함 → hyper parameters)

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%209.png)

뮤 를 빼줌으로서, 중간(중앙에 결집되는 중앙값)을 원점으로 처리하게 하거나,

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2010.png)

값들의 제곱을 평균내 곱함으로 원점 지향을 하게 만든다면,

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2011.png)

좌측의 그래프에서 우측 그래프처럼 gradient descent라던지 다양한 loss function을 최소화하는 과정에서 훨씬 용이하게 결과값을 얻을 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2012.png)

layer을 만들 때에, 조심해야 하는 것이, vanishing 혹은 exploding gradients이다. 만약에 W의 값들이 모두 1.5라고 할때, 처음에는 별일이 없지만, W의 개수가 커지면 explode하게 되며, 반대로 0.5라면 0으로 수렴하게 될 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2013.png)

이를 위해서 W 값을 n의 값의 역수로 비례하게 설정을 해놓는다면 explode하거나 vanishing하는 것을 방지 할 수 있다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2014.png)

이러한 W 값의 조정은, 어떠한 activation 함수를 이용하냐에 따라 차이점이 있는데, sigmoid와 relu는 위 사진의 가장 위 식을 이용하며, sigmoid는 상수 1, relu는 2를 이용하면 좋다고 경험적으로 연구되어왔다고 이야기 해주었다. (경험하여.) 마찬가지로 tanh에서는 xavie,r initializatoin을 이용하여 이용하는 것이 tanh에 경험적으로 좋다고 이야기 해주었다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2015.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2015.png)

explode나 vanishing을 방지하기 위해 하나하나 전부 다 계산하는 것이 아닌, mini batch를 이용하여 algorithm을 돌려서 결과값을 얻는데, 위의 경우 처럼 일부분을 발췌해서 하는 경우, 너무 많은 변수로 인해 일전에 말한 explode, vanishing이 방지되지만, 아래의 그림처럼 우측 그림처럼 cost function이 그리 좋지 않게 결과값이 나올 수 있음을 유의해야 한다.

![Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2016.png](Machine%20Learning%20Stanford%20Lecture%2012%20ac5a3572b977462890b9aab1ce0a5fc3/Untitled%2016.png)