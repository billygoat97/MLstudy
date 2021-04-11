# Machine Learning Stanford Lecture 10

Class: CS229
Created: Jul 24, 2020 12:39 AM
Reviewed: Yes
Type: Lecture

내용요약

- Decision Tree
- Ensemble Methods
- Bagging
- Random Forest
- Boosting (이건 5분만에 흐지부지 설명함)

# Decision Tree

linear 하지 않음! → 1차 방정식 형태가 아니라는 것. 그냥 x,y축에 평행한 직선형태

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled.png)

x축은 달, y축은 경도이다. 스키를 탈수 있는 경우를 표시하도록 되어있다. 이 경우에는 linear로 표현하기에는 어렵기 때문에 partitioning해야 한다.

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%201.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%201.png)

분리의 원칙은 다음을 통해서 이루어진다

1. greedy
2. top-down
3. recursive partitioning

top down은 parent에서 분리해 내려가기 떄문

recursive는 계속해서 분리해 나가기 떄문

greedy는 그중에서 가장 좋은 것을 가져가기 때문이다.

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%202.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%202.png)

위의 그림과 같이 partition을 분리해서 쪼갤수 있다!

그렇다면 partitioning을 (split)을 어떻게 하는가?

Loss를 최소화 하는 방식으로 해야 한다. 이를 위해서는 loss function을 R(region) dependent하게 이용해야 한다.

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%203.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%203.png)

이리하여 분리하는 경우, 아래와 같이 같은 분리라도 값이 다르게 분리 될 경우 어떤것이 더 나아 보이는지에 대해 판단해야 한다.

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%204.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%204.png)

그냥 봤을 떄는 좌측의 split이 더 좋아 보인다. 하지만 전체 loss라는 측면에서 보았을 때에는 일정하게 100이라는 숫자가 유지된다. → 해결이 안되는데? → then?

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%205.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%205.png)

따라서 우리는 cross entropy loss 를 이용한다 (엔트로피 하면 또 분산~ 이런 개념일 것)

cross entropy measure the number of bits needed to specify the outcome given that the distribution is known.

즉, 분리하는데에 필요한 질문의 개수를 cross entropy라고 한다. (분류)분리함에 있어서 질문 개수를 줄여야 하는 것. 그런 의미에서 상기의 분류에서는 왼쪽이 오른쪽보다는 더 유리하다고 할 것이다.

## cross entropy

[초보를 위한 정보이론 안내서 - Cross Entropy 파헤쳐보기](https://hyunw.kim/blog/2017/10/26/Cross_Entropy.html)

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%206.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%206.png)

위의 그래프를 보면, cross entropy loss는 볼록함이 strict하기 떄문에, children loss의 합은 parent loss보다 작음이 확실하나, misclassification은 볼록함이 guarentee하지 않기 때문에 모른다고 할 수 있다. 따라서 cross entropy loss가 loss를 줄일수 있음을 알 수 있다.

따라서 마지막으로 정리를 할 경우,

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%207.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%207.png)

final prediction for region R is mean of all values!

## categorical variables

yes or no로 표현할수도 있지만, 카테고리별로 variables들을 분리할 수도 있다. ski 관련 split을 다음과 같이 분리할수도 있다.

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%208.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%208.png)

너무 많은 카테고리를 설정하면 안되는 것은, possible question은 power set P(S) 은 2^(s)이기 때문이기 때문에 너무 많은 question이 될 여지가 있다. → expensive

.

## regularization

- minimum leaf size
- maximum depth
- maximum number of node
- minimum decrease in loss (not so good idea)
- pruning(misclassification)

split을 반복적으로 하라고 했지만, 멈출 기준을 잡아야 하는데 위의 기준을 두고 진행해야 한다.

중복되는 값이 줄어들면(cardinality가 줄어들면) , leaf의 크기가 기준점보다 낮아지면 (leaf size)(너무 개수가 자잘해지는것 방지)

depth가 정해진 기준보다 높으면, ( 너무 많이 분리하는 것이면)

leaf node가 정해진 개수보다 많으면 중단

## RunTime

big O notation으로 DT의 시간을 알아보는데,

n 예시, f input(feature), tree의 depth d로 표현한다면,

O(d) ≤ O(logn)

하나의 feature안에 들어가는 시간을 O(f), n개의 갯수가 있으므로 총 드는 시간은,

O(d*f*n) 이다. 이는 측정되는 시간이 훨씬 줄어드는 것을 알 수 있다. (다른 algo에 비해)

## 단점

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%209.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%209.png)

이런 경우에는 좌측처럼 바보같이 waste가 심하다

ㅋㅋㅋㅋㅋ 그래서 사실 정리하자면 쉽고, 빠르고 다양한곳에 이용 가능하지만, 사실 좋지 않다 → 결론

우리는 이것을 어떻게 이용해야 하는가? → ensemble!

# Ensemble method

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2010.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2010.png)

iid일 경우 상단의 공식이 성립,

만약에 independent하다는 공식을 빼버리면? → i.d 일 경우, 

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2011.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2011.png)

위의 식과 같이 표현될 수 있다 (강의에서는 계산식 안알려주고 결과만 알려줌)

우리가 ensemble할수 있는 방법 4가지

1. different algorithms
2. different training set
3. bagging(random forest)
4. boosting(addboost, xgboost)

1,2번은 다르게 하고 평균냄으로 할 수 있다

그외의 경우를 설명하겠다

# bagging

# bootstrap

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2012.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2012.png)

[부트스트랩에 대하여 (Bootstrapping)](https://learningcarrot.wordpress.com/2015/11/12/%EB%B6%80%ED%8A%B8%EC%8A%A4%ED%8A%B8%EB%9E%A9%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC-bootstrapping/)

random forest

[](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-5-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8Random-Forest%EC%99%80-%EC%95%99%EC%83%81%EB%B8%94Ensemble)

부트스트랩이란 중복을 허용하여 일부 뽑고, (이를 위해 independent를 제거함), train하고, 평균치를 냄.

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2013.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2013.png)

위 식에서 볼 수 있듯이, 전체 train 개수인 M개수를 증가시키면 variance를 줄일수 있음을 알 수 있다.

boostrap을 통해, bias slightly 증가하지만, variance를 줄일 수 있음을 알 수 있다.

우리는 위의 결과를 통해, bagging과 DT의 ensemble의 적절성을 알 수 있다.

bagging을 통해 DT의 엄청난 variance를 줄이고, 낮은 bias를 조금(slightly) 올리게 함으로서 신뢰도가 올라감을 알수 있다.

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2014.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2014.png)

가장중요한 bagged dt 는 random forest를 cover하기 위함이다.

이는 위의 링크에서 볼수 있듯이, 여러개의 feature에서 일부를 랜덤하게 뽑아 decision트리를 통해 여러 예측 값을 내놓고, 그중에서 가장 확률이 높은 것을 (다수결) 선택하여(평균)(variance감소, increase bias) 결과를 내놓는다.

# Boosting → 아 이거 설명 왜 제대로 안되어있냐고!

used for reduction of bias

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2015.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2015.png)

The key idea is that we then track which examples
the classifier got wrong, and increase their relative weight compared to the
correctly classified examples. We then train a new decision stump which will
be more incentivized to correctly classify these ”hard negatives.” We continue
as such, incrementally re-weighting examples at each step, and at the end we
output a combination of these weak learners as an ensemble classifier.

즉, 오답 가중치를 더욱 강하게 준다는 요지, boosting은 순차적으로 실행하기 때문에 틀린 결과값에 그때 그때 바로 가중치를 주어서 결과에 반영함. classify하기 어려운 것들에 가중치를 높게 두고, 정답에 가중치를 낮게 줌으로서 그 다음에 train할때 오답에 집중 가능

weak learner 들을 결합하여 strong learner로 만듬

[Bagging과 Boosting 그리고 Stacking](https://swalloow.github.io/bagging-boosting/)

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2016.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2016.png)

![Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2017.png](Machine%20Learning%20Stanford%20Lecture%2010%2089b1368dbbde406b93252784df613db7/Untitled%2017.png)