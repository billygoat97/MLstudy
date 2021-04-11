# Machine Learning Stanford lecture2

Class: CS229
Created: Jul 9, 2020 9:53 PM
Reviewed: Yes
Type: Lecture

# Linear Regression

how to represent hypothesis function h as linear equation. 

which means that how to find approximate y out of x

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled.png)

d ⇒ number of input(features)

x0 =1 (dummy feature)

$ɵ$ ← parameters ( weights)

mapping from  X → Y.

(x,y) → training example

(x(i), y(i)) → ith training example

given training set, how do we pick or learn parameters $ɵ$?

is to make h$ɵ$ close to y(target) as possible.

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%201.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%201.png)

Goal : to minimize the cost function

(reason why we set 1/2 is to make math easier later on.)

# LMS algorithm (least mean squares)

in  order to minimize cost function, we have to initial guess with $ɵ$**.**

consider  "Gradient descent algorithm".

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%202.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%202.png)

alpha is called "Learning Rate"

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%203.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%203.png)

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%204.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%204.png)

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%205.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%205.png)

this should be repeated until convergence.

this method looks at every example → so called "batch gradient descent"

cons about batch gradient descent 

1. can stop at local minimum ⇒ so it is done in linear regression →  only global minimum
2.  if there is large set of data, it takes too much time.

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%206.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%206.png)

in this algorithm, it will lead to minimum difference as repeatedly change.

(because every stage, the ɵ is updated)

stochastic gradient descent method uses part of the data at that stage, so it can caculate faster.

stochastic gradient descent is much faster than batch type. ( but it will never lead to minimum)

so if data set is large, stochastic gradient descent is often preferred.

# normal equation

we'll minimize J by taking derivatives(도함수). respect to ɵj and setting them to ZERO.

## matrix derivatives

we can define derivative of f with respect to A to be:

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%207.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%207.png)

## least squares revisted

using matrix derivatives, we can find ɵ minimizing J(ɵ).

assume X(input) as :

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%208.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%208.png)

assume Y(target) as:

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%209.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%209.png)

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%2010.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%2010.png)

we can show J(theta) with matrix above.

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%2011.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%2011.png)

by derivative matrix, we can show j(theta) as last equation.

so value of theta that minimizes J(theta) is 

![Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%2012.png](Machine%20Learning%20Stanford%20lecture2%2081dc95d9837b4f0fa3e9621499b18fca/Untitled%2012.png)