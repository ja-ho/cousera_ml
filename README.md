# 1. Introduction

### what is machine learning

machine learning definition

* arthur samuel(1959) - machine learning: Field of study that gives computers the ability to learn without being explicitly programmed.

recent definition

* tom michell(1998) - Well-posed learning problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.  
experience E : 학습 데이터
task T : 수행하는 행동
performance measure P : 제대로 수행할 확률

다양한 machine learning algorithms 이 있다.
크게 2가지로 나눌 수 있다.
    -	supervised learning
    -	unsupervised learning
others – reinforcement learning, recommender systems.
also talk about: practical advice for applying learning algorithms.

##### supervised learning - computer에게 "right answer"를 제공
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.
>examples

1 classfication problem - discrete valued output(0 or 1)

2 regression problem : our goal is to predict a continuous valued output.
support vector machine - neat mathematical trick. 컴퓨터가 infinite한 features 들을 다룰 수 있게 도움.

##### unsupervised learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.
>examples

- Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

- Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

#2. Linear Regression with One Vairable( = univariate linear regression)
>notation
>m = Number of training examples
>x's = "input" variable/features
>y's = "output" variable/"target" variable
>(x, y) -> single row of training example
>(x(i), y(i)) -> ith row of training example

##### model representation
![model_representation](./images/model_representation.png)

##### cost function
![cost_function](./images/cost_function.png)


##### cost function intuition_1
hθ(x) = θ0 + θ1*x
인 원래의 hypothesis를 간단하게 나타내기 위해 θ0 = 0인 경우를 살펴본다.
![cost_function_intuition_1](./images/intuition_1_1.png)
![cost_function_intuition_1](./images/intuition_1_2.png)

##### const function intuition_2
θ0가 0이 아닌 경우 parameter가 2개이므로 3차원으로 나타나게 된다.
*contour plot (= contour feature)
![contour_plot](./images/contour_plot.gif)
A contour plot is a graphical technique for representing a 3-dimensional surface by plotting constant z slices, called contours, on a 2-dimensional format. That is, given a value for z, lines are drawn for connecting the (x,y) coordinates where that z value occurs.

![cost_function_intuition_2](./images/intuition_2_1.png)
![cost_function_intuition_2](./images/intuition_2_2.png)
![cost_function_intuition_2](./images/intuition_2_3.png)

#3. Parameter learning
##### gradient descent
![gradient_descent_1](./images/gradient_descent_1.png)
![gradient_descent_1](./images/gradient_descent_2.png)

##### gradient descent intuition
![gradient_descent_intuition](./images/gradient_descent_intuition_1.png)
![gradient_descent_intuition](./images/gradient_descent_intuition_2.png)
![gradient_descent_intuition](./images/gradient_descent_intuition_3.png)

##### gradient descent for linear regression
![gradient_descent_for_linear_regression](./images/gradient_descent_linear_regression.png)
![gradient_descent_for_linear_regression](./images/gradient_descent_linear_regression_2.png)

#4. Multivariate Linear Regression

##### Multiple Features
![Multiple Features](./images/multiple_features.png)
##### gradient descent for multiple variables
![gradient_descent_for_multiple_variables](./images/gradient_descent_for_multiple_variables.png)
##### gradient_descent_in_practice_1_feature_scaling
![gradient_descent_in_practice_1_feature_scaling](./images/gradient_descent_in_practice_1_feature_scaling.png)
feature scaling에서 목표는 -1<= x <= 1이지만
-3 <= <= 3, -1/3 <= <=1/3 까지는 허용. 이거보다 작거나 크면 feature scaling이 필요.

##### gradient_descent_in_practice_2_learning_rate
![gradient_descent_in_practice_2_learning_rate](./images/gradient_descent_in_practice_2_learning_rate.png)
![gradient_descent_in_practice_2_learning_rate_2](./images/gradient_descent_in_practice_2_learning_rate_2.png)
gradient descent가 제대로 되고 있는지 확인하기 위해 "debugging"을 해야 한다. 그리고 이를 통해 learning rate α를 선택할 수 있다.
debugging은 gradient descent를 몇 번 iteration을 하는 지를 x-axis로, iteration 후의 θ에 따른 cost function j(θ)을 y-axis로 하는 그래프를 plot하여 할 수 있다. 이 경우 그래프를 통해 제대로 gradient descent가 이뤄지고 있는지 확인이 가능하다. 제대로 되었다면 convergence가 이루어짐을 확인할 수 있다.
또한, automatic convergence test를 통하여 한 번에 iteration 시 10^(-3)과 같은 작은 thresshold를 지정하여 convergence를 체크할 수 있지만 어떤 값을 선택해야 할 지 어렵다. (그냥 plot 해라)

learning rate은 너무 클 경우 대부분 convergence가 이뤄지지 않고 오히려 증가한다. (이뤄질 경우도 있지만 느리게 local minima가 됨)
또한, 너무 작을 경우 지나치게 느리게 convergence가 이뤄진다.
대부분의 경우 debugging을 위해 plot을 그려보고 제대로 gradient descent가 이뤄지지 않는다면 smaller alpha(learning rate)을 이용하면 해결된다.(수학적으로 증명: learning rate이 충분히 작다면 j(θ)는 every iteration마다 decrease한다.)

plot을 한 그림에서 flatten한 부분이 있다면 그곳이 convergence 지점.

대강 learning rate의 범위를 정하고 각각 plot하여 가장 급격하게 decrease하는 모양을 가진 것을 택해라

α = 
...., 0.0001, 0.0003, 0.001, 0.03, 0.1, .., 1, ....
등등 보통 *3배를 해가면서 체크
이 범위로 debugging을 하는 것이 일반적
##### Features and Polynomial Regression
![Features and Polynomial Regression](./images/Features_and_Polynomial_Regression.png)









