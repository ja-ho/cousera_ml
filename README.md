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










