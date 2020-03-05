# Foundations of ML

## What is Machine Learning

?> Machine learning is programming computers to optimise the **performance** on a particular **task** by **generalising** from examples of **past experiences** to **predict** what is occur in future experience.

A computer is said to learn

-   Some class of tasks **T**
-   From experience **E**
-   Measured by performance **P**

If its perormance at tasks in **T**, as measured by **P** improves with experience **E**

>   Spam example. Given an email, we detect if the email is spam or not.
>
>   **T:** Decide wether if it's a spam or not
>
>   **P**: Compare with the actual label given.
>
>   **E:** Some experience from the past which email is scam or not

### Task: Unknown Target Function

The task can be expressed as **unknown target function**

$$y = f(x)$$

-   Attributes of the task: \\(x\\)

-   Unknown function: \\(f(x)\\)

-   Output of the function: \\(y\\)

ML finds a Hypothesis, h, which is a function (or model) which approximatees the unknown target function (the function of the line)

$$h(x) = f(x)$$

-   The hypothesis is **learnt from experience**
-   A good hypothesis has a **high evaluation from performance measure**
-   The Hypothesis generalises to **predict the output of instances from outside the Experience.**

### Experience

The Experience is denoted D such as

$$ D = \{X, f(X)\}$$

-   Attribute of the task: \\(X\\)
-   Output of the Unknown function: \\(f(X)\\)

### Perfornmance

?> Numerical measure tht determines how well the hypothesis matches the experience.

This is measured against the experience.



## Type of Machine Learning Problems

### Supervised learning

In **supervised learning**, the output is known

$$y = f(x)$$

**Experience**: examples of input-output pairs

**Task**: Learns a model that maps input to desired output. Predict the output for new "unseen" inputs

**Performance**: Error message how closely the hypothesis predicts the target output.

>   Two main types of supervised learning:
>
>   -   Classfication
>   -   Regression

### Unsupervised learning

**In unsupervised learning**, the output is unknown

$$ ? = f(x)$$

**Experience:** Data set with values for some or all attributes

**Task**: 'invent' a suitable output. Identify trends and patterns between data points

**Performance**: How well the "invented" output matches the dataset



### Reinforcement Learning

In **reinforcement learning**, the target function is to learn an optimal policy, which is the best "action" for a **dynamic agent** to perform at any point in time

$$a = \pi * (S)$$

**Experience**: A transition function, the result of performing any action in a state

**Task**: Learn the optimal actions required the agent to achieve a goal

**Performance**: Reward ( or reinforment ) to perform certain action(s).

>   Reinforcement learning shares similarities with supervised and unsupervised learning:
>
>   -   The output (action, \\(a\\)) is unknown, however
>   -   The experience gives an "output" of performing action in states: \\((s, a) \rightarrow s'\\) 
>   -   The performance measures the "worth/reward" of each experience instance: \\(R(s,a)\\)
>   -   The performance acts as a proxy for the "actual" output, since in simple terms, it is the best "reward", that is accumulated over time as the agent conducts actions.