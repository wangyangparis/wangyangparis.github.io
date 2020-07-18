---
layout: maths
name: Gradient descent
category: Theory
---

**Gradient descent**

$\ell$ the loss function to minimize:

$\omega_{t+1}=\omega_t - \alpha \nabla \ell$

The most popular forms of gradient descent are:

![image](/assets/img/global-gradient-descent.png){: height="15%" width="15%"}

![image](/assets/img/stochastic-gradient-descent.png){: height="15%" width="15%"}

![image](/assets/img/stochastic-random-gradient-descent.png){: height="15%" width="15%"}

The main advantage of stochastic gradient is that it avoids computing the gradient descent on all the observations (greedy). However, in doing so, the gradient descent is subject to noise and can take longer to reach the optimum.

*Note*: the gradient is the derivation w.r.t $\omega$

*Note (stochastic and random gradient descent)*: the draw can be done with or without replacement.

Proof of the formula:

<figure>
    <img src="/assets/img/GradientDescent_proof.png">
</figure>

<ins>Learning rate optimization</ins>

Note:

\- one epoch = one forward pass and one backward pass of all the
training examples.

\- batch size = the number of training examples in one forward/backward
pass. The higher the batch size, the more memory space needed.

\- number of iterations = number of passes, each pass using \[batch
size\] number of examples. To be clear, one pass = one forward pass +
one backward pass.

*Learning rate decay*

Simple idea: reduce the learning rate progressively.

E.g. $1/t$ decay:

$$\alpha_t=\frac{1}{(t+1)}$$

*Momentum*

Momentum is a method that helps accelerate SGD in the relevant direction
and reduce oscillations.

General idea:

$0 \le \gamma \le 1$

$M_{t_0}=x_0$

$M_{t_1}=\gamma M_{t_0} + x_1$

$M_{t_2}=\gamma M_{t_1} + x_2$

Let's develop $M_{t_2}$ to have a better view of momentum effect:

$M_{t_2} = \gamma (\gamma M_{t_0} + x_1) + x_2 = \gamma^2 x_0 + \gamma x_1 + x_2$

We can see that more importance is given to the most recent value
($x_2$) and the least to the past values.

In practice, $\gamma = 0.9$ gives good results.

**Advantage**: less dependent on noise

*Adagrad - Adaptive Gradient Algorithm (2011)*

Divide the learning rate by \"average\" gradient:

$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\Sigma_{i=0}^t(\nabla f_i^2)}}\nabla f$

*RMSProp - Root Mean Squared Propagation*

Same as AdaGrad, but with an exponentially decaying average of past
squared gradients.

$\theta_t = \theta_{t-1} - \frac{\alpha}{\sigma_{t-1}}\nabla f$

Where:

$\sigma_{t} = \sqrt{\alpha (\sigma_{t-1})^2+(1-\alpha)(\nabla f_t)^2}$

*Adam - Adaptive moment estimation*

Adam = RMS + momentum =\> use of a exponential decaying average of past
**squared** gradients and past gradients
