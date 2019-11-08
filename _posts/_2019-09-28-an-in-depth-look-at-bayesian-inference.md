---
layout: post
title: An In-depth Dive into Using Bayesian Inference
description: >
  A deep dive into inference based statistics, how it's different from frequentist approaches and what we can do with it in applied data science contexts.
noindex: true
comments: true
---
Welcome back to the blog folks! Sorry for the long gap between posts. I've been pretty busy and the technical aspects of today's post has been pretty involved. As a result though, I have A LOT to share (yayn)! Today I'm giving some technical depth to a powerful data science technique that I worked with in my graduate work by giving the full Bayesian treatment to parameterizing a multivariate, non-linear, analytical model. I'll go through doing this in two different software packages, with two different models and a comparitive  analysis of all four parameterizations produced. First, let's do a recap of Bayes' theorem and its usefulness as data science tool.


## Bayes' Theorem, Bayesian Statistics and Bayesian Data Science

**Bayes' Theorem**  

$$
\begin{aligned}
  P\left(\Theta \vert D\right) = \frac{P\left(D \vert \Theta\right)P\left(\Theta\right)}{P\left(D\right)}
\end{aligned}
$$

In the above, $$\Theta$$ are the parameters of our model (or specifically our force field parameters) and $$D$$ is our data or evidence that we're "training" with. All the terms of this equation also have specific names. $$P\left(\Theta\right)$$ is called a prior distribution; it represents our beliefs and previous knowledge about parameters before we introduce any data. Distinctions between classes of priors are made by the amount of information they give regarding our prior knowledge on the parameters of interest. For example, an uninformative prior might be a uniform distribution simply setting the bounds of what the parameter could be while a slightly more informative prior might be gaussian with a ballpark guess of the mean and a large estimate of the standdard deviation (for a wide distribution). The next term of Bayes' Theorem is $$P\left(D \vert \Theta\right)$$; this is our likelihood distribution. This is the part of Bayes' Theorem that allows us to quantitatively explore parameter space and see which combinations of parameters are well matched to reproduce the evidence we're using. The form of the likelihood function is different depending on how we will consider the errors to be distributed. Generally, we'll default to a gaussian unless we have good reason to believe otherwise. The next part of Bayes' is $$P\left(\Theta \vert D\right)$$, or the posterior distribution. This is our belief in the distribution of parameters given our evidence. It's the main thing that want to find. Given our data/evidence, what are the most probabilistically liekly parameter spaces? For every combination of parameters that we sample from, the posterior distribution encodes a probability that that is likely. Finally, our last term is $$P\left(D\right)$$ referred to as the marginal likelihood. This is a scaling factor (as in it's not a distribution, it's just a constant) that normalizes our posterior distribution. For most Bayesian inference calculations, it doesn't factor in (luckily for us, because it's actually quite difficult to calculate). Since we use some kind of Markov Chain Monte Carlo in order to sample from our posterior distribution, exact realizations aren't really important, as long as they're all scaled by the same factor. Therefore, we often write Bayes' Theorem as follows.

**Also Bayes' Theorem**  

$$
\begin{aligned}
  P\left(\Theta \vert D\right) \propto P\left(D \vert \Theta\right)P\left(\Theta\right)
\end{aligned}
$$  

This is to say that when we sample from our posterior distribution of parameters using a stochastic method like MCMC, we are actually sampling from a distribution whose realizations are proportional to the true posterior distribution by a factor of $$P\left(D\right)$$, which is just as well and WAY easier.

So, why do we care to take a Bayesian approach to model parameterization rather than something frequentist like least-squares? One primary reason is exploring the possibility of multidimensional parameter distributions. Sometimes, there are numerous parameter combinations that are equally likely for our model, given our data. A Bayesian approach to parameterization can make this very apparent, very quickly and can inform us on ways to improve our model form that we're fitting to or the sampling approach to exlore those multiple optima efficiently (things that are hard to diagnose with frequentist data fitting methods). Another, BIG, reason we might use a Bayesian parameterization approach is the robust and simple way that it quantifies uncertainty in parameters and model estimates. Parameter uncertainties are easily encoded because the process gives you DISTRIBUTIONS of parameters. We can estimate these distributions as something familiar (like a gaussian or cauchy or what have you) and get robust parameter uncertainties from that. Additionally, we can easily account for error in our fitting data by making it part of our likelihood function. Below is an example of how we could modify a likelihood function with normally distributed errors, to also be constrained by the measurement error from our data. 

[//]: # (Add example of adding data error to gaussian likelihood error model)

**Likelihood Function for a Normally Distributed Observation**

$$
\begin{aligned}
  P\left(D \vert \Theta\right) = \Pi_{j=1}^{n} \left(2\pi\sigma^2\right)^{-1/2}\exp\left(-\frac{1}{2}\frac{\left(D_j-M(\Theta)\right)^2}{\sigma^2}\right)
\end{aligned}
$$ 

In the above, $$D_j$$ is the $$j^{th}$$ data point in our pool of $$n$$ evidence points, $$M(\Theta)$$ is the forward data model evaluated at current parameter state $$\Theta$$ and $$\sigma$$ is the uncertainty in our forward data model. Oftentimes, we estimate that $$\sigma$$ as a parameter in our inference. If we had a measurement uncertainty (let's say $$s$$) associated with each data point in our pool of evidence, then we could account for it by replacing each instance of $$\sigma^2$$ with $$(\sigma^2+s_j^2)$$ in the likelihood function (where $$s_j$$ is the measurement uncertainty of evidence data point $$j$$). This data uncertainty would likely change how the parameters are constrained and the uncertainty of our final parameter distributions.
   
## What was my primary contribution to the Open Force Field Intiative and what are the most useful skills that I learned? 
One HUGE bottleneck to Bayesian inference driven force field parameterization is the time it takes to run our forward data model at each new set of parameters as we sample from the posterior distribution. For a [liquid phase simulation of some biologically relevant small molecules, it takes about 1 hour of wall time on a GPU](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005659) to get good data and statistics. That is ONE forward model realization at ONE point in parameter space for a few HUNDRED dimensional problem. If we were to take this vanilla forward data model approach, the problem is intractable; it simply takes too long. My job in the Open Force Field was how to figure out ways to shorten that. I settled on using a hierarchical scheme in order to estimate the thermophysical properties that we use to compare to our evidence. Below is a simple figure illustrating the scheme. 


[Reference model problem](https://physics.stackexchange.com/questions/388238/liquid-density-as-a-function-of-pressure-and-temperature-how-to-model-experimen)
