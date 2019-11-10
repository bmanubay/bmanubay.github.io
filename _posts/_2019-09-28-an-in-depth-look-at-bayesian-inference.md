---
layout: post
title: An In-depth Dive into Using Bayesian Inference
description: >
  A deep dive into inference based statistics, how it's different from frequentist approaches and what we can do with it in applied data science contexts.
noindex: true
comments: true
---
Welcome back to the blog folks! Sorry for the long gap between posts. I've been pretty busy and the technical aspects of today's post has been pretty involved. As a result though, I have A LOT to share (yay)! Today, I'm giving some technical depth to a powerful data science technique that I worked with in my graduate work by giving the full Bayesian treatment to parameterizing a multivariate, non-linear, analytical model. I'll go through doing this in two different software packages, with two different models and a comparitive  analysis of all four parameterizations produced. First, let's do a recap of Bayes' theorem and its usefulness as a data science tool.


## Bayes' Theorem, Bayesian Statistics and Bayesian Data Science

**Bayes' Theorem**  

$$
\begin{aligned}
  P\left(\Theta \vert D\right) = \frac{P\left(D \vert \Theta\right)P\left(\Theta\right)}{P\left(D\right)}
\end{aligned}
$$

Above is the basis of all inference-based statistics: Bayes' Theorem! It gives us a means to quantify the probability of an event or configuration based on prior knowledge that may be related. $$\Theta$$ are the parameters of our model (or the space containing the event or configuration that we're interested in) and $$D$$ is our data or evidence (this contains the information on our prior knowledge) that we're "training" with. All the terms of this equation also have specific names. $$P\left(\Theta\right)$$ is called a prior distribution; it represents our beliefs and previous knowledge about parameters before we introduce any data. Distinctions between classes of priors are made by the amount of information they give regarding our prior knowledge on the parameters of interest. For example, an uninformative prior might be a uniform distribution simply setting the bounds of what the parameter could be while a slightly more informative prior might be gaussian with a ballpark guess of the mean and a large estimate of the standard deviation (giving a wide distribution). The next term of Bayes' Theorem is $$P\left(D \vert \Theta\right)$$; this is our likelihood distribution. This is the part of Bayes' Theorem that allows us to quantitatively explore parameter space and see which combinations of parameters are well matched to reproduce the evidence we're using. The form of the likelihood function is different depending on how we will consider the errors to be distributed. Generally, we'll default to a gaussian unless we have good reason to believe otherwise. The next part of Bayes' is $$P\left(\Theta \vert D\right)$$, or the posterior distribution. This is our belief in the distribution of parameters given our evidence. It's the main thing that want to find. Given our data/evidence, what are the most probabilistically liekly parameter spaces? For every combination of parameters that we sample from, the posterior distribution encodes a probability that they are likely. Finally, our last term is $$P\left(D\right)$$, referred to as the marginal likelihood. This is a scaling factor (as in it's not a distribution, it's just a constant) that normalizes our posterior distribution. For most Bayesian inference calculations, it doesn't factor in (luckily for us, because it can be difficult to impossible to calculate, sometimes). Since we use some kind of Markov Chain Monte Carlo in order to sample from our posterior distribution, exact realizations aren't really important, as long as they're all scaled by the same factor. Therefore, we often write Bayes' Theorem as follows.

**Also Bayes' Theorem**  

$$
\begin{aligned}
  P\left(\Theta \vert D\right) \propto P\left(D \vert \Theta\right)P\left(\Theta\right)
\end{aligned}
$$  

This is to say that when we sample from our posterior distribution of parameters using a stochastic method like MCMC, we are actually sampling from a distribution whose realizations are proportional to the true posterior distribution by a factor of $$P\left(D\right)$$, which is just as well and WAY easier.

So, why do we care to take a Bayesian approach to statistics (specifically for today, model parameterization) rather than something frequentist like least-squares? One great reason is exploring the possibility of multimodal parameter distributions. Sometimes, there are numerous parameter combinations that are equally likely for our model, given our data. A Bayesian approach to parameterization can make this very apparent, very quickly and can inform us on ways to improve our model form that we're fitting to or the sampling approach to exlore those multiple optima efficiently (things that are hard to diagnose with frequentist data fitting methods). Another, BIG, reason we might use a Bayesian parameterization approach is the robust and simple way that it quantifies uncertainty in parameters AND model estimates. Parameter uncertainties are easily encoded because the process gives you DISTRIBUTIONS of parameters. We can estimate these distributions as something familiar (like a gaussian or cauchy or what have you) and get robust parameter uncertainties from that. Additionally, we can easily account for error in our fitting data by making it part of our likelihood function. Below is an example of how we could modify a likelihood function with normally distributed errors, to also be constrained by the measurement error from our data. 

[//]: # (Add example of adding data error to gaussian likelihood error model)

**Likelihood Function for a Normally Distributed Observation**

$$
\begin{aligned}
  P\left(D \vert \Theta\right) = \Pi_{j=1}^{n} \left(2\pi\sigma^2\right)^{-1/2}\exp\left(-\frac{1}{2}\frac{\left(D_j-M(\Theta)\right)^2}{\sigma^2}\right)
\end{aligned}
$$ 

In the above, $$D_j$$ is the $$j^{th}$$ data point in our pool of $$n$$ evidence points, $$M(\Theta)$$ is the forward data model evaluated at current parameter state $$\Theta$$ and $$\sigma$$ is the uncertainty in our forward data model. Oftentimes, we estimate that $$\sigma$$ as a parameter in our inference. If we had a measurement uncertainty (let's say $$s$$) associated with each data point in our pool of evidence, then we could account for it by replacing each instance of $$\sigma^2$$ with $$(\sigma^2+s_j^2)$$ in the likelihood function (where $$s_j$$ is the measurement uncertainty of evidence data point $$j$$). This data uncertainty would likely change how the parameters are constrained and the uncertainty of our final parameter distributions.

## Introducing the Problem
The project that we're doing today is [based on a post I found on Physics Stack Exchange](https://physics.stackexchange.com/questions/388238/liquid-density-as-a-function-of-pressure-and-temperature-how-to-model-experimen). While I am a bit disappointed that the provided data does not have accompanying uncertainties (which therefore makes its quality suspect at best), it IS well-behaved and provided something nice and simple to work with (even if it may be synthetic, I have no idea). Maybe in a future post I'll grab some nice data from the NIST ThermoML database (with vetted values and uncertainties). Anyway, the problem for today is fitting an analytical model for liquid densities as a function of temperature and pressure. We're using the 5 parameter model from [a reference text by D. J. Furbish](https://global.oup.com/academic/product/fluid-physics-in-geology-9780195077018?cc=us&lang=en&) written as:

**Furbish's Liquid Density Equation of State**

$$
\begin{aligned}
  \rho\left(T,p\right) = \rho_0\left\[1-\alpha\left(T-T_0\right)+\beta\left(p-p_0\right)\right\]
\end{aligned}
$$ 

In the above, $$T$$ is temperature, $$p$$ is pressure and $$\rho$$ is mass density. The parameters $$\alpha$$ and $$\beta$$ are physically relevant quantities known as the isobaric coefficient of thermal expansion and local isothermal compressibility, respectively. The remaining parameters $$\rho_0$$, $$T_0$$ and $$p_0$$ are reference values of density, temperature and pressure likely to occur in the range of values given by the problem of interest. 

## Setting Up the MCMC Model
For this little project, the primary differences between the parameterizations were in how the MCMC models were setup and implemented, so it's important to get a detailed run down of what I did. The first model was very vanilla. Uniform priors were used for all of our parameters where the bounds for $$rho_0$$, $$T_0$$ and $$p_0$$ were given by the evidence and the bounds of $$\alpha$$ and $$\beta$$ were just guessed to be between 0 and 1. I also included a sixth parameter for estimate measurement uncertainty ($$\sigma$$) with a uniform prior from 0 to 1 (given the scale of the evidence values). I used a likelihood function assuming normally distributed errors. Below is a nice symbolic summary. 

**Model Summary**
*Parameters*
$$
\begin{aligned}
  \Theta = {\rho_0,\alpha,T_0,\beta,p_0,\sigma}
\end{aligned}
$$   
$$
*Priors*
$$
\begin{aligned}
  \P\left(\rho_0\right) ~ \mathit{U}\left(0.6,0.9\right)
  \P\left(\alpha\right) ~ \mathit{U}\left(0,1\right)
  \P\left(T_0\right) ~ \mathit{U}\left(40,390\right)
  \P\left(\beta\right) ~ \mathit{U}\left(0,1\right)
  \left(p_0\right) ~ \mathit{U}\left(2000,27000\right)
  \left(\sigma\right) ~ \mathit{U}\left(0,1\right)
\end{aligned}
$$  
*Likelihood*
$$
\begin{aligned}
   P\left(D \vert \Theta\right) = \Pi_{j=1}^{n} \left(2\pi\sigma^2\right)^{-1/2}\exp\left(-\frac{1}{2}\frac{\left(D_j-M(\Theta)\right)^2}{\sigma^2}\right)
\end{aligned}
$$
Where $$M(\Theta)$$ (our forward data model) is Furbish's from earlier.

## Implementations in `emcee` and `PyMC3`

## Analysis

[//]: # (Do a quick parameterization in scipy or sklearn for comparison to the Bayesian fits)

## Conclusions
