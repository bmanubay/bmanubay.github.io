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

**Uninformative Prior Model Summary**
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

The second model used changed the uninformative flat priors to weakly informative normal priors on all of the parameters and, additionally, hyperpriors on the mean and standard deviations of the normal priors on $$\rho_0$$ and $$\alpha$$ (making the entire model weakly hierarchical). Ultimately, this choice was made becuase of the strong correlations between the two parameters and the difficulty of `PyMC3`'s gradient-enhanced sampler exploring the parameter space because of it. The choice of likelihood remained the same. Again, this is nicely summarized symbolically.

**Hierarchical Normal Prior Model Summary**
*Priors*
$$
\begin{aligned}
  \P\left(\mu_{\rho_0}\right) ~ \mathcal{N}\left(\mu=0.75,\sigma=0.01\right)
  \P\left(\sigma_{\rho_0}\right) ~ \mathcal{HC}\left(s=0.01\right)
  \P\left(\rho_0\right) ~ \mathcal{N}\left(\mu_{\rho_0},\sigma_{\rho_0}\right)
  \P\left(\mu_{\alpha}\right) ~ \mathcal{N}\left(\mu=5x10^{-3},\sigma=10^{-3}\right)
  \P\left(\sigma_{\alpha}\right) ~ \mathcal{HC}\left(s=10^{-3}\right)
  \P\left(\alpha\right) ~ \mathcal{N}\left(\mu_{\alpha},\sigma_{\alpha}\right)
  \P\left(T_0\right) ~ \mathcal{N}\left(\mu=200,\sigma=10^2\right)
  \P\left(\beta\right) ~ \mathcal{N}\left(\mu=0,\sigma=10^{-5}\right)
  \left(p_0\right) ~ \mathcal{N}\left(\mu=14000,\sigma=6x10^{3}\right)
  \left(\sigma\right) ~ \mathcal{HC}\left(s=0.05\right)
\end{aligned}
Where $$\mathcal{HC}\left(s\right)$$ is the half-cauchy distribution centered around 0 with scale parameter $$s$$.
$$  

## Implementations in `emcee` and `PyMC3`
The two MCMC software packages we used to carry out the posterior sampling, `emcee` and `PyMC3`, are both implemented in Python3. There are clear advantages to both (from speed to diagnostic capability) which will become more clear when we view our results. First, let's import all of our relevant packages that we'll use for our different sampling methods.
~~~python
%matplotlib notebook #ignore this is you're not using a Jupyter Notebook. This sets the rendering backend.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy as sp
import corner
import arviz as az
from IPython.display import display, Math
from multiprocessing import Pool

import seaborn as sns
import time

import emcee as mc
import pymc3 as pm

import pickle
import h5py

sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
%config InlineBackend.figure_format = 'svg'
np.random.seed(111)
~~~
Now we read in and configure the mass density versus temperature and pressure data, which I have mad available [here](https://github.com/bmanubay/bayes_parameter_estimation_for_blog/blob/master/liquid_mass_dens_vs_T_P.csv).
~~~python
#Read and configure data
df = pd.read_csv("liquid_mass_dens_vs_T_P.csv")

df["Rho, g/cm3"] = df["Rho, g/cm3 * 10^4"]/10000.
df = df[["Temp, F","Press, psia", "Rho, g/cm3"]]

T = df["Temp, F"].values
P = df["Press, psia"].values
y = df["Rho, g/cm3"].values
~~~
Now, we can set up the probability backend models in `emcee`. 
~~~python
# Define our probability models
# We define our probability models as their natural log analogs for numerical robustness
# Key here, models will pull T & P directly from the evidence objects, so we don't attempt to estimate those as a parameter
# The theta model will be run variationally only including
#ρ=ρ(T0,P0)*(1−α(T−T0)+β(P−P0))

def rho(theta,T,P):
    #forward data model for rho as a function of T, P and parameters (theta) 
    rho_T0_P0,alpha,T0,beta,P0,eps = theta
    return np.array(rho_T0_P0*(1 - alpha*(T-T0) + beta*(P-P0)))

def lnprior(theta):
    #model defining the prior contribution to our log posterior density
    rho_T0_P0,alpha,T0,beta,P0,eps = theta
    if 0.6 < rho_T0_P0 < 0.9 and 0. < alpha < 1. and 40 < T0 < 390 and 0. < beta < 1. and 2000. < P0 < 27000. and 0. < eps < 0.05:
        return 0.0
    return -np.inf

def log_like(theta,T,P,y):
    #model defining our likelihood contribution (given T, P, y pairs --> i.e. evidence) to our log posterior density
    rho_T0_P0,alpha,T0,beta,P0,eps = theta
    # Product of all lnlikelihoods at given theta for ALL evidence (ALL T and P)
    loglike = []
    for i in range(len(y)):
        loglike.append(-(1./2.)*np.log(2.*np.pi) - (1./2.)*np.log(eps**2) - (1./(2.*eps**2))*(y[i]-rho(theta,T[i],P[i]))**2)
    return np.array(loglike)

def lnprob_blobs(theta, T, P, y):
    #model defining how to calculate our log posterior density from prior and likelihood
    #returns the posterior density, prior and likelihood calculation for every sampling step
    rho_T0_P0,alpha,T0,beta,P0,eps = theta
    prior = lnprior(theta)
    like_vect = log_like(theta, T, P, y)
    like = np.sum(like_vect)
    return like + prior, prior, like
~~~
Note that we've defined all of our probability models as their natural log analogs. This helps with numerical robustness because instead of multiplying iid events on the interval $$\[0,1\]$$ (which quickly go to zero), we add the log analogs on the interval $$\[-\infty,0\]$$. Now that we have all the backends set up to sample with `emcee`, the last major choice we have to make is "how and where to initialize the sampling?" There are many potential options. We could set start points randomly (which is a nice, robust choice to test chain convergence), we could use some relatively inexpensive [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) technique to give a pretty good estimate of where the chain will end up and thus just end up sampling our high probability areas using MCMC. Given that this probability space is relatively simple (smooth, not apparently multimodal or disjointed, etc.), I chose to use the [maximum a posteriori (or MAP)](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) as a starting point. In general, this is not wise. The MAP (which is also the mode of the posterior) [does not always look like a typical sample from the distribution](https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/) and can be a very poor starting point. Additionally, the gradient of the posterior at the MAP is ZERO by definition ([which can be a big problem for gradient-enhanced sampling methods like those implemented by default in `PyMC3`](https://docs.pymc.io/api/inference.html#module-pymc3.sampling)) However, in low dimensional problems (like ours) it's usually okay. We can initialize our sampling with the following snippets.
~~~python
#get MAP estimate
np.random.seed(45)
nll = lambda *args: np.sum(-log_like(*args))
initial = np.array([0.7,0.01,50,0.01,3000.,0.01]) + 0.1*np.random.randn(6)
soln = minimize(nll, initial, bounds=((0.6,0.9),(0,1),(40,390),(0,1),(2000,27000),(0,0.05)),args=(T, P, y))
rho_T0_P0_ml, alpha_ml, T0_ml, beta_ml, P0_ml, eps_ml = soln.x
~~~
~~~python
pos0 = soln.x + 1e-4*np.random.randn(30, 6)
nwalkers, ndim = pos0.shape

max_n = 30000 #we'll draw 30000 samples from 30 iid simulations

# If you want to save the samples like I did, this is how!
# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "emcee_UI_priors_trace.h5"
backend = mc.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
dtype = [("log_prior", float), ("log_like", float)] #emcee blob dtypes

with Pool() as pool: #I'm running my simulation on multiple cores using the Multiprocessing module
    sampler = mc.EnsembleSampler(nwalkers, ndim, lnprob_blobs, args=(T,P,y), pool=pool, backend=backend, blobs_dtype=dtype)
    print("Begin production")
    # Now we'll sample for max_n steps
    sampler.run_mcmc(pos0, max_n, progress=True)
~~~
One of the great things about `emcee` is how lightweight and speedy it is. A simulation this size only take about 45 minutes to an hour. We can set up the weakly informative backend similarly.
~~~python
def rho(theta,T,P):
    rho_T0_P0,alpha,T0,beta,P0,eps = theta
    return rho_T0_P0*(1 - alpha*(T-T0) + beta*(P-P0))

def lnprior(theta):
    rho_T0_P0,alpha,T0,beta,P0,eps = theta

    rho0_mu_hyper = sp.stats.norm.rvs(loc=0.75, scale=1.e-1)
    rho0_std_hyper = sp.stats.halfcauchy.rvs(loc=0., scale=1.e-1)
    
    alpha_mu_hyper = sp.stats.norm.rvs(loc=0.005, scale=1.e-3)
    alpha_std_hyper = sp.stats.halfcauchy.rvs(loc=0., scale=1.e-3)
    
    return sp.stats.norm.logpdf(rho_T0_P0,loc=rho0_mu_hyper,scale=rho0_std_hyper)+sp.stats.norm.logpdf(alpha,loc=alpha_mu_hyper,scale=alpha_std_hyper)+sp.stats.norm.logpdf(T0,loc=200.,scale=100.)+sp.stats.norm.logpdf(beta,loc=0.,scale=1.e-5)+sp.stats.norm.logpdf(P0,loc=14000.,scale=6.e3)+sp.stats.halfcauchy.logpdf(eps,loc=0.,scale=0.05)
    
def log_like(theta,T,P,y):
    rho_T0_P0,alpha,T0,beta,P0,eps = theta
    # Product of all lnlikelihoods at given theta for ALL evidence (ALL T and P)
    loglike = []
    for i in range(len(y)):
        loglike.append(-(1./2.)*np.log(2.*np.pi) - (1./2.)*np.log(eps**2) - (1./(2.*eps**2))*(y[i]-rho(theta,T[i],P[i]))**2)
    return np.array(loglike)

def lnprob_blobs(theta, T, P, y):
    rho_T0_P0,alpha,T0,beta,P0,eps = theta
    prior = lnprior(theta)
    like_vect = log_like(theta, T, P, y)
    like = np.sum(like_vect)
    return like + prior, prior, like
~~~
Again, the only thing that changed is the prior model. The MAP estimate and sampling block are set up identically, except the name you choose for your HDF5 save file (if you choose to). The speed of this simulation is not noticeably affected by the change in prior model.

If we want to set up the sampler with `PyMC3` we can follow a similar workflow. For the uninformative prior model, the probability backends are set up with the following block.
~~~python
#Now let's try with PyMC3
#The data can stay formatted the same way
#Also, wow, look how clean the model and sampling set up is
with pm.Model() as model:
    
    rho0 = pm.Uniform('rho0', lower=0.6, upper=0.9)
    alpha = pm.Uniform('alpha', lower=0., upper=1.)
    T0 = pm.Uniform('T0', lower=40., upper=390.)
    beta = pm.Uniform('beta', lower=0., upper=1.)
    P0 = pm.Uniform('P0', lower=2000., upper=27000.)
    eps = pm.Uniform('eps', lower=0., upper=1.)
     
    rho = rho0*(1-alpha*(T-T0)+beta*(P-P0))
    
    # Likelihood (sampling distribution) of observations
    rho_obs = pm.Normal('rho_obs', mu=rho, sd=eps, observed=y)
~~~
Setup with `PyMC3` is super clean and easy. All of the computation and probability backends currently rely on `theano` (which will be switching hands to the `tensorflow` probability people with `PyMC4`). Setting up the MCMC sampling machinery is similarly very easy.
~~~python
with model:   
    #We'll just use vanilla MH for ease, we can explore NUTS with a better model later 
    step = pm.Metropolis()
    map_estimate = pm.find_MAP(model=model)
    prior = pm.sample_prior_predictive()
    #Draw 30000 posterior samples from 30 chains
    trace = pm.sample(draws=30000, chains=30, cores=4, step=step, start=map_estimate)
    posterior_predictive = pm.sample_ppc(trace)
~~~
There are a ton of bells and whistles for `PyMC3`'s sampling machinery. Choice of step method, initialization method, step adapatation and much more can be user controlled. Additionally, and this will become a big deal in the next section, `PyMC3`'s sampling diagnostics are REALLY GOOD. It will warn you of divergent samples, if the chain is too short for good statistics, if the samples are highly correlated and further sampling should be done. It's great! Also, saving the outputs is as easy as using `Python`'s built-in `pickle` method (which gives surprising errors in `emcee`). An unfortunate drawback to `PyMC3` is the speed though. It's bulky and is best optimized on a HPC node. The simulation above took around 10 hrs on my four core CPU. And even worse it was with vanilla Metropolis-Hastings, so the efficiency of drawing uncorrelated samples is very low and convergence is slow. Defining the new probability backends for the weakly informative hierarchical prior model can be done with the following code block.
~~~python
with pm.Model() as model:
    mu_rho0 = pm.Normal('mu_rho0', 0.75, sd=1.e-1)
    sd_rho0 = pm.HalfCauchy('sd_rho0', 1.e-1)
    rho0 = pm.Normal('rho0', mu=mu_rho0, sd=sd_rho0)
    
    mu_alpha = pm.Normal('mu_alpha', 0.005, sd=1.e-3)
    sd_alpha = pm.HalfCauchy('sd_alpha', 1.e-3)
    alpha = pm.Normal('alpha', mu=mu_alpha, sd=sd_alpha)
    
    T0 = pm.Normal('T0', mu=200., sd=1.e2)    
    beta = pm.Normal('beta', mu=0., sd=1.e-5)    
    P0 = pm.Normal('P0', mu=14000., sd=6.e3)
    eps = pm.HalfCauchy('eps', 0.05)
     
    rho = rho0*(1-alpha*(T-T0)+beta*(P-P0))
    
    # Likelihood (sampling distribution) of observations
    rho_obs = pm.Normal('rho_obs', mu=rho, sd=eps, observed=y)
~~~
For this model, I used `PyMC3`'s flagship No U-Turn Sampler (or NUTS). It's an extension of [Hamiltonian Monte Carlo that requires little to no hand-tuning to sample efficiently](https://arxiv.org/abs/1111.4246). Below is the setup for the sampling block.
~~~python
with model:    
    #Little higher product sampler
    trace_prod = pm.sample(draws=3000,chains=10,cores=4,target_accept=0.9,init='advi+adapt_diag') #start sampler with variational inference estimate
    prior_prod = pm.sample_prior_predictive()
    posterior_predictive_prod = pm.sample_ppc(trace_prod)  
~~~
As the developers claim, NUTS has very efficient sampling such that the convergence is very fast and nearly all of the samples are uncorrelated. However, sampling can be VERY slow if the sampler gets stuck due to low gradient calculations. The above simulation took about 24 hrs to complete. However, the results were very robust, the chains were converged, there were very few divergent samples and there were a high number of uncorrelated samples. Now, that we have all of the machinery out of the way, let's look at the results!
## Analysis

|![emcee UI traces](/assets/img/blog3/emcee_UI_KDE_and_traces.png)|![emcee Weak HA traces](/assets/img/blog3/emcee_weak_KDE_and_traces.png)|
| ------------- |:-------------:|
|![PyMC3 UI traces](/assets/img/blog3/PyMC3_UI_KDE_and_traces.png)|![PyMC3 Weak HA traces](/assets/img/blog3/PyMC3_weak_KDE_and_traces.png)|
*Fig 1. A trace and 1-D marginal posterior comparison of each software and model tested. For each, we show the traces for all 6 variables as well as a KDE estimate for each marginal posterior.*

|![emcee UI ppc plot](/assets/img/blog3/emcee_UI_KDE_and_traces.png)|![emcee Weak HA ppc plot](/assets/img/blog3/emcee_weak_KDE_and_traces.png)|
| ------------- |:-------------:|
|![PyMC3 UI ppc plot](/assets/img/blog3/PyMC3_UI_KDE_and_traces.png)|![PyMC3 Weak HA ppc plot](/assets/img/blog3/PyMC3_weak_KDE_and_traces.png)|
*Fig 1. A trace and 1-D marginal posterior comparison of each software and model tested. For each, we show the traces for all 6 variables as well as a KDE estimate for each marginal posterior.*



[//]: # (Do a quick parameterization in scipy or sklearn for comparison to the Bayesian fits)

## Conclusions
