---
layout: post
title: An In-depth Dive into Using Bayesian Inference
description: >
  A deep dive into inference based statistics, how it's different from frequentist approaches and what we can do with it in applied data science contexts.
noindex: true
comments: true
---
Welcome back to the blog folks! Today's post is going to give more depth on my "real" introduction to data science and software development and what my research in graduate school was centered around. I use "real" (note the quotation marks) because while I already had had my soft intro to useuful stats and software as an undergrad, this was my whole job and study now and I jumped in the deep end. In the [previous post](https://bmanubay.github.io/2019-09-17-welcome-to-my-blog/) I spoke briefly about my work in molecular dynamics (MD) research and how it got me more deeply involved and studied in applied math and stats. This post will shed some light on what classical mechanics simulations are, how they work and specifically what my work was contributing to them. 

**Bayes' Theorem**  

$$
\begin{aligned}
  P\left(\Theta \vert D\right) = \frac{P\left(D \vert \Theta\right)P\left(\Theta\right)}{P\left(D\right)}
\end{aligned}
$$

In the above, $$\Theta$$ are the parameters of our model (or specifically our force field parameters) and $$D$$ is our data or evidence that we're "training" with. All the terms of this equation also have specific names. $$P\left(\Theta\right)$$ is called a prior distribution; it represents our beliefs and previous knowledge about parameters before we introduce any data. A simple, adequate example of a prior distribution on a bond length parameter in a spring potential would be a uniform distribution on the physically relevant ranges of that bond length. The next term of Bayes' Theorem is $$P\left(D \vert \Theta\right)$$; this is our likelihood distribution. This is the part of Bayes' Theorem that allows us to quantitatively explore parameter space and see which combinations of parameters are well matched to reproduce the evidence we're using. The next part of Bayes' is $$P\left(\Theta \vert D\right)$$, or the posterior distribution. This is our belief in the distribution of parameters given our evidence. It's the main thing that want to find. Given our data/evidence, what are the most probabilistically likely force fields? For every combination of parameters that we sample from, the posterior distribution encodes a probability that that is likely. Finally, our last term is $$P\left(D\right)$$ referred to as the marginal likelihood. This is a scaling factor (as in it's not a distribution, it's just a constant) that normalizes our posterior distribution. For most Bayesian inference calculations, it doesn't factor in (luckily for us, because it's actually quite difficult to calculate). Since we use some kind of Markov Chain Monte Carlo in order to sample from our posterior distribution, exact realizations aren't really important, as long as they're all scaled by the same factor. Therefore, we often write Bayes' Theorem as follows.

**Also Bayes' Theorem**  

$$
\begin{aligned}
  P\left(\Theta \vert D\right) \propto P\left(D \vert \Theta\right)P\left(\Theta\right)
\end{aligned}
$$  

This is to say that when we sample from our posterior distribution of parameters using a stochastics method like MCMC, we are actually sampling from a distribution whose realizations are proportional to the true posterior distribution by a factor of $$P\left(D\right)$$, which is just as well and WAY easier.

Alright, back to the force field problem. The burdens of choice (like "which potential forms to use?" & "which force field in this equally likely family is best?") have systematic solutions by using a Bayesian approach. For example, the multiple optima issue is captured using Bayes. With a Bayesian approach, we sample from a N-dimensional posterior distribution (where N is the number of force field parameters we're training) and end up with a probabilistic landscape of likely force fields. The different force fields that we end up sampling equally can be compared quantitavely by direct simulation or use of Bayes' factors. Additionally, the families can often be shrunk by adding new evidence to constrain the parameter space. Since different thermophysical observables (like mass density or heat capacity or dielectric constant) are affected differently by different parameters of the force field, they constrain the force field differently when we add them to the pool of evidence used in the inference process. The model complexity problem can be compared using sophisticated sampling methods such as Reversible-Jump Markov Chain Monte Carlo, wherein we have separate Markov chains with different model choices that our walker(s) can jump between. When more samples are drawn from a certain chain, it is more likely given our data. 
   
## What was my primary contribution to the Open Force Field Intiative and what are the most useful skills that I learned? 
One HUGE bottleneck to Bayesian inference driven force field parameterization is the time it takes to run our forward data model at each new set of parameters as we sample from the posterior distribution. For a [liquid phase simulation of some biologically relevant small molecules, it takes about 1 hour of wall time on a GPU](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005659) to get good data and statistics. That is ONE forward model realization at ONE point in parameter space for a few HUNDRED dimensional problem. If we were to take this vanilla forward data model approach, the problem is intractable; it simply takes too long. My job in the Open Force Field was how to figure out ways to shorten that. I settled on using a hierarchical scheme in order to estimate the thermophysical properties that we use to compare to our evidence. Below is a simple figure illustrating the scheme. 


[Reference model problem](https://physics.stackexchange.com/questions/388238/liquid-density-as-a-function-of-pressure-and-temperature-how-to-model-experimen)
