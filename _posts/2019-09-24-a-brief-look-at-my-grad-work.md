---
layout: post
title: A Brief Look at my Graduate Work
description: >
  Initial blog post: More on what my graduate work was and my intro to applied stats and data science.
noindex: true
---
Welcome back to the blog folks! Today's post is going to give more depth on my "real" introduction to data science and software development and what my research in graduate school was centered around. I use "real" (note the quotation marks) because while I already had had my soft intro to useuful stats and software as an undergrad, this was my whole job and study now and I jumped in the deep end. In the [previous post](https://bmanubay.github.io/2019-09-17-welcome-to-my-blog/) I spoke briefly about my work in molecular dynamics (MD) research and how it got me more deeply involved and studied in applied math and stats. This post will shed some light on what classical mechanics simulations are, how they work and specifically what my work was contributing to them. 
## What are MD simulations, how do they work and what are they good for?
Before we get into my graduate work, specifically, let's give a short, high-level overview of MD, how it works and what it's good for. In very simple terms, MD is a simulation tool used to study how things move at a molecular or atomic scale. In its most common form, trajectories of spatial positions of atoms over time are determined by solving Newton's Laws of Motion. 

**Newton's Laws of Motion** 
$$
\begin{aligned}
  &\frac{\partial\mathbf{x}}{\partial t} = \mathbf{v} \\[2em]
  &\frac{\partial\mathbf{v}}{\partial t} = \frac{F\left(\mathbf{x}\right)}{\mathbf{m}} \\[2em]
  &F\left(\mathbf{x}\right) = -\nabla U\left(\mathbf{x}\right)
\end{aligned}
$$

Where $$\mathbf{x}$$ is position, $$\mathbf{v}$$ is velocity, $$F\left(\mathbf{x}\right)$$ is interatomic force as a function of position and $$U\left(\mathbf{x}\right)$$ is interatomic potenital energy as a function of position. Specifically, the simulation makes very small time steps (often only a few femtoseconds), at each step calculating forces on each particle in the system and updating the positions and velocities of said particles at each step using [some kind of time symmetric integration method](https://en.wikipedia.org/wiki/Verlet_integration) (there are many ways to do this and is a very active field of research in itself). This process of take a step, calculate force, update position and velocity repeats for (hopefully) some physically relevant time-scale in order to observe a molecular phenomena or process under study. MD simulations are very accurate when used correctly. They are often used to find drug leads (everything from simulating whether or not the molecule is stable to how it binds with a therapeutic target to its crystalline polymorphism and bioavailability), calculate physical properties of different species with high accuracy and visualize protein folding and ligand binding.  

<figure>
  <img alt="An image with a caption" src=/assets/img/dna_Si3N4_diff.gif class="lead" />
  <figcaption>A short visualization I made from trajectory data ([found here](http://www.ks.uiuc.edu/Training/Tutorials/nanobio/) showing DNA diffusing through a  nanopore of Si\textsubscript{3)N\textsubscript{4} for testing as a higher throughput gene-sequencing material.</figcaption>
</figure>

Overall, this is a pretty simple method for simulating molecules. No explicit quantum mechanics and just simple classical physics that we all learn from high school. The only lingering question is **how the heck do we calculate that potential energy that we need in order to find our forces**? A great question and at its cores lies one of the most confounding parts of all MD and classical mechanics simulation. How we actually calculate the potential is quite simple. We use a set of potential equations and chemistry specific parameters in order to calculate the short-range, long-range and bonded potentials of the system as a function of position. We call this set of equations and parameters a "Force Field". Most of these potentials are easy to understand; think like Lennard-Jones non-bonded potential energy, spring potentials for the energy of a bonded pair of atoms, etc. The more complicated part (and the part that's a bit of a black box for most non-expert users of the method) is the parameter part of a force field. Where do these parameters come from? How do we assign them? We'll cover this conundrum in the next section.     

## What is the [Open Force Field Initiative](https://openforcefield.org/) and what are they doing to change the field?
Up until now, determining force fields has been treated as a very high dimensional model optimization problem wherein we train a set of models with upwards of 1000 distinct parameters to find a single optimal force field. Some common implementations include use of clever [evolutionary](https://www.sciencedirect.com/science/article/pii/S0020169399005769) [algorithms](https://pubs.acs.org/doi/abs/10.1021/acs.jpca.5b00218) for both initial parameterization and further refinement as well as [automated objective function minimization schemes](https://pubs.acs.org/doi/10.1021/jz500737m). There are obviously a few issues with this. First of all, the idea of a single optima is naive at best. It's reasonable to believe that in a parameter space that complex that there would be many disparate optima that are essentially equal and deserving of comparison. Second, choice of potential models is not necessarily a constant or obvious as to what's best. Let's take, for example, long-range, non-bonded potentials. The following are just a few common choices of model:

- 12-6 Lennard Jones Potential: $$V_{\text{LJ}}=\varepsilon \left[\left({\frac {r_{\text{m}}}{r}}\right)^{12}-2\left({\frac {r_{\text{m}}}{r}}\right)^{6}\right]$$ 
- Buckingham Potential: $$V_{\text{B}}=\gamma \left[e^{-r/r_{1}}-\left({\frac {r_{0}}{r}}\right)^{6}\right]$$ 
- Morse Potential: $$V'(r)=D_{e}(1-e^{-a(r-r_{e})})^{2}$$ 
- Morse/Long-Range Potential: $$V(r)={\mathfrak {D}}_{e}\left(1-{\frac {u(r)}{u(r_{e})}}e^{-\beta (r)y_{p}^{r_{\rm {eq}}}(r)}\right)^{2}$$ 

Given all of these potential choices, which one do we choose for our force field? How do we quantitatively compare them? An answer that the Open Force Field Initiative came to was reframing the force field optimization problem as a Bayesian inference problem. Let's take a quick aside in order to look at Bayes' Theorem. 

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

<figure>
  <img alt="An image with a caption" src=/assets/img/hierarchical_property_calculation_pyramid.png class="lead" />
  <figcaption>A representation of the hierarchical property calculation scheme. </figcaption> 
</figure>

It is a three level, multi-fidelity calculation scheme. We start with a full MD simulation in order to generate a robust configuration ensemble. From that single, well-described state point we can estimate properties (specifically the thermophysical properties that we are going to use to constrain our force field) at many local state points using a multi-state statistical reweighting technique like the [Multistate Bennett Acceptance Ratio (or MBAR)](https://arxiv.org/abs/1704.00891). A nice feature of ensemble configuration simulations, such as MD or Monte Carlo, is that the configurations they sample contain a huge amount of information and it turns out that you can expand the results of your original simulation to a sufficiently close state point (or pointS!!). In particular, as long as the configuration space (or probability distributions) sampled by the state0s are sufficiently similar and overlap, we can get very good estimates of properties at other state points using the configurations of just one simulation. The form turns out to be very simple.

**Reweighting Expectation**
$$
\begin{aligned}
{\langle O \rangle}_i = \frac{1}{N} \sum_{n=1}^N O\left(\vec{x}_n\right) \left(\frac{p_i\left(\vec{x}_n\right)}{p_j\left(\vec{x}_n\right)} \right)
\end{aligned}
$$

The above shows how to find the expectation of observables in some state(s) *i*, from N configuration ($$\vec{x}$$) samples in state(s) *j*. The trick is that we can find the ratio of probability distributions from a ratio of internal energy (or enthalpy) distributions. We simply use the configurations that we sampled from state(s) *j* and re-evaluate the energies at state(s) *i*. This is why the configurations in the new state(s) need to sufficiently overlap with the original(s).

Okay, great, now we have a set of minimum variance, unbiased observables in a local parameter space with pretty minimal computational cost. However, we can go one step further in generating realizations of our forward data model as a function of force field parameter rapidly (like analytical model, rapidly). The final stage of our multifidelity calculation scheme is to construct a regression-type model of our observables over parameter space from the calculations we made using reweighting. There are many possibilities for choice of regression technique, but a robust starting point we ended up on was Gaussian process regression or kriging. Kriging is a good choice because, first, estimates have been shown to be stable and accurate for parameter spaces up to around 20 dimensions and, second, realizations are generated extremely fast (even though it's not a non-parametric regression and is more similar to reweighting in how it works). We formalise the estimation of some quantity $$Z$$ at unknown location $$x_0$$ ($$Z\left(x_0\right)$$) from N pairs of observed values, $$w_i\left(x_0\right)$$ and $$Z\left(x_i\right)$$, where $$i = 1,...,N$$, with the following:

**Kriging form**
$$
\begin{aligned} 
  \hat{Z}\left(x_0\right) = \sum_{i=1}^N w_i\left(x_0\right) \times Z\left(x_i\right) 
\end{aligned}
$$
               
We find our weight matrix, **W**, by minimizing **W** subject to the following system of equations:

$$
\begin{aligned}
  &\underset{W}{\text{minimize}}& & W^T \cdot \operatorname{Var}_{x_i} \cdot W - \operatorname{Cov}_{x_ix_0}^T \cdot W - W^T \cdot \operatorname{Cov}_{x_ix_0} + \operatorname{Var}_{x_0} \\              	
  &\text{subject to}
  & &\mathbf{1}^T \cdot W = 1
\end{aligned}
$$                	

Where the literals $$\left\{\operatorname{Var}_{x_i}, \operatorname{Var}_{x_0}, \operatorname{Cov}_{x_ix_0}\right\}$$ stand for
$$\left\{\operatorname{Var}\left(\begin{bmatrix}Z(x_1)&\cdots&Z(x_N)\end{bmatrix}^T\right), \operatorname{Var}(Z(x_0)), 
\operatorname{Cov} \left(\begin{bmatrix}Z(x_1)&\cdots&Z(x_N)\end{bmatrix}^T,Z(x_0)\right)\right\}$$

The weights summarize important procedures of the inference process like the structural closeness of samples to the estimation location, $$x_0$$.They also have a desegregating effect, to avoid bias caused by sample clustering.










 
