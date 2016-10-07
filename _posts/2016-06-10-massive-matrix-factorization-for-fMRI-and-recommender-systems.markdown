---
layout: post
title:  "Massive Matrix Factorization for fMRI and Recommender Systems"
date:   2016-06-10 10:19:34 +0200
categories: research
images:

 - url: /assets/figures/brains.png
 - alt: 10x speed-up for brain decomposition
 - title: 10x speed-up for brain decomposition

---

As ICML approaches, I will give a quick overview of our work on efficient
matrix factorization for very large datasets. Our focus was to scale *matrix
factorization* techniques for functional MRI, a domain where data to
decompose is now at terabyte scale. Along the way, we also designed a encouraging proof-of-concept
experiment for collaborative filtering.

{% include mathjax.html %}

We'll start by reviewing matrix factorization techniques for interpretable data
decomposition. $$\def\EE{\mathbb E}    \def\RR{\mathbb R}    \def\PP{\mathbb P}    \def\A{\mathbf A} \def\D{\mathbf D} \def\M{\mathbf M} \def\X{\mathbf X} \def\b{\mathbf b} \def\a{\mathbf a} \def\d{\mathbf d} \def\x{\mathbf x} \def\balpha{\boldsymbol{\alpha}} \def\argmin{\text{argmin}}$$

## Understanding data with matrix factorization

Unsupervised learning aim at finding patterns in a sequence of n samples
$(x_t)t$, living in a $p$ dimensional space. Typically, this involve finding a few statistics that describe data in a *compressed* manner. Our dataset can be seen as a large matrix $\X \in \RR^{n \times p}$. Factorizing such matrix has proven a very flexible manner to extract interesting pattern. Namely, we want to find two *small* matrices $\D$ (the *dictionary*) and $\A$ (the *code*) with $k$ columns/rows whose product approximates $\X$

<img src="/assets/drawings/poster_model_sparse.png" width="80%" style="display: block; margin: 0 auto;" title="Model" />

Small can mean several things here : we may impose $k$ to be small, which amounts to search for a low rank representation of the matrix $\X$, and thus a subspace of $\RR^p$ that approximately include all samples. For interpretability, it can be useful, as in the drawing above, to impose sparsity on $\D$ -- this is what we'll do in fMRI.

In other settings, we may have $k$ large but impose $\A$ *sparse*, leading to an overcomplete dictionary learning setting, that generalize the k-means algorithm. This setting won't interest us today, although we use its terminology.

### fMRI example

We can already instantiate matrix factorization for fMRI as this will make things clearer. We study resting-state functional imaging : 500 subjects go four times in a scanner, to get their brain activity recorded during 15 minutes while at rest -- roughly, a 3D image of their brain activity is acquired every second. This yields 2 millions 3D images of brain activity, each of them with 200 000 *voxels* -- **2TB** of dense data. We want to extract spatial activity maps that constitute a good basis for these images:

<img src="/assets/drawings/poster_fmri_dl_flat.png" width="80%" style="display: block; margin: 0 auto;" title="Model" />


What we are most interested in is the dictionary $D$, that holds, say, 70 sparse spatial maps. We expect those to capture functional networks, segmenting the auditory, visual, motor cortex, etc. Sparsity and low-rank are key for pattern discovery: we want to find few maps, with few activated regions.

## Matrix factorization for large datasets

A little math should be introduced to better grasp our problem. Decomposing $\X$ into the product $\D \A$ can be done by solving an optimization problem (see **[Olshausen '97]** for the initial problem setting):

$$\min_{\D \in \mathcal{C}, \A \in \RR^{k\times p}} \Vert \X - \D \A \Vert_2^2 + \lambda \Omega(\balpha)$$

where structure and sparsity can be imposed via constraints (convex set $\mathcal{C}$)
and penalties. For example, we may impose dictionary columns to live in $\ell_1$ balls, to get a sparse dictionary.

Solving this minimization problem is where all the honey is : let's see what methods can be used when $X$ grows large.

A naive solver alternatively minimize the loss function over $\A$ and $\D$. Meaning, given $\X$ and $\A$, find the best $\D$, given $\X$ and $\D$, find the best $\A$, and repeat. If we look at it from a dictionary oriented point of view, we define

$$\A(\D) = \argmin_{\A \in \RR^{k \times n}} \Vert \X - \D \A \Vert_F^2  + \lambda \Omega(\A)$$

$$\balpha_i(\D) = \argmin_{\A \in \RR^{k \times n}} \Vert \x_i - \D \balpha_i \Vert_F^2  + \lambda \Omega(\balpha_i)$$

where the second equality has used the colummns $(\balpha_i)$ of $\A$ -- we'll see why in a minute. The naive algorithm simply consist in doing

$$\begin{align}
\D_t &= \argmin_{\D \in \mathcal{C}} \Vert \X - \D \A(\D_{t-1}) \Vert_F^2 \\
&= \min_{D} \sum_{i=1}^n \Vert \x_i - \D \balpha_i(\D_{n-1})) \Vert_F^2
\end{align}$$

This takes time, as the whole data $\X$ is loaded at each iteration. In fact, it quickly becomes intractable: beyond 1 million entry in $\X$, it already takes hours.

### Going online

A very efficient way to get past this intractability was introduced by **[Mairal '10]**. Computing $\A$ for the whole dataset is costly, and overkill for a single step of improving the dictionary: we can maintain an approximation of this code by streaming the data and optimizing the dictionary along the stream.

<img src="/assets/drawings/poster_model_sparse_online.png" width="80%" style="display: block; margin: 0 auto;" title="Model" />

As the drawing above indicates, we look at data sample $x_t$ after
sample. At iteration $t$t, we use the current dictionary to compute the associated loadings
$\balpha_t$:

$$\balpha_t(\D) = \argmin_{\A \in \RR^{k \times n}} \Vert \x_t - \D_{t-1} \balpha_t \Vert_F^2  + \lambda \Omega(\balpha_t)$$

We then solve, at each iteration

$$\D_t = \argmin_{\D \in \mathcal{C}} \sum_{i=1}^t \Vert \x_i - \D \balpha_i \Vert_F^2$$

This look very much like the original update, except we use outdated
$\balpha_t$ to approximate our objective function. The essential idea here is
start solving the problem with a very inaccurate approximation of it, and
improve it by looking at more data.

A single iteration of the algorithm depend on $p$ but no longer on $n$, and the
algorithm empirically converges in a few epochs on data. This is very efficient
when data dimension $p$ is reasonable -- as a matter of fact the online algorithm
was initially designed to handle large sequences of 16x16 image patches -- **a
very low p compared to fMRI setting**.

## Handling large sample dimension

This is where our contribution begins. We want to provide an algorithm that
 scales not only in the number of samples but also in the sample dimension. To
 scale in the number of samples, we went from using $\X$ to using $\x_t$ at
 each iteration, allowing around n time faster iterations. Here, $\x_t$ is
 still too large, and **we want to acquire information even faster**.

This is where we introduce *random subsampling*: can we improve the dictionary
with only a *fraction* of a sample at each iteration. The answer is yes, as we'll
now show.  The algorithm we propose loads a fraction of a sample $\x_t$ at each
iteration and use it to update the approximation of the optimization problem.
The fraction is different at each iteration: this way, we are able to obtain
information about the whole feature space, in a stochastic manner. We go a step
beyond in randomness:

![Random subsampling](/assets/drawings/poster_next_level.png)

$\M_t \x_t$ corresponds to a subsampling of $\X_t$, choosing $\M_t$ to be a $[0, 1]$ diagonal matrix with, say, 90% zeros.

The whole difficulty lies in constructing the right approximations so that the
problem we solve at each iteration looks more and more like the original
optimization problem -- just like the online algorithm does.

The online algorithm relies on a few low dimensional statistics that
 sufficiently describe the approximate problem. These are updated in a
 $\mathcal{O}(p)$ cost -- ensuring scalable single iteration, and hence the online magic.

Our objective here is to speed up iteration of a constant factor, that
corresponds to the factor of dimension reduction. We must therefore ensure that
everything we do at iteration t scales in $\mathcal{O}(s)$, where $s$ is the *reduced* dimension.
That way, we gain a constant factor (from 2 to 12 on large datasets, as we'll
see) on single iteration complexity (*computational speed-up*), and we expect not to
loose it because of the approximation we introduce (*approximation errance*).

This is because **very large datasets have often many redundancies**, accessing
a stochastic part of sample does not reduce much the information acquired at
each iteration.  As we'll see, on large datasets, the balance is therefore very
much on the side of single iteration computational speed-up.

The constraint we introduce on iteration complexity restrains much what we are able to do. To sum up, we have to adapt the three steps of the online algorithm
- Computing the code from past iterate : we rely on a *sketched* version of code computation, where we only look at $\M_t$ features of $\x_t$ and $\D_{t-1}$

$$\begin{align}\balpha_t(\D) &= \argmin_{\A \in \RR^{k \times n}} \Vert \M_t(\x_t - \D_{t-1} \balpha_t) \Vert_F^2\\
&\phantom{=} +\lambda \frac{s}{p} \Omega(\balpha)\end{align}$$

- Aggregating this partial sample and code in an approximative objective, as we do by summing $t$ factors in the online algorithm. We have to do this in a clever manner so that we only update statistics of size in s and not in p. This includes keeping tracks of the number of time we saw a feature in the past.

- Updating the dictionary: we can't update the full $\D$ at each iteration as this is $\mathcal{O}(p)$ costly. It makes sense to update the features of the dictionary atoms that were seen in $\M_t$, ensuring that $\D$ remains in $\mathcal{C}$ by projection.

I skipped the math in the two last parts, but you can access it in more detail [on these slides](docs/presentations/icml_presentation.pdf). You will also find a detailed comparison between our algorithm and the original online algorithm.

## Results

Let's get to the most important part: do we get desired speed-up, is the dictionary
we compute as good as those we would obtain with previous algorithms ?

**On fMRI, we can push the reduction up to x12 and obtain x10 speed-up compared to the online algorithm**. Remember that a single pass on the data would take 235h using the online algorithm. We'll use the obtained maps as a baseline. Maps are blobish, with noiseless contours.

In no more than 10h, our algorithm, using a 12-fold reduction, is able to recover maps that are almost as epxloitable as the baseline one. In comparison, the original algorithm stopped after 10h yields very poor results: noisy maps with many blobs.

Displaying the contour of these maps makes it clearly appear:

![Brains](/assets/figures/brains.png)

We can quantify the speed-up we obtain by looking at convergence curve, that decribe how good the dictionary perform as a basis on a test set, against time spent in computation.

<img src="/assets/figures/bench.png" width="70%" style="display: block; margin: 0 auto;" title="Bench" />

**Convergence is obtained x10 more quickly** with a 12 times reduction.
This is very valuable for practioners ! Information is indeed acqired faster,
as the speed-up we obtained is close to the reduction we imposed.

## Collaborative filtering

Our setting imposes masks on data to speed up learning. Quite interestingly,
collaborative filtering brings us a setting where we can only acces *masked*
data, that corresponds to, for example, the few movies that a user has rated.
Matrix factorization is there used to reconstructe the incomplete matrix $\X$ (see, for instance **[Szabo '11']**).
To evaluate the performance of our model, we look at rating prediction on a
test set. We compare our algorithm with a fast coordiate descent solver from
[spira](http://github.com/mathieublondel/spira), that does not involve setting any
hyperparmeter -- our algorithm is, unlike SGD, not very dependant on
hyperparameters. We get good results on large datasets (Netflix,
Movielens 10M), as these benches show. On **Netflix**, our algorithm is **7x faster** than the coordinate descent solver, which was the fastest well-packaged collaborative filtering algorithm we could find.

<img src="/assets/figures/rec_bench.png" width="100%" style="display: block; margin: 0 auto;" title="Collaborative filtering benches" />

Our model is very simple (minimization of an $\ell_2$ loss), and we do not get
state of the art prediction on Netflix. However, this experiment shows that our
algorithm is able to learn a decomposition even with non random masks, and
demonstrate the efficiency of imposing the complexity constraints explained
above.

## Conclusion

Leveraging random subsampling with online learning is thus a very efficient manner to perform matrix factorization on datasets large in both direction. Our algorithm had no convergence guarantee at the time of contribution (February), but we now have a slightly adapted algorithm that converges with the same guarantee as in the original online matrix factorization paper -- we rely on the stochastic majorization minimization framework **[Mairal '13]**.

[A Python package](http://github.com/arthurmensch/modl) is available for reproducibility. We hope to integrate this algorithm in more well-known library in the long term.

I hope that this post was readable enough and has interested you. You'll find
more details in our [paper](https://hal.archives-ouvertes.fr/hal-01308934),
[poster](/docs/posters/icml_poster.pdf) and [slides](/docs/presentations/icml_presentation.pdf). I
will present this work in ICML New York Monday June 20th at 11h45. Discussions are
welcome !

## References

- **[Mairal '10]** Mairal, Julien, Francis Bach, Jean Ponce, and Guillermo Sapiro. “Online Learning for Matrix Factorization and Sparse Coding.” The Journal of Machine Learning Research, 2010.

- **[Mairal '13]** Mairal, Julien. “Stochastic Majorization-Minimization Algorithms for Large-Scale Optimization.” In Advances in Neural Information Processing Systems, 2013.

- **[Olshausen '97]** Olshausen, Bruno A., and David J. Field. “Sparse Coding with an Overcomplete Basis Set: A Strategy Employed by V1?” Vision Research, 1997.

- **[Szabo '11]** Szabó, Zoltán, Barnabás Póczos, and András Lorincz. “Online Group-Structured Dictionary Learning.” In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2011.

- See also [these slides](/docs/presentations/icml_presentation.pdf)
