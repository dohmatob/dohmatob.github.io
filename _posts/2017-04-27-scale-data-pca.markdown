---
layout: post
title:  "The importance of standardizing data before running PCA"
date:   2016-10-07 14:01
categories: research
images:

 - url: /assets/scaling_files/scaling_6_1.png
 - alt: Sobolev regularization meets online DL
 - title: Sobolev regularization meets online DL

---


Standardization is important in PCA since the latter is a variance maximizing exercise.
It projects your original data onto directions which maximize the variance. If your
features have different scales, then this projection may get screwed!

- References: <a href="https://stats.stackexchange.com/a/69159/156791">stackexchange discussion</a>

The experiments
===============

{% highlight python %}
# main imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

%matplotlib inline

{% endhighlight %}



{% highlight python %}
# load data
import os
url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/USArrests.csv"
in_file = os.path.basename(url)
df = pd.read_csv(in_file if os.path.exists(in_file) else url)
del df["Unnamed: 0"]
{% endhighlight %}


{% highlight python %}
# import estimators
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
{% endhighlight %}


{% highlight python %}
# fit pipelines
n_components = 4
pipelines = {}
for with_scaling in [True, False]:
    name = "with%s scaling" % ("" if with_scaling else "out")
    steps = []
    if with_scaling:
        name = "with scaling"
        steps.append(("scaler", StandardScaler()))
    else:
        name = "without scaling"
    steps.append(("pca", PCA(n_components=n_components)))
    pipeline = Pipeline(steps).fit(df)
    pipelines[name] = pipeline
{% endhighlight %}


{% highlight python %}
# plotting

def plot_pipeline(pipeline, title=None):
    pca = pipeline.steps[-1][1]
    _, (ax2, ax1) = plt.subplots(1, 2)
    if title is not None:
        plt.suptitle(title)
        
    im = ax1.matshow(pca.get_covariance(), cmap="viridis")
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("covariance matrix")
    ax1.axis("off")
    ax1.set_aspect('auto')
    
    ax2.bar(range(n_components), pca.explained_variance_)
    ax2.set_xticks(.5 + np.arange(n_components))
    ax2.set_xticklabels(["PC #%02i" % (c + 1) for c in range(n_components)])
    ax2.set_ylabel("explained variance")
    
    plt.tight_layout()
    return ax1, ax2
{% endhighlight %}


{% highlight python %}
for name, pipeline in pipelines.items():
    plot_pipeline(pipeline, title=name)
{% endhighlight %}


Results
=======
<img src="/assets/scaling_files/scaling_6_0.png"/>
<img src="/assets/scaling_files/scaling_6_1.png"/>

