---
layout: post
title: An introduction to resampling methods and why we should use them.
description: >
  An introduction to resampling methods and why it is such a powerful, useful tool for quantifying uncertainty in predictive models. 
noindex: true
comments: true
---
Welcome back to the blog folks! Hopefully, you all are enjoying your holiday season, but also are still ready for some interesting stats methods. Today we're going to make a small foray into the world of resampling methods. Specifically, we are going use them to quantify the uncertainty in the accuracy of a machine learning model that we will code up in `Python`, using the powerful machine learning package `sklearn` or ['Scikit-learn'](scikit-learn.org). The data set for the classification problem I'm looking at today can be found [here](https://github.com/bmanubay/blog4_notebook_resampling/blob/master/breast-cancer-wisconsin-data.csv) on a GitHub repo I made for this post. It's a ~30 feature diagnostic data set where the target was the malignancy or benignness of a breast tumor. Orginally, I found the dataset on Kaggle. Let's start by introducing the concept of resampling and briefly discussing the methods that I'm going to be going over today. 

## What is resampling?
Let's start by introducing the idea of resampling. Resampling methods are processes of repeatedly drawing samples from a data set and calculating a statistic with an estimator in order to learn more about said estimator. In modern contexts, resampling is often used as a robust way to estimate the uncertainty on predictions of a fitted model. When training a machine learning algorithm, we split the available pool of data into distinct sets: one for training and another for testing. The utility here is to test the accuracy of the model on data that it has never seen before (which is obviously a more robust validation). We can use different resampling methods in choosing our train-test splits in order to place uncertainty bounds on the performance metric (or prediction) of our model.

In the next section we'll use `Python`'s 'Scikit-learn' package in order to train a binary classifier and use several resampling methods in order to put uncertainty bounds on their performance metrics. We'll be investigating the following resampling schemes:

1. Leave-one-out
2. K-fold
3. Random permutation
4. Bootstrapping 

## Implementing resampling methods in a binary classification problem
Before we start introducing the resampling methods, let's train our model with a standard train-test split method and look at some of its performance diagnostics. For this initial training, I'm using an 80-20 train-test split. The implementation of training a random forest classifier using `Python`'s 'Scikit-learn' package is shown below.
### Initial look at our classification problem using a vanilla train-test split
~~~python
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sea
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.model_selection import RFECV
from yellowbrick.classifier import ClassificationReport

file_name = "breast-cancer-wisconsin-data.csv"

df = pd.read_csv(file_name)
df=df.drop(columns=['Unnamed: 32']) #There's some weird dummy column showing up that's all nan values (probably has something to do the delimiter reading)
df=df.dropna() #clean nan values from the getgo
df.diagnosis = pd.Series(np.where(df.diagnosis.values == 'M', 1, 0), df.index) #Change diagnosis identifiers to 1==M and 0==B for ease of fit and score
X=df.values[:,2:]
y=df.values[:,1]

test_size = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
result_test_train = model.score(X_test, y_test)
print("Percent Accuracy: %s" % (result_test_train*100.0))
~~~
~~~
Percent Accuracy: 96.49122807017544
~~~

Great! Looks like our model predicts data it hasn't seen incredibly well! Let's check a few more diagnostics.
~~~python
cancer_cm = yel.classifier.ConfusionMatrix(model, classes=['B','M'], label_encoder={0: 'B', 1: 'M'})
cancer_cm.fit(X_train, y_train)
cancer_cm.score(X_test, y_test)

cancer_cm.show(outpath="confusion_matrix_resamp.png",bbox="tight")
~~~

![confusion matrix_](/assets/img/blog4/confusion_matrix_resamp.png)


The confusion matrix echoes well what the cross validation accuracy did. The vast majority of the predicted and true classes matched. 
~~~python
cv = model_selection.StratifiedKFold(5)
visualizer = RFECV(RandomForestClassifier(n_estimators=100), cv=cv, scoring='f1_weighted')

visualizer.fit(df.loc[:, 'radius_mean':], df['diagnosis'])        # Fit the data to the visualizer
visualizer.show(outpath="recursive_feature_elim_resamp.png",bbox="tight")           # Finalize and render the figure
~~~
![recursive feature elimination](/assets/img/blog4/recursive_feature_elim_resamp.png)

If we look at a recursive feature elimination plot, it appears that our accuracy score stops appreciably changing at around 10 features. We could use this to make the fitting process more efficient, but it's not really necessary for the purposes of our demonstrations today. The fitting is already fast and isn't a bottleneck.
~~~python
# Instantiate the classification model and visualizer
model = RandomForestClassifier(n_estimators=100)
visualizer = ClassificationReport(model, classes=['B','M'], label_encoder={0: 'B', 1: 'M'}, support=True)

visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show(outpath="classification_report_resamp.png",bbox="tight")
~~~

![classification report](/assets/img/blog4/classification_report_resamp.png)


Finally, looking at a classification report we see that, in addition to accuracy being very high, the precision (the ability of the classifier not to label as positive a sample that is negative) and recall (the ability of the classifier to find all the positive samples) are also very high. Nothing looks suspect, the classifier works well. 

### Using resampling to place uncertainty of the performance of our classifier
As I had mentioned earlier, it's good practice to place uncertainty bounds on any model that you fit/train. With a classifier, that isn't super straight forward, but one good practice is to propagate an uncertainty on the performance metrics that you use to rate it. In our case, we'll be doing that with predictive accuracy of data that the trained algorithm has never seen before. The accuracy of our test predictions is given as:

$$
\begin{aligned}
\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)
\end{aligned}
$$

Where $$\hat{y}_i$$ is the predicted value of the $$i^{th}$$ sample and $$y_i$$ is the corresponding true value. Let's start by looking at the leave-one-out method.

**Leave-one-out**
As the name implies, leave-one-out splitting is when you train on all of the data points in your set, but one, and then test on that one left out sample. This is done for every point in the data set to get a distribution of test estimates. A visual of the splitting is below.  

![LOO split](/assets/img/blog4/LOO_split_diagram.png)


Since we are testing on only a single point at a time, the only possible scores for the accuracy are 1 or 0 (right or wrong). This can lead to large estimates of uncertainty. Below is the `Python` implementation. 
~~~python
loo = model_selection.LeaveOneOut()
model = RandomForestClassifier(n_estimators=100)
result_LOO = model_selection.cross_val_score(model, X, y, cv=loo)
print("Percent Accuracy: %s (+/- %s)" % (result_LOO.mean()*100.0, result_LOO.std()*100.0))
~~~
~~~
Percent Accuracy: 96.30931458699473 (+/- 18.853312241692635)
~~~
Along with the uncertainty being very large, this method is VERY computationally expensive and is best used for smaller data sets. If we want a less robust uncertainty method with a shorter runtime, we can use a different resampling method like K-Fold.

**K-Fold**
As opposed to the previous method, where we leave out every sample one-at-a-time, the K-Fold method splits our data into K different sets and we train our model K times using all combinations of folds as train/test sets. A vizualization of a 10-fold splitting is below.

![K-Fold split](/assets/img/blog4/KFold_split_diagram.png)


Since our test sets are larger, the accuracy score per fold is closer to the mean (the average of roughly $$\frac{n}
{K}$$ test samples), hence the uncertainty estimate is lower (probably less robust though). Below is the `Python` implementation.
~~~python
kfold = model_selection.KFold(n_splits=10)
model = RandomForestClassifier(n_estimators=100)
result_kfold = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Percent Accuracy: %s (+/- %s)" % (result_kfold.mean()*100.0, result_kfold.std()*100.0))
~~~
~~~
Percent Accuracy: 95.78947368421053 (+/- 2.956543780061881)
~~~
This method is much better for larger data sets since we only need to train the model K times. However, the uncertainty estimate you get is less robust. If you have a large data set and want an uncertainty estimates, K-Fold splitting with a low K is a great first test.

**Shuffle Split (repeated random permutations)**
If we want to further reduce uncertainty (and possible systematic error) by randomizing our test sets. We can use a shuffle split method. With this we prescribe a test/train split size, shuffle our data set and draw our training set without replacement (the leftover is our test set). This is repeated predetermined number of times to get an uncertainty estimate of our score. The visual is below.

![Shuffle & split](/assets/img/blog4/shuffsplit_split_diagram.png)


The diagram shows random permutation splitting for an 80/20 train/test split for 10 replicates. Below is the implementation in `Python`.
~~~python
shuffsplit = model_selection.ShuffleSplit(n_splits=10, test_size=test_size)
model = RandomForestClassifier(n_estimators=100)
result_shuffsplit = model_selection.cross_val_score(model, X, y, cv=shuffsplit)
print("Percent Accuracy: %s (+/- %s)" % (result_shuffsplit.mean()*100.0, result_shuffsplit.std()*100.0))
~~~
~~~
Percent Accuracy: 95.96491228070177 (+/- 2.294157338705617)
~~~
This method can be very useful when trying to minimize the uncertainty estimate and/or diagnose correlations in the data set. 

**Bootstrapping**
The final method I'm demonstrating is bootstrapping. This has become a gold-standard for estimating uncertainty of an estimator. It is generalizable, easy to set up, estimator/algorithm agnostic and extremely useful for getting variance measurements when it is otherwise difficult or impossible. The bootstrap functions by repeatedly sampling observations from the original data set. These generated subsets can be used to estimate the distribution of a statistic just given our limited, original sample. Unlike the random-permutation method, these subsets are created by sampling with replacement from our full data set. Randomizing you train/test split ratios can provide another level of robustness to your estimator uncertainty.

~~~python
#reconfigure data values for easier bootstrap indexing
vals=df.values
#configure bootstrap
n_iterations = 1000
n_size = int(len(df)*0.8)
result_bootstrap = []
predictions = []
test_vals = []
#run bootstrap
for i in range(n_iterations):
    if i%10==1:
        print(i)
    #prepare train and test sets
    train = resample(vals,n_samples=n_size)
    test = np.array([x for x in vals if x.tolist() not in train.tolist()])
    #fit model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train[:,2:], train[:,1])
    #evaluate model
    prediction = model.predict(test[:,2:])
    score = model.score(test[:,2:],test[:,1])
    result_bootstrap.append(score)
    predictions.append(prediction)
    test_vals.append(test[:,1])
result_bootstrap=np.array(result_bootstrap)    
print("Percent Accuracy: %s (+/- %s)" % (result_bootstrap.mean()*100.0, result_bootstrap.std()*100.0))
~~~
~~~
Percent Accuracy: 95.51536528772262 (+/- 1.2476688605638024)
~~~
This method is powerful, robust, reliable, but expensive. If your data set is very large, it can be difficult to implement. We can secondarily compare all of our resampling methods by plotting all of the distributions of our model scores.

![Overlay KDEs](/assets/img/blog4/resample_dist_overlays.png)


If we look more closely at our shuffle split and bootstrap distributions (which are probably the most reliable), they look very similar (unsuprisingly). 

![Overlay KDEs B&SP](/assets/img/blog4/resample_dist_overlays_just_Shuffsplit_Bootstrap.png)


## Conclusions
There are many potential resampling methods for estimating uncertainty in a model. These are some good, generalizable ones that can be easily implemented for machine learning algorithms. The bootstrap is likely the most robust and general and is my go-to quick method for uncertainty estimation. It might get its own post in the future because it is very useful! Hope you learned and enjoyed! Until next time, folks. 
