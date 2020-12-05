.. profit curves and imbalanced classes

*******************
Imbalanced classes
*******************

Unit objectives
-----------------

By the end of this module you will be able to:

+-------------------+--------------------------------------------------------------------------------------------------+
| Class imbalance   |  Explain the issues that arise with class imbalance and implement common methods to address them | 
+-------------------+--------------------------------------------------------------------------------------------------+

Many machine learning algorithms used for classification have underlying assumptions about the presence of relatively
equal numbers of samples in each class.  In reality, there are many situations where the class of interest is
underrepresented in the overall population, which can lead to data with proportions that reflect the wider population.
For example you might train a logistic regression model to predict whether or not an patient has a specific disease, or
whether a company is likely to retain a subscribing customer. The models that have underlying assumptions about class
balance ultimatly perform better on data that have relatively equal proportions of each class.

A common example from business that is brought up when discussing imbalanced classes is fraud detection.  An example might
be a service that handles payments and they might be interested in flagging a transaction as potentially fraudulent.  The
true cases of fraud may only be a small fraction of a percentage, but the impact to both the company and to the
affected customers can be significant. 

When it comes to mitigation there the idea to keep in mind is that you wish to avoid being overwhelmed by the majority class.
Accuracy, that is the proportion of predictions that are correct, is an intuitive and commonly used metric for classification,
but when using it to gauge model performance the results can be very misleading with imbalanced classes.
For example, let's say we have a 2-class image classification data set that is made of 1% commercial trucks and the rest of
the images are passenger cars.    A model that *always* predicts "not fraudulent" will be 99% accurate, but it will never
identify a commercial truck.

.. warning:: Accuracy is not an appropriate metric for imbalanced classes.  Use F1 Score or another metric instead.

As we know F1 Score and other related metrics combine the elements of a confusion matrix in a slightly different
way than accuracy.  The elements: true positives, true negatives, false positives, and false negatives are used
in different ways depending on the field.  Here is a table that `summarizes the evaluation of binary classifiers <https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers>`_.  By convention the minority class is designated as the positive
class. As a reminder precision and recall can be defined as:

.. math::

    \textrm{precision} = \frac{tp}{tp + fp}

.. math::

    \textrm{recall} = \frac{tp}{tp + fn}

Having better metrics for measuring performance in imbalanced classification is the first step.  Here is a list of
other mitigation techniques.

* under and over-sampling
* sample re-weighting
* synthetic sampling techniques
* Using models that naturally account for imbalanced classes
* re-formulate your problem as an outlier or novelty detection problem

Sampling techniques
----------------------

The most commonly used approaches are sampling based or more accurately re-sampling based.  Between up-sampling and down-sampling
(also called over-sampling and under-sampling), down-sampling is a bit simpler conceptually. Fundamentally, if you have a
minority class or classes that that is noticeably underrepresented.  In a random way you can drop some of those from the
training data so that the proportions are more closely matched across classes.  There are additional methods, one of which
is inspired by `K Nearest Neighbors (KNN) <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_ that can improve on pure
random selection.

A major caveat to down-sampling is that we are not using all of our data.  Over-sampling techniques also come in several
flavors, from random or naive versions to classes of algorithms like the Synthetic Minority Oversampling Technique (SMOTE)
:cite:`2002:SMOTE` and the Adaptive Synthetic (ADASYN) :cite:`2008:ADASYN` sampling method.  Synthetic Minority Over-sampling
Technique for Nominal and Continuous (SMOTENC) is an extension of the original SMOTE method designed to handle a mixture
of categorical and continuous features.  SMOTE has a number of other variants including ones that make use of Support Vector
Machines and K-means clustering to improve on the synthetic samples.  You may also `combine over and under-sampling
techniques <https://imbalanced-learn.readthedocs.io/en/stable/combine.html>`_.

.. important::

    Always perform your test/train split before oversampling techniques. Over-sampling before splitting the data can could produce
    duplicate observations in both the train and test sets.

Python package: imbalance-learn
--------------------------------

* `Guide to over-sampling <https://imbalanced-learn.org/en/stable/over_sampling.html>`_
* `Guide to under-sampling <https://imbalanced-learn.org/en/stable/under_sampling.html>`_
* `SMOTE and variants <https://imbalanced-learn.org/en/stable/over_sampling.html#smote-variants>`_
* `Examples using imbalance-learn <https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/index.html>`_


Models that naturally handle imbalance
-----------------------------------------

Some models are more sensitive to imbalanced classes than others.  Neural networks for example are very sensitive to imbalanced
classes.  Support Vector Machines on the other hand and to some extent tree based methods are more resilient.  If present
the `class_weight` argument should be used when working with imbalanced classes.

For example,

.. code-block:: python

   clf = SVC(kernel='linear',
             class_weight='balanced',
             probability=True)


See this `this scikit-learn example using SVMs and unbalanced data <https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html>`_
for more details.


.. admonition:: CFU

   Which variant of SMOTE is most appropriate when you have a mixture of categorical nad continuous variables?

   .. container:: toggle

      .. container:: header

         * **(A)**: KMeansSMOTE
         * **(B)**: BorderlineSMOTE
         * **(C)**: SVMSMOTE
         * **(D)**: SMOTENC
         * **(E)**: SMOTE

      **ANSWER**:

         **(D)** The SMOTENC is an extension of the SMOTE algorithm for which categorical data are treated differently
         :cite:2002:SMOTE.  The NC stands for nominal and continuous.  See the `imblearn SMOTENC <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTENC.html>`_ page for more information.


Additional resources
^^^^^^^^^^^^^^^^^^^^^^^^

   * `useful review paper about resampling techniques <https://arxiv.org/abs/1608.06048>`_
