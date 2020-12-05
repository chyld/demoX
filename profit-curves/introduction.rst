.. course title

*************************************   
Introduction
*************************************

Learning Objectives
======================

By the end of this unit, you will be able to 

+-------------------+---------------------------------------------------------------------------------+
| Class Imbalance   |  Explain issues with imbalanced data and implement strategies to deal with them | 
+-------------------+---------------------------------------------------------------------------------+


Section 1: What is imbalanced classes
=======================================


If you have been working on classification problems for a while, chances are you have encounted data with imbalcanced classes: the size of some class(es) is largely outnumbered by the others. Data is often considers imbalanced when a minority class consist less than 33%. In reality, datasets can get far more imbalanced than this. 

Examples of Imbalanced Classes
""""""""""""""""""""""""""""""
- About 2% of credit card accounts are defrauded per year. In general most fraud detection data are highly imbalanced.

- Medical screening for a condition is usually performed on a large population of people without the condition, to detect a small minority with it (e.g., HIV prevalence in the USA is ~0.4%).

- Disk drive failures are approximately ~1% per year.

- The conversion rates of online ads has been estimated to lie between :math:`10^{-3}` to :math:`10^{-6}`.

- Factory production defect rates typically run about 0.1%.

Here we liken the imbalanced classes problems to find needle in a haystack, where machine learning classifiers are used to sort through huge population of negative (uninteresting) cases to find out the small number of positive (interesting) cases.

Section 2: Challenges of dealing with imbalanced dataset
==========================================================
The challenges to classify imbalanced data come from two sources: 

**Algorithm Problem** Conventional machine learning algorithms usually bias toward majority classes. Algorithms such as Logistic Regression, Naive Bayes and Decision Trees, there is a likelihood of the wrong classification of the minority class. It's because they are designed to maximize accuracy (or equivalently, minimize error rate), without taking the data distribution into account. With such optimization objective, instances belong to miority class are likely to be classified as majority class. In case of credite card defraud detection where about only 2% defraud occurs, a classifier can be trained to predict all instances to be negative to achieve 98% accuracy. However the model is not useful to us to detect the defraud, the interesting cases in the minority class.


**Cost Sensitive Problem** In real life, the cost of false negative is ususally much higher than the cost false positive, yet machine learning algorithm usually treat them the same. For instance, in case of creditcard defraud, false negatives lead to the loss of both the loan principle and the associated interest, which is usually much more costly than the false positive: interests earned from a loan. In cancer diagnosis, false negative prediction could lead to fatal result when patience miss the opportunity to receive treatment. That is enormous comparing to the cost of false positive predictions, the treatment side effects.

Section 3: Approaches to handle imbalanced data
================================================
Learning from imbalanced data has been `active studied over decades <https://arxiv.org/pdf/1505.01658.pdf>`_. A large number of techniques has been tried with varying results. However there is no single generic method can solve all the problems caused by imbalanced data: what methods to choose depend on both your data and your objectives. We provide a list of approaches losely ranked in order of efforts. You should try them out and let the classification result tell you which one(s) work best for your case. 

**Data-based approaches: balancing training set**


- Upsample the minority class

- Downsample the majority class

- Sythesize new samples for minority class

**Algorithm-based approaches**

- Adjust the class weight

- Adjust the decision threshold

- Choose algorithms / modify the existing algorithm to be sensitive to the minority class

**Cost Sensitive Learning**

- Use the original data and standard learning algorithms, only calculate the profit of each model's predictions by considering the cost-benefit of correcting misclassifications.

Section 4: How to Evaluate Classifier on Imbalanced Data
==================================================================


As you have seen throughout this specialization we tend to use
`sklearn's classification_report <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>`_
to summarize model performance.  If we had a model to predict whether a customer would be retained, or would put their
account on hold, we might observe something like the following table.

.. code-block:: python

    from sklearn.metrics import classification_report
    y_true = [0,1,0,2,1,0,0,2,1,1,0,1]
    y_pred = [0,0,0,2,1,0,2,1,2,1,0,1]
    target_names = ['retained customer', 'unretained customer', 'on hold customer']
    print(classification_report(y_true, y_pred, target_names=target_names))


.. code-block:: none

                  precision    recall  f1-score   support

        retained       0.80      0.80      0.80         5
      unretained       0.75      0.60      0.67         5
         on hold       0.33      0.50      0.40         2

        accuracy                           0.67        12
       macro avg       0.63      0.63      0.62        12
    weighted avg       0.70      0.67      0.68        12


Looking at each class in terms of the `F1_score <https://en.wikipedia.org/wiki/F1_score>`_ is a convenient way to summarize
the classifier's results.  We are going to quickly review how we obtain the numbers in this table.

To get F1_score and its constituent measures of precision and recall we need to first start with a
`confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.

+----------------------+---------------------------------------+--------------------------------------------------+
|                      | Predicted False :math:`(\hat Y = 0)`  | Predicted True :math:`(\hat Y = 1)`              |
+======================+=======================================+==================================================+
| True :math:`(Y = 0)` | True Negatives :math:`(TN)`           | False Positive :math:`(FP)`                      |
+----------------------+---------------------------------------+--------------------------------------------------+
| True :math:`(Y = 1)` | False Negatives :math:`(FN)`          | True Positives :math:`(TP)`                      |
+----------------------+---------------------------------------+--------------------------------------------------+

.. code-block:: python

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

.. code-block:: none

    [[3 0 1]
     [1 1 1]
     [0 1 1]]

It can be a useful communication tool to embed an image of the confusion matrix in reports and dashboards.

.. plot:: ./scripts/classifier-metrics-example.py


Note that the images rescale the TP, FP, TN, FN counts to a 0-1 scale so that one confusion matrix may easily be
compared to another.

Classification metrics review
"""""""""""""""""""""""""""""""

- **Confusion Matrix**: a table breaks down the predictions of a classifier into the counts of correct and incorrect predictions for each class. The confusion matrix shows the ways your classifier is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors (false positive, false negative) that are being made.

From the confusion matrix we can derided other metrics, including accuracy, precison, recall, F1 etc. By default confusion matrix and the derided metrics use fixed probability threshold (0.5) to distinguish class label. Different thresholds will result in different metric-values. 

- **Accuracy** = :math:`\frac{TN+TP}{FP+FP+TN+TP}`: how often the model gets it right.

.. note::  
    When we have balanced classes, accuracy is OK to use; but when we deal with imbalanced classes, accuracy can be missleading unless you compare the performance of your model with a baseline model which uses majority voting. 

    For imbalanced classes, we should consider both precision and recall (F1 score) rather than just accuracy.
 

- **Precision** = :math:`\frac{TP}{TP+FP}`: proportion predicted true that are correct

.. note:: 
    When False Positive is costly and need to be low, we want a classifier with high precison.

- **Recall** =  :math:`\frac{TP}{TP+FN}`: proportion of true that are predicted correctly

.. note::
    When False Negative is costly and need to be low, we want a classifier with high recall.

- **F1_score** = the `harmonic mean <https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers>`_ of precision and recall.

.. math::

    \mbox{F1_score} = \frac{\mbox{precision} \times \mbox{recall}} { \mbox{precision} + \mbox{recall}}

The F1_score is actually a special case of the :math:`F_{\beta}` score, the weighted harmonic mean of precision and recall.

.. math::

    F_{\beta} = (1 + \beta^{2})
    \frac{\mbox{precision} \times \mbox{recall}} {(\beta^{2} \times \mbox{precision}) + \mbox{recall}}

The :math:`\beta` parameter determines the weight of recall in the combined score. :math:`\beta <1` lends more weight to precision, while :math:`\beta >1` favors recall (:math:`\beta \rightarrow 0` considers only precision, :math:`\beta \rightarrow \inf` only recall).
We would consider increasing :math:`\beta` in a situation where false negatives are more costly i.e. high recall is desirable.  For example, if your model was used
to screen X-Ray images of airport luggage for manual inspection, a false positive is not as potentially costly as a false negative, where we should use :math:`\beta >1`. 

- **AUC-ROC** the area under the Receiver Operating Characteristic Curve 

Receiver Operating Characteristic curve (**ROC**) is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) for different thresholds between 0.0 and 1.0. While ROC Curves summarize the trade-off between the true positive rate and false positive rate for a classifier using different probability thresholds, **AUC-ROC** measures the entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1) and is an aggregate measure of performance across all possible classification thresholds.

- **AUC-PRC** the area under the Precision-Recall Curve 

A Precision Recall Curve (**PRC**) a plot of the precision (y-axis) and the recall (x-axis) for different thresholds between 0.0 and 1.0. While PRC summarizes the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds, **AUC-PRC** measures the entire two-dimensional area underneath the entire PRC curve from (0,0) to (1,1) and is an aggregate measure of performance across all possible classification thresholds.

Because the majority class in a imbalanced data is negative and minority is positive, a classifier produces high value for true negatives. ROC Curve uses true negatives in the False Positive Rate, therefore  ROC curve tends to give an over-optimistic view on classifier's performance on imbalanced data. In contract, Precision-Recall curve carefully avoids using this rate and is more capable than ROC curves to evaluate classifier performance for imbalanced data.

Check out this blogpost:

* `How to Use ROC Curves and Precision-Recall Curves for Classification in Python <https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/>`_

Session 5: Models that naturally handle imbalance
==================================================================

Some models are more sensitive to imbalanced classes than others.  Neural networks for example are very sensitive to imbalanced
classes.  On the other hand Support Vector Machines and to some extent tree based methods such as Random Forest are more resilient.  If present the `class_weight` argument should be used when working with imbalanced classes.

For example,

.. code-block:: python

   clf = SVC(kernel='linear',
             class_weight='balanced',
             probability=True)


See this `scikit-learn example using SVMs and unbalanced data <https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html>`_
for more details.


.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   clf = RandomForestClassifier(n_estimators=10, class_weight='balanced')


Checkout these references:

*  `Important Techniques in RandomForest to Improve Machine Learning Model Performance with Imbalance Datasets <https://towardsdatascience.com/working-with-highly-imbalanced-datasets-in-machine-learning-projects-c70c5f2a7b16>`_

*  `Bagging and Random Forest for Imbalanced Classification <https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/>`_


*  `Using Random Forest to Learn Imbalanced Data <https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf>`_


Checks for Understanding
--------------------------

.. admonition:: QUESTION 1

    If we have a situation where false positive is not as potentially costly as a false negative say flagging comments
    for manual review based on suspected unlawful activity, which of the following is the best approach to consider?

    .. container:: toggle

        .. container:: header

            * **(A)**: Only look at the f-score of the negative class for evaluation
            * **(B)**: Use recall as the evaluation metric
            * **(C)**: Use precision as the evaluation metric
            * **(D)**: Set beta to 0.5 in the fscore
            * **(E)**: Set beta to 2.0 in the fscore

        **ANSWER**:

            **(E)**

            .. math::

                F_{\beta} = (1 + \beta^{2})
                \frac{\mbox{precision} \times \mbox{recall}} {(\beta^{2} \times \mbox{precision}) + \mbox{recall}}


            If we set, for example, :math:`\beta = 2` then the metric weighs recall higher than precision.
            Conversely, with :math:`\beta = 0.5` precision is given more importance than recall.

.. admonition:: QUESTION 2

    True/False.  All classifiers in scikit-learn do multi-class classification out-of-the-box.  The classifiers can
    differ in their approach though (e.g one-vs-all or one-vs-one).

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(True)** Some classification models are inherently multi-class like the naïve Bayes. Some classifiers
            default to a one-vs-one, where a different classifier is trained for all pairwise model comparisons. Other
            classifiers use a one-vs-all approach where there is a classifier for each model.

Summary 
==============

In this unit, you discovered what issues arise with imbalanced data and how to implement strategies to deal with them.


Specifically, you learned:

- what is classification with imbalanced classes and what are the typical examples  

- what make classification with imbalanced data challenging

- what are the strategies to deal imbalanced class

- what metrics are suitable for imbalanced classification

- what models are able to naturally handle imbalanced classification


Useful Readings
=================

`A Survey of Predictive Modelling under Imbalanced
Distributions <https://arxiv.org/pdf/1505.01658.pdf>`_


`Learning from Imbalanced Classes <https://www.svds.com/learning-imbalanced-classes/>`_


