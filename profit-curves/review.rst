
*************************************   
Review
*************************************

+------+------------------------------+--------------------------------------------------------------------------+
| Unit | Main Topics                  | Learning Objectives                                                      |
+======+==============================+==========================================================================+
| 1    | Predictive Linear Models     | Explain the use of linear models in supervised learning applications     |
+------+------------------------------+--------------------------------------------------------------------------+
| 2    | GLMs                         | Describe the difference between GLMs and GLMMs and name examples of each |
+------+------------------------------+--------------------------------------------------------------------------+
| 3    | statsmodels                  | Use the availabe model building tools in stats models                    |
+------+------------------------------+--------------------------------------------------------------------------+
| 4    | CASE STUDY: GLMs             | Build a GLM using statsmodels and properly interpret the output          |
+------+------------------------------+--------------------------------------------------------------------------+


Unit 1
===========================

Supervised learning is the focus of this course, but it should be seen in the context of the other learning fields.

* `supervised learning <https://en.wikipedia.org/wiki/Supervised_learning>`_
* `unsupervised learning <https://en.wikipedia.org/wiki/Unsupervised_learning>`_
* `semi-supervised learning <https://en.wikipedia.org/wiki/Semi-supervised_learning>`_
* `reinforcement learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_

As a reminder the type of `supervised learning <https://en.wikipedia.org/wiki/Supervised_learning>`_ depends on the data
type of the target. The *supervised learning* problem is referred to as either

**Regression** (when :math:`Y` is real-valued)
   e.g., if you are predicting price, demand, or number of subscriptions.

or

**Classification** (when :math:`Y` is categorical)
   e.g., if you are predicting fraud or churn

Regression
"""""""""""""

The two following metrics are the most commonly used.

.. math::

    \textrm{MAE} = \frac{1}{N} \sum^{n}_{i=1} \left| \hat{y}_{i} - y_{i} \right|


.. math::

    \textrm{RMSE} = \sqrt{\frac{1}{N} \sum^{n}_{i=1} \left( \hat{y}_{i} - y_{i} \right)^{2}}

The `root mean square error (RMSE) <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ can be calculated in
several ways and it is equivalent to the
`sample standard deviation <https://en.wikipedia.org/wiki/Sample_standard_deviation>`_ of the differences.
The mean absolute error (MAE) is another commonly used metric in regression problems.  A major advantage of RMSE and
MAE is that the values are interpreted in the same units as the original data.  MAE is the average of the absolute
difference between the predicted values and observed value. Unlike RMSE all of the individual scores are weighted
equally during the averaging.   **The squaring of the term in RMSE results in a higher penalty on larger differences
when compared to MAE**.

Classification
""""""""""""""""

Most classification metrics start from a `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.

+----------------------+---------------------------------------+--------------------------------------------------+
|                      | Predicted False :math:`(\hat Y = 0)`  | Predicted True :math:`(\hat Y = 1)`              |
+======================+=======================================+==================================================+
| True :math:`(Y = 0)` | True Negatives :math:`(TN)`           | False Positive :math:`(FP)`                      |
+----------------------+---------------------------------------+--------------------------------------------------+
| True :math:`(Y = 1)` | False Negatives :math:`(FN)`          | True Positives :math:`(TP)`                      |
+----------------------+---------------------------------------+--------------------------------------------------+

.. math::

    \textrm{accuracy} = \frac{tp+tn}{tp + fp + tn + fn}

.. math::

    \textrm{precision} = \frac{tp}{tp + fp}

.. math::

    \textrm{recall} = \frac{tp}{tp + fn}

The F1_score is the `harmonic mean <https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers>`_ of
precision and recall.  There are different variants of the F1_score, notably the :math:`F_{\beta}` score.

.. math::

    \mbox{F1_score} = \frac{2}{ \frac{1}{\mbox{recall}} + \frac{1}{\mbox{precision}}}

There are also several ways to average the F1_score when working in multi-class applications (e.g. weighted, micro,
macro). The ``average`` parameter can significantly change the behavior and performance of your model especially when
the classes are imbalanced.

Linear models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of these models make use of the following function:

.. math::

    \hat{y}(\mathbf{x},\mathbf{w}) = w_{0} + w_{1} x_{1}, + \ldots + w_{p} x_{p},

The target :math:`\mathbf{y}` could be a column vector or a matrix in the multivariate case.  There are :math:`p` features
and :math:`p` coefficients.  The intercept is written here as :math:`w_{0}`.  If :math:`p>1` then we are under the
category of `multiple linear regression <https://en.wikipedia.org/wiki/Multiple_linear_regression>`_ a very common variant
in the data science application space.  The :math:`w_{i}`:'s are parameters or *weights*.

Many of the linear models that are commonly used in data science like linear regression, the t-test and ANOVA are
examples of the `general linear model <https://en.wikipedia.org/wiki/General_linear_model>`_.  If we relax the
assumption that residuals can only be normally distributed and we introduce the concept of a link function then
we extend into `generalized linear models <https://en.wikipedia.org/wiki/Generalized_linear_model>`_, of which logistic
regression is the best example.  One extension further from GLMs brings us into the family of models known as
`generalized linear mixed models (GLMM) <https://en.wikipedia.org/wiki/Generalized_linear_mixed_model>`_.  GLMMs contain
some of the most flexible and useful linear models available with the best example being
`multilevel models <https://en.wikipedia.org/wiki/Multilevel_model>`_.

With each extension comes the need for more sophisticated model inference methods.  For example, GLMMs generally require
Bayesian inference methods like `MCMC <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ sampling, while simple
linear regression can be carried out with
`ordinary least squares <https://en.wikipedia.org/wiki/Ordinary_least_squares>`_ approaches.

Gradient decent can also be used for inference for several methods including support vector machines and logistic
regression.  It is a powerful and flexible way to carry out inference on linear models and the results can compare
favorably to even more sophisticated models.

Linear models also have a number of extensions including kernels and splines that enable non-linear functions.  The
level of model interpretation that is available with linear models and ease of implementation make them a safe choice
for a baseline model.  A baseline model is the one that you default to if a more sophisticated model cannot be shown to
have superior performance.

Quiz
--------------

Evaluation metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

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


Linear models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

    Which of the following is not an example of a *generalized linear model* (GLM)?

    .. container:: toggle

        .. container:: header

            * **(A)**: ANOVA
            * **(B)**: t-test
            * **(C)**: F-test
            * **(D)**: KNN regression
            * **(E)**: Logistic regression

        **ANSWER**:

            **(B)** The `K nearest neighbors <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_
            algorithm can be effectively used for regression, but it is not an example of a
            `Generalized linear model <https://en.wikipedia.org/wiki/Generalized_linear_model>`_.  The generalized
            linear models are special cases of the
            `general linear models <https://en.wikipedia.org/wiki/General_linear_model>`_.

.. admonition:: QUESTION 2

    True/False.  Models in the generalized linear mixture model (GLMM) family like multilevel models are generally optimized
    using sophisticated techniques like MCMC sampling.

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(True)**  With each extension from general linear models to generalized linear models to GLMMs comes the
            need for more sophisticated model inference methods.  GLMMs generally require Bayesian inference methods
            like `MCMC <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ sampling, while simple linear
            regression can be carried out with
            `ordinary least squares <https://en.wikipedia.org/wiki/Ordinary_least_squares>`_ approaches.

TUTORIAL: Watson NLU
^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

    When you use Watson Services like Watson NLU via the Python SDK, what are the three items that need to be saved?
    These items are generally saved on a local machine and included in scripts and notebooks as imported variables.

    .. container:: toggle

        .. container:: header

            * **(A)**: service version, service API key, service JSON map
            * **(B)**: service URL, service JSON map, service API key
            * **(C)**: service API key, service version, service URL
            * **(D)**: service version, service IAMAuthenticator, service URL
            * **(E)**: service API key, service URL, service IAMAuthenticator

        **ANSWER**:

            **(C)** The service uses JSON to pass messages, but there is no map to keep track of.  The IAMAuthenticator
            is the name of a class used from the SDK to access the services.  The URL, the API key and the version are
            the three critical pieces of information to keep track of.

.. admonition:: QUESTION 2

    Which of the following does **not** describe a feature of the Watson NLU service?

    .. container:: toggle

        .. container:: header

            * **(A)**: Perform document classification tasks using a custom model built from text
            * **(B)**: Identify high-level concepts that aren't necessarily directly referenced in the text
            * **(C)**: Find people, places, events, and other types of entities mentioned in your content
            * **(D)**: Recognize when two entities are related, and identify the type of relation
            * **(E)**: Analyze the sentiment toward specific target phrases and the sentiment of the document as a whole

        **ANSWER**:

            **(A)** The natural language understanding service can build custom models for entities and relations,
            but classification is not one of the features.  If needed there is a natural language classification
            Watson service. All of the other answers describe readily accessible features.

            * `Watson Natural Language Understanding <https://www.ibm.com/cloud/watson-natural-language-understanding>`_



CASE STUDY: NLP
^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

    After investigating the relationship between churn and accuracy What can we say about how the website version might
    be affecting the rate of churn?

    .. container:: toggle

        .. container:: header

            * **(A)**: The website version tends to increase churn
            * **(B)**: The website version tends to decrease churn
            * **(C)**: The website version did not appear to affect churn
            * **(D)**: The website version played a strong role in explaining churn
            * **(E)**: The website version directly affected the accuracy

        **ANSWER**:

            **(D)** See the solution for more details, but the website version and the accuracy together can be used
            to expalin 100% of the variance in the model.  This makes sense if you look carefully at the function
            used to create the data.  The role of the website version both positively and negatively affected churn,
            which can be seen in the plot.

.. admonition:: QUESTION 2

    Which of the following is not an example of a relevant question when tuning a NLP classification pipeline?

    .. container:: toggle

        .. container:: header

            * **(A)**: Should I use bag-of-words or a vector embedding representation?
            * **(B)**: Which stop words do I include?
            * **(C)**: Which n-grams do I include?
            * **(D)**: Should I use a TF or a tf-idf transformation?
            * **(E)**: Should I use RMSE or MAE as an evaluation metric?

        **ANSWER**:

            **(E)** RMSE and MAE are evaluation metrics for *regression* not *classification*.  All of the other
            questions are relevant.

Tree-based methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

    True/False.  Bagging and boosting ensemble methods both use only decision trees as base classifiers.  The difference
    is in the bias and variance of the individual trees.

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(False)** Bagging and boosting ensemble methods are not limited to decision trees as a choice for the
            base classifier. We showed in the bagging example the use of KNNs and SVMs as base classifiers.

.. admonition:: QUESTION 2

    True/False.  A decision tree classifier is useful as a model for the AAVAIL subscriber churn data.

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(True)** The shallow decision tree used as an example did a good job detecting the root cause of the
            differences between Singapore and USA based users.  Specifically, it was able to clearly identify that the
            problem was related to subscription type.  Additionally, the model was useful because it has comparable
            performance metrics to more sophisticated models and the tree can be visualized and accordingly used as
            a communication tool.

Neural networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

    Which of the following was not discussed as a tunable parameter of neural networks?

    .. container:: toggle

        .. container:: header

            * **(A)**: Hardware availability:
            * **(B)**: Activation functions: sigmoid, tanh, softmax, ReLU, leaky ReLU
            * **(C)**: Regularization techniques: weight decay, early stopping, dropout
            * **(D)**: Training method: Loss function, learning rate, batch size, number of epochs
            * **(E)**: Structure: the number of hidden layers, the number of nodes in each layer

        **ANSWER**:

            **(A)** The hardware availability is not a parameter of the neural network itself.  It is configurable
            in TensorFlow and in most deep-learning environments, but it is related to the compute environment rather
            than the model itself.  You can query the available devices with

            .. code-block:: python

                from tensorflow.python.client import device_lib
                print(device_lib.list_local_devices())

.. admonition:: QUESTION 2

    Transfer learning is a recent advancement to come out of the field of reinforcement learning.

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(False)** `Transfer learning <https://en.wikipedia.org/wiki/Transfer_learning>`_ is a strategy where you
            leverage vetted neural network architectures and pre-trained weights to get good performance on limited
            data.  `Reinforcement learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_ is an area of
            machine learning like supervised and unsupervised learning.  It generally uses neural network feature
            extraction and prediction abilities to learn from an environment.


TUTORIAL: Watson visual recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

    When training a custom classifier in Watson Visual Recognition the negative images should be:

    .. container:: toggle

        .. container:: header

            * **(A)**: As visually similar to the positive images as possible
            * **(B)**: Background images without the positive images
            * **(C)**: As random as possible to establish a background
            * **(D)**: Randomly generated from the positive images
            * **(E)**: Visually distinct from the positive images

        **ANSWER**:

            **(A)** When we train a custom classifier using the Watson Visual Recognition service a best practice is to
            include both positive and negative examples. Crucially, to increase the accuracy of the classifier, the
            negative examples should be as visually similar to the positive examples as possible.

            * `Watson Visual Recognition best practices <https://developer.ibm.com/articles/cc-build-with-watson-tips-best-practices-custom-classifiers-visual-recognition/>`_

.. admonition:: QUESTION 2

    True/False.  The Watson Visual Recognition service can only be accessed using an API key via Python or curl.

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(False)** Other SDKs are available including Swift, Unity, Ruby, Java, Node, Go and .net. See the
            `Watson Visual Recognition docs <https://cloud.ibm.com/apidocs/visual-recognition/visual-recognition-v4>`_
            for examples of each.



CASE STUDY: TensorFlow
^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

    Which of the following use cases is the least appropriate use case for a convolutional neural network?

    .. container:: toggle

        .. container:: header

            * **(A)**: Image classification
            * **(B)**: Image retrieval
            * **(C)**: Image composition
            * **(D)**: Object detection
            * **(E)**: Image segmentation

        **ANSWER**:

            **(C)** Image composition was never discussed in the context of neural networks or as part of these materials.
            Image composition is more generally called photography composition and it deals with arranging the elements
            in an image to suit an idea or accomplish a goal.  All of the other applications were mentioned as
            practical applications of CNNs.

            * `Image classification <https://en.wikipedia.org/wiki/Contextual_image_classification>`_
            * `Image segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`_
            * `Image retrieval <https://en.wikipedia.org/wiki/Image_retrieval>`_
            * `Object detection <https://en.wikipedia.org/wiki/Object_detection>`_

.. admonition:: QUESTION 2

    True/False.  A typical convolutional neural network is constructed using a combination of convolutional, pooling and
    dense layers.

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(True)** The statement is true.  The **convolutional layer** of a deep neural network uses a
            convolutional filter to slide over an input matrix to identify patterns. The **pooling layer** reduces
            the size of its input matrix (or matrices) that are generally created by a convolutional layer.  The **dense layers** are the fully
            connected hidden layers generally used near the output layer part of the networks.
