.. m1-m2

.. figure:: ./images/ibm-cloud.png
   :scale: 100%
   :align: center
   :alt: ibm-logo
   :figclass: align-center

Missing data
###########################

Unit objectives
-----------------

+---------+-----------------------------------------------------------------------------------------------------------+
| 1       | Develop a plan for dealing with data that contains missing values                                         |
+---------+-----------------------------------------------------------------------------------------------------------+
| 2       | Employ Python to efficiently impute missing values                                                        |
+---------+-----------------------------------------------------------------------------------------------------------+
| 3       | Explain how the results of multiple imputations vary depending on the imputation scheme                   |
+---------+-----------------------------------------------------------------------------------------------------------+


|

.. raw:: html

    <iframe src="https://player.vimeo.com/video/349969184" width="600" height="400"  frameborder="0" allowfullscreen></iframe>

|

Exploratory data analysis is mostly about gaining insight through visualization and hypothesis testing.  Recall from the
beginning of this unit that the ETL process is useful for holding the data to a minimum standard with respect to quality
assurance.  This unit deals with the imputation of missing values and it is where EDA and ETL meet.  Missing value
imputation could exist as part of the ETL process, but it is not often clear which strategy is the best until we can
make comparisons.  The comparisons are best made by evaluating model performance with a using a hold-out data set. One
missing value strategy may be better for some models, but for others another strategy may show better predictive performance.

Missing data is a common problem in most real-world scientific datasets. While the best way for dealing with missing data
will always be preventing the occurrence in the first placei, it will still remain a problem.  Sometimes data is collected
from sensors that fail to record or data collection is distributed across individuals and the merged data does not
harmonize well. There are a variety of ways for dealing with missing data, from more simplistic to very sophisticated, but
a standard metric by which we measure utility will still be model performance.

Strategies for missing data
--------------------------------

Flag missing values
^^^^^^^^^^^^^^^^^^^^^^^^^^^

One strategy for accounting for missing values is to simply ignore them. This is usually not a good idea because you
have little insight into how the missing data influenced the results.

Many machine learning models require input data to be complete, so if your project plan includes such a model. At minimum,
you may need to convert missing elements to a flag that can be ingested by the model. For example, when dealing with
missing values in a column of categorical data, an option is to treat “missing” as one of the categories for the model
to train on. In such a scenario, it is important to follow the standard practice of “dummifying” the categorical variable,
such as with Pandas’ `get_dummies <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html>`_ method.

Converting a *numerical* missing value to a flag to denote that it is missing is generally not good practice,
because the choice of the value of the flag will have implications on how your model treats such data. It is usually best
to use an imputation technique, such as those discussed below. The fact that a value was missing may be useful to track.
You can do so even when you have imputed a replacement value, by adding a new column to the dataset that flags whether values
from the newly complete column were originally missing.

Complete case analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

The opposite strategy from tracking all your missing values is to delete either rows (observations) or columns (features),
from your training data. In the case of deleting rows, this is called complete case analysis, and is quite common.

Complete case analysis can lead to undesirable results, but the degree to which it does depends on the *category of missingness*.
The categories of missingness are detailed in the next section.

If you plan to use a predictive model on the data that you have, you will still need a plan to account for missing values.
Complete case models in training can yield additional problems in the future.  Sometimes complete-case analysis is used
as a first pass through the workflow.

Deleting rows by filtering out rows of Pandas DataFrames that have missing data is straightforward:

.. code:: python

    import pandas as pd
    from numpy import nan

    df = pd.DataFrame({'name':['apple','banana','orange'],
                       'price':[1.95, 3.00, nan], 'inventory':[nan, 12, 23]})

    print(df)

         name  price  inventory
    0   apple   1.95        NaN
    1  banana   3.00       12.0
    2  orange    NaN       23.0

    print(df.dropna())

         name  price  inventory
    1  banana    3.0       12.0

The default behavior of the `.dropna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html>`_
method is to return a new DataFrame containing only the complete rows.

This method naturally has additional functionality, such as dropping columns rather than rows:

.. code:: python

    print(df.dropna(axis = 'columns'))

         name
    0   apple
    1  banana
    2  orange

Note that dealing with missing values from a column-wise perspective really falls under the
heading of Feature Engineering (since there are a number of reasons why one might exclude a
column/feature from a model, beyond just whether that feature has missing values). This topic
is covered in more detail in the next module.

Categories of missingness
---------------------------

The category of missingness can have important implications for statistical bias and power.  The three categories of missingness are:

Missing completely at random or MCAR:
   When data are MCAR, missing cases are, on average, identical to non-missing cases, with respect the feature matrix.
   Complete case analysis will reduce the power of the analysis, but will not affect bias.

Missing at random or MAR:
   When data are MAR the missing data often have some dependence on on measured values, and models can be used to help
   impute what the likely data would be.  For example, in an MLB survey, there may be a gender bias when it comes to
   completing all of the questions.

Missing not at random or MNAR:
   In this case the missing data depend on unmeasured or unknown variables. There is no information available to account
   for the missingness.

The best case scenario is that the data are MCAR.  It should be noted that imputing values under the other two types of
missingness can result in an increase in bias.  This is a reminder of why it is so important to have train/test splits.
Two of the more sophisticated strategies are Bayesian imputation and multiple imputation.

In this unit we will illustrate multiple imputation to account for missing data in a simple analysis. In the case study
that follows this unit we will also exemplify the iterative process for deciding which strategy is best.


Simple imputation
---------------------

Once you have decided to fill in missing values (and grappled with the implications of doing so) the
simplest approach is to treat each column/feature separately and use some chosen value, such as the mean
of the available data as the imputation value.

For example, using the data defined above:

.. code:: python

    from sklearn.impute import SimpleImputer

    features = ['price', 'inventory']
    imp = SimpleImputer()

    # Use .values attribute bc sklearn works with arrays rather than DataFrames
    imp.fit(df[features].values)

    print(imp.transform(df[features].values))

    [[ 1.95  17.5  ]
     [ 3.    12.   ]
     [ 2.475 23.   ]]



As always, the best choice for exactly how and whether to impute missing data will depend on
the nature of the data at hand and the overall goals of the project. You may need to try a few
different options and compare how your model performs on some hold out data. Of course, at this
point in the data science workflow you haven't built a model yet, so you would just want to make a
note to reconsider your imputation methods once you get to the back-and-forth between the feature engineering
and modeling phases of the project. This sort of back-and-forth also underlies the big picture
approach to missing values taken with multiple imputation.

Multiple imputation
---------------------

The practice of imputing missing values introduces uncertainty into the results of a data science project.
One way to deal with that additional uncertainty is to try a range of different values for imputation and
measure how the results vary between each set of imputations. This technique is known as multiple imputation.

Generally speaking, the range of imputation values are generated using an iterative modeling setup, where each
feature with missing values is modeled as a function of the other features in a round-robin fashion. To
generate different imputation values you can repeat this process using different random initializations
and/or differing hyper-parameter specifications.

scikit-learn has an `IterativeImputer <https://scikit-learn.org/stable/modules/impute.html#multivariate-feature-imputation>`_
tool (listed as experimental as of 2019) for modeling missing values. It can be called repeatedly to
generate a number of different datasets with varying imputed values, as described `here <https://scikit-learn.org/stable/modules/impute.html#multiple-vs-single-imputation>`_.
Then, later in your data science workflow after settling on a particular modeling pipeline, you would use
these different datasets as inputs and evaluate how the outputs from your pipeline differ depending on the
missing value imputations used.

While it may be beneficial to use the complex machinery of IterativeImputer when working with a dataset
with a lot of inter-related missing values across several features, it is possible to build multiple
imputed datasets piece-by-piece using more standard parts of scikit-learn library. This may be a better
tactic when the number of features that are incomplete is relatively small.

An example using a made-up dataset can illustrate such a procedure for multiple imputation:

.. code:: python

    import numpy as np
    from sklearn.datasets import make_regression

    # Generate some fake data
    X, y = make_regression(n_samples=50, n_features = 5,
                           n_informative = 5, random_state = 0)


    # Set a portion (of size num_nulls) of the data to NaN
    np.random.seed(0)
    num_nulls = 10
    null_rows = np.random.choice(X.shape[0], size = num_nulls, replace = False)
    null_col = np.random.randint(X.shape[1])
    X[null_rows, null_col] = np.nan

    # Check where the NaNs are:
    print(np.isnan(X).sum(axis = 0))

    [10  0  0  0  0]


For simplicity, only one feature has missing values, in this case it's column 0. We can use the
other features in the X matrix to "predict" the missing values of column 0. Since this feature
is numeric, an obvious choice for modeling these values is linear regression. The assumption here
is that we are making use of correlations between the features of the training data, so it is
good idea to check that by looking at a correlation matrix, and/or using Pandas'
`scatter_matrix <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html>`_
utility.

.. code:: python

    from sklearn.linear_model import LinearRegression, Lasso

    y_impute = X[:, 0].copy()
    X_impute = X[:, 1:].copy()
    missing = np.isnan(y_impute)
    lr = LinearRegression()
    lr.fit(X_impute[~missing], y_impute[~missing])

    print(lr.predict(X_impute[missing]))
    [ 0.36934727 -0.22428957  0.33800874 -0.12450563  0.14428445 -0.06667202
      0.39542781 -0.01727914  0.9775308   0.08102224]

Here we have modeled the missing values in the data, which we could fill back into the original
data, but to better understand how doing so would affect our later modeling results we will generate
a few more imputation sets. Some number between 3 and 10 versions is usually sufficient to capture
the variability of interest.

A good way to vary these imputation sets is by adjusting the flexibility of the model. With linear regression,
this can be accomplished via regularization. Here we use L1 regularization, or LASSO, to generate a few
imputation sets to go along with the one above. This is mathematically equivalent to setting alpha to zero.

.. code:: python

    col0_impute_vals = []
    for al in (0.01, 0.1, 1):
        l1_lr = Lasso(alpha = al)
        l1_lr.fit(X_impute[~missing], y_impute[~missing])
        col0_impute_vals.append(l1_lr.predict(X_impute[missing]))

    print(col0_impute_vals)

    [array([ 0.37585184, -0.19367399,  0.32997347, -0.08576458,  0.15000377,
            -0.03391817,  0.39567839,  0.01013396,  0.90420705,  0.08503995]),
     array([0.31884568, 0.12789675, 0.27126193, 0.15705541, 0.20161252,
            0.16377634, 0.28521764, 0.18821386, 0.35227229, 0.13296356]),
     array([0.21034679, 0.21034679, 0.21034679, 0.21034679, 0.21034679,
            0.21034679, 0.21034679, 0.21034679, 0.21034679, 0.21034679])]

Increasing the hyper-parameter alpha increases the penalty on the coefficients in the linear
regression model and in this case with alpha = 1 they are forced to zero so that the model
just predicts the mean of the available values for column 0 (i.e. we could have just used
SimpleImputer to have gotten that version).

Bayesian imputation
^^^^^^^^^^^^^^^^^^^^^^

While missing values are usually handled prior to the modeling phase of a data science project, it is
worth noting an exception where missing values can be handled automatically as part of the modeling process.
This is is the case when a model is treated in a *fully Bayesian* way, that is priors are used to govern
parameters of the model.  Then `Expectation-Maximization <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_,
`Markov Chain Monte Carlo (MCMC) <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ or another
method of inference can be use to infer both the parameters, hyper-parameters and missing values.

See the following resources to learn more.

`PyMC3 <https://docs.pymc.io/>`_
    package for probabilistic programming in Python.

`TensorFlow Probability <https://www.tensorflow.org/probability>`_
    another package for Python that enables the Bayesian treatment of models

`PyMC3 Getting Started <https://docs.pymc.io/notebooks/getting_started.html>`_
    see Case study 2 to see how missing values are automatically imputed during inference


.. admonition:: CFU

    In the continuing AAVAIL streaming case study example, one of the data features that can be useful
    in answering questions about customer churn is the total number of streams that a
    customer has watched. Imagine that you are working with a dataset where 10% of customers
    are missing this feature. A good place to start would be to go back and see if it's
    possible to gather this information from the user logs, but assuming that this initiative is
    unsuccessful, you will have to decide what to do about this missing data. Which course of
    action is **LEAST** likely to be helpful in modeling churn?

    .. container:: toggle

        .. container:: header

            * **(A)**: Replace the missing stream count with the mean stream count among users where this information is available.
            * **(B)**: Replace the missing stream count with a -1 to flag that it is unknown for a given user.
            * **(C)**: Use the other features in the dataset in a model to predict the missing stream counts.


        **ANSWER**:
           **(B)** Replacing a missing count with a -1 will not (naturally) be interpreted as a flag by a machine
           learning model. A model would treat a count of -1 as "less than zero," without any further interpretation.
           Mathematically speaking this would group the users with missing counts as even more extreme than
           users who signed up but never streamed. Obviously this is not ideal, and choices **(A)** or **(C)** are
           more reasonable approaches.


.. admonition:: CFU

    What is the main reason for using multiple imputation?

    .. container:: toggle

        .. container:: header

            * **(A)**: Multiple imputation is necessary when more than one feature in the training data has missing values.
            * **(B)**: Multiple imputation is a way to increase the size of your training dataset.
            * **(C)**: Multiple imputation helps to better characterize the error introduced by replacing missing/unknown data with some chosen values.


        **ANSWER**:
           **(C)** Imputing missing values introduces a new source of error when those values are used in modeling.
           Multiple imputation uses several methods to create a number of versions of the data each with different values imputed. Examining
           how predictions differ between these methods applied to validation data, can be informative in
           trying to understand how much error is being introduced, which in turn can be useful for deciding how
           simple or complex an imputation scheme to use.
