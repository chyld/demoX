.. course title

*************************************   
Assignment
*************************************


Thought no single method can solve all imbalanced classification
problems, there is generic strategy that can help you to navigate
through the maze and find high performing solution:

::

   - use stratification when splitting data into train and test

   - change the evaluation metric from accuracy to F_score (using fixed threshold 0.5), AUC_PRC (using thresholds across 0.0 to 1.0).

   - change algorithm that can handle imbalanced classes

   - rebalance trainning data by various resampling methods

   - cost sensity learning: use routine algorithms and the original trainning data but consider the cost benefit of different errors

To familiar yourself this strategy, here you are guided to walk through
a highly imbalanced classification problem: detecting credit card fraud
which accounts for less than 1% of the total transactions.

The `Credit Card Fraud Detection
Data <https://www.kaggle.com/mlg-ulb/creditcardfraud>`__ is provided on
Kaggle. According to Kaggle,

   The datasets contains transactions made by credit cards in September
   2013 by european cardholders.

::

   This dataset presents transactions that occurred in two days, where
   we have 492 frauds out of 284,807 transactions. The dataset is highly
   unbalanced, the positive class (frauds) account for 0.172% of all
   transactions.

   It contains only numerical input variables which are the result of a
   PCA transformation. Unfortunately, due to confidentiality issues, we
   cannot provide the original features and more background information
   about the data. Features V1, V2, … V28 are the principal components
   obtained with PCA, the only features which have not been transformed
   with PCA are ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds
   elapsed between each transaction and the first transaction in the
   dataset. The feature ‘Amount’ is the transaction Amount, this feature
   can be used for example-dependant cost-senstive learning. Feature
   ‘Class’ is the response variable and it takes value 1 in case of
   fraud and 0 otherwise

1. Do Explorary Data Analysis (EDA) on the data. Find out what is the
   fraction of Fraud among all samples. Is the distribution for features
   ‘Amount’ the same for Fraud and non-Fraud? How about for feature
   ‘Time’? How are you going to scale these two features?


2. What is the baseline model you should choose for this data?


3. What are the metrics you should use to evaluate the model
   performance? compare AUC-ROC curve and AUC-PRC curve on your model.


4. What machine learning algorithms should you choose to deal with
   imbalanced data problem?


5. To rebalance the trainning data, explore the following various
   resampling options:


-  Under-sampling the majority class (i.e.
   `RandomUnderSampler <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html>`__,
   `TomekLinks <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.TomekLinks.html>`__)

-  Over-sampling the minority class (i.e.
   `RandomOverSampler <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html>`__)

-  Combine over and under-sampling (i.e.
   `SMOTEENN <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTEENN.html>`__)

-  Synthesis new samples for minority class (i.e.
   `SMOTE <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html>`__,
   `SMOTEOMEK <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTETomek.html>`__,
   `SMOTENC <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTENC.html>`__)

Use dimensionality reduction techniques such as
`T-SNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`__,
`PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`__
and `Truncated
SVD <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`__
to visualize the separability of the different classes both before and
after trainning data resampling.

6. (Optional) Create a class called Classifiers to organize your code and data for
   easy model comparison.

The **attributes** of the class could include

   - classifiers: a list of estimators

   - classifier_params: a list of dictionaries corresponding to classifiers, each dictionary contains the hyperparameters of the classifier to be tuned.

   - etc.

The **methods** of the class could include

   - train()

   - create_pipelines()

   - classification_reports()

   - plot_roc_curves()

   - plot_aucprc_curves()

   - etc. 


