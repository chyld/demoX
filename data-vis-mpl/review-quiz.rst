.. figure:: ../images/galvanize-logo.png
   :scale: 100%
   :align: center
   :alt: ibm-logo
   :figclass: align-center


Review and Quiz
=================

The learning objectives from this unit were:


+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| #     | Unit                            | Week               | Learning Objectives                                                               |
+=======+=================================+====================+===================================================================================+
| 1     | M2 Objectives                   | Data Visualization | Explain the principal steps in exploratory data analysis                          |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 2     | Data visualization              | Data Visualization | Explain the use case for Python tools (pandas, matplotlib, and Jupyter) in EDA    |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 3     | Data visualization              | Data Visualization | List several best practices concerning EDA and data visualization                 |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 4     | Missing values                  | Data Investigation | Describe strategies for dealing with missing data                                 |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 5     | CASE STUDY: Data visualization  | Data Visualization | Explain the role of communication in EDA                                          | 
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 6     | TUTORIAL: IBM Watson Dashboard  | Data Visualization | Create a simple dashboard in Watson Studio                                        |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 7     | Hypothesis Testing              | Data Investigation | Employ common distributions to answer questions about event probabilities         |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 8     | Hypothesis Testing              | Data Investigation | Apply null hypothesis testing as an investigative tool using Python               |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+
| 9     | CASE STUDY: multiple testing    | Data Investigation | Explain several methods for dealing with multiple testing                         |
+-------+---------------------------------+--------------------+-----------------------------------------------------------------------------------+


Review
--------------


M2 objectives
^^^^^^^^^^^^^^^^^^

   * The main goals of EDA are:

      1. Provide summary level insight into a data set
      2. Uncover underlying patterns and structure in the data
      3. Identify outliers, missing data and class balance issues
      4. Carry out quality control checks

   * The principal steps in the process of EDA are:

      1. **Summarize the data**     - Generally done using dataframes in R or Python
      2. **Tell the Story**         - Summarize the details of what connects the dataset to the business opportunity
      3. **Deal with missing data** - Identify the strategy for dealing with missing data
      4. **Investigate**            - Using data visualization and hypothesis testing delve into the relationship between the dataset and the business opportunity
      5. **Communicate**            - Communicate the findings from the above steps

Data visualization
^^^^^^^^^^^^^^^^^^^^^^^

   * Jupyter notebooks in combination with pandas and simple plots are the basis for modern EDA when using Python as a principal language
   * Advantages of Jupyter notebooks:

      * They are portable: then can be used locally on private servers, public cloud, and as part of IBM Watson Studio
      * They work with `dozens of languages <https://github.com/jupyter/jupyter/wiki/Jupyter-kernels>`_
      * They mix markdown with executable code in a way that works naturally with storytelling and investigation
   * matplotlib itself and its numerous derivative works like seaborn are the core of the Python data visualization landscape
   * pandas and specifically the dataframe class works naturally with Jupyter, matplotlib and downstream modeling frameworks like sklearn
     
   * EDA and Data Visualization best practices

      1. The majority of code for any data science project should be contained within text files.  This is a software engineering best practice that
	 ensures re-usability, allows for unit testing and works naturally with version control.
	 >In Python the text files can be executable scripts, modules, a full Python package or some combination of these.
      2. Keep a record of plots and visualization code that you create.  It is difficult to remember all of the details of how visualizations were created.  Extracting
	 the visualization code to a specific place will ensure that similar plots for future projects will be quick to create.
      3. Use you plots as a quality assurance tool.  Given what you know about the data it can be useful to make an educated guess before you execute the cell or run the
	 script.  This habit is surprisingly useful for quality assurance of both data and code.


Missing values
^^^^^^^^^^^^^^^^^^^^^^^

   * Dealing with missing data sits at the intersection of EDA and data ingestion in the AI enterprise workflow
   * Ignoring missing data may have unintended consequences in terms of model performance that may not be easy to detect
   * Removing either complete rows or columns in a feature matrix that contain missing values is called **complete case analysis**
   * Complete case analysis, although commonly used, can lead to undesirable results---the extent to which depends on the category of missingness
   * The categories of missingness are:

      * **Missing completely at random or MCAR**
      * **Missing at random or MAR**
      * **Missing not at random or MNAR**
	
   * The best case scenario is that the data are MCAR. It should be noted that imputing values under the other two types of missingness can result in an increase in bias.
   * In statistics the process of replacing missing data with substituted values is known as **imputation**.
   * It is a common practice to perform multiple imputations.
   * The practice of imputing missing values introduces uncertainty into the results of a data science project.
   * One way to deal with that additional uncertainty is to try a range of different values for imputation and measure how the results vary between each set of
     imputations. This technique is known as **multiple imputation**.

	 
CASE STUDY: Data visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	 
   * It can be easy to get lost in the details of the findings when communicating the finding from EDA to business stakeholders.  Project planning and milestones are
     important so remember to talk about what you:

      1. have done
      2. are doing
      3. and plan to do

   * Remember that deliverables are generally a presentation or a report and they should use a portable format (e.g. PDF or HTML)
   * Deliverables should should be concise and clear.  Appendices are useful as supplemental materials to a deliverable and they help keep them free
     of unnecessary items.
   * Visual summaries are a key component of EDA deliverables
   * There is no single right way to communicate EDA, but a minimum bar is that the data summaries, key findings, investigative process, conclusions are made clear.
	 

TUTORIAL: IBM Watson Dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the analytics dashboard, you can

   * build sophisticated visualizations of your analytics results
   * communicate the insights that you’ve discovered in your data on the dashboard
   * share the dashboard with others

The visualizations can tell the story of an investigative process or they can be made to summarize and communicate data in a way that is difficult to do with
simple plots.
     
Hypothesis Testing
^^^^^^^^^^^^^^^^^^^^^^^

   * Statistical inference and hypothesis testing can be used together to carry out investigations into the data
   * When carrying out a hypothesis test, the central question, null hypothesis and alternative hypothesis should be stated **before** the data are collected
   * Simulation based hypothesis testing like permutation tests provide a flexible alternative to more classical approaches
   * The bootstrap can be used to quantify the uncertainty around a parameter estimate and the two combined can be used as an investigative tool
   * Bayesian methods bring to the table a number of way to think differently about hypothesis testing.  They generally require more time to implement, but
     the quantification of uncertainty can be useful when making important business decisions.
   * The t-test is a simple way to care out 1 or 2 sample hypothesis tests.
   * There are a number of variants on the t-test, but the unequal variances t-test is commonly used.
   * The t-test and ANOVA (more than two groups) test whether group means have differences between each other

	 
CASE STUDY: Multiple comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   * *p*-values themselves are not a source of ground truth, but they are nonetheless quite useful if used appropriately.
   * There are a number of ways hack your way to significant results using *p*-values
   * Running more than one hypothesis test, on the same data, results in the  multiple comparisons problem.
   * Multiple comparisons are an issue because there is an expected false positive rate for running one test, and if we run multiple tests say using different combinations of features this expected rate should be higher.
   * The Bonferroni correction is commonly used to mitigate the multiple comparisons problem, but it is generally too conservative for large data sets.   
   * A number of other methods are available including the Benjamini/Hochberg correction that is based on the **false discovery rate**.
   * Permutation experiments are offer an additional method to correct for multiple comparisons that require fewer assumptions.

Quiz
--------------

M2 objectives
^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

   Which of the following is *NOT* normally a part of the EDA process

   .. container:: toggle

      .. container:: header
	
         * **(A)**: Visual summaries of the data
         * **(B)**: Connecting the data to the business opportunity
	 * **(C)**: Investigation through hypotheisis testing
         * **(D)**: Communication to stakeholders
         * **(E)**: Predictive linear or logistic regression
	    
      **ANSWER**:

         **(C)** This is an example of *p*-value hacking.  All of the other options are valid ways
	 to deal with the multiple comparisons problem.

.. admonition:: QUESTION 2

   True/False.  The EDA process is decoupled from modeling and cannot be used to help esitmate the time it will take to complete a modeling procedure

      .. container:: toggle

         .. container:: header

            **True/False**
			
         **ANSWER**:

	    **(False)** EDA helps provide insight into data through exploration.  One aspect of this is to identify missing values, the presence of
	    outliers and issues with the data like class inbalance.  These data issues along with summary level data can guide the process of selection
	    appropriate models.  Knowing which models will be tried and compared is a valuable piece of information to help estimate the time it will take
	    to get to the last stages of the workflow.


Data visualization
^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

   True/False.  The software engineering best practice of saving a maximum amount of code in text files for management under version control has become
   the norm in data science

      .. container:: toggle

         .. container:: header

            **True/False**
			
         **ANSWER**:

	    **(True)** This is True and particularly important when it comes to collaboration and reproducible research.


TUTORIAL: IBM Watson Dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

   Which of the following is the least valid statement when it comes to dashboards?

   .. container:: toggle

      .. container:: header

			
          * **(A)**: Dashboards are an easy way to share summaries and findings
          * **(B)**: Dashboards have interactive functionality that helps create a rich experience for the user
	  * **(C)**: Dashboards are generally used after serveral iterations of the AI workflow
          * **(D)**: Dashboards are quick way to create portable simple plots
          * **(E)**: Dahsboards can be used to tell the story of investigative visualizations
	    
      **ANSWER**:

	 **(D)** Dashboards are a powerful tool to communicate insights at many levels, but in general they are not
	 quick to produce simple plots when compared to other commonly used tools.

Missing values
^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

   The three types of missingss discussed during this module were:

   .. container:: toggle

      .. container:: header
	
         * **(A)**: MRAR, MAR, MCAR
         * **(B)**: MNAR, MRAR, MCAR
	 * **(C)**: MNAR, MAR, MARC
         * **(D)**: MAR, MRAR, MCAR
         * **(E)**: MCAR, MNAR, MAR
	    
      **ANSWER**:

         **(E)** 

         * Missing completely at random or MCAR
         * Missing at random or MAR
         * Missing not at random or MNAR

CASE STUDY: Data visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

   Which statement is the least true about using Jupyter notebooks in the context of EDA

   .. container:: toggle

      .. container:: header
	
         * **(A)**: They natually lend themselves to version control systems
         * **(B)**: They can be ported from one environment to another easily
	 * **(C)**: They are helpful because a mixture of code and markdown enables storytelling
         * **(D)**: They are integrated with the plotting library matplotlib
         * **(E)**: They are integrated with the data manipulation library pandas
	    
      **ANSWER**:

         **(A)** One key to working well with version control is that the files are plain text.  Jupyter notebooks use a JSON format
	 and they are not easy to read or compare to one another.  They can be saved in version control systems,
	 but comparing two notebooks is not as intuitive as comparing two .py files.

	 
Hypothesis Testing
^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

   A data scientist at Company Z sorted the survey responses by whether the respondents
   used Product 1 or Product 2 and then compiled their ages:

   .. code-block:: python

      p1_ages = [25., 32., 20., 18., 28., 32., 31., 19., 34., 34., 23., 29., 17.,
                 23., 25., 31., 32., 29., 29., 24., 22., 28., 26., 24., 23.]

      p2_ages = [20., 25., 27., 19., 22., 26., 24., 27., 24., 20., 25., 28., 18.,
                 19., 23., 28., 19., 19., 19., 25., 29., 26., 23., 23., 22.]
    
   Of the hypothesis test discussed in these contents what one is the most appropriate for testing the following hypothesis?

      There is no age difference, on average, between the users of product 1 and the users of product 2	      
	  
      .. container:: toggle

         .. container:: header
		
	    * **(A)**: A 1-sample t-test
            * **(B)**: A 2-sample t-test assuming equal variance
	    * **(C)**: Z-Test with continuity correction
            * **(D)**: A 2-sample unequal variances t-test
            * **(E)**: Binomial Test
	    
         **ANSWER**:
            *(D)* A reasonable test to use for this problem is a 2-sample t-test assuming assuming unequal variances.  We have no reason to
	    believe that the variance should be the same for the two populations.
	      
            .. code-block:: python

               test_statistic, pvalue = stats.ttest_ind(p1_ages, p2_ages, equal_var = False)
	       print("p-value: {}".format(round(pvalue,5)))

	    .. code-block:: none

	       p-value: 0.0134


.. admonition:: QUESTION 2

   Suppose that on average 2.5% of visitors to your website sign up for your newsletter.
   In a recent week, 2701 visitors out of a total of 108879 signed up.

   Using a binomial distribution.  What is the probability that number of visitors who signed up is 2701 *or fewer*?

   .. container:: toggle

      .. container:: header

			
          * **(A)**: 0.125
          * **(B)**: 0.346
	  * **(C)**: 0.414
          * **(D)**: 0.007
          * **(E)**: 0.015
	    
      **ANSWER**:

         **(B)** Having k people sign up for your newsletter out of a total of n visitors, can be modelled with a 
         to follow a Binomial distribution, where the probability of signing up is estimated
         to be 2.5%. A Cumulative Distribution Function (CDF) captures :math:`P(X \leq x)`,
         which we can use to answer the question:

         .. code-block:: python

	    from scipy import stats	    
            k = 2701
            n = 108879
            p = 0.025
            signup_prob = stats.binom.cdf(k = k, n = n, p = p)
            print("Prob of %s signups or fewer: %s "%(k, signup_prob))

            Prob of 2701 signups or fewer: 0.34647434249300585


.. admonition:: QUESTION 3

   True/False.  If there customer churn were quantified using a Poisson distribution, then a bootstrap could be used to quantify the uncertainty
   associated with the estimate.

   .. container:: toggle

      .. container:: header

         **True/False**
	    
      **ANSWER**:

	 **(True)** The bootstrap can be used to provide an empirical confidence interval around any statistic.  This includes the estimate of the rate
	 parameter for a Poisson distribution.
	      
CASE STUDY: Multiple comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: QUESTION 1

   Which of the following is *NOT* and example of a valid strategy to deal with the multiple comparisons problem?

   .. container:: toggle

      .. container:: header

			
          * **(A)**: Benjamini/Hochberg correction based on False discovery Rates
          * **(B)**: Create a null distribution using permutations to help provide context
	  * **(C)**: Perform all comparisons then only keep the single test that performs the best
          * **(D)**: If appropriate use an alternative modeling framework like generalized linear models
          * **(E)**: Bonferroni Correction
	    
      **ANSWER**:

	 **(C)** This is an example of *p*-value hacking.  All of the other options are valid ways
	 to deal with the multiple comparisons problem.
