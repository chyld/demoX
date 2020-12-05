.. r statistics

**********************
Statistics Essentials
**********************

Within data science (and perhaps at its core) is the field of
`Machine learning <https://en.wikipedia.org/wiki/Machine_learning>`_, which seeks to accomplish two main objectives:

  * **Supervised learning** - learn a mapping from inputs :math:`x` to outputs :math:`y`
  * **Unsupervised learning** - given only :math:`x`, learn interesting patterns in :math:`x`
     
These tasks are a form of artificial intelligence that endow a computer with the capability to represent a general
class of patterns.

Then through that representation they have the ability to **predict outputs** and **identify patterns**.  Note that
this is different than explicitly hard-coding some data relationship into a computer as though the specific
relationship was already known beforehand.

.. container:: toggle

    .. container:: header

        **Show More**

    In order to identify which specific patterns (out of a general class of patterns) are present in the data, machine
    learning makes extensive use of `linear algebra <https://en.wikipedia.org/wiki/Linear_algebra>`_---the branch of
    mathematics that works directly with matrices---in conjunction numerical optimization procedures.

    This process of identifying a specific instance (out of a general class of patterns) that looks as similar to the
    data as possible is called "model fitting".  Once a machine has such a model representation of the data, then it has
    *learned* the pattern in the data and can use it as a part of other programatic instructions designed to accomplish
    some objective.

|

Machine Learning versus Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`What is the difference between statistics and machine learning? <https://www.quora.com/What-is-the-difference-between-statistics-and-machine-learning>`_

**Statistics** and **Machine Learning** represent related, but distinct quantitative analysis traditions.
Both disciplines are rooted in the common enterprise of "data analysis".   In recent years the line between the two has
become increasingly blurred.  Nonetheless, some general statements related to the traditional domains and expertise
claimed by each discipline can be made:

**Statistics**

  * utilizes confidence intervals, hypothesis tests, and optimal estimators  
  * places paramount importance on characterizing uncertainty in estimation 
  * bases methodological development on distribution and asymptotic theory  

**Machine Learning**

  * utilizes nonparametric and non-linear models with a major focus on prediction
  * places paramount importance on "out of sample" generalizability/performance
  * bases methodological development on empirical and computational techniques 
    
Objectives
^^^^^^^^^^

The purpose of this short course is to:

   1.  equip you with actual quantitative tools that you can apply to more effectively tackle problems you're
       interested in using data, and
   2.  to provide you with a appropriate foundation on which you can effectively build a synergistic data science skill
       set that leverages:
    
  * `Descriptive statistics <https://en.wikipedia.org/wiki/Descriptive_statistics>`_ - mean, median, skewness... 
  * `Inferential statistics <https://en.wikipedia.org/wiki/Statistical_inference>`_  - hypothesis testing, interval estimation...
  * `Predictive analytics <https://en.wikipedia.org/wiki/Predictive_analytics>`_ - supervised learning: regression, classification...
  * `Prescriptive analytics <https://en.wikipedia.org/wiki/Prescriptive_analytics>`_ - unsupervised learning, recommender systems...


Many of the descriptive statistics you likely already know.

+------------------------------+----------------------------------------------------------------------------------+
| Function                     | Description                                                                      |
+==============================+==================================================================================+
| ``mean()``                   | `mean <https://en.wikipedia.org/wiki/Mean>`_                                     |
+------------------------------+----------------------------------------------------------------------------------+
| ``median()``                 | `median <https://en.wikipedia.org/wiki/Median>`_                                 |
+------------------------------+----------------------------------------------------------------------------------+
| ``sd()``                     | `standard deivation <https://en.wikipedia.org/wiki/Standard_deviation>`_         |
+------------------------------+----------------------------------------------------------------------------------+
| ``var()``                    | `variance <https://en.wikipedia.org/wiki/Variance>`_                             |
+------------------------------+----------------------------------------------------------------------------------+
| ``cor()``                    | `correlation <https://en.wikipedia.org/wiki/Correlation_and_dependence>`_        |
+------------------------------+----------------------------------------------------------------------------------+
| ``cov()``                    | `covariance <https://en.wikipedia.org/wiki/Covariance>`_                         |
+------------------------------+----------------------------------------------------------------------------------+


Resources for further study
-----------------------------

  * `Khan Academy - statistics and probability <https://www.khanacademy.org/math/statistics-probability>`_
  * `Visualizing probability and statistics <http://students.brown.edu/seeing-theory/index.html>`_


