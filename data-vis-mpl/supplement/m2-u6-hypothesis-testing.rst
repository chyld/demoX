.. m1-m2

.. figure:: ./images/ibm-cloud.png
   :scale: 100%
   :align: center
   :alt: ibm-logo
   :figclass: align-center

Hypothesis testing
###########################

Unit objectives
-----------------

By the end of this unit you will be able to:

+---------+-------------------------------------------------------------------------------------------------------+
| 1       | Explain event probabilities using common distributions                                                |
+---------+-------------------------------------------------------------------------------------------------------+
| 2       | Apply null hypothesis testing as an investigative tool using Python                                   |
+---------+-------------------------------------------------------------------------------------------------------+

|

.. raw:: html

    <iframe src="https://player.vimeo.com/video/355136255" width="600" height="400"  frameborder="0" allowfullscreen></iframe>

|


Overview
---------------

Data scientists employ a broad range of statistical tools to analyze data and reach conclusions
from messy and incomplete data. This unit focuses on the foundational techniques of estimation
with probability distributions and extending these estimates to apply null hypothesis significance tests.

Estimation and hypothesis testing build upon probability. Specifically, they are built on probability
distributions, which in turn depend on the concept of random variables and this unit assumes you are
familiar with these topics.

   * `Khan Academy resource on random variables and distributions <https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library>`_

Scientists use statistical models to make general statements about real-world observations.
Many types of data collected from many different sources often follow recognizable patterns. For example,
many of the measurable things in our world (e.g., IQ scores) follow a normal or Gaussian distribution.

When data are observed to follow probability distributions, and the data were collected in a way that
maximizes the chances that they are representative of the larger population from which they were taken,
it is possible to make estimates about the properties of that population. Note that this task of working
with data that has been drawn from some unknown probability distribution marks the transition between the
topics of probability and statistics.

Making inferences about some population by matching a sample of data with a probability
distribution and estimating its parameters requires that the sample meet certain criteria. Specifically, you
should have reason to believe that:

   * the data follow the chosen type of distribution, and
   * each point is independent and identically distributed (IID).

That is, the data were collected in a way that makes it likely that each data point is statistically independent
from every other one, and each was drawn from a distribution with identical parameters. The nuances of how to
collect data in ways that maximize the chances that your dataset satisfy these criteria are referred
to as `sampling techniques <https://en.wikipedia.org/wiki/Sampling_(statistics)>`_ and fall outside the scope
of this course.

Assuming you have a dataset that can be reasonably parameterized with a probability distribution, the
process of estimating those parameters, also known as
`statistical inference <https://en.wikipedia.org/wiki/Statistical_inference>`_ is relatively straightforward.

Statistical Inference
-------------------------------

Statistical inference is a very complex discipline, but fortunately there are tools that make its application
routine.  Statistical inference is generally involved in answering certain types of questions:

* **Hypothesis Testing**: How well does the data match some assumed (null) distribution?

* **Point Estimation**: What instance of some distributional class does it match well?

* **Uncertainty Estimation**: How can we quantify our degree of uncertainty about our estimates?

* **Sensitivity Analysis**: Do our results rely heavily on our distributional assumptions?

The suite of methods to carry out statistical inference are varied and include:

* Numerical Optimization
   * Maximum Likelihood
   * Least Squares
   * Expectation Maximization (EM)
* Simulation of Null Distributions
   * Bootstrapping
   * Permutation Testing
   * Monte Carlo Methods
* Estimation of Posterior Distributions
   * Markov Chain Monte Carlo (MCMC)
   * Variational Methods
* Nonparametric Estimation
   * Bayesian Non-parametrics

Hypothesis testing as an investigative tool
----------------------------------------------

In the context of data science, hypothesis tests often take the form of A/B tests where samples receive
two different treatments and testing between the impact of the two is the test. To demonstrate A/B
testing in a business context we will use the example of visitors to a AAVAIL's website that are randomly
sent to version A or version B, that are slightly different from each other. Let's assume that version B has a
new marketing scheme for getting a user to click 'subscribe' and version A is the default version. In order
to investigate whether version A has a greater impact on purchase decisions we will track the number of
visitors to each version and the proportions that click the buttons, then apply hypothesis testing.

Hypothesis testing
^^^^^^^^^^^^^^^^^^

1. Pose your **question**
    *Do visitors to sites A and B convert (i.e. become subscribers) at different rates?*

2. Find the relevant **population**
    *What is the conversion rate from each site?*

3. Specify a **null hypothesis** :math:`H_0`
    *The conversion rate is the same between sites A and B*

4. Choose **test statistic** to test :math:`H_0`
    *The observed conversion rate*

5. Collect **data**
    *Track visitors to site for x period of time, randomly sending each to either A or B*

6. Calculate the **test statistics**
    *Count the total number of conversions out of the total visitors to each version*

7. Reject the null  **null hypothesis**
    *If the test statistic appears to be significantly different compared to its *sampling distribution* under the null hypothesis. Otherwise, *fail to reject the null hypothesis*

**Simulating Data**

There are a few different ways of modeling this scenario. If the test statistic is the site conversion
rate then that can be reduced to an estimate of the probability of success. The probability of success
in a single trial follows a Bernoulli distribution. We can simulate visitors to the site by running
repeated Bernoulli trials and specifying a probability of success (which of course is what we'd be trying
to measure in an actual A/B test).

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import stats

   n = 100
   pconvert = 0.12 # our specified probability of success
   results = st.bernoulli(pconvert).rvs(n)
   converts = sum(results)
   print("We observed %s conversions out of %s"%(converts, n))

   We observed 17 conversions out of 100

**Null Hypothesis**

We simulated having 100 visitors to one version of the site, and in a typical A/B test we would be comparing
two versions of the site running concurrently. In that setup, we could run the above code again, possibly
changing pconvert to simulate another sample conversion rate. This would be a two-sample hypothesis test.
For simplicity, let’s start with a one sample test. We have already simulated the sample, and now need to
compare that sample to a baseline. In our example, this baseline could be the long-term conversion rate of
the site, if this has been stable over time.

The sample simulated above came from repeated Bernoulli trials, and these follow a Binomial distribution.
So we can specify the baseline as follows:

.. code-block:: python

   p = 0.1 # long term conversion rate
   rv = stats.binom(n,p)
   mu = rv.mean()
   sd = rv.std()
   print("The expected distribution the site is mu=%s, sd=%s"%(mu,sd))

   The expected distribution the site is mu=10.0, sd=3.0

**Binomial Test**

Now we can calculate the probability of observing a result at least as extreme as our sample
conversion rate from the baseline distribution. That is, the probability of seeing 17 visitors
out of 100 convert on the site that has a long term conversion rate of 10%.

.. code-block:: python

   print("binomial test p-value: %s"%stats.binom_test(converts, n, p))

   binomial test p-value: 0.028435297266484586

Remember that when calculating p-values it is important to define a threshold for rejecting the
null hypothesis *before* doing the actual calculation.

**Z-Test with continuity correction**

Our A/B test example has a known probability distribution, but you are likely to encounter scenarios
where the distribution is not quite so well defined. While it is convenient to calculate p-values
from the binomial distribution for relatively small numbers of trials, it becomes cumbersome as the
number of trials, or in our case, the number of visitors, increases beyond a few thousand. In both
situations, the solution is to switch to the **normal distribution**. Thanks to the central limit
theorem, the sampling distribution of sample means approaches the normal distribution. In our example,
the conversion or success rate is a special case of a mean.

Calculating p-values from a normal distribution is traditionally done by comparing the sample statistics
with a **standard normal** which is a normal with mean of zero and variance of one. This standardized
version of the sample statistic is known as a *z-score*, and this particular form of hypothesis test
is a Z-test.

Here we are approximating a discrete distribution with a continuous distribution. In doing so, we must
apply a `continuity correction <https://en.wikipedia.org/wiki/Continuity_correction>`_. When
starting with a binomial distribution, the calculation of the z-score goes from:
:math:`z = (\bar{x} - \mu) / \sigma` to :math:`z = (\bar{x} - \frac{1}{2}- \mu) / \sigma`.

.. code-block:: python

   z = (converts-0.5-mu)/sd
   print("normal approximation p-value: %s"%(2*(1 - st.norm.cdf(z))))

   normal approximation p-value: 0.030260280020471653

The p-values from the Binomial test and the Z-test are pretty close with the relatively small number
of trials at n = 100. Running the simulation with a greater number of trials should align them further.

**Permutation Test**

It is also possible to take a numerical approach to calculating these probabilities. NumPy's random
module allows you to draw repeated samples from the distribution of your choice. Thus, in our example,
we can repeatedly generate success counts from a binomial distribution with n = 100, and p = 0.1. We would
then track how many of those success counts were equal to or exceeded the observed number of conversions. The
proportion that do so will converge towards the p-value of this observation as the number of repeats
increases.

.. code-block:: python

   nsamples = 100000
   xs = np.random.binomial(n, p, nsamples)
   print("simulation p-value: %s"%(2*np.sum(xs >= converts)/xs.size))

   simulation p-value: 0.0413


Maximum Likelihood Estimation (MLE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the above approaches depend on specifying some underlying population distribution, and
comparing the observations with the characteristics of that distribution. Instead, you could
use `bootstrapping <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_
to characterize the underlying distribution.

.. code-block:: python

   bs_samples = np.random.choice(results, (nsamples, len(results)), replace=True)
   bs_ps = np.mean(bs_samples, axis=1)
   bs_ps.sort()

   print("Maximum Likelihood Estimate: %s"%(np.sum(results)/float(len(results))))
   print("Bootstrap CI: (%.4f, %.4f)" % (bs_ps[int(0.025*nsamples)], bs_ps[int(0.975*nsamples)]))

   Maximum likelihood 0.17
   Bootstrap CI: (0.1, 0.25)

Bayesian Estimation
^^^^^^^^^^^^^^^^^^^

The Bayesian approach estimates the posterior distribution or the updated belief about the parameters
given the prior belief and the observed data. The approach uses the posterior distribution to make point
and interval estimates about the parameters. The calculations we demonstrate here have
`analytic solutions <https://en.wikipedia.org/wiki/Closed-form_expression>`_.
For most real life problems the necessary statistical models are more complex
and estimation makes use of advanced numerical simulation methods.

In the case of characterizing a probability of success, in our case the site conversion rate, we
can make use of the `Beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`_ as our
prior distribution, which updates to posterior. This is also a Beta distribution
of Beta distributions, known as being `self-conjugate <https://en.wikipedia.org/wiki/Conjugate_prior>`_).

In this scenario the Bayesian update to the Beta distribution's parameters :math:`\alpha`
and :math:`\beta` (for example with :math:`\alpha = 1`, :math:`\beta` = 1, the distribution is Uniform),
yields :math:`\alpha + k` and :math:`\beta + n - k` in the posterior distribution, where :math:`k` is the
number of successes out of :math:`n` trials.

.. code-block:: python

   fig  = plt.figure()
   ax = fig.add_subplot(111)

   a, b = 1, 1
   prior = st.beta(a, b)
   post = st.beta(converts+a, n-converts+b)
   ci = post.interval(0.95)
   map_ =(converts+a-1.0)/(n+a+b-2.0)

   xs = np.linspace(0, 1, 100)
   ax.plot(prior.pdf(xs), label='Prior')
   ax.plot(post.pdf(xs), label='Posterior')
   ax.set_xlim([0, 100])
   ax.axhline(0.3, ci[0], ci[1], c='black', linewidth=2, label='95% CI');
   ax.axvline(n*map_, c='red', linestyle='dashed', alpha=0.4)
   ax.legend()
   plt.savefig("ab-test.png")

.. note::

   **EXERCISE**

   Use the Python code above to play around with the *prior specification*. Does it seem to influence the resulting posterior distribution?
   If so, how? How would you describe these effects to a client?


Business scenarios and probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two-sample independent t-test

We are interested in comparing the amount of time elapsed between a client request and stream availability for the company AAVAIL's streaming servers.  Specifically we want to compare our locally hosted servers to a cloud service in terms of speed. The data are arrival times (in seconds) for a stream, meaning the time it takes from submission to receive a link with the modified version of the steam.

Remember to formalize your hypothesis.

1. Pose your **question**
   *Is it faster, on average, to process streams for viewing on a cloud service compared to our locally hosted servers?*    

2. Find the relevant **population**
   *The population consists of all possible steams*
    
3. Specify a **null hypothesis** :math:`H_0`
   *There is no difference, on average, between local and hosted services for stream processing times
   location after I submit my ride request.*

4. Set the significance level, :math:`\alpha=0.05`

5. Collect your data.   
    
.. code-block:: python

   local_arrivals = np.array([3.99, 4.15, 6.88, 4.53, 5.65, 6.75, 7.13, 2.79, 6.20,
                              3.72, 7.28, 5.23, 4.72, 1.04, 4.25, 4.71, 2.16, 3.46,
			      3.41, 7.98, 0.75, 3.64, 6.25, 6.86, 4.71]) 
   hosted_arrivals = np.array([5.82, 4.83, 7.19, 6.98, 5.82, 5.25, 5.71, 5.59, 7.93,
                               7.09, 6.37, 6.31, 6.28, 3.12, 6.02, 4.84, 4.16, 6.72,
			       7.44, 6.28, 7.37, 4.27, 6.15, 4.88, 7.78])		

The test statistic will be calculated as part of the following code block..

.. code-block:: python

   test_statistic, pvalue = stats.ttest_ind(local_arrivals,hosted_arrivals)
   print("p-value: {}".format(round(pvalue,5)))

.. code-block:: none

   p-value: 0.0069

In this case we would reject the **null hypothesis** in favor of the alternative that the average times are not the same.

**Unequal variances t-test**

The use of a `Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_, accounts for a specific bias that the Gaussian distribution does not, because it has heavier tails.

The t-distribution always has mean 0 and variance 1, and has one parameter, the **degrees of freedom**. Smaller degrees of freedom have heavier tails, with the distribution becoming more and more normal as the degrees of freedom gets larger.

The default version of a Student's t-test assumes that the sample sizes and variances of your two samples are equal.  In the case of our arrival times above we cannot state that the variances of the two samples are suppose to be the same.  The unequal variances t-test, also called `Welch's t-test <https://en.wikipedia.org/wiki/Welch's_t-test>`_ is a more appropriate variant of the t-test for this example.

There are many variants of the t-test and depending on the field of study some have different names for the same variant.  The unequal variances t-test in Python can be accessed with the `equal_var` keyword argument. 

.. code-block:: python

   test_statistic, pvalue = stats.ttest_ind(local_arrivals, hosted_arrivals, equal_var = False)
   print("p-value: {}".format(round(pvalue,5)))

.. code-block:: none

   p-value: 0.00735

Variants on t-tests
^^^^^^^^^^^^^^^^^^^^^^^^^

There a number of variants on t-tests available through the `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ submodule.

ttest_1samp
    Calculate the t-test for the mean of ONE group of scores.

ttest_ind
    Calculate the t-test for the means of two independent samples of scores.

ttest_ind_from_stats
    t-test for means of two independent samples from descriptive statistics.

ttest_rel
    calculate the t-test on two related samples of scores, a and b. (paired t-test)

One-way Analysis of Variance (ANOVA)
--------------------------------------------

The previous scenarios have concerned distinguishing between a sample and a baseline,
and between two samples. Suppose you want to distinguish between three or more samples,
that is your data fall into three-plus categories and you want to establish whether there
is a difference in outcomes based on those categories.

Example: A clothing stores runs a few different types of promotions and wants to know which
promotion type has the greatest impact on its daily income.
The promotion types are: 20% off sale, buy one get one free on select items, and every
$50 in purchases earns a $10 gift card for future use.

1. Pose your **question**
    *Does income differ depending on the type of promotion running at the store?*

2. Find the relevant **population**
    *The population consists of all the days when the store is running a promotion.*

3. Specify a **null hypothesis** :math:`H_0`
    *Daily store income does not depend on the nature of the promotion available to customers of the store.*

4. Set the significance level, :math:`\alpha=0.05`
    
5. Collect your data
    
.. code-block:: python

   twnty_pct = np.array([13374.67, 14788.77,  1413.77, 13373.73,  7847.34,
               14664.43, 13549.71, 10728.61,  7671.43, 15237.58])

   bogo = np.array([13256.61, 18098.51, 15176.5 , 18269.76, 14580.62, 12648.66,
       15126.09, 16674.32, 18757.55, 15591.71])

   giftcard = np.array([11751.06, 13799.08,  9215.4 , 10993.85,  9043.87,
              16607.26, 16665.58,  9905.32,  5729.78, 12161.7 ])

.. code-block:: python

   all_income = np.vstack([twnty_pct, bogo, giftcard])
   print("The global mean income is: %s"%np.round(all_income.mean(), decimals=2))

   for promo, income in zip(['20% off', 'BOGO', 'Gift Card'], all_income.mean(axis=1)):
       print("Mean income for the %s promo is %s"%(promo, np.round(income, decimals=2)))

   The global mean income is: 12890.11

   Mean income for the 20% off promo is 11265.0
   Mean income for the BOGO promo is 15818.03
   Mean income for the Gift Card promo is 11587.29

7. Reject the *null hypothesis*
   *If the test statistic does not fit or match its sampling distribution under the null hypothesis. Otherwise, fail to reject the null hypothesis*

   When comparing across three or more groups (in this case types of promotions) an
   appropriate test is a `one-way ANOVA <http://www.biostathandbook.com/onewayanova.html>`_, which compares
   between group variation and within group variation. The relevant probability
   distribution if the `F distribution <https://en.wikipedia.org/wiki/F-distribution>`_,
   and that is the name used in the relevant method in Scipy:

.. code-block:: python

   print(st.f_oneway(twnty_pct, bogo, giftcard))

   F_onewayResult(statistic=5.424995346717885, pvalue=0.010459786642859003)

In this example, it seems likely that the differences in income have something to do
with the type of promotion being run at the store. When digging deeper to determine which
type of promotion is best, one needs to be mindful of the `multiple comparison problem <https://en.wikipedia.org/wiki/Multiple_comparisons_problem>`_.


