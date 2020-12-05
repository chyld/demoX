.. name of course

*************************************   
cuDF
*************************************

+------+------------------------------+-----------------------------------------------------------------------+
| 2    | cuDF                         | Describe the use cases of and apply cuDF                              |
+------+------------------------------+-----------------------------------------------------------------------+

Usage and limitations
========================

.. admonition:: TODO

   This unit ia a placeholder


Check for Understanding
===========================


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

Additional Resources
========================

Try to end each of your units with a section where learners can go to learn more


