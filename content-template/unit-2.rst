.. name of course

*************************************   
Unit 2 Title 
*************************************

+------+------------------------------+--------------------------------------------------------------------------+
| 1    | GLMs                         | The learning objective for the unit                                      |
+------+------------------------------+--------------------------------------------------------------------------+

* Courses are made up of 3-5 units that address 3-5 learning objectives.
* Each unit correponds 1-to-1 with a learning objective.
* Units are a single page of rst.
* Units have some number of sections.
* Units have some in-line CFUs (check for understanding). 
* Each course will link to the github repo with associated exercises.
* Each course will link a checkpoint that has 3-5 questions to assess mastery of the learning objectives.

Use sections to break up your content.  We want to avoid a wall of text.  The section grouping is the
most common grouping that you will be using.

Section Title (H3)
====================

One great way to avoid the wall of text is to embed code into your content.

Subsection (H4)
----------------

Content block for a subsection.

Subsubsection (H5)
^^^^^^^^^^^^^^^^^^^^

Content block for a subsubsection.

Paragraph title (H6)
""""""""""""""""""""""

An if for some reason subsubsections are not enough you have paragraphs.


Embedding images
==================

.. figure:: ../images/galvanize-logo.png
   :scale: 35%
   :align: center
   :alt: galvanize-logo
   :figclass: align-center


Notes warnings and the like
==============================

Note, warning, hint and important are the important ones
	      

.. note::
   
   Programming practices, editors, version control, software engineering, and other related topics are
   not part of the scope of this short course.

.. important::

   When you run the installer be sure that you click the checkbox that says `add to system path`.

.. warning::

   The `C` hyperparameter in SVMs is treated differently depending on the implementation it may be `1/C`

.. hint::

   There are several was you can edit the fonts when using matplotlib.  Looks at the ...
   

Embedding code
====================

If the code is meant to be run from the command line use a `~$` as an additional indicator.

.. code-block:: bash
      
    ~$ brew install git

code blocks work in many languages including lisp!
    
.. code-block:: lisp

    CL-USER> (defun hello ()
             (format t "Hello, World!~%"))
    HELLO
    CL-USER> (hello)
    Hello, World!
    NIL
    CL-USER> 


    
You can also use the interactive version of python syntax

>>> import numpy as np
>>> x = np.random.randint(0,100,50)


Literal include
-------------------

.. literalinclude:: ./scripts/linspace-example.py
  :language: python



Citations
===============

To train most neural networks an important algorithm called
`backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_ is used :cite:`1986:Rumelhart`. The algorithm is
used to compute the gradient, while another algorithm, such as stochastic gradient descent, carries out the rest of
learning process.

LaTeX
==============

.. math::

    \textrm{MAE} = \frac{1}{N} \sum^{n}_{i=1} \left| \hat{y}_{i} - y_{i} \right|


.. math::

    \textrm{RMSE} = \sqrt{\frac{1}{N} \sum^{n}_{i=1} \left( \hat{y}_{i} - y_{i} \right)^{2}}


You can also use :math:`\LaTeX` inline like this :math:`\sum^{n}_{i=1}`


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


