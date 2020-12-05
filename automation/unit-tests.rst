********************************
Feedback loops and unit tests
********************************

An organized, automated, repeatable process of deployment is key to iterating quickly, receiving feedback and
advancing the project.  It also sets a stage for systematic checks for model improvement via feedback loops.

Feedback Loops
===================

A reusable deployment process, with Docker images as templates, will save you, and those who work closely with you, a lot
of time.  Feedback loops represent all of the possible ways you can return to an earlier stage in the AI enterprise
workflow.  We discussed feedback loops in the first course of this specialization, *Business Priorities and Data Ingestion*.

.. figure:: ./images/workflow_conceptual.png
   :scale: 50%
   :align: center
   :alt: data science workflow
   :figclass: align-center

Common feedback loops to keep in mind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

production --> business opportunity
    The business opportunity that was refined and decided on in the beginning is in some ways a statement of purpose for
    your models.  If a model has less of an impact on the business than originally anticipated this is often the first
    feedback loop that you will visit.  It is a place to discuss the other potentially relevant feedback loops.  Once
    all of the least time-consuming feedback loops have been explored this is also the place where you discuss the
    opportunity cost of continued workflow iteration.

production --> data collection
    This is a very common feedback loop especially when using deep-learning models.  Because of their flexibility, neural
    networks can overfit a model.  You may plot
    `learning curves <https://scikit-learn.org/stable/modules/learning_curve.html>`_ to help guide the decision to obtain
    more data.  In some cases, obtaining more data means *labeling* more data which can be costly so ensure that you
    engage in discussions to determine the best course of action.

production --> EDA
    This is an important and often overlooked feedback loop.  Once a model has been in production for some time, it
    becomes necessary to investigate the relationship between model performance and the business metric.  This can be thought
    of as an extension of EDA, where visualization and hypothesis testing are the most important tools.  Investigations
    into the underlying causes of model performance drift can re-purpose much of the code developed during EDA.

production --> model selection and development
    If a model performs poorly in production, perhaps due to latency issues or because there is an over-fitting issue,
    it is reasonable to return to try a different model. If it is an overfitting scenario and obtaining
    more data is not an option, choosing a model with lower complexity (e.g.
    `SGDClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_) is
    a reasonable next step.  Spark ML models tend to have more latency than deployed scikit-learn models.
    `Latency <https://en.wikipedia.org/wiki/Latency_(engineering)>`_ is the effective runtime for a prediction. You
    can run simulations to test different models, which can help optimize for latency.  Another reason to return to the
    ``models`` stage from production is if we observe performance drift (a topic covered in the next unit).


There are other feedback loops such as trying different data transformations to improve a model's performance or optimizing
the way data are collected to reduce the number of transformations that are necessary to run a machine learning model. The most 
aspect of all feedback loops is that they end with a return to a previous stage of the workflow.  This is the only way to ensure
that your entire workflow does not contain a weak link, and also the best way to keep track of each stage of the workflow.  

Unit Tests
-------------

`Unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_ is the process of testing small portions of the software,
also known as *units*.  This is done one test at a time, to verify that an expected result is returned under controlled
conditions. Importantly, the unit tests are usually organized as a suite and return *objective evidence*, in
the form of a boolean value, which is a key element that enables workflow automation.  The boolean value indicates whether
or not each and every part of the software that was tested performed as expected. Ideally every portion of the code would
be tested under every conceivable combination of conditions, however this is clearly not possible in the real world. The
amount of `source code <https://en.wikipedia.org/wiki/Source_code>`_ that is actually tested when compared to the total
amount of testable code is known as `test coverage <https://en.wikipedia.org/wiki/Code_coverage>`_. There is even a
`package in Python called coverage <https://coverage.readthedocs.io/en>`_ that estimates the total coverage of your
tests.

.. important::

    There is an important trade-off in data science between the amount of *test coverage* and prioritizing other tasks.
    In many ways this trade-off is the same as the one that software engineers face, except data science has a sizable
    component of *experimentation*.  This means that many models that get created never see production and many models
    that see production never come to fruition.  There are many reasons for this and the AI workflow presented here is
    designed to minimize this risk, but nonetheless many modeling efforts are shelved.  Because of this, we present as
    part of the overall workflow a way to properly include unit tests, but we do so in a way that includes only a minimum
    number of tests along with the scaffolding to expand once a model or service proves its worth.

It is important to think about *opportunity cost* when determining the appropriate amount of test coverage.  We
will refer to the unit testing approach presented here as a `test harness <https://en.wikipedia.org/wiki/Test_harness>`_,
because it is implemented as an *automated test framework*.  Much like *data ingestion*, the idea is to have the necessary
components of a task bundled under a single script. In this case it will be called ``run-tests.py``.  To help ensure that
our unit tests are a *test harness* we will use the script to setup a `hook <https://en.wikipedia.org/wiki/Hooking>`_.

The field of `software testing <https://en.wikipedia.org/wiki/Software_testing>`_ is out of the scope of this specialization,
but it is worth noting that there are many viable testing frameworks and technologies that can be used in place of the
approach presented here.  One of the reasons to create unit tests is to ensure that iterative improvements
to code do not break the functionality of the model or API.  This is known as
`regression testing <https://en.wikipedia.org/wiki/Regression_testing>`_, because when the code does not perform as expected
it is a *regression*.  Including regression testing, here is a summary of the principal reasons to package unit tests with
your deployed machine learning model:

* **Regression Testing**: ensure that previously developed and tested software still performs after a change.
* **Code Quality**: promote the use of well-written code along with well-conceived designs
* **Documentation**: unit tests are a form of documentation that can help you and members of your team understand the details of how the software works, unit-by-unit
* **Automatic Performance Monitoring**: having a suite of unit tests that are kicked off when training is performed can help monitor for performance drift in an automated way

Unit tests also helps ensure that when software fails, it fails *gracefully*. This means it stops execution
without causing additional errors and takes any steps, such as closing open connections or saving data to a file that
may be necessary for recovery.  This is an important aspect of software design that can save significant amounts of time
when debugging a problematic query.

Unit testing in Python
^^^^^^^^^^^^^^^^^^^^^^^^^

Three of the most useful libraries in Python to carry out unit testing are:

* `pytest <https://docs.pytest.org/en/latest/>`_
* `nose <https://nose.readthedocs.io/en/latest/testing.html>`_
* `unittest <https://docs.python.org/3.5/library/unittest.html>`_

We will use the ``unittest`` library in this example.  Here is a simple working example of how we will build tests.

.. literalinclude:: ./scripts/test-example.py

To run the script, save it as ``test-example.py`` and then execute the following on the command line:

.. code-block:: bash

    ~$ python -m unittest test-example.py

You should see the following output. Note the four small dots indicate four passing tests. If one of the tests had failed, 
you would see an 'F' in place of one of the dots, followed by additional information describing the test result and failure.

.. code-block:: none

    ....
    ----------------------------------------------------------------------
    Ran 4 tests in 0.022s

    OK

Notice that we are proactively trying to catch errors inside of the function.  We used the
`Python regular expression <https://docs.python.org/3/library/re.html>`_ module to help validate the input as it is a
very useful tool for quality assurance on inputs.  The are many types of
`built-in exceptions <https://docs.python.org/3/library/exceptions.html>`_ in Python.  We simply use the generic
``Exception`` class in these examples. One way to further polish your code would be to use the closest matching
exception class or to write your own class, inheriting from `Exception <https://docs.python.org/3/library/exceptions.html#Exception>`_.

The tests shown in the example above are unit tests because they are small.  If one of the tests were
more comprehensive, for example an API test that tested multiple functions, it would likely fall under the umbrella of
`integration testing <https://en.wikipedia.org/wiki/Integration_testing>`_.  Both unit tests and integration tests are
part of the CI/CD pipeline (see section below).  There are several types of testing that can be used and the smaller,
faster ones should be the most numerous, with the more comprehensive ones being fewer.  The number of tests are often
illustrated as a `test pyramid <https://developer.ibm.com/apiconnect/2019/07/16/ibm-api-connect-practical-api-test-pyramid/>`_.
The goal of this module is to get you up and running with unit testing. Keep in mind that this is an active and
important area of software engineering that you should continue to study.

Test-Driven Development (TDD)
-------------------------------

Traditionally, software developers write software by first writing their functions, algorithms, classes, etc... and then,
once they are satisfied that everything is working, they write a series of unit tests to provide objective evidence that
it works as expected. The downside to this approach is that, without a defined completion criteria, it may result in
writing more code than is necessary, and, without a clear definition of the expected outcome, the programmer might not 
know what completion criteria they are working towards until late in the process.

Test-Driven Development extends the idea of unit testing by recognizing that the **sucessful completion** of the test is
the most important outcome of the software development process. Assuming the test is well-written and has sufficient coverage, 
any code that produces an 'OK' is ready for production; any code that does more than this is simply superfluous. TDD can have the same effect as using
`pseudocode <https://en.wikipedia.org/wiki/Pseudocode>`_ to template a piece of software or a script before writing the
code.  When working on large software projects, it is easy to get caught up in non-essential portions of the code. TDD and 
pseudocode can serve as a checklist of tasks that need to be completed to obtain that all-import boolean result: 'OK'. This 
can help keep you within a pre-defined set of boundaries all the way through the development process, saving time and effort.

To this end, TDD starts by clearly defining the expected outcomes under various conditions *first*. We then write only enough 
code to achieve successful unit test results. There are other methodologies to produce completion criteria, some of which use
`requirement analysis <https://en.wikipedia.org/wiki/Requirements_analysis>`_ as part of the
`software design <https://en.wikipedia.org/wiki/Software_design>`_ process, but here we show a simple approach to
demonstrate how unit testing through TDD can be used to build a *test harness*.

.. admonition:: SCREENCAST

    * provide a walk-through of a typical unit test suite
    * show an example of test driven development to build out the training part of a flask API

.. raw:: html

    <iframe src="https://player.vimeo.com/video/87110435" width="640" height="360"  frameborder="0" allowfullscreen></iframe>

|

CI/CD
----------

In software engineering CI/CD, sometimes written as CICD, refers to `continuous integration <https://en.wikipedia.org/wiki/Continuous_integration>`_ 
and `continuous delivery <https://en.wikipedia.org/wiki/Continuous_delivery>`_.
Depending on the context, CD may also refer to `Continuous Deployment <https://en.wikipedia.org/wiki/Continuous_deployment>`_.  
CI/CD is a concept to be aware of when learning about the DevOps side of data
science. Continuous Integration is the practice of merging all developers' changes *frequently* (usually daily) into a
single central source often called the `trunk <https://en.wikipedia.org/wiki/Trunk_(software)>`_.  

Continuous Delivery refers to the iteration on software in short cycles using a straightforward and repeatable deployment
process.  Continuous Delivery uses **manual deployments**, which is in contrast to Continuous Deployment which makes use of
**automated deployments**.  The unit testing framework presented here can be readily integrated into several CI/CD
pipelines.  This is not a course in DevOps nor is it a course in data engineering, so we will not go so
far as to make recommendations about deployment systems and architectures, but being aware of the terminology can promote
cross-team functionality.

Historically, software updates were deployed infrequently, perhaps once per year, and only after extensive, months-long
testing cycles. CICD improves this process tremendously, allowing developers to see the results of their efforts almost
immediately. However, the increased pace of change comes at the risk of introducing bugs. For this reason, CICD depends
heavily on a robust testing process. Without automated testing, CICD would not be possible.

Additional resources
^^^^^^^^^^^^^^^^^^^^^^

* `Why you should use microservices and containers <https://developer.ibm.com/technologies/devops/articles/why-should-we-use-microservices-and-containers>`_
* `Managed CI/CD Kubernetes services (IBM) <https://www.ibm.com/us-en/marketplace/kubernetes-and-devops-consulting>`_
* `GitHub actions <https://github.com/features/actions>`_
* `Jenkins <https://jenkins.io/>`_

------------------------------------------------------------------

.. admonition:: CFU

    Which of the following was **not** mentioned as a reason to bundle unit tests with a deployed model?

    .. container:: toggle

       .. container:: header

          * **(A)**: promotes code quality
          * **(B)**: regression tests
          * **(C)**: they perform the workflow feedback loops
          * **(D)**: automate performance monitoring tasks
          * **(E)**: can be readily integrated into CI/CD pipelines

       **ANSWER**:

          **(C)** When used in the context of feedback loops, unit tests are generally tasked with
          *checking whether or not* a feedback loop is necessary.  A unit test may have detected that the model
          performance has fallen below some threshold.  Corresponding feedback loops like swapping one model for another
          or collecting and labeling more data are more often than not a heavier lift than a unit test is designed
          to accomplish.  The other answers are all viable reasons to adopt unit tests.
