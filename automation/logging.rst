.. figure:: ./images/ibm-cloud.png
   :scale: 100%
   :align: center
   :alt: ibm-logo
   :figclass: align-center

Performance Monitoring
##############################

The course learning objectives.

+---+---------------------------------------------+----------------------------------------------------------------------------+
| 2 | Performance monitoring and business metrics | * Describe processes for monitoring model performance in production        |
|   |                                             | * Describe general principles for understanding models in business contexts|
+---+---------------------------------------------+----------------------------------------------------------------------------+


.. raw:: html

    <iframe src="https://player.vimeo.com/video/373939573" width="640" height="360"  frameborder="0" allowfullscreen></iframe>

|

.. admonition:: OUR STORY

    You have seen how other data scientists on the AAVAIL team work and you now have a good idea what the lifecycle of a
    deployed model or service looks like.  One thing that you have observed is that once a model has been running for some
    time, either management or senior members of the data science team will ask about how the model is doing.
    You have also noticed that most members of the team respond by talking about the same fundamental concepts:
    performance drift, load, latency, and average runtime.

    However, an important consideration that often gets overlooked is *business value*; is the model having a significant effect on 
    business metrics as intended?  It is important to be able to use log files that have been standardized across the team to
    answer questions about business value as well as performance monitoring.  

.. admonition:: THE DESIGN THINKING PROCESS

    Concerns about performance and monitoring are generally not raised in the design thinking process until the *prototype* or *test* phases. 
    Indeed, performance monitoring is historically treated as an afterthought during implementation or long-term production support and 
    is occasionally left as a consideration for other members of the team with specialized skills in systems optimization.  
    Needless to say, planning for performance monitoring early in the process yields dividends down the line and eases the transition 
    from development to production.

Logging
-----------

Like all problems in data science, performance monitoring starts with collecting the right data in the right format. 
Because performance monitoring is a concern in nearly all customer-facing computer systems, there is a well-established set of 
tools and techniques for collecting this data. Data for performance monitoring is generally collected using
`log files <https://en.wikipedia.org/wiki/Log_file>`_.
Recall the following best practice:

.. important::

    Ensure that your data are collected at the most granular level possible. This means each data point should represent
    one user making one action or one event.

Naturally, collecting very granular data will result in very large data sets. If there is a need, you can always summarize 
the data after it has been collected. Summary level data may mask important patterns and generally it is not possible to 
go from summary data to granular data. Log files with several million entries can be analyzed on a single node or on a
personal machine with little overhead.

.. note::

    If the logging is handled by another member of your team or by another another team you should ensure that the
    minimally required data discussed here are available or it will be difficult to monitor your model's performance
    and/or debug performance related issues.

Minimal requirements for log files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are data that are minimally required for performance monitoring for most model deployment projects.  There are other
features that fall into this category that are situation dependent, like `user_id` in a recommendation system, so do not
view this list as comprehensive, simply keep it as a reference starting point.

runtime
    The total amount of time required to process the request.  This is a factor that directly affects the end user's
    experience and should be monitored.
timestamp
    Timestamps are needed to evaluate how well the system handles load and concurrency. Additionally, timestamps
    are useful when connecting predictions to labels that are acquired afterwards.  Finally, they are needed for the
    investigation of events that might affect the relationship between the performance and business metrics.
prediction
    The prediction is, of course, the primary output of a predition model. It is necessary to track the prediction for 
    comparison to feedback to determine the quality of the predictions. Generally, predictions are returned as a `list` 
    to accommodate multi-label classification.
input_data_summary
    Summarizing information about the input data itself.  For the predict endpoint this is the shape of the input
    feature matrix, but for the training endpoint the features and targets should be summarized.
model_version_number
    The model version number is used to better understand the influence of model improvements (or bugs) on performance

Additional features that can be optionally logged
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the features that are considered nice to have, but they are not always relevant to the circumstances or sometimes
there can be practical limitations (e.g. disk space or computational resources) that limit the ability to save these features.

request_unique_id
    Each request that has been made should correspond to an entry in the log file.  It is possible that a request
    corresponds to more than one entry in the log file for example if more than one model is called.  This is also
    known as `correlation_id <https://en.wikipedia.org/wiki/Identity_correlation>`_.
data
    Saving the input features that were provided at the time of a predict request makes it much easier to debug broken
    requests.  Saving the features and target at the time of training makes it easier to debug broken model training.
request_type
    Relevant attributes about the request (e.g. webapp request, browser request)
probability
    Probability associated with a prediction (if applicable)

The value of logging most of the mentioned data is fairly intuitive, but saving the data itself might seem unnecessary.
If we save the input features, when a predict endpoint was hit, we can reconstruct the individual prediction, stepping
through each part of the prediction process.  For training, the archiving of all the data is often unnecessary, because
there is a system in place, like a centralized database, that can re-create the training data for a given point in time.
One option is to archive only the previous iteration of training data.

If very granular levels of performance monitoring are needed, we could model the distribution of each feature in the
training data matrix and determine if new batches of data fall outside the normal range. We could also use
one of the models we have discussed for
`novelty detection <https://scikit-learn.org/stable/modules/outlier_detection.html>`_, but insight would be at the level
of observations across all features rather than at the feature level.  For most models the latter option is sufficient.

.. warning::

    if you decide to log the data, be aware of disk space and read/write bottlenecks. It is also important to ensure compliance 
    with company policies or regulations such as `HIPAA <https://www.dhcs.ca.gov/formsandpubs/laws/hipaa/Pages/1.00WhatisHIPAA.aspx>`_, 
    or `GDPR <https://en.wikipedia.org/wiki/General_Data_Protection_Regulation>`_ concerning personally identifiable or sensitive 
    information, depending on jurisdiction.


Logging in Python
^^^^^^^^^^^^^^^^^^^^

Python has a `logging module <https://docs.python.org/3/library/logging.html>`_ which can be used for performance
monitoring, but we will show logging through the use of the `csv module <https://docs.python.org/3/library/csv.html>`_ to keep the
process as simple as possible.  The following code shows how to create a log file.

.. code-block:: python

    import os, csv, time, uuid

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    logfile = "aavail-predict-churn-{}-{}.log".format(today.year, today.month)

    ## write the data to a csv file
    header = ['unique_id','timestamp','y_pred','y_proba','x_shape','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.asctime(),y_pred,y_proba,query.shape,MODEL_VERSION,runtime])
        writer.writerow(to_write)

The above code snippet is not a fully working example, because it does not include the model, but it does show the key
pieces for logging to a file that changes automatically each month.  The 'append' or `'a'` mode is used here since we
do not want to overwrite the existing file.  See the
`Python input/output documentation <https://docs.python.org/3/tutorial/inputoutput.html>`_ if you need to review the other
modes.  It is reasonable to use `JSON <http://json.org>`_ or a centralized database as a target destination.  There are
numerous other tools like `Elasticsearch <https://www.elastic.co>`_ and
`Apache Commons Logging <https://commons.apache.org/proper/commons-logging/>`_. We use
simple CSV formatted files because they help keep the Docker container isolated from other environments and they are a
convenient format for most data scientists.

For a full working version of a script that demonstrates logging for performance monitoring purposes download and spend
some time with the following example.  We encourage you to look through and run the file to see how it works since the
next step is to use the same code from within a Flask API.

:download:`example-logging.py <./scripts/example-logging.py>`

.. note::

    In these materials, we log everything to either a train or a predict file, but depending on your environment you
    may want to separate log files in a different way (e.g. debugging, performance, storage).

Model performance drift
-------------------------

With a system for logging in place, the overall goal is to keep the performance of a model high over time, and ideally to 
see continuous improvement. The log
files are key to identifying when a change has occurred, but it helps to know what kind of performance drift
to expect. When we monitor model performance, we look for
**any significant changes** in model performance, in order to both identify issues early and capitalize on changes that
result in performance improvements.

.. admonition:: Software decay

    `Software decay <https://en.wikipedia.org/wiki/Software_rot>`_ or *software rot* occurs when there is any decrease 
    in model performance.

Common forms of performance drift or software decay include concept drift, sampling bias changes, selection bias changes, 
software changes, and data changes.  Each of these is covered below.  


Concept drift
^^^^^^^^^^^^^^^^^^^^^^^

`Concept drift <https://en.wikipedia.org/wiki/Concept_drift>`_, is when the statistical distribution of a target
variable changes over time.  One example of this would be fraud detection.  If fraud was a fraction of a percentage
of all known cases before we deployed a machine learning algorithm, it would be reasonable to assume that the percent
of fraud will decrease over time, effectively changing the distribution.  The change will likely have consequences on
model performance.  This type of drift would appear as decreased model performance, but you could also detect
it by checking the training log files.

Sampling bias and selection bias changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After a model is deployed, any newly introduced `sampling bias <https://en.wikipedia.org/wiki/Sampling_bias>`_.  
could result in subgroups of the data being under or over represented and the model would not generalize well
to new data, which would decrease model performance.  Any newly encountered
`selection bias <https://en.wikipedia.org/wiki/Selection_bias>`_ is also likely to affect model performance.  

For example, imagine a model was built to diagnose a specific medical condition from a chest X-ray.  Perhaps the standards
and technology have changed the way the radiologist makes a diagnosis, implying that the way the labels were initially generated is
different today than it was in the past.  Supervised learning in its current form requires accurate and consistent labeling
of targets.  If the process for labeling data has changed, it will likely affect model performance.  We often
observe the change in model performance through a detected outlier, but it requires some investigation before the reason
for performance drift can be confidently identified.

Software changes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Another cause of performance drift can come from changes in the the model and container software.  This is why we explicitly use
a model version, used in conjunction with version control.  If a library dependency, code optimization,
or feature addition were to blame for the performance drift it should be easy to track based on the model version.  

For example, imagine you have just converted a neural network into the newest version of TensorFlow or another deep-learning
package.  This change should be tied to a specific model version.  You can create
`releases in GitHub <https://help.github.com/en/github/administering-a-repository/creating-releases>`_ or you may directly
add `tags to your docker image <https://docs.docker.com/engine/reference/commandline/tag/>`_.  Additionally, there are many
`features in GitHub <https://github.com/features>`_ that help you track, review and ready version changes for code
for deployment.  There are `version control strategies specific to AI applications <https://medium.com/ibm-watson/a-version-control-strategy-for-ai-applications-f421d5b91934>`_ as well.

Data changes
^^^^^^^^^^^^^^^^^^^^^

It is worth noting that performance drift can arise from changes in the data itself, and it can be anticipated by directly monitoring the
features in the data.  There are several methods that can be used to compare an established distribution to a new one,
e.g., from a new batch of training data.  It is also possible that for a given use case there is a specific feature or two
that are of critical importance and checks on those features should be implemented as a step for quality assurance.  Two commonly used
methods to compare distributions are:

* `Kullback–Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
* `Wasserstein_metric <https://en.wikipedia.org/wiki/Wasserstein_metric>`_

We will show in the following screencast how to implement performance monitoring at the level of evaluation metric using
a model-based approach.  This example could serve as a template to add more granular feature-level monitoring.

.. admonition:: SCREENCAST

    * Adding logging to the API
    * Create a script for drift detection

.. raw:: html

    <iframe src="https://player.vimeo.com/video/87110435" width="640" height="360"  frameborder="0" allowfullscreen></iframe>

--------------------------

.. admonition:: CFU

    True/False.  When logging for the predict endpoint, `runtime` is considered an optional feature to be monitored.

    .. container:: toggle

        .. container:: header

            **True/False**

        **ANSWER**:

            **(False)**  The runtime, or time it takes to return a prediction, should be monitored
            over time.  In general, any factor that directly affects the end user's experience, should be considered an
            essential feature to be monitored through log files.
