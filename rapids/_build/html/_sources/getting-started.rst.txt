.. name of course

*************************************   
Getting Started
*************************************

+------+------------------------------+----------------------------------------------------------------------------------------+
| 1    | Getting Started              | Describe the main components of the RAPIDS ecosystem and install a working environment |
+------+------------------------------+----------------------------------------------------------------------------------------+


We will be installing an environment to leverage NVIDIA `RAPIDS <https://developer.nvidia.com/rapids>`_, an open-source
suite of software libraries, built on `CUDA-X <https://www.nvidia.com/en-us/technologies/cuda-x>`_.  Historically,
At the core of this system is the `parallel computing <https://en.wikipedia.org/wiki/Parallel_computing>`_ platform,
`CUDA® <https://en.wikipedia.org/wiki/CUDA>`_.

CUDA-X has two main library variants: CUDA-X AI and CUDA-X-HPC.  The term CUDA-X uses a number of
`GPU accelerated libraries <https://developer.nvidia.com/gpu-accelerated-libraries>`_ as building blocks.  Some of the
most important ones are:

|

`cuBLAS <https://developer.nvidia.com/cublas>`_
    The linear algebra library BLAS.

`CUDA Math Library <https://developer.nvidia.com/cuda-math-library>`_
    Basic math functions and some statistics

`cuRAND <https://developer.nvidia.com/curand>`_
    Random number generation

`cuSPARSE <https://developer.nvidia.com/cusparse>`_
    BLAS for sparse matrices

`cuDNN <https://developer.nvidia.com/cudnn>`_
    Library of primatives for deep neural networks

`NVIDIA Jarvis <https://developer.nvidia.com/nvidia-jarvis>`_
    Platform for developing AI powered conversation apps

To use the above libraries much of the code needs to be written in C/C++.  The packages
`PyCUDA <https://documen.tician.de/pycuda/>`_ and `scikit-cuda <https://scikit-cuda.readthedocs.io/en/latest/>`_ are
reasonable ways to access these libraries directly from the Python ecosystem.  However, it is possible that some of your code
(called the kernel) would still need to be in C++. It is good to know that these libraries are accessible, but
much of the day-to-day data science that needs to be carried out requires that most of these libraries function
*under the hood*.  RAPIDS was created with users in mind that are one level removed from this level of coding abstraction.

Installing RAPIDS locally
=============================

See the `RAPIDS getting started documentation <https://rapids.ai/start.html>`_.  If you do not have access to a GPU
the skip this section and begin with the `Installing RAPIDS in Google Colab`.

.. warning::

    RAPIDS can only be installed on Ubuntu and CentOS systems

.. important::

    RAPIDS no longer uses PIP and the base install
    `now uses conda <https://medium.com/rapids-ai/rapids-0-7-release-drops-pip-packages-47fc966e9472>`_

Enusre that you refer to the above documentation for the latest information on installation.
If you do not have one of these operating systems installed then you should skip to the Google Colab section as well.

Prerequisites
-----------------

* GPU: NVIDIA Pascal™ or better with compute capability 6.0+
* OS: Ubuntu 16.04/18.04/20.04 or CentOS 7 with gcc/++ 7.5+
* Docker: Docker CE v19.03+ and nvidia-container-toolkit
* CUDA & NVIDIA Drivers: One of the following supported versions:

    10.0 & v410.48+    10.1.2 & v418.87+    10.2 & v440.33+

The following script can help you determine what is installed on your system.

.. literalinclude:: ./scripts/check-system.sh
     :language: bash

.. code-block:: none

    Distributor ID:	Ubuntu
    Description:	Ubuntu 20.04 LTS
    Release:	20.04
    Codename:	focal
    Python 3.7.8
    Docker version 19.03.12, build 48a66213fe
    01:00.0 VGA compatible controller: NVIDIA Corporation GP104BM [GeForce GTX 1070 Mobile] (rev a1)
    driver_version
    440.100
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Sun_Jul_28_19:07:16_PDT_2019
    Cuda compilation tools, release 10.1, V10.1.243

.. important::

    If you are missing CUDA, conda can install it for you, but you will need the NVIDIA driver installed before running
    the next command.

If you install with conda it will likely look something like the following, but again use the
`RAPIDS getting started documentation <https://rapids.ai/start.html>`_ to perform your install ensuring that you
construct the install args correctly.

.. code-block:: bash

    ~$ docker pull rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04
    ~$ docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04


See the `RAPIDS getting started documentation <https://rapids.ai/start.html>`_ for more information on other variants.

Installing RAPIDS in Google Colab
=====================================

.. code-block::

    # Install RAPIDS
    !git clone https://github.com/rapidsai/rapidsai-csp-utils.git
    !bash rapidsai-csp-utils/colab/rapids-colab.sh stable

    import sys, os

    dist_package_index = sys.path.index('/usr/local/lib/python3.6/dist-packages')
    sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.6/site-packages'] + sys.path[dist_package_index:]
    sys.path
    exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())


.. admonition:: TODO

   Need to complete this with our own Colab notebooks


A hello world
==================

The data are available from as a traffic accident data set.

* https://www.kaggle.com/sobhanmoosavi/us-accidents/data

.. literalinclude:: ./scripts/hello-world.py
     :language: python

.. code-block:: none

    Distance(mi)          0.281617
    Precipitation(in)     0.015983
    Temperature(F)       61.935119
    Wind_Speed(mph)       8.219025
    Severity              2.339929
    0:00:23

    Distance(mi) 0.28
    Precipitation(in) 0.02
    Temperature(F) 61.94
    Wind_Speed(mph) 8.22
    Severity 2.34
    0:00:02

Ecosystem Overview
======================

.. figure:: ./images/cuda-x-ai-ecosystem.png
   :scale: 50%
   :align: center
   :alt: galvanize-logo
   :figclass: align-center


RAPIDS is capable of executing end-to-end data science and analytics pipelines entirely on GPUs and in this lesson
we will survey the available tooling, provide some context for how the different tools work, get hand-on with several
examples of this exciting addition to the data science ecosystem.

Key features
------------------

1. **Hassle-Free Integration** - Accelerate your Python data science toolchain with minimal code changes and no new tools to learn
2. **Model Accuracy** - Increase machine learning model accuracy by iterating on models faster and deploying them more frequently
3. **Reduced Training Time** - Reduce iteration time on your development workflow
4. **Open Source** - Customizable, extensible software supported by NVIDIA and built on `Apache Arrow <https://arrow.apache.org/>`_

End-to-end data science
-------------------------

NVIDIA makes the claim that RAPIDS is an end-to-end data science platform.  See
the `GPU accelerated blog <https://developer.nvidia.com/blog/gpu-accelerated-analytics-rapids/>`_.  The following figure
makes the claim that the data professional can spend more time in analysis mode.

.. figure:: ./images/a-day-in-the-life.png
   :scale: 50%
   :align: center
   :alt: galvanize-logo
   :figclass: align-center

.. important::

    There is a trade-off between the speed of our code and the quality of our code when we can go from idea to minimal
    viable product.  Is it worth it to spend our initial development iterations on an idea using RAPIDS?


.. figure:: ./images/rapids-pipeline.png
   :scale: 50%
   :align: center
   :alt: galvanize-logo
   :figclass: align-center

Component libraries
--------------------------

`cuDF <https://github.com/rapidsai/cudf>`_:
    A GPU DataFrame library with a pandas-like API. cuDF provides operations on data columns including unary and binary
    operations, filters, joins, and groupbys. cuDF currently comprises the Python library PyGDF, and the C++/CUDA GPU
    DataFrames implementation in libgdf. These two libraries are being merged into cuDF. See the documentation for more
    details and examples.
cuSKL:
    A collection of machine learning algorithms that operate on GPU DataFrames. cuSKL enables data scientists, researchers,
    and software engineers to run traditional ML tasks on GPUs without going into the details of CUDA programming from Python.
XGBoost:
    XGBoost is one of the most popular machine learning packages for training gradient boosted decision trees. Native
    cuDF support allows you to  pass data directly to XGBoost while remaining in GPU memory.
cuML:
    a GPU-accelerated library of machine learning algorithms including Singular Value Decomposition (SVD), Principal
    Component Analysis (PCA), Density-based Spatial Clustering of Applications with Noise (DBSCAN).
ml-prims
    A library of low-level math and computational primitives used by cuML.


Partners
------------

.. figure:: ./images/rapids_community.png
   :scale: 50%
   :align: center
   :alt: galvanize-logo
   :figclass: align-center


Dask integrates well with RAPIDS, but what about Spark.  See the
`Dask compared to Spark document for more info <https://docs.dask.org/en/latest/spark.html>`_


Check for Understanding
===========================


.. admonition:: QUESTION 1

    placeholder, placeholder

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

* `RAPIDS getting started documentation <https://rapids.ai/start.html>`_

