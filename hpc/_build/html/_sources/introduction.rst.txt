.. hpc

***************
Introduction
***************

There is an appropriate sequence to creating programs:

   1. Make it work
   2. Ensure it is right
   3. Make it fast

Concentrating on the last step before the previous two can result in significantly more total work.  Sometimes our
programs are fast enough and we do not even need the last step.  If we have some evidence that our programs work and
that the details are correct then the following sections should help you make it faster.

There are plenty of examples where a well-written script addresses a specific need even without the use of
machine learning.  What if we were interested in optimizing how maintenance crews are deployed to jobs or how packages
need to be delivered for a given day?  This would both be variants on the
the `traveling salesman problem <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_ where you could write
a brute-force algorithm or use some variant on a
`minimal spanning tree <https://en.wikipedia.org/wiki/Minimum_spanning_tree>`_ to solve it. Either way, these are
both very useful tools that do no rely on machine learning algorithms.

At a higher level there are two important areas of data science where a trove of these algorithms can be found.

1. `optimization <https://en.wikipedia.org/wiki/Mathematical_optimization>`_
2. `graph theory <https://en.wikipedia.org/wiki/Graph_theory>`_

These tools do not rely on an in-depth knowledge of models or machine-learning they require an awareness of algorithmic
solutions.

The first rule, **make it work**, is meant to ensuring that you are optimizing your code in a smart way.  You should
look around for implementations before spending significant amounts of time on creating one.  The
`scipy.optimize <https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html>`_ submodule has a number of
optimizers and algorithms (some of them general purpose) already implemented.  If your problem is in graph space like
`customer journeys <https://en.wikipedia.org/wiki/Customer_experience>`_ or social networks then check out the
`algorithms implemented by NetworkX <https://networkx.github.io/documentation/stable/reference/algorithms/index.html>`_
before you set off building your own.

Finally, we come to the scripts or blocks of code that need speed improvements, but you have come to
the conclusion that there is no optimized code readily available.  The task of optimizing the code then falls to you.
The first step is to identify which parts of your code are bottlenecks.  This is done using
`profiling <https://en.wikipedia.org/wiki/Profiling_(computer_programming)>`_ or more specifically `Python
profilers <https://docs.python.org/3/library/profile.html>`_.  Once the specific pieces of code that
need to be optimized are identified, then there are a number of common tools that may be used to improve the speed of
programs.  Several of these tools make use of the fact that modern computers have multiple available processor cores on
a machine.  To see how many processor cores are available on your machine or compute resource try the following code.

A list of commonly used techniques and tools to optimize code:

Use appropriate data containers
    For example, a Python set has a shorter look-up time than a Python list.  Similarly, use dictionaries and NumPy
    arrays whenever possible.

`Multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_
    This is a package in the standard Python library and it supports spawning processes (for each core) using an API
    similar to the threading module. The multiprocessing package offers both local and remote concurrency.

`Threading <https://docs.python.org/3/library/threading.html#module-threading>`_
    Another package in the standard library that allows separate flows of execution at a lower level than
    multiprocessing.

`Subprocessing <https://docs.python.org/3/library/subprocess.html>`_
    A module that allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
    You may run **and control** non-Python processes like Bash or R with the subprocessing module.

`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_
    MPI for Python provides bindings of the Message Passing Interface (MPI) standard for the Python programming
    language, allowing any Python program to exploit multiple processors.

`ipyparallel <https://ipyparallel.readthedocs.io/en/latest/>`_
    Parallel computing tools for use with Jupyter notebooks and IPython.  Can be used with mpi4py.

`Cython <https://cython.org/>`_
     An optimizing static compiler for both the Python programming language and the extended Cython programming language
     It is generally used to write C extensions for slow portions of code.

`CUDA (Compute Unified Device Architecture) <https://en.wikipedia.org/wiki/CUDA>`_
    Parallel computing platform and API created by `Nvidia <https://www.nvidia.com/en-us/>`_ for use with CUDA-enabled
    GPUs.  CUDA in the Python environment is often  run using the package `PyCUDA <https://documen.tician.de/pycuda/>`_.


Why run code in parallel?
============================

There are numerous reasons to run your code in parallel

   * Modern computers have multiple cores and `hyperthreading <https://en.wikipedia.org/wiki/Hyper-threading>`_
   * Graphics processing units (GPUs) have driven many of the recent advancements in data science
   * Many of the newest *i7* processors have 8 cores
   * The is a lot of **potential** but the overhead can be demanding for some problems
   * When we call a python script e.g.

     .. code-block:: bash

         python run.py

   only a single core is dedicated to this process by default.
   `Parallel computing <https://en.wikipedia.org/wiki/Parallel_computing>`_ can help us make better use of the
   available hardware.

How many cores do I have?

.. code-block:: python

   from multiprocessing import Pool, cpu_count
   total_cores = cpu_count()
   print('total cores: ', total_cores)

.. code-block:: none

    total cores: 8

.. hint::

    If you are on a Linux or MacOS machine try using **htop** (installed via brew or apt)
   
So if we have 32 cores... the time taken in (HH:MM:SS)

+----------------------------------+------------------------------+
| 1 core                           | 32 cores                     |
+==================================+==============================+
| 1 minute (60s)                   | 00:00:02                     |
+----------------------------------+------------------------------+
| 1 hour (3,600s)                   | 00:01:52                    |
+----------------------------------+------------------------------+
| 1 Day (86,400s)                   | 00:45:00                    |
+----------------------------------+------------------------------+
| 1 Month (2,628,288s)               | 22:48:54                   |
+----------------------------------+------------------------------+
| 1 Year (3.156e7s)                | 11 days, 09:45:00            |
+----------------------------------+------------------------------+

Of course that is in an ideal world.  We still have to consider read/write operations memory allocation and all the
other procedural overhead.

High-Performance Computing
============================

We mentioned in the previous section that inference can be difficult to optimize and that one way around this is to add
more GPUs.  The general idea of using an aggregation of compute resources to dramatically increase available compute
resources is known as high-performance computing (HPC) or `supercomputing <https://en.wikipedia.org/wiki/Supercomputer>`_.
Within this field there is the important concept of `parallel computing <https://en.wikipedia.org/wiki/Parallel_computing>`_,
which is exactly what we enable by adding multiple GPUs to compuation tasks.

Supercomputers and parallel computing can help with model training, prediction and other related tasks, but it is worth
noting that there are two laws that constrain the maximum speed-up of computing:
`Amdahl's law <https://en.wikipedia.org/wiki/Amdahl%27s_law>`_ and
`Gustafson's law <https://en.wikipedia.org/wiki/Gustafson%27s_law>`_.  Listed below is some of the important terminology
in this space.

`Symmetric multiprocessing <https://en.wikipedia.org/wiki/Symmetric_multiprocessing>`_
    Two or more identical processors connected to a single unit of memory.
`Distributed computing <https://en.wikipedia.org/wiki/Distributed_computing>`_
    Processing elements are connected by a network.
`Cluster computing <https://en.wikipedia.org/wiki/Computer_cluster>`_
    Group of loosely (or tightly) coupled computers that work together in a way that they can be viewed as a single system.
`Massive parallel processing <https://en.wikipedia.org/wiki/Massively_parallel>`_
    Many networked processors usually > 100 used to perform computations in parallel.
`Grid computing <https://en.wikipedia.org/wiki/Grid_computing>`_
    distributed computing making use of a middle layer to create a `virtual super computer`.

An important part of this course is dealing with data at scale, which is closely related to both code
optimization and parallel computing.

If we talk about scale in the context of a program or model, we may be referring to any of the following questions.
Let the word **service** in this context be both the deployed model and the infrastructure.

* Does my service train in a reasonable amount of time given a lot more data?
* Does my service predict in a reasonable amount of time given a lot more data?
* Is my service ready to support additional request load?

It is important to think about what kind of scale is required by your model and business application in terms of which
bottleneck is most likely going to be the problem associated with scale.  These bottlenecks will depend heavily on
available infrastructure, code optimizations, choice of model and type of business opportunity.  The three questions
above can serve as a guide to help put scale into perspective for a given situation.

Additional resources
=======================

* `scipy-lectures tutorial for optimizing code <https://scipy-lectures.org/advanced/optimizing/index.html>`_
* `mpi4py tutorial <https://mpi4py.readthedocs.io/en/stable/tutorial.html>`_
* `ipyparallel demos <https://ipyparallel.readthedocs.io/en/latest/demos.html>`_
* `Cython tutorial <https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html>`_
* `A talk about parallel computing in R (Luke Tierney) <http://homepage.stat.uiowa.edu/~luke/talks/uiowa03.pdf>`_

