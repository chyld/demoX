.. cnrs, created by ARichards

******************
Multiprocessing
******************

Once you have finished with the **multiprocessing** unit we expect that you will be able to:

+------+------------------------------+-------------------------------------------------------------------------+
| 1    | Multiprocessing              | Use the multiprocessing module to parallelize operations in python      |
+------+------------------------------+-------------------------------------------------------------------------+


In Python
=============

.. code-block:: python

   from multiprocessing import Pool, cpu_count
   total_cores = cpu_count()
   print('total cores: ', total_cores)

.. code-block:: none

    total cores: 8


Why run code in parallel?
---------------------------

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

When to go parallel
######################

   * Sometimes it is difficult to make code more efficient otherwise
   * Sometimes it is `embarrassingly parallel <http://en.wikipedia.org/wiki/Embarrassingly_parallel>`_
   * Try to think about future development
   * Sometimes we (think that we) only need to run the code once

Examples of embarrassingly parallel applications:

   * Multiple chains of MCMC
   * Bootstrap for confidence intervals
   * Power calculations by simulation
   * Permutation-resampling tests
   * Fitting same model on multiple data sets
   * Distance matrices

This is a package in the standard python library. Here is the `documentation <https://docs.python.org/3.8/library/multiprocessing.html>`_.
For some problems it can avoid the hassle of chunking your analysis into wedges and reassembling the parts.

.. code-block:: python
   
   from multiprocessing import Pool, cpu_count
   totalCores = cpu_count()
   print totalCores


Using the futures object with multiprocessing
===============================================

When you have many jobs:

The `futures` object gives fine control over the process, such as adding
callbacks and canceling a submitted job, but is computationally
expensive. We can use the `chunksize` argument to reduce this cost when
submitting many jobs.


.. code-block:: python

   import numpy as np
   from multiprocessing import pool
   from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

   def f(x):
       return x*x

   with ProcessPoolExecutor(max_workers=4) as pool:
       result = pool.map(f, range(10))
       result = np.array(list(result))
       print(result)

Note that the methods of a pool should only ever be used by the process which created it.

Threading
=============

If you are looking for the same functionality but at a lower level see
the `threading module <https://docs.python.org/3.8/library/threading.html>`_.

Threading for subprocessing
------------------------------

Here is class shell that you can use to control a subprocess with threads.  Scroll to the bottom to see how to use it.

.. literalinclude:: ./scripts/run-subprocess.py

.. admonition:: Assignment

    Given what you have seen above can you use multiprocessing to speed-up the original `great_circle` function?


.. important::

    For those of you who want to dig deeper into high performance computing, continue into the remaining sections.
    Multiprocessing is the most important section and it was the main learning objective.
