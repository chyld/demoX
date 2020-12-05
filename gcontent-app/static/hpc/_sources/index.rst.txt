
.. hpc



###########################
High-Performance Computing
###########################

Learning objectives
-------------------

+------+------------------------------+-------------------------------------------------------------------------------------------+
| Unit | Main Topics                  | Learning Objectives                                                                       |
+======+==============================+===========================================================================================+
| 1    | Parallel Programming         | Describe some algorithms and data structures that contribute to fast code.                |
+------+------------------------------+-------------------------------------------------------------------------------------------+
| 2    | Multiprocessing              | Use the multiprocessing module parallelize operations in python                           |
+------+------------------------------+-------------------------------------------------------------------------------------------+
| 3    | Sparse Matrices              | Explain when to use sparse matrices during the machine learning model development process |
+------+------------------------------+-------------------------------------------------------------------------------------------+
| 4    | Subprocessing                | Use the subprocess module to optimize python and other types of code                      |
+------+------------------------------+-------------------------------------------------------------------------------------------+

There are so many ways to pull in additional compute power to speed up your machine learning through the use of remote
computing, be it AWS, another cloud service or an in-house cluster.
**This does not mean you have to scale that way**.
Before you invest time and money into additional compute resources, you should be sure that your code is fully
optimized for performance, first.

.. important::

    Throwing more hardware at a problem should not be the first step in an a code optimization process... the first
    step should be better code.

Modern computers have multiple cores and hyper-threaded processors now.  You may be surprised by how much more your
computer can do when you consider this and a few other things.  This course will help you better harness the potential
of your machine and it will provide a set of guidelines to help you scale using well-written and optimized code.

Data scientists have more *tools* available today than ever before to solve problems.  With the availability of more
efficient tools than ever before there is less of a focus on optimized code.  Also, data science has been increasing in
popularity so many practicing data scientists only have a few years on the job experience, which means that they are
less likely to be **aware** of the importance of code optimization.  This course will survey the best practices and
modern tools used to increase the speed of your code. For some business applications, like
`recommender systems <https://en.wikipedia.org/wiki/Recommender_system>`_, speed can be directly linked to revenue.

Table of Contents
------------------

.. toctree::
   :maxdepth: 1
   :caption: HPC

   introduction
   parallel-programming
   multiprocessing
   sparse-matrices
   subprocessing

.. toctree::
   :maxdepth: 1
   :caption: APPENDICES

   cython

	     
.. toctree::
   :maxdepth: 1
   :caption: LEARN
	     
   Lesson Dashboard <http://127.0.0.1:5000/dashboard>
   Checkpoints Overview <https://learn-2.galvanize.com>
