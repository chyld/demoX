.. course title

*************************************   
Introduction
*************************************

The company `NVIDIA® <https://www.nvidia.com/en-us/>`_. has been working hard
`to accelerate machine learning and analytics <https://developer.nvidia.com/machine-learning>`_ by leveraging GPUs.  It
is no secret that NVIDIA exists to sell
`Graphics processing units <https://en.wikipedia.org/wiki/Graphics_processing_unit>`_ (GPUs) and the company has for
sometime now been an important contributor to both `open-source projects <https://developer.nvidia.com/open-source>`_ and
to `scientific computing in general <https://developer.nvidia.com/hpc>`_.  One interesting example of how NVIDIA has
contributed to scientific technology and tooling while simultaneously promoting the sale of GPUs is in their initiatives
to help `advance the state of  COVID-19 research <https://blogs.nvidia.com/blog/2020/06/22/fighting-covid-19-scientific-computing/>`_.

RAPIDS and the rest of the developer tools that have been created to help data professionals were born out of this same
mindset and the result is a powerful ecosystem that is well-integrated with the
`PyData ecosystem <https://stackoverflow.com/questions/18168400/the-pydata-ecosystem/20822664>`_.

Learning Objectives
======================

By the end of this lesson you will be able to:

+------+------------------------------+----------------------------------------------------------------------------------------+
| Unit | Main Topics                  | Learning Objectives                                                                    |
+======+==============================+========================================================================================+
| 1    | Getting Started              | Describe the main components of the RAPIDS ecosystem and install a working environment |
+------+------------------------------+----------------------------------------------------------------------------------------+
| 2    | Analytics                    | Describe the use cases of and apply cuDF                                               |
+------+------------------------------+----------------------------------------------------------------------------------------+
| 3    | Machine Learning             | Describe the use cases of and apply cuML and cuSKL                                     |
+------+------------------------------+----------------------------------------------------------------------------------------+
| 4    | end-to-end example           | Implement and end-to-end data science pipeline using RAPIDS                            |
+------+------------------------------+----------------------------------------------------------------------------------------+

|

.. figure:: ./images/nvidia-logo.png
   :scale: 60%
   :align: center
   :alt: galvanize-logo
   :figclass: align-center

|

NVIDIA created RAPIDS in a way that makes use of CUDA primitives for low-level compute optimization.  The developers at
NVIDIA and other contributors have abstracted away most of the implementation details and the result is a number of
interfaces that are both intuitive and aligned with the norms of the Python data science ecosystem.

Useful Resources
=================

* `Medium blog for RAPIDS <https://medium.com/rapids-ai>`_
* `NVIDIA developer news for RAPIDS <https://news.developer.nvidia.com/?search_theme_form=rapids&s=rapids>`_
* `NVIDIA developer blogs <https://developer.nvidia.com/blog/?search_theme_form=rapids>`_
* `RAPIDS manuscript on arxiv <https://arxiv.org/pdf/2002.04803.pdf>`_


