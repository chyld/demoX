.. learning objectives

.. figure:: ../images/galvanize-logo.png
   :scale: 100%
   :align: center
   :alt: ibm-logo
   :figclass: align-center


Data Visualization
====================================

Once you have finished with this unit you will be able to:

+-------+---------------------+-----------------------------------------------------------------------------------+
| 1     | Data visualization  | Explain the use case for Python tools (pandas, matplotlib, and Jupyter) in EDA    |
+-------+---------------------+-----------------------------------------------------------------------------------+

|

.. raw:: html

    <iframe src="https://player.vimeo.com/video/87110435" width="600" height="400" frameborder="0" allowfullscreen></iframe>

|

Data visualization is one the most creative outlets that data scientists have.  At the end of the day the results of a model, or
the results of EDA, are visualizations surrounded by text to provide context.  The combination of well-constructed visualizations
with annotating text is an incredibly powerful tool to communicate insight.  

Data visualization is arguably the most important tool for communicating your results to others, especially business stakeholders.
To convey results, patterns in the data, model comparisons and so much more visualization will always be the greatest enabler
of effective communication.  Data visualization includes tools like tabular summaries and data plotting.  In this unit we will
discuss descriptive tabular tools such as aggregate groupbys, but the principal focus will be plotting.  

.. important::

   For some business opportunities, a sound solution can be easily identified through thoughtful data visualization.
   Models are not always a part of the data science workflow.

|

Unit Materials
---------------

   * :download:`world-happiness dataset <./data/world-happiness.csv>`
   * :download:`pdf version of slides for data visualization <../images/galvanize-logo.png>`

     
Data visualization in Python
-----------------------------

It is expected that you already know and use both pandas and matplotlib on a regular basis. For those of you who are comfortable with plotting---meaning that you can readily produce any of the several dozen types common plots used in data science, then this unit will serve as a reminder to keep the best practices in mind.  This module is not a comprehensive overview of the data visualization landscape.  We haved touched on essential tools and best practices.  If necessary, use the resources below to improve your data visualization skills and provide more context.

If you would like additional context a few links are available below:

* `Anaconda's article on 'moving toward convergence <https://www.anaconda.com/blog/developer-blog/python-data-visualization-moving-toward-convergence/>`_
* `Anaconda's article on the future of Python visualization libraries <https://www.anaconda.com/blog/developer-blog/python-data-visualization-2018-where-do-we-go-from-here>`_
* `Tutorials for matplotlib <http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html>`_

Best practices
^^^^^^^^^^^^^^^^
  
Best practices as a data scientist generally require that all work be saved as text files:

   1. `simple scripts <https://docs.python.org/3/using/cmdline.html>`_
   2. `modules <https://docs.python.org/3/tutorial/modules.html>`_
   3. `Python package <https://www.pythoncentral.io/how-to-create-a-python-package>`_

Remember to save a maximum amount of code within files, even when using Jupyter. Version control is a key component to effective collaboration and reproducible research.

* `Nature article on Git and scientific reproducibility <http://blogs.nature.com/naturejobs/2018/06/11/git-the-reproducibility-tool-scientists-love-to-hate/>`_

Essentials of matplotlib
^^^^^^^^^^^^^^^^^^^^^^^^^
Matplotlib has a "functional" interface similar to MATLAB® that works via the pyplot module for simple interactive use, as well as an object-oriented interface that is more pythonic and better for plots that require some level of customization or modification. The latter is called the artist interface. There is also built in functionality from within pandas for rapid access to plotting capabilities.

* `matplotlib tutorial on lifecycle of a plot <https://matplotlib.org/3.1.1/tutorials/introductory/lifecycle.html>`_
* `pandas visualization <pandas visualization>`_
* `matplotlib pyplot interface <matplotlib pyplot interface>`_
* `matplotlib artist interface <https://matplotlib.org/users/artists.html>`_

The video in this module only touches on a couple types of plots and yet there are dozens that are commonly used.  Here
are a few examples of other essential data science plots.

* `swarm plot <https://seaborn.pydata.org/generated/seaborn.swarmplot.html>`_
* `joint plot <https://seaborn.pydata.org/generated/seaborn.jointplot.html>`_
* `violin plot <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.violinplot.html>`_
* `contour plot <https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_demo.html#sphx-glr-gallery-images-contours-and-fields-contour-demo-py>`_
* `distribution plots <https://seaborn.pydata.org/tutorial/distributions.html>`_

Two scales
-----------------------

Sometimes we wish to use multiple scales in a single plot.  This can be used to help understand relationships between featuresor a bi-variate target.

.. literalinclude:: scripts/two_scales.py  

.. plot:: scripts/two_scales.py


  
Other visualization tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Bokeh <https://bokeh.pydata.org/en/latest/index.html>`_
   Bokeh supports streaming, real-time data and is used to create interactive, web-ready plots, which can output as JSON objects, HTML documents, or interactive web applications.
`plotly <https://plot.ly/>`_
   While plotly is widely known as an online platform for data visualization, it can be accessed from a Python notebook. Like Bokeh, Plotly’s strength lies in making interactive plots, and it offers some charts not found in most libraries, like contour plots.
`dash <https://plot.ly/dash/>`_
   Dash is Python framework for building web applications. It built on top of Flask, Plotly.js, React and React Js. It enables you to build dashboards using pure Python. 
`ggplot <http://ggplot.yhathq.com/>`_
   ggplot is a python visualization library based on R’s ggplot2 and the Grammar of Graphics. It lets you construct plots using high-level grammar without thinking about the implementation details.
`folium <https://python-visualization.github.io/folium/>`_
   based on leaflet.js it is a great tool for working with geographic data for example choropleth visualizations
`Analytics dashboard in IBM Watson Studio <https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/analytics-dashboard.html>`_
   You can build sophisticated visualizations of your analytics results, communicate the insights that you’ve discovered in your data on the dashboard and then share the dashboard with others.

----------------------------------------------------

.. admonition:: CFU

   Which of the following is an example of a data manipulation that is **NOT** considered reproducible research?

   .. container:: toggle

      .. container:: header

         * **(A)**: Saving classes and functions in a Python file to be called by Jupyter
         * **(B)**: Code blocks in Jupyter notebooks
         * **(C)**: The use of proprietary tools to carry out research
         * **(D)**: Graphics, plots and other visualizations
         * **(E)**: Copy and paste actions in a spreadsheet

      **ANSWER**:

         **(E)** Any manipulation in a spreadsheet tool is considered **NOT** reproducible.  Code and even better readable
         code in the form of text files that can be tracked with version control is reproducible.  Code blocks in Jupyter
         do not work well with version control, but the notebooks themselves can be shared and a history of manipulations
         can be constructed.  Proprietary tools are used all of the time in research.  Knowing the version of the tool and
         the sequence of manipulations ideally through code helps with reproducibility.  Plots and other visualization tools
         also can be created with code in very reproducible ways.


.. admonition:: CFU

   True/False. The seaborn pairplot and other seaborn plotting functions exist as distinct tools from the plots available through matplotlib.

      .. container:: toggle

         .. container:: header

            **True/False**

         **ANSWER**:

	    **(False)** seaborn was
	    `built on top of matplotlib <https://seaborn.pydata.org/introduction.html#introduction>`_. Often you can pass
	    a matplotlib axes object directly into seaborn functions.  There is also some custom functionality
            to interface with the underlying matplotlib objects.

..
