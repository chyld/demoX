.. advanced plotting


Matplotlib - animations
================================

Sometimes a plot is not enough we need to tell a story with our plot.
A series of plots might cut it but really animations are just that---
a series of plots.  There are a number of examples of `matplotlib animations <http://matplotlib.org/examples/animation/index.html>`_ and we will go through a few to illustrate the underlying themes.

If we run an animation example (`plt.show()`) then a matplotlib plot instance shows up and we see the animation.  This is fine if we create something for our selves but it is good to know how to save these animations as videos.  You will need `the encoding library FFmpeg <https://ffmpeg.org>`_ installed to make this happen.

On Ubuntu:
  sudo apt-get install ffmpeg 

On OSX:
  brew install ffmpeg  

You will see that in the next example we are setting things like frames per second (fps) and bitrate--bet you did not think you would be making movies! Lets get started with this one.


.. literalinclude:: scripts/lines_animation.py
		      
.. raw:: html

  <video controls src="_static/lines.mp4"></video>

There is also the `moviewriter way to do this <http://matplotlib.org/examples/animation/moviewriter.html>`_ in case you cannot get this methods working for your example.

As a side note if you want to embed this into a jupyter notebook.  Here is a page that `shows you how <http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/>`_.  Basically, it is just the following tow lines added to your notebook.

>>> from IPython.display import HTML
>>> HTML(ani.to_html5_video())

.. topic:: time-series animation

	   "I really like the way plot.ly makes time series plots so I would like you to know how to use them.
	   However, sometimes I want to watch the time series progress rather than having a global view.
	   Can you please come up with something that can animate a time-series plot for me."

Hints:
	   
  * use the following `plot.ly time series plot <https://plot.ly/pandas/time-series/>`_
  * do not get too caught up with plot.ly, but know it is a great visualization tool
  * Use the same time-series data as used in the example
  * np.sin is just a function
  * `the subplots animation <http://matplotlib.org/examples/animation/subplots.html>`_ might be helpful

Resources and notable examples
---------------------------------

  * `Decent blog post with more details <https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/>`_
  * `Animating the Schrodinger Equation <http://jakevdp.github.io/blog/2012/09/05/quantum-python>`_
  * `Bayes update <http://matplotlib.org/examples/animation/bayes_update.html>`_
  * `Histogram <http://matplotlib.org/examples/animation/histogram.html>`_
