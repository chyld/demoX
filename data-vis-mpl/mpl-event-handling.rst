.. advanced plotting

Matplotlib - event handling
================================

Sometimes we have a lot going on inside a data visualization and it
can be useful to interact with the plot itself in order to aid data
exploration efforts.

`Event handling in MPL
<http://matplotlib.org/users/event_handling.html>`_ starts to get into
an area that can quickly get complicated.  There are many different
interface toolkits: wxpython, tkinter, qt4, gtk, and macosx, but still
MPL can work with all of these interfaces.

**Go through the mentioned event handling webpage to understand how you can bind clicks, mouse movement and much more**.  This is one place in the MPL docs that is well done.

Also `you can use the examples <http://matplotlib.org/examples/event_handling/index.html>`_.

Binding to keypress events
----------------------------

This is basically the minimal working example.  Run it and click on the plot. 

.. literalinclude:: scripts/keypress_basics.py

.. plot:: scripts/keypress_basics.py
		    

Selecting points
-------------------------------

The following script plots a basic scatter of points, but when you run
it you can **draw** a lasso around an arbitrary group of points.  If you look at
your terminal after you have selected the points you will be asked to hit a key.
Once you do the points you drew are then printed.  

See the `lasso selector <http://matplotlib.org/examples/widgets/lasso_selector_demo.html>`_ demo
in the docs for more information.
    
The print statement after capturing some points is an example of event
handling, where that could be replaced by plot again or save to file
etc.

    
Creating plots on click
--------------------------

Imagine that you have a whole bunch of posteriors that you want to
compare.  As an example, maybe I am looking at the results of a model
that predicts radon levels in houses and the model accounts for an
effect at the level of county.  Well that would yield a posterior
distribution for each county. Instead of looking through potentially
hundreds of plots you could summarize the means in a scatter plot
standard deviations.  Ideally, when I click on a given point another
plot will pop up displaying the posterior for that point.

This is what the following code would accomplish.

.. literalinclude:: scripts/pick_event_demo2.py
	   
If you want to do this in as single plot with subplots you can use the `data browser example <http://matplotlib.org/examples/event_handling/data_browser.html>`_.
	   
Where do we go from here?
-----------------------------

If capturing events are not enough and you want to stay in the MPL
environment, then there are a couple of directions you can go.  First
you could create a widget (checkbuttons, sliders etc).  If a couple of
buttons and a slider will not suit your needs then you could embed
matplotlib in a fully featured user interface.  This is well beyond
the scope of what we are trying to accomplish here, but it is good to
know that working examples are maintained as part of MPL for a number
of toolkits.  Desktop application (i.e. user interfaces) have largely
been replaced by web-applications, but they do still exist.

My suggestion just to know what you can do if you wanted would be to
at download a couple of the examples from the widgets (e.g. slider
demo) and run them.

  * `widgets examples <http://matplotlib.org/examples/widgets/index.html>`_
  * `user interface examples <http://matplotlib.org/examples/user_interfaces/index.html>`_


Football games data example
-------------------------------

.. topic:: football fan

	   "You manager is a (American) football fan and she gets
           really excited when the season gets rolling.  She is also
           quite busy so she does not have time to follow every game,
           but she loves data visualization.  The goal here is to
           impress her with a visualization tool quickly that will
           allow you to navigate football game results as the season
           progresses."

Here are the results of 10 season's of football results 1981, 1983-1986, 1988-1992. The source for these data is `Andrew Gelman's book <http://www.stat.columbia.edu/~gelman/book/data>`_.
	   	   
This is an open ended question, but here are some ideas.

  * [easy] Modify **lasso example** to print to the terminal information about those games (just plot spreads)
  * [harder] Modify a **creating plots on a click** example to show spreads when you click on a team
  * [even harder] Plot the lasso example for teams ranked by number of wins then using the `data browser example <http://matplotlib.org/examples/event_handling/data_browser.html>`_ display additional win/loss/spread information with a click.
  * [hint] - You can limit your data to a given year to make it easier to work with

The data are available in the csv below.  If you are curious how to parse the data into a csv the script is below as well.

  * :download:`script-used-to-make-csv <./scripts/munge_football.py>`
  * :download:`football.csv <./data/football.csv>`
