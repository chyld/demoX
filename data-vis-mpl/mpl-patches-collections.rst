.. advanced plotting

Matplotlib - patches and collections
========================================

Recall the two different ways we can interface with matplotlib.

  * `pylab interface <http://matplotlib.org/users/pyplot_tutorial.html#pyplot-tutorial>`_
  * `artist interface <http://matplotlib.org/users/artists.html#artist-tutorial>`_

    
Unfortunately, these interfaces are both used and not everyone can
agree on one being better than another.  Generally the **artist**
interface is preferred, because it is more explicit.  As we get into
interactive plotting we will see that it useful to use the more
succinct **pylab** interface.  Just as a reminder we have discussed a
number of plotting functions in MPL.

+-----------------+---------------------------+
| Command         | Description               |
+=================+===========================+
| plot            | plot lines and/or markers |
+-----------------+---------------------------+
| bar             | bar plot                  |
+-----------------+---------------------------+
| error bar       | error bar plot            |
+-----------------+---------------------------+
| boxplot         | boxplot                   |
+-----------------+---------------------------+
| histogram       | histogram                 |
+-----------------+---------------------------+
| pie             | pie charts                |
+-----------------+---------------------------+
| imshow          | heatmaps/images           |
+-----------------+---------------------------+
| scatter         | scatter plots             |
+-----------------+---------------------------+

This section will introduce some of the other commands MPL has to offer.

Patches
-------------------------------

Patches are used when we need access to shapes.  The  `patches documentation <http://matplotlib.org/api/patches_api.html>`_ is limited in how helpful it is, but looking at examples of code helps us understand the potential utility. 

.. literalinclude:: scripts/ellipses.py  

.. plot:: scripts/ellipses.py

Some of the most useful shapes to import from `matplotlib.patches` are: **Arrow**, **Circle**, **Ellipse**, **Polygon**, **Rectangle**, and **Wedge**.  Notice that here we are specifying the ellipses (`ells`) in a list then we add them to the axis individually.

Collections
-------------------------------

Looping through data adding lines or patches to an axes object can be tedious.  This is why `collections exists <http://matplotlib.org/api/collections_api.html>`_ in MPL.
	  
.. literalinclude:: scripts/ellipse_collection.py  

.. plot:: scripts/ellipse_collection.py

If you use collections then play with an easy example like this to get a better understanding.   Here we used the `EllipseCollection`, but there are others.  Again refer to `<the collections documentation http://matplotlib.org/api/collections_api.html>`_  to know more about which collections exist as well as what the arguments represent.

The following collections demo is a pretty good resource for a variety of collection types.

.. plot:: scripts/collections_demo.py

