.. r into (powerbayes)

**************
Python Basics
**************

Learning objectives
=====================

+---------+-------------------------------------------------------------------------------------------------------+
| 1       | Work with variables and different data types in the Python environment                                |
+---------+-------------------------------------------------------------------------------------------------------+

.. tip::
    Open a Python environment and follow along as you proceed through this unit.  If you want to take it a step further
    you could use a Jupyter notebook (to take notes) where you alternate code and text cells.

Getting Help
=================

From an interactive Python session you can use the ``help`` command to, for example, learn more about the ``sum()``
function.

.. code-block:: python

    >>> help(sum)

We have mentioned the word *function* and we will return to functions, but some description will help lay the groundwork.
In Python you may create or you may use pre-defined blocks of code that perform a specific function.  The function
``sorted()`` for example is built-in to Python and it is used to sort an iterable.  An iterable in Python is a
data container like lists and sets (that we will discuss in this unit).  Both data containers and functions will prove
to be quite useful, but first let's dive into the different data types that are available in Python.

The documentation that you see from within the Python environment comes from the
`official Python documentation <https://docs.python.org/3/>`_.

Data types
=================

A `data type <https://en.wikipedia.org/wiki/Data_type>`_ or simply called a ``type`` in the context of computer
programming is a notion exists across programming langauages.  More specifically we are discussing in this section
`primitive data types <https://en.wikipedia.org/wiki/Primitive_data_type>`_, that is the data types that are fundamental
to a language implementation.

Numerics, integers, logicals and characters
---------------------------------------------------------

Variables in Python are assigned with the equal sign (``=``).

>>> x == 4

Try out the following code block and notice how we use the built-in functions ``class`` and ``isinstance``.

.. code-block:: r
   
   > x <- 4
   > type(x)
   > isinstance(x,int)

The output here implies that by default when you assign a number to a variable it is of a *numeric* type.  This makes
sense as the default since we often want to perform division and other operations on our variables and
`integer division <http://mathworld.wolfram.com/IntegerDivision.html>`_ is often not what is intended.

Additional Resources
=========================

    * `Python Getting started tutorial <https://docs.python.org/3/tutorial/>`_
