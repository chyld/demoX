.. r into (powerbayes)

**************
R Basics
**************

Learning objectives
=====================

+---------+--------------------------------------------------------------------------------------------------+
| 1       | Work with variables and different data types in the R environment                                |
+---------+--------------------------------------------------------------------------------------------------+

.. tip::
    Open an R environment and follow along as you proceed through this unit.  If you want to take it a step further you
    could use a Jupyter notebook (to take notes) where you alternate code and text cells.

Getting Help
=================

Use the `help`, `args` and `example` functions.

.. code-block:: r

    > help(log10)

In place of ``help`` you may also use:

.. code-block:: r

   > ?log10

For a reminder on the arguments that a function might take:

.. code-block:: r

    > args(mean)

A convenient way to quickly recall how a function is used.

.. code-block:: r

    > example(median)

.. hint::

   If you want to see something that your are more likely to use in practice try ``example(glm)``.

We have mentioned the word *function* a couple of times.  In R you may create or you may use pre-defined blocks of
code that perform a specific function.  The function ``median`` for example is built-in to R and it is used to calculate
the median over a data container.  We will get back to functions and they will prove quite useful, but first let's
dive into the different data types in R, along with the different data containers.

R also provides a lot of documentation.  You can view the table of contents with:

.. code-block:: r

   > help.start()

The documentation that you see from within R environments comes from the
`official R documentation <https://cran.r-project.org/manuals.html>`_.

.. tip::

    It is particularly convenient to view documentation when from the RStudio environment since it appears in one of the
    windows as part of the GUI.

Data types
=================

A `data type <https://en.wikipedia.org/wiki/Data_type>`_ or simply called a ``type`` in the context of computer
programming is a notion exists across programming langauages.  More specifically we are discussing in this section
`primitive data types <https://en.wikipedia.org/wiki/Primitive_data_type>`_, that is the data types that are fundamental
to a language implementation.

Numerics, integers, logicals and characters
---------------------------------------------------------

Variables in R are assigned either with the arrow, ``<-`` or with ``=``, but the arrow is a convention so it should
be used in general.

.. code-block:: r

    x <- 4

Try out the following code block and notice how we use the built-in functions ``class`` and ``is.integer`` to

.. code-block:: r
   
   > x <- 4
   > is.numeric(x)
   > is.integer(x)
   > class(x)

The output here implies that by default when you assign a number to a variable it is of a *numeric* type.  This makes
sense as the default since we often want to perform division and other operations on our variables and
`integer division <http://mathworld.wolfram.com/IntegerDivision.html>`_ is often not what is intended.

There is an `integer <https://en.wikipedia.org/wiki/Integer>`_ (whole number) type in R and here we demonstrate with
the ``as.integer`` function how to *cast* from a *numeric* to an *integer* type.

.. code-block:: r

    > y <- as.integer(3.1)
    > z <- as.integer(TRUE)
    > class(FALSE)
    > z <- x > y


The two lines that involve the variable ``z`` introduce the `boolean <https://en.wikipedia.org/wiki/Boolean_data_type>`_
data type.  In R booleans are called *logicals*.  Note that the logical type can be cast to an integer and that logical
expressions return a logical that can then be assigned to variables (last line).

We have seen *numeric*, *integer* and *logical* types, but what of
`strings <https://en.wikipedia.org/wiki/String_(computer_science)>`_ ?  A string in R is known as a *character*.  These
are where we store words, descriptions, names and generally anything that is not a number or a boolean.

.. code-block:: r

   > as.character(99.9)

Printing variable contents
------------------------------

Often when we run a script or some lines of code we need to print to the screen some information.  There are several
ways of doing this with the most common being the function *print*.  In an interactive R environment there is a lot of
information printed regularly to the screen, but if your code lived in a file (also called a script) then you would need
to explicitly print information to the screen that you would like to see.  This is commonly done when displaying results
or debugging code.

.. code-block:: r

    > print(2+2)

It is often the case that the print function by itself is not flexible enough to display exactly what is needed.  So
it is commonly used along with the *paste* function.

.. code-block:: r

    > paste(2, 3, 5, 7, 11, 13, 17, sep = ":")
    > print(paste("the answer is:",2+2),sep = " ")

If you are familiar with C-style printing there is a wrapper for the C function ``sprintf`` that can be used.

.. code-block:: r

    > a <- paste("warm","bread",sep=" ")
    > sprintf("%s likes %s %d times a day", 'Omar', a, 2)

Another convenient way to print strings involve the sub and gsub functions.

.. code-block:: r

    > a <- "In the afternoon I enjoy black tea"
    > sub("black","green",a)

The ``sub()`` function will replace the first pattern it detects in a string.  The ``gsub()`` function will replace
all of the detected patterns globally.

.. code-block:: r

    > a <- "In the afternoon I enjoy black tea. My sister prefers black tea."
    > sub("black","green",a)
    > gsub("black","green",a)

Data containers
####################

It is common to aggregate primitive data types into data containers.

vectors
-------------

The function ``c()`` is used to combine values into a list or a vector. The default method combines the provided
elements to form a vector.

.. code-block:: r

   > x <- c(1,2,3,4)
   > is.list(x)
   > is.vector(x)
   > y <- c("a", "b", "c", "d", "e")
   > z <- c(x,y)

.. caution::

    All arguments are coerced to a common type which is the type of the returned value.

.. code-block:: r

    > x <- c(1,2,3,4)
    > class(x)
    > x <- c(1,2,3,'4')
    > class(x)
    > mode(x)

.. note::

    A data type when referring to containers in R is referred to as a *mode*.


Here are some different ways that vectors can be used.  Try some different examples to get used to the behavior.

.. code-block:: r

   > x <- c(1,2,3,4)
   > x + 5
   > y <- 1:4
   > x + y
   > u <- c(10, 20, 30)
   > v <- c(1, 3:10)
   > u + v

As a reference here are some of the arithmetic operators available to you in R.

+------------------+----------------------------------------------------------------------------------------------+
| Operator         | Description                                                                                  |
+==================+==============================================================================================+
| ``%*%``          | Matrix multiplication                                                                        |
+------------------+----------------------------------------------------------------------------------------------+
|  ``*``           | Elementwise multiplication                                                                   |
+------------------+----------------------------------------------------------------------------------------------+
|  ``%/%``         | Integer Division                                                                             |
+------------------+----------------------------------------------------------------------------------------------+
|  ``^`` or ``**`` | Power                                                                                        |
+------------------+----------------------------------------------------------------------------------------------+
|  ``%%``          | Modulus                                                                                      |
+------------------+----------------------------------------------------------------------------------------------+
|  ``outer()``     | Outer product                                                                                |
+------------------+----------------------------------------------------------------------------------------------+

.. important::

    Matrix operations follow the rules of linear algebra whereas array operations execute element by element
    operations.

Matrices
--------------

Data science is in many ways powered by `linear algebra <https://en.wikipedia.org/wiki/Linear_algebra>`_.  Linear
algebra is concerned with several types of mathematical objects including scalers, vectors, matrices and tensors.
Accordingly, it is very common to work with vectors and matrices in data science so you should be aware of the matrix
data container in R.

.. code-block:: r

   > A <- matrix(c(1, 2, 3, 4, 5, 6), nrow=2, ncol=3, byrow = TRUE)
   > A
   > A <- matrix(c(1, 2, 3, 4, 5, 6), nrow=2, ncol=3, byrow = FALSE)
   > A
   > dim(A)
   > A[2,3]
   > A[2,]
   > A[,c(1,3)]

The number of row and the number of columns are specified by the ``nrow`` and ``ncol`` arguments respectively.  This
can also be found with the ``dim()`` function.  The subsequent lines will help you understand how to access elements
of a matrix either individually or by slice.

.. warning::

    Python uses `zero-based numbering <https://en.wikipedia.org/wiki/Zero-based_numbering>`_ or indexing, but R uses
    a 1-based system. This means that matrices and vectors you will index with 1 as the first element not 0.

Often you will be working with matrices that are too large to be printed.  So you will need to slice and index to print
subsets of the matrix.  To learn more about matrices see the
`R documentation on matrices <https://cran.r-project.org/doc/manuals/r-release/R-intro.html#Arrays-and-matrices>`_.

Lists and data frames
-----------------------

Lists can be thought of as a higher-level container that contains different objects.  They can be used to represent
data sets for example. We will work mostly with data frames which can be thought of as a specific type of list.  It
is typical to build a list using key:value pairs.

.. code-block:: r

    > l <- list(a=cos(1:3), b=c("a","b","c"), c=pi)
    > l$a
    > l[['c']]

.. note::
    There were two ways to access the values stored in the list.

Also notice that the values do not have to be the same data type or size---think of lists as a generic container.

.. code-block:: r

   > a <- c(2, 3, 5)
   > b <- c("aa", "bb", "cc")
   > c <- c(TRUE, FALSE, TRUE)
   > df <- data.frame(a, b, c)
   > df

If you are already familiar with
`Pandas data frames <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ then this data
container will feel very comfortable.  In fact, Pandas data frames were inspired by the data frames in R because of
their utility in working with data sets.  In contrast to lists data frames expect the values associated with keys to
have the same size.

Additional Resources
=========================

    * `A collection of resources for R documentation <https://www.r-project.org/other-docs.html>`_
    * `Khan Academy intro to matrix multiplication <https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/v/matrix-multiplication-intro>`_
    * `R documentation vector arithmetic <https://cran.r-project.org/doc/manuals/r-release/R-intro.html#Vector-arithmetic>`_
    * `R-bloggers tutorial on data types <http://www.dataperspective.info/2016/02/basic-data-types-in-r.html>`_
