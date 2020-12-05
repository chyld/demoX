.. r datasets

**************
R Data Sets
**************

Learning objectives
=====================

+---------+--------------------------------------------------------------------------------------------------+
| 1       | Load and explore R's built-in data sets                                                          |
+---------+--------------------------------------------------------------------------------------------------+
| 2       | Work effectively with different types of input/output common to data science                     |
+---------+--------------------------------------------------------------------------------------------------+
| 3       | Demonstrate several of the ``apply`` family of function in the context of data frames            |
+---------+--------------------------------------------------------------------------------------------------+

Accessing Built-in Datasets
==============================

The standard data sets distributed with R are available as part of the ``datasets`` package.  We have not discussed
packages in R until now, but the ecosystem they provide is extensive.  The packages available through the The
Comprehensive R Archive Network or `CRAN <https://cran.r-project.org/>`_ are numerous and when considered collectively
it is one of the major reasons why R is a popular language for statistics and data science.   We will introduce packages
as part of these materials individually. To begin lets load the ``datasets`` sets package.  Use either the ``library``
or ``require`` function to load a package into an R workspace.

.. code-block:: r

    > library(datasets)
    > help(package='datasets')

.. tip:: The ``help`` function can be used to load the documentation for a specific package.

The package ``datasets`` is already loaded by default in most R environments.  You can check which packages are already
available in your workspace with:

.. code-block:: r

    > search()

The data sets that are built into R (and other languages) are **very** helpful for learning and familiarizing yourself
with the the specifics of a language.  To see exactly which data sets are available use the function ``data()``.

.. code-block:: r

    > data()

If we use the data set ``Orange`` as an example.  You can look the first few rows of the data with ``head()`` or
similarly you can look at the last few rows with ``tail()``.

.. code-block:: r

    > head(Orange)

.. code-block:: none

      Tree  age circumference
    1    1  118            30
    2    1  484            58
    3    1  664            87
    4    1 1004           115
    5    1 1231           120
    6    1 1372           142

If you have some experience with the package Pandas or with Data Frames from Apache Spark this will be familiar.  We
will continue to work with data sets, specifically with Data Frames throughout this unit.

Input and Output
====================

+------------------------------+----------------------------------------------------------------------------------+
| Function                     | Description                                                                      |
+==============================+==================================================================================+
| ``source()``                 | Run the commands in a specified file                                             |
+------------------------------+----------------------------------------------------------------------------------+
| ``read.table()``             | Read in data from file                                                           |
+------------------------------+----------------------------------------------------------------------------------+
| ``read.csv()``               | Read a csv file                                                                  |
+------------------------------+----------------------------------------------------------------------------------+
| ``read.delim()``             | Read in delimited text files                                                     |
+------------------------------+----------------------------------------------------------------------------------+
| ``download.file()``          | Download a file from the Internet                                                |
+------------------------------+----------------------------------------------------------------------------------+
| ``write()``                  | Write an object to a file                                                        |
+------------------------------+----------------------------------------------------------------------------------+
| ``writelines()``             | Write text lines to a connection                                                 |
+------------------------------+----------------------------------------------------------------------------------+
| ``write.table()``            | Write a table to a file                                                          |
+------------------------------+----------------------------------------------------------------------------------+

Reading and writing files is a major component of all things data science.  If you are to call one language from another
then you could pass information between the two with files.
`Comma-separated value <https://en.wikipedia.org/wiki/Comma-separated_values>`_ (CSV) files are very common in data
science as are JavaScript Object Notation or `JSON <https://en.wikipedia.org/wiki/JSON>`_ files.

.. code-block:: r

    > write.csv(Orange,"orange.csv",quote=FALSE,row.names=FALSE)

When you are in an R environment, be it a script or an interactive one, there is the notion of a working directory.
Specifically this refers to what directory is the R interpreter working under.  The line above saved our Orange data
frame to a CSV filed, but where exactly?  Whether you are in a standard R GUI or RStudio there is a global default
working directory---normally your home directory.  The default behavior can be customized for example see
`the article about working directories in RStudio <https://support.rstudio.com/hc/en-us/articles/200711843-Working-Directories-and-Workspaces>`_.
Otherwise, the ``getwd()`` and ``setwd()`` functions can be useful to help ensure you are reading and writing from
the correct locations.

A best practice when working with files is to use
``fully qualified path names <https://en.wikipedia.org/wiki/Fully_qualified_name>`_ and this can be done consistently
across operating systems with ``file.path``.

.. code-block:: r

    > file.path("f:", "path", "to","directory")
       [1] "c:/path/to/directory"

Or something more realistic---these lines assume you wrote the *orange.csv* file and your working directory was your
user home directory when you ran the command.

.. code-block:: r

    > orange_file <- file.path(path.expand('~'),"orange.csv")
    > df = read.csv(orange_file)
    > head(df,3)
      Tree age circumference
    1    1 118            30
    2    1 484            58
    3    1 664            87

Working with data frames
============================

The functions in the *apply* family are very useful and very commonly used.  Looping with ``for`` and ``while`` are not
computationally efficient in R language, which is a major reason you should consider these functions whenever it is
relevant.  The **apply** family of methods are a more concise and a more resource efficient way of looping in R.

``apply()``
    It takes Data frame or matrix as input and returns a vector, list or array.

.. code-block:: r

    > A <- matrix(c(1, 2, 3, 4, 5, 6), nrow=2, ncol=3, byrow = TRUE)

.. code-block:: none

          [,1] [,2] [,3]
    [1,]    1    2    3
    [2,]    4    5    6

.. code-block:: r

    > apply(A,1,sum)

.. code-block:: none

    [1]  6 15

.. code-block:: r

    > apply(A,2,sum)

.. code-block:: none

    [1] 5 7 9

.. hint::

    When using apply you have to specify the MARGIN (2nd argument)

    * ``MARGIN=1``: the function is carried out on rows
    * ``MARGIN=2``: the function is carried out on columns


``lapply()``
    Useful to perform operations on list, vector or data frame objects.  It returns a list object of same length.

If we return to the orange data set the ``age`` is given in days and the circumference is given
in `mm`.  If we had a list with ``lapply()``.  First let's return to the `Orange` data set.   The age is given in days.
Recall that we can access the data within a column with the ``$`` or by indexing.

.. code-block:: r

    > names(Orange)

    [1] "Tree"          "age"           "circumference"

    > Orange["age"]
    > summary(Orange$age)

      Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
      118.0   484.0  1004.0   922.1  1372.0  1582.0

    > Orange['age.years'] <- Orange["age"] / 365
    > head(Orange)

      Tree age circumference age.years
    1    1 118            30 0.3232877
    2    1 484            58 1.3260274
    3    1 664            87 1.8191781

In addition to an ``lapply()`` application the above code shows two common data frame manipulations.

1. The `summary()` function which can be applied to a column or to the data frame itself.
2. The assignment of a new column ``age.years``

We can use the ``lapply()`` function in this case to round.  The following line of code applies over all of the columns
in the data frame.


.. code-block:: r

    > new_orange <- data.frame(lapply(Orange, function(y) if(is.numeric(y)) round(y, 2) else y))
    > head(new_orange)
      Tree  age circumference age.years
    1    1  118            30      0.32
    2    1  484            58      1.33
    3    1  664            87      1.82
    4    1 1004           115      2.75
    5    1 1231           120      3.37
    6    1 1372           142      3.76


``sapply()``
    The same as ``lapply()`` except that it returns vector or matrix instead of list object

Here is a simple example.

.. code-block:: r

    > a <- lapply(1:10, function(x) if(x<5) x^2 else(0))
    > b <- sapply(1:10, function(x) if(x<5) x^2 else(0))
    > c <- unlist(lapply(1:10, function(x) if(x<5) x^2 else(0)))
    > identical(a,b)
    [1] FALSE
    > identical(b,c)
    [1] TRUE

.. note::

    There is a lack of curly braces ``{}``.  The above code uses the shorthand version for writing a function.

There are other methods that are in the *apply* family like ``tapply()`` and ``mapply()``, but these three are enough
to get started.

Additional Resources
=========================

    * `R Documentation for import/export <https://cran.r-project.org/doc/manuals/r-release/R-data.html>`_
    * `R-bloggers: using apply, sapply and lapply <https://www.r-bloggers.com/using-apply-sapply-lapply-in-r/>`_
