.. galvanize, created by ARichards

******************
Sparse Matrices
******************

Once you have finished with the **sparse matrices** unit we expect that you will be able to:

+------+------------------------------+-------------------------------------------------------------------------------------------+
| 1    | Sparse Matrices              | Explain when to use sparse matrices during the machine learning model development process |
+------+------------------------------+-------------------------------------------------------------------------------------------+


Sometimes the optimization of our code has less to do with the algorithm or model and more to do with optimizing how
the data is stored and fetched.  There are many applications in which we deal with matrices that are nearly all zeros.

Deploying your AI application is only one portion of the solution.  Another significant portion is the ability
to ingest new data and ensure that models have the ability to predict and refresh themselves with new data.
​
The full analysis pipeline should not be finalized during the initial stages of development of the machine learning
model it feeds.  Finalizing the process of data ingestion *before* models have been run and your hypotheses about the
business use case have been tested often leads to an ineffective use of time. Early experiments almost always fail and
you should be careful about investing large amounts of time in building proper data ingestion pipeline until there is
enough accumulated evidence that a deployed model will help the business.


Data staging
--------------




* Data scientists will often use sparse matrices during the development and testing of a machine learning model
* Python libraries available in SciPy package to work with sparse matrices.

Instead of building a complete data ingestion pipeline, data scientists will often use sparse matrices during the
development and testing of a machine learning model. Sparse matrices are used to represent complex sets of data
(e.g., word counts) in a way that reduces the use of computer memory and processing time.

The code block below imports this library as well as NumPy for calculations:

.. code-block:: python

    import numpy as np
    from scipy import sparse


Sparse matrices offer a middle-ground between:

  * A comprehensive data warehouse solution with extensive test coverage
  * A directory of text files and database dumps

Sparse matrices offer a middle-ground between a comprehensive data warehouse solution with extensive test coverage and
a directory of text files and database dumps. Sparse matrices do not work for all data types, but in situations where
they are an appropriate technology you can leverage them even under load in production.

.. note::

    If the number of zero-valued elements divided by the size of the matrix is greater than 0.5 then it is considered
    **sparse**.

A sparse matrix is one in which most of the values are zero. If the number of zero-valued elements divided by the size
of the matrix is greater than 0.5 then it is considered sparse.  The following code will generate an array of 100,000
random integers between 0 and 2, reshape that array into a 100x1000 matrix, and then compute the sparsity.

.. code-block:: python

    A = np.random.randint(0,2,100000).reshape(100,1000)
    sparsity = 1.0 - (np.count_nonzero(A) / A.size)
    print(round(sparsity,4))

The convenience of sparse matrices
=====================================

  * Very large non-sparse matrices require significant amounts of memory.
  * Sparse matrices allow you to manage large amounts of data in a memory-efficient and time-efficient manner.

Very large matrices require significant amounts of memory. For example, If we make a matrix of counts for a document or
a book where the features are all known English words, the chances are high that your personal machine does not have
enough memory to represent it as a dense matrix. Sparse matrices have the additional advantage of getting around
time-complexity issues that arise with operations on large dense matrices.

The following code will create a 10x100 array of random numbers drawn from a Poisson distribution, cast that
matrix into a sparse matrix in coordinate format, and then return it to a dense matrix.

.. code-block:: python

    A = np.random.poisson(0.3, (10,100))
    B = sparse.coo_matrix(A)
    C = B.todense()

    print("A",type(A),A.shape,"\n"
          "B",type(B),B.shape,"\n"
          "C",type(C),C.shape,"\n")

.. code-block:: none

    A <class 'numpy.ndarray'> (10, 100)
    B <class 'scipy.sparse.coo.coo_matrix'> (10, 100)
    C <class 'numpy.matrix'> (10, 100)



Working with sparse matrices
===============================
When there are repeated entries in the rows or cols, we can remove the redundancy by indicating the location of the
first occurrence of a value and its increment instead of the full coordinates.

csc_matrix
 When the repeats occur in columns we use a CSC, or
 `compressed sparse column <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix/>`_ format.

csr_matrix
 When the repeats occur in rows we use a CSR, or
 `compressed sparse row <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix/>`_ format.

coo_matrix
 A sparse matrix in `coordinate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix/>`_ format.

.. code-block:: python

    A = np.random.poisson(0.3, (10,100))
    B = sparse.csc_matrix(A)

.. hint::

    Because the coordinate format is easier to create, it is common to create it first then cast to another more
    efficient format. Let us first show how to create a matrix from coordinates

.. code-block:: python

    rows = [0,1,2,8]
    cols = [1,0,4,8]
    vals = [1,2,1,4]

    A = sparse.coo_matrix((vals, (rows, cols)))
    print(A.todense())

Then to cast it to a CSR matrix:

.. code-block:: python

    B = A.tocsr()
    print(B)

.. code-block:: none

    (0, 1)	1
    (1, 0)	2
    (2, 4)	1
    (8, 8)	4


Because this introduction to sparse matrices is applied to data ingestion you would need to be able to:

* concatenate matrices (e.g., add a new user to a recommender matrix)
* read and write the matrices to and from disk


.. code-block:: python

    ## concatenate example
    C = sparse.csr_matrix(np.array([0,1,0,0,2,0,0,0,1]).reshape(1,9))
    print(B.shape,C.shape)
    D = sparse.vstack([B,C])
    print(D.todense())

.. code-block:: python

    (9, 9) (1, 9)

    [[0 1 0 0 0 0 0 0 0]
     [2 0 0 0 0 0 0 0 0]
     [0 0 0 0 1 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 4]
     [0 1 0 0 2 0 0 0 1]]

Reading and writing sparse matrices
=====================================


.. code-block:: python

    ## read and write
    file_name = "sparse_matrix.npz"
    sparse.save_npz(file_name, D)
    E = sparse.load_npz(file_name)

As you can see the syntax is very similar to NumPy.

Sparse matrices in Pandas
=====================================
Pandas supports Sparse data structures to efficiently store sparse data, and these functions are interoperable with scipy sparse. 

`arrays.SparseArray <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.SparseArray.html#pandas.arrays.SparseArray>`_
 **ExtensionArray** for storing an array of sparse values. It is a 1-dimensional ndarray-like object storing only values distinct from the ``fill_value``:

.. code-block:: python

    In [13]: arr = np.random.randn(10)

    In [14]: arr[2:5] = np.nan

    In [15]: arr[7:8] = np.nan

    In [16]: sparr = pd.arrays.SparseArray(arr)

    In [17]: sparr
    Out[17]: 
    [-1.9556635297215477, -1.6588664275960427, nan, nan, nan, 1.1589328886422277, 0.14529711373305043, nan, 0.6060271905134522, 1.3342113401317768]
    Fill: nan
    IntIndex
    Indices: array([0, 1, 5, 6, 8, 9], dtype=int32)

A sparse array can be converted to a regular (dense) ndarray with ``numpy.asarray()``

.. code-block:: python

    In [18]: np.asarray(sparr)
    Out[18]: 
    array([-1.9557, -1.6589,     nan,     nan,     nan,  1.1589,  0.1453,
            nan,  0.606 ,  1.3342])

`SparseArray.dtype <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.SparseDtype.html#pandas.SparseDtype>`_
 property stores **dtype** of the non-sparse values and the **scalar fill value**

.. code-block:: python

    In [19]: sparr.dtype
    Out[19]: Sparse[float64, nan]

The default fill value for a given NumPy dtype is the “missing” value for that dtype, though it may be overridden.

.. code-block:: python

    In [21]: pd.SparseDtype(np.dtype('datetime64[ns]'),
    ....:                fill_value=pd.Timestamp('2017-01-01'))
    ....: 
    Out[21]: Sparse[datetime64[ns], 2017-01-01 00:00:00]

The string alias ``'Sparse[dtype]'`` may be used to specify a sparse dtype in many places

.. code-block:: python

    In [22]: pd.array([1, 0, 0, 2], dtype='Sparse[int]')
    Out[22]: 
    [1, 0, 0, 2]
    Fill: 0
    IntIndex
    Indices: array([0, 3], dtype=int32)

`.sparse accessor <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#api-frame-sparse>`_
 Pandas provides a ``.sparse`` accessor, similar to ``.str`` for string data, ``.cat`` for categorical data, and ``.dt`` for datetime-like data. This namespace provides attributes and methods that are specific to sparse data.

.. code-block:: python

    In [23]: s = pd.Series([0, 0, 1, 2], dtype="Sparse[int]")

    In [24]: s.sparse.density
    Out[24]: 0.5

    In [25]: s.sparse.fill_value
    Out[25]: 0

This accessor is available only on data with SparseDtype, and on the **Series** or **DataFrame** class itself for creating a Series with sparse data from a scipy COO matrix with.

`Numpy ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
 Sparse calculations let you apply NumPy ufuncs to ``SparseArray`` and get a ``SparseArray`` as a result.

.. code-block:: python

    In [26]: arr = pd.arrays.SparseArray([1., np.nan, np.nan, -2., np.nan])

    In [27]: np.abs(arr)
    Out[27]: 
    [1.0, nan, nan, 2.0, nan]
    Fill: nan
    IntIndex
    Indices: array([0, 3], dtype=int32)

The `ufunc` is also applied to ``fill_value``. This is needed to get the correct dense result.

.. code-block:: python

    In [28]: arr = pd.arrays.SparseArray([1., -1, -1, -2., -1], fill_value=-1)

    In [29]: np.abs(arr)
    Out[29]: 
    [1.0, 1, 1, 2.0, 1]
    Fill: 1
    IntIndex
    Indices: array([0, 3], dtype=int32)

    In [30]: np.abs(arr).to_dense()
    Out[30]: array([1., 1., 1., 2., 1.])


Additional resources:
=====================

* `SciPy docs for Sparse Matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
* `Pandas docs for Sparse Matrices <https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html>`_


.. admonition:: Assignment

    * download: `Assignment 1 <exercises/assignment-1.md>`_
