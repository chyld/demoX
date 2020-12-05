.. linear algebra

***********************   
Matrix Decompositions
***********************

The idea of **Matrix decomposition** also known as **matrix factorization**

* Matrix decompositions are an important step in solving linear systems in a computationally efficient manner
* Numerous decomposition exist examples include: Cholesky Decomposition, LU Decomposition, QR decompositon and Eigendecomposition

A **system** of equations** is a collection of equations that you deal with all together.
  

LU Decomposition and Gaussian Elimination
=============================================

LU stands for 'Lower Upper', and so an LU decomposition of a matrix
:math:`A` is a decomposition so that

.. math:: A= LU

where :math:`L` is lower triangular and :math:`U` is upper triangular.

Now, LU decomposition is essentially gaussian elimination, but we work
only with the matrix :math:`A` (as opposed to the augmented matrix).

Let's review how gaussian elimination (ge) works. We will deal with a
:math:`3\times 3` system of equations for conciseness, but everything
here generalizes to the :math:`n\times n` case. Consider the following
equation:

.. math:: \left(\begin{matrix}a_{11}&a_{12} & a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{matrix}\right)\left(\begin{matrix}x_1\\x_2\\x_3\end{matrix}\right) = \left(\begin{matrix}b_1\\b_2\\b_3\end{matrix}\right)

For simplicity, let us assume that the leftmost matrix :math:`A` is
non-singular. To solve the system using ge, we start with the 'augmented
matrix':

.. math:: \left(\begin{array}{ccc|c}a_{11}&a_{12} & a_{13}& b_1 \\a_{21}&a_{22}&a_{23}&b_2\\a_{31}&a_{32}&a_{33}&b_3\end{array}\right)

We begin at the first entry, :math:`a_{11}`. If :math:`a_{11} \neq 0`,
then we divide the first row by :math:`a_{11}` and then subtract the
appropriate multiple of the first row from each of the other rows,
zeroing out the first entry of all rows. (If :math:`a_{11}` is zero, we
need to permute rows. We will not go into detail of that here.) The
result is as follows:

.. math::

   \left(\begin{array}{ccc|c}
   1 & \frac{a_{12}}{a_{11}} & \frac{a_{13}}{a_{11}} & \frac{b_1}{a_{11}} \\
   0 & a_{22} - a_{21}\frac{a_{12}}{a_{11}} & a_{23} - a_{21}\frac{a_{13}}{a_{11}}  & b_2 - a_{21}\frac{b_1}{a_{11}}\\
   0&a_{32}-a_{31}\frac{a_{12}}{a_{11}} & a_{33} - a_{31}\frac{a_{13}}{a_{11}}  &b_3- a_{31}\frac{b_1}{a_{11}}\end{array}\right)

We repeat the procedure for the second row, first dividing by the
leading entry, then subtracting the appropriate multiple of the
resulting row from each of the third and first rows, so that the second
entry in row 1 and in row 3 are zero. We *could* continue until the
matrix on the left is the identity. In that case, we can then just 'read
off' the solution: i.e., the vector :math:`x` is the resulting column
vector on the right. Usually, it is more efficient to stop at *reduced
row eschelon* form (upper triangular, with ones on the diagonal), and
then use *back substitution* to obtain the final answer.

Note that in some cases, it is necessary to permute rows to obtain
reduced row eschelon form. This is called *partial pivoting*. If we also
manipulate columns, that is called *full pivoting*.

It should be mentioned that we may obtain the inverse of a matrix using
ge, by reducing the matrix :math:`A` to the identity, with the identity
matrix as the augmented portion.

Now, this is all fine when we are solving a system one time, for one
outcome :math:`b`. Many applications involve solutions to multiple
problems, where the left-hand-side of our matrix equation does not
change, but there are many outcome vectors :math:`b`. In this case, it
is more efficient to *decompose* :math:`A`.

First, we start just as in ge, but we 'keep track' of the various
multiples required to eliminate entries. For example, consider the
matrix

.. math::

   A = \left(\begin{matrix} 1 & 3 & 4 \\
                              2& 1& 3\\
                              4&1&2
                              \end{matrix}\right)

We need to multiply row :math:`1` by :math:`2` and subtract from row
:math:`2` to eliminate the first entry in row :math:`2`, and then
multiply row :math:`1` by :math:`4` and subtract from row :math:`3`.
Instead of entering zeroes into the first entries of rows :math:`2` and
:math:`3`, we record the multiples required for their elimination, as
so:

.. math::

   \left(\begin{matrix} 1 & 3 & 4 \\
                              (2)& -5 & -5\\
                              (4)&-11&-14
                              \end{matrix}\right)

And then we eliminate the second entry in the third row:

.. math::

   \left(\begin{matrix} 1 & 3 & 4 \\
                              (2)& -5 & -5\\
                              (4)&(\frac{11}{5})&-3
                              \end{matrix}\right)

And now we have the decomposition:

.. math::

   L= \left(\begin{matrix} 1 & 0 & 0 \\
                              2& 1 & 0\\
                              4&\frac{11}5&1
                              \end{matrix}\right)
                             U = \left(\begin{matrix} 1 & 3 & 4 \\
                              0& -5 & -5\\
                              0&0&-3
                              \end{matrix}\right)

.. code:: python

    import numpy as np
    import scipy.linalg as la
    np.set_printoptions(suppress=True) 
    
    A = np.array([[1,3,4],[2,1,3],[4,1,2]])
    
    L = np.array([[1,0,0],[2,1,0],[4,11/5,1]])
    U = np.array([[1,3,4],[0,-5,-5],[0,0,-3]])
    print(L.dot(U))
    print(L)
    print(U)


.. parsed-literal::

    [[ 1.  3.  4.]
     [ 2.  1.  3.]
     [ 4.  1.  2.]]
    [[ 1.   0.   0. ]
     [ 2.   1.   0. ]
     [ 4.   2.2  1. ]]
    [[ 1  3  4]
     [ 0 -5 -5]
     [ 0  0 -3]]


We can solve the system by solving two back-substitution problems:

.. math:: Ly = b

and

.. math:: Ux=y

These are both :math:`O(n^2)`, so it is more efficient to decompose when
there are multiple outcomes to solve for.

Let do this with numpy:

.. code:: python

    import numpy as np
    import scipy.linalg as la
    np.set_printoptions(suppress=True) 
    
    A = np.array([[1,3,4],[2,1,3],[4,1,2]])
    
    print(A)
    
    P, L, U = la.lu(A)
    print(np.dot(P.T, A))
    print
    print(np.dot(L, U))
    print(P)
    print(L)
    print(U)


.. parsed-literal::

    [[1 3 4]
     [2 1 3]
     [4 1 2]]
    [[ 4.  1.  2.]
     [ 1.  3.  4.]
     [ 2.  1.  3.]]
    
    [[ 4.  1.  2.]
     [ 1.  3.  4.]
     [ 2.  1.  3.]]
    [[ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]]
    [[ 1.      0.      0.    ]
     [ 0.25    1.      0.    ]
     [ 0.5     0.1818  1.    ]]
    [[ 4.      1.      2.    ]
     [ 0.      2.75    3.5   ]
     [ 0.      0.      1.3636]]


Note that the numpy decomposition uses *partial pivoting* (matrix rows
are permuted to use the largest pivot). This is because small pivots can
lead to numerical instability. Another reason why one should use library
functions whenever possible!

Cholesky Decomposition
=========================

Recall that a square matrix :math:`A` is positive definite if

.. math:: u^TA u > 0

for any non-zero n-dimensional vector :math:`u`,

and a symmetric, positive-definite matrix :math:`A` is a
positive-definite matrix such that

.. math:: A = A^T

Let :math:`A` be a symmetric, positive-definite matrix. There is a
unique decomposition such that

.. math:: A = L L^T

where :math:`L` is lower-triangular with positive diagonal elements and
:math:`L^T` is its transpose. This decomposition is known as the
Cholesky decompostion, and :math:`L` may be interpreted as the 'square
root' of the matrix :math:`A`.

Algorithm:
--------------

Let :math:`A` be an :math:`n\times n` matrix. We find the matri
:math:`L` using the following iterative procedure:

.. math::

   A = \left(\begin{matrix}a_{11}&A_{12}\\A_{12}&A_{22}\end{matrix}\right) =
   \left(\begin{matrix}\ell_{11}&0\\
   L_{12}&L_{22}\end{matrix}\right)
   \left(\begin{matrix}\ell_{11}&L_{12}\\0&L_{22}\end{matrix}\right)

1.) Let :math:`\ell_{11} = \sqrt{a_{11}}`

2.) :math:`L_{12} = \frac{1}{\ell_{11}}A_{12}`

3.) Solve :math:`A_{22} - L_{12}L_{12}^T = L_{22}L_{22}^T` for
:math:`L_{22}`

Example:
~~~~~~~~

.. math:: A = \left(\begin{matrix}1&3&5\\3&13&23\\5&23&42\end{matrix}\right)

.. math:: \ell_{11} = \sqrt{a_{11}} = 1

.. math:: L_{12} = \frac{1}{\ell_{11}} A_{12} = A_{12}

:math:`\begin{eqnarray*} A_{22} - L_{12}L_{12}^T &=& \left(\begin{matrix}13&23\\23&42\end{matrix}\right) - \left(\begin{matrix}9&15\\15&25\end{matrix}\right)\\ &=& \left(\begin{matrix}4&8\\8&17\end{matrix}\right)\\ &=& \left(\begin{matrix}2&0\\4&\ell_{33}\end{matrix}\right) \left(\begin{matrix}2&4\\0&\ell_{33}\end{matrix}\right)\\ &=& \left(\begin{matrix}4&8\\8&16+\ell_{33}^2\end{matrix}\right) \end{eqnarray*}`

And so we conclude that :math:`\ell_{33}=1`.

This yields the decomposition:

.. math::

   \left(\begin{matrix}1&3&5\\3&13&23\\5&23&42\end{matrix}\right) = 
   \left(\begin{matrix}1&0&0\\3&2&0\\5&4&1\end{matrix}\right)\left(\begin{matrix}1&3&5\\0&2&4\\0&0&1\end{matrix}\right)

Now, with numpy:

.. code:: python

    A = np.array([[1,3,5],[3,13,23],[5,23,42]])
    L = la.cholesky(A)
    print(np.dot(L.T, L))
    
    print(L)
    print(A)


.. parsed-literal::

    [[  1.   3.   5.]
     [  3.  13.  23.]
     [  5.  23.  42.]]
    [[ 1.  3.  5.]
     [ 0.  2.  4.]
     [ 0.  0.  1.]]
    [[ 1  3  5]
     [ 3 13 23]
     [ 5 23 42]]


Cholesky decomposition is about twice as fast as LU decomposition
(though both scale as :math:`n^3`).


QR decompositon
==================

As with the previous decompositions, :math:`QR` decomposition is a
method to write a matrix :math:`A` as the product of two matrices of
simpler form. In this case, we want:

.. math::  A= QR

where :math:`Q` is an :math:`m\times n` matrix with :math:`Q Q^T = I`
(i.e. :math:`Q` is *orthogonal*) and :math:`R` is an :math:`n\times n`
upper-triangular matrix.

This is a form of the Gram-Schmidt orthogonalization of the columns
of :math:`A`. The G-S algorithm itself is unstable, so various other
methods have been developed to compute the QR decomposition.

The first :math:`k` columns of :math:`Q` are an orthonormal basis for
the column space of the first :math:`k` columns of :math:`A`.

Iterative QR decomposition is often used in the computation of
eigenvalues.

Eigendecomposition
======================

Let :math:`A` be an :math:`n \times n` matrix and :math:`\mathbf{x}`
be an :math:`n \times 1` nonzero vector. An **eigenvalue** of
:math:`A` is a number :math:`\lambda` such that

.. math::

   A \boldsymbol{x} = \lambda \boldsymbol{x}


A vector :math:`\mathbf{x}` satisfying this equation is called an **eigenvector** associated with :math:`\lambda`


>>> a = np.diag((1, 2, 3))
>>> a
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3]])
>>> w,v = np.linalg.eig(a)
>>> w;v
array([ 1.,  2.,  3.])
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])

Eigenvectors and eigenvalues are important mathematical identities that play many roles across a range of disciplines


Singular Values
===================

For any :math:`m\times n` matrix :math:`A`, we define its *singular
values* to be the square root of the eigenvalues of :math:`A^TA`. These
are well-defined as :math:`A^TA` is always symmetric, positive-definite,
so its eigenvalues are real and positive. Singular values are important
properties of a matrix. Geometrically, a matrix :math:`A` maps the unit
sphere in :math:`\mathbb{R}^n` to an ellipse. The singular values are
the lengths of the semi-axes.

Singular values also provide a measure of the *stabilty* of a matrix.
We'll revisit this in the end of the lecture.


Singular Value Decomposition
================================

Another important matrix decomposition is singular value decomposition
or SVD. For any :math:`m\times n` matrix :math:`A`, we may write:

.. math:: A= UDV

where :math:`U` is a unitary (orthogonal in the real case)
:math:`m\times m` matrix, :math:`D` is a rectangular, diagonal
:math:`m\times n` matrix with diagonal entries :math:`d_1,...,d_m` all
non-negative. :math:`V` is a unitary (orthogonal) :math:`n\times n`
matrix. SVD is used in principle component analysis and in the
computation of the Moore-Penrose pseudo-inverse.


PCA and SVD
================

   * :download:`./slides/pca_svd.pdf`
   * :download:`./notebooks/svd-as-recommender.ipynb`

For a more in-depth resourse see:

`SciPy's official tutorial on Linear
algebra <http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html>`_
