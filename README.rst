gContent
============

:Version: 0.0.1
:Authors: Galvanize
:Web site: https://github.com/GalvanizeDataScience/gcontent
:Copyright: Galvanize Inc.
:License: None public

About
---------

This is the source directory for documentation for data science repositories.

Essentially, the bulk of the content is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ (reST).
reST is not all that different than markdown.  reST has a number of features that give it an advantage over markdown.

   * Built in support for extensions
   * Native support for tables
   * Autogenerates things like table of contents
   * Can include comments
   * Can do so much more when used through `Sphinx <http://www.sphinx-doc.org/en/stable/>`_  

With Sphinx we can most importantly:

   1. Render professional LaTeX (PDF) and HTML outputs simply
   2. Include citations `using BibTeX <sphinxcontrib-bibtex>`_

As an aside markdown can be converted to reST and vice versa with tools like `Pandoc <https://pandoc.org>`_.  

.. code-block:: bash

   pandoc [filename] -f markdown -t rst -o [filename].rst

To learn more about markdown and reST

   * `reST vs markdown for technical docs <http://eli.thegreenplace.net/2017/restructuredtext-vs-markdown-for-technical-documentation>`_
   * `As comparison of reST and markdown syntax <http://www.unexpected-vortices.com/doc-notes/markdown-and-rest-compared.html>`_

System Setup
---------------

Install the Python packages

.. code-block::

    ~$ pip install -r requirements.txt 
    ~$ jupyter-nbextension install rise --py --sys-prefix
    ~$ jupyter-nbextension enable rise --py --sys-prefix

.. note:: 

    If you do not have Python installed see the Anaconda install link listed under Resources below.


Getting started
------------------

After ensuring that everthing is installed.  To build, for example, ``stats-fundamentals``:

.. code-block:: bash

   ~$ cd stats-essentials
   ~$ make html
   
The if you are on a OSX

.. code-block:: bash

   ~$ open _build/html/index.html

Or if you are on Ubuntu

.. code-block:: bash

   ~$ xdg-open _build/html/index.html


.. note::
   On Windows systems you will need to install Chocolately
   
Now you simple edit any *.rst document and type ``make html`` again. The entire HTML folder 
is what is transferred to the target repo.


Helpful
---------------------------------

To convert a single rst file to a pdf use for example

.. code-block::

   pandoc OUTLINE.rst --template=template.tex --pdf-engine=xelatex -o OUTLINE.pdf


To install a spellchecker for Jupyter notebooks

.. code-block::
   
   ~$ pip install jupyter_contrib_nbextensions
   ~$ jupyter contrib nbextension install --user
   ~$ jupyter nbextension enable spellchecker/main

Resources
-------------------

   * `Anaconda's into restructured text <https://docs.anaconda.com/restructuredtext/detailed/>`_
   * `restructured text official docs <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_
   * `install Anaconda <https://docs.anaconda.com/anaconda/install/>`_
