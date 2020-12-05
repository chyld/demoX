
***********************
Environment and Setup
***********************

Learning objectives
=====================

+---------+-------------------------------------------------------------------------------------------------------+
| 1       | Install and maintain a local Python environment that readily works with Jupyter                       |
+---------+-------------------------------------------------------------------------------------------------------+

Install Python
=================

`Anaconda <https://docs.anaconda.com/anaconda/>`_ is a convenient way to manage both Python and R.  In this unit we will
detail the install process for Python and Git.  If you would like to setup an R environment that works with your Python
environment follow the directions to setup an
`R environment using the navigator <https://docs.anaconda.com/anaconda/navigator/tutorials/create-r-environment/>`_
after you have completed this unit.

.. warning::

    If you already have working Python environment you may skip this section, but you should at least read the
    contents to ensure that you have an environment that is compatible with these learning materials.

The following questions may be helpful to determine if you environment is fully compatible:

    * Can you launch a Jupyter notebook server and execute code?
    * Can you save Python code to a file and execute the file from the command line?

Install Anaconda
---------------------------

If you already have a version of Anaconda and you would like to start fresh use the following command (from a terminal)
to remove the old version.

.. code-block::
   bash

   ~$ rm -rf ~/anaconda*

.. hint::

    On Windows, macOS, and Linux, it is best to install Anaconda for the local user.  This type of install **does not**
    require administrator permissions. However, if you need to, you can install Anaconda system wide, which does
    require administrator permissions.  See the `Anaconda install docs <https://docs.anaconda.com/anaconda/install/>`_
    for more information.

Once you have installed the package Anaconda you will be using `conda <https://conda.io/en/latest>`_ to manage the
packages in your environment.  Anaconda is a powerful data science platform that allows you to maintain
`several versions of Python <https://docs.anaconda.com/anaconda/navigator/tutorials/use-multiple-python-versions/>`_.
Let's get started by installing the Anaconda package.

Anaconda for Windows
^^^^^^^^^^^^^^^^^^^^^^

1. Open https://www.anaconda.com/download/#windows
2. Download the Python 3 installer for Windows.
3. Install Python 3 using all of the defaults for installation except make sure to check
   **Make Anaconda the default Python**.

Anaconda for Mac OS
^^^^^^^^^^^^^^^^^^^^^^
1. Open https://www.anaconda.com/download/#macos
2. Download the Python 3 installer for OS X.
3. Install Python 3 using all of the defaults for installation.

Anaconda for Linux
^^^^^^^^^^^^^^^^^^^^^^^

1. Open https://www.anaconda.com/download/#linux
2. Download the Python 3 installer for Linux.
3. Install Python 3 using all of the defaults for installation.
4. Open a terminal window and navigate to the downloads folder where you enter the following command.

.. code-block::
    bash

    ~$ bash Anaconda3-20XX.XX-Linux-x86_64.sh

.. note::

    The exact file name will depend on the version of Anaconda that you downloaded.

5. Follow the prompts ensuring that you prepend Anaconda to your PATH (this makes the Anaconda distribution the
default Python).


You can work directly from a terminal for all materials in this course.  However, the
`Anaconda Navigator <https://docs.anaconda.com/anaconda/navigator/>`_ is a useful tool for launching different
environments especially if you are not particularly comfortable on the command line yet.  See the
`guide for using the navigator <https://docs.anaconda.com/anaconda/navigator/getting-started/>`_ as a good resource for
getting started.

* From Windows you can find it using the *Start Menu*.
* In MacOS it can be found under *Applications*
* Under both MacOS and Linux it can be started from the command line with:

.. code-block::
    bash

    ~$ anaconda-navigator

Once you have the navigator open it will look like this:

.. figure:: ./images/anaconda-nav.png
   :scale: 30%
   :align: center
   :alt: anaconda-navigator
   :figclass: align-center

Anaconda is a very useful tool because it allows you to install and maintain multiple programming environments
side-by-side.  If you need to install another version of Python or R you will use the environments interface.

.. figure:: ./images/anaconda-nav-environments.png
   :scale: 30%
   :align: center
   :alt: anaconda-navigator-environments
   :figclass: align-center

For the purposes of this Python course you will not need an additional environment, but it is good to keep in mind.

.. tip::

    You may also install any other software, like JupyterLab, that you might use as part of your development environment.

Install Git
=================

Git is a `version control <https://en.wikipedia.org/wiki/Version_control>`_ system that lets you track changes to files
containing text.  Often these files are scripts or other files that contain computer programming code.
`Git <https://git-scm.com/>`_ through the use of `GitHub <https://github.com/>`_ helps enable collaboration, resource
sharing and reproducible analytics.  You will need a
`GitHub supported web browser <https://help.github.com/en/enterprise/2.15/user/articles/supported-browsers>`_ to complete
the assignments and tasks that are part of this learning experience.  You will also need to
`sign up for a GitHub account <https://github.com/join>`_ if you do not have one already.  Basic GitHub accounts are
free. Please create one now if you do not have one already.

.. tip::

    There are a number of settings associated with your GitHub account.  For example, if you would like to keep your
    email address private see the `setting your email address documentation <https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address>`_.

Git for Windows
-----------------

    1. Download the `Git for Windows installer <https://git-for-windows.github.io/>`_.
    2. Run the installer and follow the steps below:
    3. Click on "Next" four times (two times if you've previously installed Git). You don't need to change anything in
       the Information, location, components, and start menu screens.
    4. Select "Use the nano editor by default" and click on `Next`.

    .. important::

        Keep **Use Git from the Windows Command Prompt** selected and click on `Next`. If you forgot to do this programs
        that you need for the workshop will not work properly. If this happens rerun the installer and select the
        appropriate option.

    5. Click on `Next`. Keep **Checkout Windows-style, commit Unix-style line endings** selected and click on `Next`.
       Select **Use Windows' default console window** and click on `Next`.

    6. Click on `Install`.
    7. Click on `Finish`.

If your "HOME" environment variable is not set (or you don't know what this is):

1. Open command prompt (Open Start Menu then type cmd and press [Enter])
2. Type the following line into the command prompt window exactly as shown:

.. code-block::
    bash

    ~$ setx HOME "%USERPROFILE%"

3. Press `[Enter]` and you should see `SUCCESS: Specified value was saved`.

4. Quit command prompt by typing `exit` then pressing `[Enter]` This will provide you with both Git and
Bash in the Git Bash program.

Git for MacOS
----------------

If it is not already on your machine install `Homebrew <https://brew.sh/>`_ (another package manager).

.. code-block::
    bash

    ~$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then install Git

.. code-block::
    bash

    ~$ brew install git

Git for Ubuntu Linux

.. code-block::
    bash

    ~$ sudo apt install git-all

See the `installing Git documentation <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ for more
information and troubleshooting.

Keeping everything updated
================================

Be sure you systems is always up-to-date and periodically use the following command to keep the packages in conda
current.

.. code-block::
    bash

    ~$ conda update --all

Additional Resources
=======================

    * `Anaconda User Guide <https://docs.anaconda.com/anaconda/user-guide/>`_
    * `To integrate Anaconda with an IDE <https://docs.anaconda.com/anaconda/user-guide/tasks/integration/>`_
    * `Anaconda's Docker images <https://docs.anaconda.com/anaconda/user-guide/tasks/docker/>`_
