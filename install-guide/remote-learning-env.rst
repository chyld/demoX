.. name of course

*************************************
Remote learning environment
*************************************

The educational products that are offered by Galvanize have two variants: synchronous and blended.
Synchronous remote learning at Galvanize is called, more specifically, *live online*.

System Requirements
=======================

Office Requirements
--------------------------------

* High speed Internet with >= 50Mbps
* Company-provided and configured VPN (if applicable)
* Webcam (integrated or dedicated)
* Headset with microphone (recommended)

Machine Requirements
-------------------------

* 32- or 64-bit computer
* 8GB Minimum Memory RAM, 16GB Memory RAM is preferred
* At least 10GB of free hard disk space will be required for most programs

Software Requirements
---------------------------

* `Slack desktop <https://slack.com/>`_
* `Zoom conferencing software <https://zoom.us/>`_
* `Git <https://git-scm.com/>`_ with a `GitHub <https://github.com/>`_ account

The rest of this document details how to install these requirements on your system.  Live online and blended sessions
will occur principally through Zoom and Slack.  Assignments and the practical application of the topics you will learn
will take place via GitHub.

Zoom
=========

1. Install Zoom using the download for you operating system

    * `Zoom downloads <https://zoom.us/download>`_

2. Sign up for an account if you do not already have one

    * `Zoom signup page <https://zoom.us/freesignup/>`_

Slack
=========

1. Install slack using the instructions for your operating system

    * `Slack Windows install page <https://slack.com/downloads/windows>`_
    * `Slack MacOS install page <https://slack.com/downloads/mac>`_
    * `Slack Linux install page <https://slack.com/downloads/linux>`_

2. Once you receive an email from Galvanize with login instructions you can use the information to login

.. note::
    Slack can be run either in the browser or installed locally.

Git and GitHub
================

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
    3. Click on `Next` four times (two times if you've previously installed Git). You don't need to change anything in
       the Information, location, components, and start menu screens.
    4. Select **Use the nano editor by default** and click on `Next`.

    .. important::

        Keep **Use Git from the Windows Command Prompt** selected and click on `Next`. If you do not do this, programs
        that you need for the workshop will not work properly. If this happens, rerun the installer and select the
        appropriate option.

    5. Click on `Next`. Keep **Checkout Windows-style, commit Unix-style line endings** selected and click on `Next`.
       Select **Use Windows' default console window** and click on `Next`.
    6. Click on `Install`.
    7. Click on `Finish`.

If your "HOME" environment variable is not set (or you don't know what this is):

1. Open command prompt (open Start Menu then type ``cmd`` and press `[Enter]`)
2. Type the following line into the command prompt window exactly as shown:

.. code-block::
    bash

    ~$ setx HOME "%USERPROFILE%"

3. Press `[Enter]` and you should see "SUCCESS: Specified value was saved".

4. Quit command prompt by typing ``exit`` and pressing `[Enter]`.

Git for MacOS
----------------

If it is not already on your machine install `Homebrew <https://brew.sh/>`_ (another package manager):

.. code-block::
    bash

    ~$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then install Git:

.. code-block::
    bash

    ~$ brew install git

Git for Ubuntu Linux
----------------------

Install Git:

.. code-block::
    bash

    ~$ sudo apt install git-all

More Git installation info
----------------------------

See the `installing Git documentation <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ for more
information and troubleshooting.

Website restrictions
==============================

Some firewalls prevent access to certain websites.  You will need access to the following websites to ensure a
successful learning process.

* `https://github.com/ <https://github.com/>`_
* `https://learn-2.galvanize.com/ <https://learn-2.galvanize.com/>`_
* `https://vimeo.com/ <https://vimeo.com/>`_

We use Vimeo for pre-recorded lecture videos. Using a centralized resource for version control
(like GitHub) is a cornerstone of modern data science.

Additional Resources
===========================

* `Slack Tutorials <https://slack.com/resources/using-slack/slack-tutorials>`_
* `Zoom Tutorials <https://support.zoom.us/hc/en-us/articles/206618765-Zoom-Video-Tutorials>`_
