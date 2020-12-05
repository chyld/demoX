.. r into (powerbayes)

**************
R Programming
**************

Learning objectives
=====================

+---------+--------------------------------------------------------------------------------------------------+
| 1       | Manage variables and objects, control flow, handle logic operations, and debug code              |
+---------+--------------------------------------------------------------------------------------------------+

Manage variables and Objects
===============================

When working in interactive environments it is especially important to be aware of which variables have been
declared and which ones have not.  If you need a reminder of the contents of a specific variable you can always use
a ``print()`` statement or simply type the variable name and hit ``[Enter]``.  Another useful function is the ``ls()``
function which displays the names of objects in your workspace.

.. code-block:: r

    > x <- 1:5
    > y <- pi
    > ls()
    > z <- ls()
    > is.vector(z)

.. hint::
    If you want more than just a vector of names you can use ``ls.str()``

We used the term *object* a moment ago when describing the entities in your workspace that are returned with `ls()`.
We have discussed vectors, matrices, data frames, and lists all in the context of data containers.  We have also
discussed the different data types: numerics, integers, logicals and characters.  In R all types of in the workspace
are in effect objects.  Objects themselves are data structures that having attributes and methods.  Objects are
instaintated from classes (the code that defines the object).  The analogy that is commonly used is that classes are
like recipes and objects are the meals or dishes that have been prepared from those recipes.

.. code-block:: r

    > class(4)
    > class(TRUE)

In this course we will not be creating classes, but we will be working with them---in fact we have already been working
with them.  You now have a better understanding of what is listed when your run the ``ls()`` function, but what do we
do when there is a variable we wish to remove?

.. code-block:: r

   > a <- sum(4:8)
   > b <- Sys.Date()
   > ls()
   > rm(a)
   > ls()


Control flow
=====================

In computer science `control flow <https://en.wikipedia.org/wiki/Control_flow>`_ deals with the order with which
statements and sections of code are executed.  There are two main ways to *control the flow* of code in R: looping and
logical statements.  The logical statements can use ``if-else`` or ``switch`` and the looping most commonly use ``for``
and ``while`` statements.

.. code-block:: r

    temperature <- 30
    if (temperature > 26){
        print("Too hot")
    } else if (temperature > 20){
        print("Just right")
    } else{
        print("Too cool")
    }

Spend some time with this example to be sure you understand the syntax.  Review the logical operators in the next
section as well.  Can you re-create the this example to describe the phase changes of water (using Celsius the
standard in most of the world and in science)

Logical operators
---------------------

Here are some of the most common logical operators.

+------------------------------+----------------------------------------------------------------------------------+
| Logical operator             | Description                                | Example                             |
+==============================+============================================+=====================================+
| ``<``  or ``<=``             | Less than or less than or equal to         |  ``7 < 4 = FALSE``                  |
+------------------------------+--------------------------------------------+-------------------------------------+
| ``>``  or ``>=``             | Greater than or greater than or equal to   |  ``7 > 4 = TRUE``                   |
+------------------------------+--------------------------------------------+-------------------------------------+
| ``==``                       | Equal to                                   |  ``7==round(7.1) = TRUE``           |
+------------------------------+--------------------------------------------+-------------------------------------+
| ``!=``                       | NOT Equal to                               |  ``7!=round(7.1) = FALSE``          |
+------------------------------+--------------------------------------------+-------------------------------------+

There are other logical operators that you should have at your disposal.  For the logical or we have:

.. code-block:: r

    > x <- 1:10
    > x[x<4]
    > x[x<4 | x>6]

There is a similar syntax for the logical ``and``.  We also introduce the useful ``seq()`` function to setup a more
complex example.

.. code-block:: r

    > x <- seq(1,99,by=3)
    > x[x%%10==0 & x > 30]

Can you predict the above output before you execute it?

Switch statements are another way to control the flow of executed code in R.  The are less common than ``if-else``, but
it is worth noting that the syntax can serve as a shorthand way to specify certain logic.  The ``switch`` function can
be used with both indices or keywords.

.. code-block:: r

    > switch(3, "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")
    > switch("earth", "mercury"=0, "venus"=0, "earth"=1, "mars"=2, "jupiter"=79, "saturn"=82, "uranus"=27, "neptune"=14)

Looping
---------------

.. code-block:: r

    for (planet in c("earth", "mars", "jupiter", "saturn", "uranus", "neptune") ){
        num_moons <- switch(planet, "earth"=1, "mars"=2, "jupiter"=79, "saturn"=82, "uranus"=27, "neptune"=14)
        print(paste("The number of moons for planet", planet, "is", num_moons))
    }

You can further control the logic with ``next``.

.. code-block:: r

    for (planet in c("mecury","venus","earth", "mars", "jupiter", "saturn", "uranus", "neptune") ){
        num_moons <- switch(planet, "earth"=1, "mars"=2, "jupiter"=79, "saturn"=82, "uranus"=27, "neptune"=14)

        if (length(num_moons) == 0 ){
        next
        }

    print(paste("The number of moons for planet", planet, "is", num_moons))
    }

The for loop iterates over a specified number of items.  If we want to iterate until some condition is met the a
``while`` loop is more appropriate.

.. code-block:: r

    entered = -1
    while (entered < 1 | entered > 10){
        if (entered!=-1){
            print("That was  not a number betweeen 1 and 10")
        }
        entered <- as.integer(readline(prompt="Enter a number between 1 and 10: "))
    }

There is also function that is really a combination of ``if-else`` and looping and it is called the vectorized if
statement.

.. code-block:: r

    > x <- 1:21
    > ifelse(x %% 7 == 0, "divisible by 7", as.character(x))

Defining functions
========================

Lets start with the temperature function that you just worked with.  If you had a sensor that was reading the temperature
in your home and you wanted to get a message back based on a specified range you would need to package the code in a
re-usable definition or as a function.

.. code-block:: r

    check_temperature <- function(temperature){
        if (temperature > 26){
            print("Too hot")
        } else if (temperature > 20){
            print("Just right")
        } else{
            print("Too cool")
        }
    }

    check_temperature(15)
    check_temperature(25)

In this function we pass a single argument, temperature, and it does nothing different than before except that is
is far easier to re-use when we have a new value for temperature.  In the next function take our check number example
and turn it into a function  One difference between this one and the previous example is that we provide a default
value for the input variable.  provide a default number to begin with.


.. code-block:: r

    check_number <- function(entered=-1){

        while (entered < 1 | entered > 10){
            if (entered!=-1){
                print("That was  not a number between 1 and 10")
            }
            entered <- as.integer(readline(prompt="Enter a number between 1 and 10: "))
        }
    }

    check_number()


Debugging
================

It is inevitable that your code will contain `https://en.wikipedia.org/wiki/Software_bug <bugs>`_.  No matter how
good a programmer you are it is a certainty that you will encounter nuances in new data or a situation that a program
you have was not prepared to execute properly.  Here are the common debugging tools used in R.

.. tip::

    RStudio allows you to step-through your code by running only highlighted portions of a script.  This in combination
    with carefully placed ``print`` statements is an effective way of both debugging and understanding code.

``print()``
    The print function is incredible useful as a debugging tool.

``traceback()``
    Prints the call stack of the last uncaught error, that is the sequence of calls that lead to the error.

`Debugging with RStudio <https://support.rstudio.com/hc/en-us/articles/205612627-Debugging-with-RStudio>`_ is perhaps
the best all around resource you currently have at your disposal to carry out a prolonged debugging process.

Assignment
=========================

Use a folder with scripts or a Jupyter notebook to complete the following assignment.

:download:`assignment-1.md <./exercises/assignment-1.md>`