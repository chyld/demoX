## Assignment Questions

1. Write a function that takes a user inputted number and prints whether it is positive, negative or zero, with "The inputted number is (positive/negative/zero)" depending.

2. Write a function that takes two user inputted numbers and prints "The first number is larger" or "The second number is larger" depending on which is larger. (**Hint**: you'll need to use `input()` twice.)

3. Write a function that computes the sum from 0 to a user inputted number. This time though, start at the user inputted number and work down. This answer will look very much like the example above, you'll just need to change a couple of things.

4. Write a function that computes the factorial of a user inputted number. If you don't know what a factorial is or need a review, check [this](https://en.wikipedia.org/wiki/Factorial) link out. Again, your solution is going to look a lot like the code above. Things you should think about:
    * What is the process of computing a factorial if you were to compute it by hand?
    * What is the common starting place when trying to compute the factorial of any number?<br><br>

5.  Write a function that computes and prints all of the divisors of a user inputted number. If you don't know what a divisor is or need a review, check out [this](https://en.wikipedia.org/wiki/Divisor) link. Things to think about:
    * How do you determine if a single number is a divisor of another?
    * How do you do this multiple times (**Hint**: it involves a for loop)?<br><br>

6.  Write a function that computes the greatest common divisor between two user inputted numbers. If you don't know what a greatest common divisor is, check out [this](https://en.wikipedia.org/wiki/Greatest_common_divisor) link. 

7.  Write a function that computes the least common multiple between two user inputted numbers. If you don't know what a least common multiple is or want a review check [this](https://en.wikipedia.org/wiki/Least_common_multiple) out. 

8. Write a function that determines whether or not a user inputted number is a prime number and prints `'The number you inputted is (not) a prime number.'` depending on what your script finds. If you don't know what a prime number is or need a review, check out [this](https://en.wikipedia.org/wiki/Prime_number) link. Things to think about:
    * How do you check if a number is divisible by another number?
    * What numbers are a prime number divisible by?
    * How do you check all of the numbers a number could be divisible by?

9.  One can use loops to compute the elements of a mathematical series. Series can be defined recursively with the value of each element depending on the one that comes before it. Consider the series created by the rules:

        > a[0]=1, a[i+1] = 2*a[i] + 1, for i >0

    Write a script that prints the `nth` element in the series as determined by input from the user. e.g. If the user inputs the number `3`, your script should print the 3rd element in the series, `15`. You're welcome to check the math! Things to think about:
    * You know you're going to use a loop to solve this problem, how?
    * How do you store each of the elements as you calculate them with the loop?
    * How many elements do you need to keep track of at any one time?<br><br>

10. Challenge: solve the equation:

    `(a + (b - c) * d - e) * f = 75`

    where a, b, c, d, e, and f are unique integers in the range [1, 6].

    Hints:
      - Computers are so fast that your program can simply try all possible valid values of a, b, c, d, e, and f until it finds one permutation of 1-6 that solves the challenge! (Btw, there is only *one* permutation that will solve it.)
      - Use 6 nested for-loops to enumerate all ways of setting each of a, b, c, d, e, and f to the values 1-6.


    Want more? Modify your program to solve all these (very similar) equations:

    ```
    (a + (b - c) * d - e) * f = 22
    (a + (b - c) * d - e) * f = 38
    (a + (b - c) * d - e) * f = 46
    (a + (b - c) * d - e) * f = 57
    (a + (b - c) * d - e) * f = 78
    (a + (b - c) * d - e) * f = 80
    (a + (b - c) * d - e) * f = 81
    (a + (b - c) * d - e) * f = 88
    (a + (b - c) * d - e) * f = 92
    (a + (b - c) * d - e) * f = 100
    (a + (b - c) * d - e) * f = 102
    (a + (b - c) * d - e) * f = 104
    (a + (b - c) * d - e) * f = 105
    ```