## Assignment 1 - Optimizing code


### Question 1

Write a Python script or create a Jupyter notebook that compares the following functions.  

* Run each of the following functions over a range of `n` using a timer function that you create.
* Save your results as a Pandas data frame.  (Hint: this can be done from a dictionary)
* Create a simple summarizing plot

If you not already aware [Pandas can create plots fairly easily](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html).

```python
def loop1(n):
    """Using for loop with function call."""
    z = []
    for i in range(n):
        z.append(math.sin(i))
    return z

def loop2(n):
    """Using local version of function."""
    z = []
    sin = math.sin
    for i in range(n):
        z.append(sin(i))
    return z

def loop3(n):
    """Using list comprehension."""
    sin = math.sin
    return [sin(i) for i in range(n)]

def loop4(n):
    """Using map."""
    sin = math.sin
    return list(map(sin, range(n)))

def loop5(n):
    """Using numpy."""
    return np.sin(np.arange(n)).tolist()
```

### Question 2

For the fastest two functions that you found in question 1, use the 
[timeit](https://docs.python.org/3/library/timeit.html#basic-examples) interface to make a direct comparison.



