## Multiprocessing Assignment


### Question:
Imagine a unit circle (radius = 1, centered on zero) inside a unit square. The area of the circle divided by the area of the 
square is (π r2) / (2 r)2 = π / 4.0. A random (x, y) point, -1 < x < 1, -1 < y < 1, must fall within the square. 
The probability that it also falls within the circle is proportional to the area of the circle. 
Thus, if you generate X random points, the number of points that fall within the circle divided by the total number of points 
comes to π / 4.0 as X approaches infinity. That is,

π = 4.0 * (number of circle points / X)

Write a parallel Python program to calculate π. Let X = 100,000. 
Run the calculation 1000 times and average the values of π computed.


### Bonus Question 

Consider G as a graph of vertices and directed edges. Given a start vertex S, return all vertices reachable from S. 
In your directory, the file edges gives a double column list of edges. The first line is the number of vertices [0 .. N) and 
edges. The rest of the lines give the source and destination vertex of an edge.

The most straightforward parallel implementation of Breadth First Search is the horizon method that keeps two lists, old and new. 
Choose a start vertex, say vertex 0, and append it to old. Now enter an iterative loop with the guard

```python
while len(old) != 0:
```
In the body of the loop, for each vertex v in old, examine the neighbors of v. Append each unvisited neighbor to new. 
At the bottom of the while loop, swap new and old. To execute the algorithm, besides old and new, you will need two other data structures:

neighbors ¬– a list of integer lists. The ith list holds the neighbors of vertex i. visited ¬– an integer list. 
The ith element is 0 if vertex i is unvisited; else 1. Initialize the list to 0.
open the file search.py in your directory (the code is the the folder). The file creates the processes, manager, and shared structures you will need to run the program. The file reads the edges file and creates the list of neighbors, edges.

