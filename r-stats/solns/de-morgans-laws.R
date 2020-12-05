#!/usr/bin/RScript

# solution to De Morgan’s laws questions

## example variables 
a <- c("A","B","C","D")
b <- c("C","D","E","F")
sample_space = c("A","B","C","D","E","F","G")

## set operations
intersect(a,b)
setdiff(a,b)
union(a,b)
complement_a <- setdiff(sample_space,a)

## Part 1 - The complement of the union of two sets is the same as the intersection of their complements
print("PART 1")
print(setdiff(sample_space,union(a,b)))
print(intersect(setdiff(sample_space,a), setdiff(sample_space,b)))

## Part 2 - The complement of the intersection of two sets is the same as the union of their complements
print("PART 2")
print(sort(setdiff(sample_space,intersect(a,b))))
print(sort(union(setdiff(sample_space,a), setdiff(sample_space,b))))