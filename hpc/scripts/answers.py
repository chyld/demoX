#!/usr/bin/env python
"""
run file for the speedy computing lecture
"""

import os,sys,subprocess
from speedylib import *

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
np.random.seed(80)

###########################################
## test function for timer decorator
###########################################

print("\n------------------------------")
print("...running decorator example")
test_function("Hello Decorator World", 1.0)

###########################################
## loop testing
###########################################
print("\n------------------------------")
print("...running loops test")
n = 1000000
print("basic loop")
loop1(n)
print("local version of function")
loop2(n)
print("list comprehension")
loop3(n)
print("using map")
loop4(n)
print("using numpy")
loop5(n)

###########################################
## subprocessing
###########################################
#print("\n------------------------------")
#print("running subprocessing example...")
#cmd = 'python speedylib.py'
#proc = subprocess.call(cmd,shell=True)

###########################################
## specify data
###########################################
n = 500000
m = np.random.randint(-360,360,n*4).reshape(n,4)
eps = np.spacing(1) 
n = n + eps

###########################################
## basic looping
###########################################
print("\n------------------------------")
print("...via basic looping")
result = great_circle_looping(m)

###########################################
## subprocessing
###########################################
#print("\n------------------------------")
#print("running subprocessing example...")
#cmd = 'python speedylib.py'
#proc = subprocess.call(cmd,shell=True)

###########################################
## numpy
###########################################
print("\n------------------------------")
print("...via numpy")
result = great_circle_numpy(m)

###########################################
## multiprocessing
###########################################
print("\n------------------------------")
print("...via multiprocessing (old)")
po = Pool(processes=cpu_count()-1)
_results = po.map_async(calculate_great_circle,(m[i,:] for i in range(m.shape[0])))
results =  _results.get()





sys.exit()




#result = great_circle_multiprocessing(m)

###########################################
## multiprocessing (old way)
###########################################
#print("...via multiprocessing (old way)")
#result = great_circle_multiprocessing_old(m)

###########################################
## cython
###########################################
print("...via cython")
result = great_circle_cython(m)
