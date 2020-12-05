import math, os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def func_timer(func):
    """Times how long the function took."""

    def f(*args, **kwargs):
        import time
        start = time.time()
        results = func(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"Elapsed: {elapsed_time:.2f}s")
        return(results)
    return(f)
    
@func_timer
def test_function(msg, sleep=1.0):
    """Delays a while before answering."""
    import time
    time.sleep(sleep)
    print(msg)

@func_timer
def loop1(n):
    """Using for loop with function call."""
    z = []
    for i in range(n):
        z.append(math.sin(i))
    return z

@func_timer
def loop2(n):
    """Using local version of function."""
    z = []
    sin = math.sin
    for i in range(n):
        z.append(sin(i))
    return z

@func_timer
def loop3(n):
    """Using list comprehension."""
    sin = math.sin
    return [sin(i) for i in range(n)]

@func_timer
def loop4(n):
    """Using map."""
    sin = math.sin
    return list(map(sin, range(n)))

@func_timer
def loop5(n):
    """Using numpy."""
    return np.sin(np.arange(n)).tolist()


def calculate_great_circle(args):
    """one step of the great circle calculation"""
    lon1,lat1,lon2,lat2 = args
    
    radius = 3956.0
    x = np.pi/180.0
    a,b = (90.0-lat1)*(x),(90.0-lat2)*(x)
    theta = (lon2-lon1)*(x)

    c =  np.arccos((np.cos(a)*np.cos(b)) +
                   (np.sin(a)*np.sin(b)*np.cos(theta)))
    return(radius*c) 

@func_timer
def great_circle_looping(mat):
    """basic great circle"""

    result = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        result[i] = calculate_great_circle(mat[i,:])
        
    return result

@func_timer
def great_circle_numpy(mat):
    """numpys great circle"""
    radius = 3956
    x = np.pi/180.0
    lon1 = mat[:,0]
    lat1 = mat[:,1]
    lon2 = mat[:,2]
    lat2 = mat[:,3]
    
    a = (90.0-lat1)*(x)
    b = (90.0-lat2)*(x)
    theta = (lon2-lon1)*(x)
    c = np.arccos((np.cos(a)*np.cos(b)) +
                  (np.sin(a)*np.sin(b)*np.cos(theta)))
    return radius*c

class GreatCircleMultiprocessing(object):
    """multiprocessing version using async pool"""
    def __init__(self,mat):
        self.mat = mat
    
    def run(self,mat):
        po = Pool(processes=cpu_count()-1)
        _results = po.map_async(great_circle,(mat[i,:] for i in range(mat.shape[0])))
        results =  _results.get()
        return(results)

    def great_circle_multiprocessing(self,mat,chunksize=500):
        """multiprocessing version of the great circle usng Ececutors"""
        with ProcessPoolExecutor(max_workers=cpu_count()-1) as pool:
            result = pool.map(great_circle,(mat[i,:] for i in range(mat.shape[0])),chunksize=chunksize)

        return(np.array(list(result)))

    
def great_circle_cython(mat):
    os.system("python setup_gc.py build_ext --inplace")
    from greatcircle import great_circle
    result = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        result[i] = great_circle(*mat[i,:])
    return(result)


if __name__ == "__main__":
    
    n = 500000
    m = np.random.randint(-360,360,n*4).reshape(n,4)

    @func_timer
    def run_slow_way(mat):
        for i in range(mat.shape[0]):
            x = great_circle(mat[i,:])

    ## slow way
    print("great circle via looping...")
    run_slow_way(m)
                   
    ## numpy way
    print("great circle via numpy...")
    great_circle_numpy(m)    
    

