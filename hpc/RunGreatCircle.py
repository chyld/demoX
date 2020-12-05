import math,time
import numpy as np

def great_circle(lon1,lat1,lon2,lat2):
    radius = 3956 #miles
    x = math.pi/180.0

    a = (90.0-lat1)*(x)
    b = (90.0-lat2)*(x)
    theta = (lon2-lon1)*(x)
    c = math.acos((math.cos(a)*math.cos(b)) +
                  (math.sin(a)*math.sin(b)*math.cos(theta)))
    return radius*c

## create the matrix
lon1,lat1,lon2,lat2 = 42,0.5,-13,-32
n = 1e06
mat = np.zeros((int(n),4),)
mat = mat + [lon1,lat1,lon2,lat2]

## run the calculations
answers = [great_circle(*mat[i,:]) for i in range(mat.shape[0])]

print(answers[:10])
