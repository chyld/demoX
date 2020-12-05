#!/usr/bin/env python
"""
simple example of how to calculate runtime
"""

import time


# start timer
run_start = time.time()

# do something
time.sleep(2)
run_stop = time.time()

# save runtime as a variable
m, s = divmod(run_stop - run_start, 60)
h, m = divmod(m, 60)
run_time = f"{int(h):d}:{int(m):02d}:{int(s):02d}"

print(run_time)

     
