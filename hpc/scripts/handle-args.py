#!/usr/bin/env python
"""
simple example to show how to handle arguments
"""

import sys,getopt

## collect args
argString = f"{sys.argv[0]} -f filepath -d [optional debug]"
try:
    optlist, args = getopt.getopt(sys.argv[1:],'f:d')
except getopt.GetoptError:
    print(getopt.GetoptError)
    raise Exception(argString)

## handle args
debug = False
filePath = None
for o, a in optlist:
    if o == '-f':
        filePath = a
    if o == '-d':
        debug = True

if filePath == None:
    raise Exception(argString)
if not os.path.exists(filePath):
    print(f"... {filePath}")
    raise Exception("bad file path")
                    
