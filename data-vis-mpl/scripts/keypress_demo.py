#!/usr/bin/env python

"""
Show how to connect to keypress events
"""
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt

class Modulator(object):
    count = 0
    
    @classmethod
    def iterate_count(cls):
        cls.count += 0.1
        return cls.count
        
def my_event(event,methd):
    print('press', event.key)
    sys.stdout.flush()

    xpos = methd()
    ax.text(xpos,0.5,event.key,
            verticalalignment='bottom',horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=25)
    fig.canvas.draw()
            
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', lambda event: my_event(event,Modulator.iterate_count))

ax.plot(np.random.rand(12), np.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')

plt.show()
