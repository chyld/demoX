#!/usr/bin/env python

## adjust the backend
import matplotlib as mpl
mpl.use("Agg")

## make imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800, codec='h264')

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("my cool animation")
ax.set_ylabel("y")
ax.set_xlabel("x")

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x),color='yellow')

def animate(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True)

ani.save('../_static/lines.mp4', writer=writer)

