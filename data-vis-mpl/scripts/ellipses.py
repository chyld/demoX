import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
delta = 45.0 # degrees

angles = np.arange(0, 360+delta, delta)
ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

for e in ells:
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.1)
    ax.add_artist(e)

ax.set_xlim(-2, 4)
ax.set_ylim(-1, 3)
plt.show()
