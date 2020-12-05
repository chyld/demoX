#!/usr/bin/env python

"""
===================================
Shaded & power normalized rendering
===================================

The Mandelbrot set rendering can be improved by using a normalized recount
associated with a power normalized colormap (gamma=0.3). Rendering can be
further enhanced thanks to shading.

The `maxiter` gives the precision of the computation. `maxiter=200` should
take a few seconds on most modern laptops.
"""
import numpy as np


def mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, np.complex64)
    for n in range(maxiter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2.1 + C[I]
    N[N == maxiter-1] = 0
    return Z, N


def main(cmap,filename,txt):
    
    xmin, xmax, xn = -2.25, +0.75, 3500/2
    ymin, ymax, yn = -1.25, +1.25, 2500/2
    maxiter = 200
    horizon = 2.0 ** 40
    log_horizon = np.log(np.log(horizon))/np.log(2)
    Z, N = mandelbrot_set(int(xmin), int(xmax), int(ymin), int(ymax), int(xn), int(yn), int(maxiter), int(horizon))

    # This line will generate warnings for null values but it is faster to
    with np.errstate(invalid='ignore'):
        M = np.nan_to_num(N + 1 -
                          np.log(np.log(abs(Z)))/np.log(2) +
                          log_horizon)

    dpi = 72
    width = 10
    height = 10*yn/xn
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)
        
    # Shaded rendering
    light = colors.LightSource(azdeg=315, altdeg=10)
    M = light.shade(M, cmap=cmap, vert_exag=1.5,
                    norm=colors.PowerNorm(0.5), blend_mode='hsv')
    plt.imshow(M, extent=[xmin, xmax, ymin, ymax], interpolation="bicubic")

    plt.text(0.5, 0.5, txt, size=60, transform=ax.transAxes,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec='black',
                       fc='wheat',
             ))


    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename,dpi=400)


if __name__ == '__main__':

    import time
    import matplotlib
    from matplotlib import colors
    import matplotlib.pyplot as plt

    text = ["Python Essentials", "Math Essentials", "Stats Essentials"]
    
    for c, cmap in enumerate([plt.cm.gnuplot,plt.cm.jet,plt.cm.rainbow]):
        filename = "mandelbrot-{}.png".format(c+1)
        print("...creating {}".format(filename))
        main(cmap,filename,text[c])
    
    print("done")
