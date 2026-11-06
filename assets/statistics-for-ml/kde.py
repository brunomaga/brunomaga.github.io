"""Draw kde.svg: a histogram and a Gaussian kernel density estimate of the six data points of the Wikipedia example.

Histogram bins of width 2, and kernels of bandwidth h = 1.5, each divided by the number of points.
usage: python3 kde.py   (needs matplotlib, numpy and the fonts of CLAUDE.md, Pictures, section 4)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts

POINTS, H = np.array([-2.1, -1.3, -0.4, 1.9, 5.1, 6.2]), 1.5
x = np.linspace(-6.5, 10.5, 681)
kernels = np.exp(-0.5 * ((x[:, None] - POINTS) / H)**2) / (H * np.sqrt(2 * np.pi) * len(POINTS))

fig = charts.figure(560, 250, 'kde')
for k in range(2):
    ax = charts.axes(fig, 60 + 280 * k, 45, 210, 180)
    if k == 0:
        ax.hist(POINTS, bins=range(-4, 9, 2), density=True, facecolor='#d5e8d4', edgecolor=charts.BLACK, lw=1.0)
    else:
        ax.fill_between(x, kernels.sum(1), facecolor='#d5e8d4', lw=0)
        ax.plot(x, kernels, color=charts.RED, linestyle=(0, (3, 3)))
        ax.plot(x, kernels.sum(1), color=charts.BLUE)
    ax.vlines(POINTS, 0, 0.008, color=charts.BLACK, lw=1.8, zorder=2.2)
    ax.set(xlim=(-6.5, 10.5), ylim=(0, 0.175), xticks=[-5, 0, 5, 10], yticks=[0, 0.05, 0.1, 0.15])
    ax.xaxis.set_major_formatter(lambda v, _: ('%g' % v).replace('-', '−'))
    ax.yaxis.set_major_formatter(lambda v, _: '%.2f' % v)
    ax.set_xlabel('𝑥', family=charts.MATH)
    ax.set_ylabel('Density function')
charts.save(fig, os.path.join(HERE, 'kde.svg'))
