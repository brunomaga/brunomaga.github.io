"""Draw l1_l2_regularization.svg: a squared loss minimized inside the regions ||w||_1 <= 1 and ||w||_2 <= 1.

Solutions are searched on the boundaries of the regions; layout of figure 3.11 of The Elements of Statistical Learning.
usage: python3 l1_l2_regularization.py   (needs matplotlib, numpy and the fonts of CLAUDE.md, Pictures, section 4)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts
from matplotlib import transforms

W_LS = np.array([0.9, 1.7])                                   # least squares solution
ANGLE, EIG = np.radians(-35), np.array([1.0, 4.0])            # axes and curvatures of the loss
R = np.array([[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]])
A = R @ np.diag(EIG) @ R.T
XLIM, YLIM = (-1.6, 3.0), (-1.6, 3.7)

t = np.linspace(0, 2 * np.pi, 100001)
circle = np.stack([np.cos(t), np.sin(t)], 1)
REGIONS = [('L1 regularization', circle / np.abs(circle).sum(1, keepdims=True)), ('L2 regularization', circle)]

fig = charts.figure(700, 310, 'l1_l2_regularization')
for k, (title, boundary) in enumerate(REGIONS):
    d = boundary - W_LS
    loss = np.einsum('ni,ij,nj->n', d, A, d)
    w, level = boundary[loss.argmin()], loss.min()
    ax = charts.axes(fig, 10 + 250 * k, 10, 230, 265)
    ax.set(xlim=XLIM, ylim=YLIM, aspect='equal')
    ax.axis('off')
    ax.fill(*boundary[::500].T, facecolor=charts.BLUE_FILL, edgecolor=charts.BLACK, lw=1.0, zorder=1)
    for f in (1 / 3, 2 / 3, 1):                                # contours at 1/9, 4/9 and 1 of the level of w
        ax.plot(*(W_LS + f * np.sqrt(level) * circle[::500] / np.sqrt(EIG) @ R.T).T, color=charts.BLACK, lw=1.0,
                zorder=1.5)
    charts.arrow(ax, (XLIM[0], 0), (XLIM[1], 0), lw=1.8, zorder=2)
    charts.arrow(ax, (0, YLIM[0]), (0, YLIM[1]), lw=1.8, zorder=2)
    charts.runs(ax, XLIM[1], 0, charts.sub('𝑤', '1'), ha='right',
                transform=transforms.offset_copy(ax.transData, fig, x=0, y=-18, units='dots'))
    charts.runs(ax, 0, YLIM[1], charts.sub('𝑤', '2'), ha='right',
                transform=transforms.offset_copy(ax.transData, fig, x=-8, y=-12, units='dots'))
    ax.plot(*W_LS, markerfacecolor=charts.BLACK, label='least squares', **charts.DOT)
    ax.plot(*w, markerfacecolor=charts.RED, label='regularized', **charts.DOT)
    ax.set_title(title, weight='bold')
    print(title, 'solution', w.round(3), 'loss', level.round(3))
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 0.95), handlelength=1.0)
charts.save(fig, os.path.join(HERE, 'l1_l2_regularization.svg'))
