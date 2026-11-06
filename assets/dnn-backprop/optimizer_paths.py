"""Draw optimizer_paths.svg: 40 steps of GD, momentum and Adam on f(theta) = theta_1^2 / 2 + 25 theta_2^2.

The updates are the equations of the Optimizers section of the post, with the exact gradient.
usage: python3 optimizer_paths.py   (needs matplotlib, numpy and the fonts of CLAUDE.md, Pictures, section 4)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts
from matplotlib import transforms

CURV = np.array([1.0, 50.0])                    # f(theta) = sum(CURV * theta^2) / 2, gradient CURV * theta
START, STEPS = np.array([-4.0, 1.0]), 40


def path(update):
    theta, state, out = START.copy(), {}, [START]
    for t in range(1, STEPS + 1):
        theta = theta - update(CURV * theta, t, state)
        out.append(theta)
    return np.array(out)


def momentum(eta, mu):
    def update(g, t, s):
        s['m'] = mu * s.get('m', 0) + g
        return eta * s['m']
    return update


def adam(eta, beta1=0.9, beta2=0.999, eps=1e-8):
    def update(g, t, s):
        s['m'] = beta1 * s.get('m', 0) + (1 - beta1) * g
        s['v'] = beta2 * s.get('v', 0) + (1 - beta2) * g**2
        return eta * (s['m'] / (1 - beta1**t)) / (np.sqrt(s['v'] / (1 - beta2**t)) + eps)
    return update


OPTIMIZERS = [('GD', charts.BLUE, momentum(0.035, 0)), ('Momentum', charts.GREEN, momentum(0.006, 0.9)),
              ('Adam', charts.RED, adam(0.2))]

fig = charts.figure(640, 270, 'optimizer_paths')
ax = charts.axes(fig, 45, 45, 480, 192)
ax.set(xlim=(-4.5, 1.5), ylim=(-1.2, 1.2), aspect='equal', xticks=range(-4, 2), yticks=[-1, 0, 1])
t = np.linspace(0, 2 * np.pi, 401)
for i in range(1, 9):                           # contours where theta_1 = 1, ..., 8 on the axis
    ax.plot(i * np.cos(t), i * np.sin(t) / np.sqrt(CURV[1]), color=charts.BLACK, lw=1.0, zorder=1)
for name, color, update in OPTIMIZERS:
    p = path(update)
    ax.plot(*p.T, color=color, label=name, marker='o', markersize=3.5, markeredgewidth=0)
    print(name, 'theta_40 =', p[-1].round(3))
ax.plot(*START, markerfacecolor=charts.BLACK, **charts.DOT)
ax.plot(0, 0, marker='x', markersize=7, markeredgewidth=1.8, color=charts.BLACK, zorder=2.2)
charts.tick_labels(ax)
charts.runs(ax, 0.5, 0, charts.sub('𝜃', '1'), transform=transforms.offset_copy(ax.transAxes, fig, y=-38, units='dots'))
charts.runs(ax, 0, 0.5, charts.sub('𝜃', '2'), ha='right',
            transform=transforms.offset_copy(ax.transAxes, fig, x=-26, y=-4, units='dots'))
ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.0))
charts.save(fig, os.path.join(HERE, 'optimizer_paths.svg'))
