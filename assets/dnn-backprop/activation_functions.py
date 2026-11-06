"""Draw activation_functions.svg: the sigmoid, tanh, ReLU and Leaky ReLU (alpha = 0.1) and their derivatives.

usage: python3 activation_functions.py   (needs matplotlib, numpy and the fonts of CLAUDE.md, Pictures, section 4)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts

ALPHA = 0.1
x = np.linspace(-3, 3, 601)
neg, pos = x[x <= 0], x[x >= 0]
sig = 1 / (1 + np.exp(-x))
FUNCTIONS = [  # name, color, dashed, curve pieces (x, y) of the function and of its derivative
    ('Sigmoid', charts.BLUE, False, [(x, sig)], [(x, sig * (1 - sig))]),
    ('tanh', charts.GREEN, False, [(x, np.tanh(x))], [(x, 1 - np.tanh(x)**2)]),
    ('ReLU', charts.ORANGE, False, [(x, np.maximum(0, x))], [(neg, 0 * neg), (pos, 1 + 0 * pos)]),
    ('Leaky ReLU', charts.RED, True, [(x, np.maximum(ALPHA * x, x))], [(neg, ALPHA + 0 * neg), (pos, 1 + 0 * pos)]),
]
PANELS = [('𝜙(𝑥)', (-1.25, 3), [-1, 0, 1, 2, 3]), ('𝜙′(𝑥)', (-0.1, 1.15), [0, 0.5, 1])]

fig = charts.figure(720, 240, 'activation_functions')
for k, (label, ylim, yticks) in enumerate(PANELS):
    ax = charts.axes(fig, 50 + 305 * k, 45, 230, 170)
    for name, color, dashed, *curves in FUNCTIONS:
        for i, (xs, ys) in enumerate(curves[k]):
            ax.plot(xs, ys, color=color, linestyle=(0, (3, 3)) if dashed else '-', label=name if i == 0 else None)
    ax.set(xlim=(-3, 3), ylim=ylim, xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=yticks)
    ax.grid(color='#b3b3b3', linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel('𝑥', family=charts.MATH)
    ax.set_ylabel(label, family=charts.MATH)
    charts.tick_labels(ax)
ax.legend(loc='upper left', bbox_to_anchor=(1.07, 1.0))
charts.save(fig, os.path.join(HERE, 'activation_functions.svg'))
