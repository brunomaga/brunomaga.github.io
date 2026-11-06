"""Draw gd_step_size.svg: gradient descent on f(x) = x^2 from x_0 = 1, with the step sizes 0.1 and 1.1.

Each iteration computes x_{i+1} = (1 - 2 gamma) x_i, the example of the post Algebra for ML.
usage: python3 gd_step_size.py   (needs matplotlib, numpy and the fonts of CLAUDE.md, Pictures, section 4)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts
from matplotlib import transforms

PANELS = [  # step size, iterations, x limits, y limit, x ticks, y ticks
    (0.1, 10, (-0.4, 1.25), 1.3, [0, 0.5, 1], [0, 0.5, 1]),
    (1.1, 5, (-2.75, 2.75), 7.0, [-2, 0, 2], [0, 2, 4, 6]),
]

fig = charts.figure(570, 250, 'gd_step_size')
for k, (gamma, n, xl, yl, xt, yt) in enumerate(PANELS):
    ax = charts.axes(fig, 45 + 295 * k, 45, 220, 170)
    x = np.linspace(*xl, 401)
    ax.plot(x, x**2, color=charts.BLACK)
    xs = [1.0]
    for _ in range(n):
        xs.append(xs[-1] - gamma * 2 * xs[-1])
    pts = [(v, v * v) for v in xs]
    ax.set(xlim=xl, ylim=(0, yl), xticks=xt, yticks=yt)
    for p, q in zip(pts, pts[1:]):
        if np.hypot(*(ax.transData.transform(q) - ax.transData.transform(p))) > 22:  # room for line and head
            charts.arrow(ax, p, q, color=charts.RED, gap0=4.5, gap1=4.5)
    ax.plot(*zip(*pts), markerfacecolor=charts.RED_FILL, **charts.DOT)
    charts.runs(ax, 1, 1, charts.sub('𝑥', '0'), ha='left',
                transform=transforms.offset_copy(ax.transData, fig, x=7, y=-10, units='dots'))
    ax.set_title('𝛾 = %g' % gamma, family=charts.MATH)
    ax.set_xlabel('𝑥', family=charts.MATH)
    ax.set_ylabel('𝑓 (𝑥)', family=charts.MATH)
    charts.tick_labels(ax)
charts.save(fig, os.path.join(HERE, 'gd_step_size.svg'))
