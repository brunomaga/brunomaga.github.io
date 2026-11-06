"""Draw coordinate_descent.svg: coordinate descent on f(x, y) = 5x^2 - 6xy + 5y^2 from (-0.5, -1).

Each step minimizes f along one coordinate, y = 0.6 x then x = 0.6 y, as in the figure of the Wikipedia article.
usage: python3 coordinate_descent.py   (needs matplotlib, numpy and the fonts of CLAUDE.md, Pictures, section 4)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts

pts = [(-0.5, -1.0)]
for i in range(12):                             # alternate the minimum along y and along x
    x, y = pts[-1]
    pts.append((x, 0.6 * x) if i % 2 == 0 else (0.6 * y, y))

fig = charts.figure(360, 360, 'coordinate_descent')
ax = charts.axes(fig, 55, 50, 270, 270)
g = np.linspace(-1.5, 1.5, 601)
X, Y = np.meshgrid(g, g)
cs = ax.contour(X, Y, 5 * X**2 - 6 * X * Y + 5 * Y**2, levels=[0.5, 1.4, 2.3, 3.2, 4.1, 5.0], colors=charts.BLACK,
                linewidths=1.0, zorder=1)
ax.clabel(cs, fmt='%.1f', fontsize=12)
ticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect='equal', xticks=ticks, yticks=ticks)
for p, q in zip(pts, pts[1:]):
    if np.hypot(*(ax.transData.transform(q) - ax.transData.transform(p))) > 14:  # room for line and head
        charts.arrow(ax, p, q, color=charts.RED)
    else:
        ax.plot(*zip(p, q), color=charts.RED, lw=1.4, zorder=2.1)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(lambda v, _: ('%.1f' % v).replace('-', '−'))
ax.set_title('𝑓 (𝑥, 𝑦) = 5𝑥² − 6𝑥𝑦 + 5𝑦²', family=charts.MATH)
ax.set_xlabel('𝑥', family=charts.MATH)
ax.set_ylabel('𝑦', family=charts.MATH)
charts.save(fig, os.path.join(HERE, 'coordinate_descent.svg'))
