"""Shared style of the charts of the posts (CLAUDE.md, Pictures): matplotlib at page scale, where 1 unit is 1 px.

Text in Source Sans 3 and formulas in New Computer Modern Sans Math, both drawn as glyph outlines, 1.8 px axes and
data lines, and the accent colors of the palette. A chart script imports this module, draws its figure and saves it
next to itself with save().
"""
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

MATH = 'NewComputerModernSansMath'
BLACK, GREY = '#000000', '#666666'
BLUE, GREEN, ORANGE, YELLOW, RED, PURPLE = '#6c8ebf', '#82b366', '#d79b00', '#d6b656', '#b85450', '#9673a6'  # accents
BLUE_FILL, GREEN_FILL, ORANGE_FILL, YELLOW_FILL, RED_FILL, PURPLE_FILL = ('#dae8fc', '#d5e8d4', '#ffe6cc', '#fff2cc',
                                                                        '#f8cecc', '#e1d5e7')  # their pastel fills
DOT = dict(marker='o', markersize=6, markeredgecolor=BLACK, markeredgewidth=1.0, linestyle='none', zorder=2.2)


def figure(width, height, name):
    """A figure of width x height px; name keeps the ids of the SVG the same on every run."""
    plt.rcParams.update({
        'font.family': 'Source Sans 3', 'font.size': 12, 'svg.fonttype': 'path', 'svg.hashsalt': name,
        'axes.linewidth': 1.8, 'axes.edgecolor': BLACK, 'axes.labelsize': 14, 'axes.labelcolor': BLACK,
        'axes.titlesize': 14, 'axes.titlepad': 8, 'lines.linewidth': 1.8, 'lines.solid_capstyle': 'butt',
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'xtick.color': BLACK, 'ytick.color': BLACK,
        'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'xtick.major.size': 4, 'ytick.major.size': 4,
        'legend.fontsize': 14, 'legend.frameon': False, 'legend.handlelength': 20 / 14, 'legend.handletextpad': 4 / 14,
        'legend.borderaxespad': 0, 'legend.borderpad': 0, 'legend.labelspacing': 0.4,
    })
    return plt.figure(figsize=(width / 72, height / 72), dpi=72)


def axes(fig, left, bottom, width, height):
    """Axes placed in px from the bottom left corner of the figure."""
    W, H = fig.get_size_inches() * 72
    return fig.add_axes((left / W, bottom / H, width / W, height / H))


def tick_labels(ax):
    """Tick labels with no trailing zeros (0.5, 1, 1.5) and a true minus sign."""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(lambda v, _: ('%g' % v).replace('-', '−'))


def runs(ax, x, y, parts, ha='center', transform=None):
    """Draw parts (text, family, size, rise in px) one after the other on the baseline at (x, y), e.g. subscripts."""
    trans, texts, dx = transform or ax.transData, [], 0.0
    for s, family, size, rise in parts:
        t = ax.text(x, y, s, family=family, size=size, ha='left', va='baseline', clip_on=False)
        texts.append((t, dx, rise))
        dx += t.get_window_extent().width
    shift = {'left': 0.0, 'center': -dx / 2, 'right': -dx}[ha]
    for t, x0, rise in texts:
        t.set_transform(transforms.offset_copy(trans, ax.figure, x=x0 + shift, y=rise, units='dots'))


def sub(base, index, size=14):
    """The parts of a math variable with a subscript: the index 2 px smaller and 3.5 px lower."""
    return [(base, MATH, size, 0), (index, MATH, size - 2, -3.5)]


def arrow(ax, p0, p1, color=BLACK, lw=1.4, gap0=0.0, gap1=0.0, zorder=2.1):
    """A line from p0 to p1 (data coordinates) with draw.io's classic head, 6.4 px long and wide, its notch at 3/4 of
    its length; gap0 and gap1 shorten the arrow at each end, in px."""
    to_px, to_data = ax.transData, ax.transData.inverted()
    a, b = to_px.transform(p0), to_px.transform(p1)
    u = (b - a) / np.hypot(*(b - a))
    n = np.array([-u[1], u[0]])
    a, tip = a + gap0 * u, b - gap1 * u
    head = [tip, tip - 6.4 * u + 3.2 * n, tip - 4.8 * u, tip - 6.4 * u - 3.2 * n]
    ax.plot(*to_data.transform([a, tip - 4.8 * u]).T, color=color, lw=lw, zorder=zorder, clip_on=False)
    ax.fill(*to_data.transform(head).T, color=color, lw=0, zorder=zorder, clip_on=False)


def save(fig, path):
    """Write the SVG with its width and height in px and no DTD, as the other SVGs of the site."""
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02, transparent=True, metadata={'Date': None})
    head, rest = open(path).read().split('<svg', 1)
    attrs, body = rest.split('>', 1)
    open(path, 'w').write(head.split('<!DOCTYPE', 1)[0] + '<svg' + attrs.replace('pt"', '"', 2) + '>' + body)
