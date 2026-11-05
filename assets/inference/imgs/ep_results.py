"""Draw ep_results.svg: the runtime of the five implementations of the expert parallelism post.

Top left, one MoE layer in prefill; the other three panels, the whole model in prefill, in decode and in
both. The time per iteration, in ms, is drawn against the input tokens per rank on log2 axes, where a
constant speedup is a constant vertical offset. The two phased implementations carry a line and a filled
marker and their non-phased counterparts the same marker, open and without a line, so that each pair reads
as one. The data comes from the paper of the post. The picture follows CLAUDE.md (Pictures): Source Sans 3
text and New Computer Modern Sans Math formulas at page size (1 unit = 1 px), drawn as glyph outlines,
1.8 px axes and data lines, the data drawn behind the axes, and one legend in one column right of the plots.

usage: python3 ep_results.py   (needs matplotlib, Source Sans 3 and New Computer Modern Sans Math)
"""
import os
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

NAMES = ['Baseline', 'MegaKernel Phased', 'MegaKernel', 'MegaKernel INC Phased', 'MegaKernel INC']
TOKENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
# time per iteration in ms, one list per implementation, in the order of NAMES
MOE_PREFILL = [[4.37, 7.12, 13.66, 22.67, 37.89, 71.21, 141.96],
               [2.53, 4.32, 8.62, 15.40, 30.22, 55.59, 115.45],
               [2.36, 4.07, 8.23, 14.87, 29.26, 54.28, 110.95],
               [2.31, 3.70, 6.46, 11.22, 20.27, 37.63, 72.98],
               [2.27, 3.62, 6.09, 10.68, 19.75, 37.04, 70.66]]
MODEL_PREFILL = [[94.13, 139.53, 253.88, 427.86, 732.97, 1389.42, 2766.20],
                 [51.00, 90.86, 172.84, 310.55, 611.16, 1138.50, 2340.94],
                 [53.00, 88.25, 165.46, 302.42, 593.72, 1118.44, 2270.58],
                 [51.96, 81.12, 137.66, 243.22, 450.41, 852.16, 1664.80],
                 [49.00, 79.07, 131.09, 234.22, 442.77, 841.88, 1624.08]]
MODEL_DECODE = [[1377.41, 1345.34, 1338.45, 1454.40, 1384.84, 1434.97, 1640.52],
                [764.06, 754.26, 759.99, 818.16, 817.74, 819.04, 1013.46],
                [772.34, 812.24, 764.72, 790.13, 792.69, 827.29, 1054.33],
                [744.29, 779.04, 777.17, 827.44, 800.79, 831.67, 997.51],
                [722.45, 758.89, 760.25, 764.46, 785.48, 885.65, 1056.12]]
MODEL_BOTH = [[1471.54, 1484.87, 1592.33, 1882.26, 2117.81, 2824.39, 4406.72],
              [815.06, 845.12, 932.83, 1128.71, 1428.90, 1957.54, 3354.40],
              [825.34, 900.49, 930.18, 1092.55, 1386.41, 1945.73, 3324.91],
              [796.25, 860.16, 914.83, 1070.66, 1251.20, 1683.83, 2662.31],
              [771.45, 837.96, 891.34, 998.68, 1228.25, 1727.53, 2680.20]]

BLACK = '#000000'
COLORS = ['#b85450', '#82b366', '#82b366', '#6c8ebf', '#6c8ebf']   # red, green and blue accents
MARKERS = ['s', '^', '^', 'o', 'o']
FACES = ['#b85450', '#82b366', '#ffffff', '#6c8ebf', '#ffffff']    # the non-phased variants are open
LINES = ['-', '-', 'none', '-', 'none']                            # and carry no line
STYLE = [dict(color=COLORS[k], linestyle=LINES[k], linewidth=1.8, marker=MARKERS[k], markersize=7,
              markerfacecolor=FACES[k], markeredgecolor=BLACK, markeredgewidth=1.0) for k in range(5)]

plt.rcParams.update({
    'font.family': 'Source Sans 3', 'font.size': 12, 'svg.fonttype': 'path',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'NewComputerModernSansMath',
    'mathtext.it': 'NewComputerModernSansMath', 'mathtext.bf': 'NewComputerModernSansMath',
    'axes.linewidth': 1.8, 'axes.edgecolor': BLACK, 'axes.labelsize': 14, 'axes.labelcolor': BLACK,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'xtick.color': BLACK, 'ytick.color': BLACK,
    'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'xtick.major.size': 4, 'ytick.major.size': 4,
    'legend.fontsize': 14, 'svg.hashsalt': 'ep_results',   # the same ids on every run
})
XLABELS = ['1k', '2k', '4k', '8k', '16k', '32k', '64k']


def panel(ax, title, series, yticks, ylabels, ylim, ylabel=None, xlabel=None):
    """One log2 panel: a line per implementation, drawn under the spines and the ticks."""
    for k, ys in enumerate(series):
        ax.plot(TOKENS, ys, **{**STYLE[k], 'marker': 'none'}, zorder=2)
    for k, ys in enumerate(series):
        ax.plot(TOKENS, ys, **{**STYLE[k], 'linestyle': 'none'}, zorder=2.1)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(TOKENS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(XLABELS)
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(ylabels)
    ax.set_xlim(870, 77000)
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(True, color='#b3b3b3', linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# the layout in px (1 inch = 72 units = 72 px): panels Wp x Hp, Gx apart (room for the y labels of the
# right column) and Gy apart (room for the tick labels of the top row and the titles of the bottom row)
Wp, Hp, Gx, Gy, L, B, R, T = 244, 168, 74, 58, 62, 52, 192, 30
FW, FH = L + 2 * Wp + Gx + R, B + 2 * Hp + Gy + T
fig = plt.figure(figsize=(FW / 72, FH / 72), dpi=72)


def axes(col, row):
    return fig.add_axes([(L + col * (Wp + Gx)) / FW, (B + (1 - row) * (Hp + Gy)) / FH, Wp / FW, Hp / FH])


XL = 'tokens per rank'
panel(axes(0, 0), 'MoE layer: prefill', MOE_PREFILL, [2, 4, 8, 16, 32, 64, 128],
      ['2', '4', '8', '16', '32', '64', '128'], (2.0, 175), ylabel='time per iteration (ms)')
model_prefill = axes(1, 0)
panel(model_prefill, 'whole model: prefill', MODEL_PREFILL, [64, 128, 256, 512, 1024, 2048],
      ['64', '128', '256', '512', '1k', '2k'], (44, 3300))
panel(axes(0, 1), 'whole model: decode', MODEL_DECODE, [512, 1024, 2048], ['512', '1k', '2k'],
      (512, 2100), ylabel='time per iteration (ms)', xlabel=XL)
panel(axes(1, 1), 'whole model: prefill and decode', MODEL_BOTH, [1024, 2048, 4096], ['1k', '2k', '4k'],
      (700, 5200), xlabel=XL)
model_prefill.legend([Line2D([], [], **style) for style in STYLE], NAMES,   # one legend for the four panels
                     loc='upper left', bbox_to_anchor=(1.05, 1.0), ncol=1, frameon=False, labelcolor=BLACK,
                     handlelength=2.4, borderaxespad=0.0)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ep_results.svg')
fig.savefig(out, bbox_inches='tight', pad_inches=0.02, transparent=True, metadata={'Date': None})
svg = open(out).read()
head, rest = svg.split('<svg', 1)
head = head.split('<!DOCTYPE', 1)[0]                  # no DTD, as in the other SVGs of the site
attrs, body = rest.split('>', 1)
attrs = attrs.replace('pt"', '"', 2)                  # the width and height in px, so that 1 unit = 1 px
open(out, 'w').write(head + '<svg' + attrs + '>' + body)
