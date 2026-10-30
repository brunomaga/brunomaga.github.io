"""Draw ep_results.svg: the runtime of Baseline, MegaKernel and MegaKernel+INC, with the INC all-to-all and with its
phased variant (G = 1), on log2 axes.

Top, the MoE layers; bottom, the whole model. Left, prefill: the time per iteration, in s, against the batch size, in
tokens per compute rank. Right, decode: the decode time, in s, against the number of concurrent sequences of 1,024
tokens. A light blue band spans MegaKernel+INC with the INC all-to-all and with its phased variant. Baseline runs one
microbatch per batch; the other implementations take, at each point, the microbatch size with the shortest time. The
data comes from the paper of the expert parallelism kernel fusion post. The picture follows CLAUDE.md (Pictures):
Source Sans 3 text and New Computer Modern Sans Math formulas at page size (1 unit = 1 px), drawn as glyph outlines,
1.8 px axes and data lines, the data drawn behind the axes, and one legend in one column right of the plots.

usage: python3 ep_results.py   (needs matplotlib, Source Sans 3 and New Computer Modern Sans Math)
"""
import os
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

NAMES = ['Baseline', 'MegaKernel', 'MegaKernel+INC', 'MegaKernel+INC\n(Phased, $𝐺 = 1$)']
TOKENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
SEQS = [1, 2, 4, 8, 16, 32, 64]
# one list per implementation: Baseline, MegaKernel, MegaKernel+INC, MegaKernel+INC (Phased, G = 1)
MOE_PREFILL = [[0.077, 0.117, 0.218, 0.327, 0.570, 1.085, 2.126],
               [0.063, 0.091, 0.164, 0.264, 0.475, 0.905, 1.736],
               [0.041, 0.067, 0.107, 0.199, 0.371, 0.700, 1.346],
               [0.048, 0.072, 0.118, 0.211, 0.384, 0.726, 1.410]]
MOE_DECODE = [[1.062, 0.912, 1.079, 1.091, 1.135, 0.987, 0.910],
              [0.700, 0.650, 0.720, 0.740, 0.750, 0.660, 0.620],
              [0.350, 0.360, 0.369, 0.386, 0.325, 0.306, 0.299],
              [0.412, 0.402, 0.401, 0.403, 0.363, 0.344, 0.364]]
MODEL_PREFILL = [[0.119, 0.159, 0.268, 0.411, 0.708, 1.338, 2.638],
                 [0.111, 0.141, 0.224, 0.357, 0.614, 1.143, 2.214],
                 [0.094, 0.120, 0.170, 0.290, 0.517, 0.965, 1.851],
                 [0.102, 0.124, 0.184, 0.305, 0.531, 0.998, 1.918]]
MODEL_DECODE = [[1.907, 1.762, 1.911, 1.974, 1.980, 1.852, 1.950],
                [1.800, 1.680, 1.800, 1.840, 1.820, 1.750, 1.840],
                [1.613, 1.595, 1.563, 1.725, 1.680, 1.522, 1.664],
                [1.712, 1.584, 1.700, 1.705, 1.695, 1.643, 1.740]]

BLACK, BAND = '#000000', '#dae8fc'                       # the band takes the blue pastel fill
COLORS = ['#b85450', '#82b366', '#6c8ebf', '#6c8ebf']    # red, green and blue accents, as in the paper
MARKERS = ['s', '^', 'o', 'o']
FACES = COLORS[:3] + ['#ffffff']                         # the phased variant has open circles
STYLES = ['-', '-', '-', '--']
STYLE = [dict(color=COLORS[k], linestyle=STYLES[k], linewidth=1.8, marker=MARKERS[k], markersize=7,
              markerfacecolor=FACES[k], markeredgecolor=BLACK, markeredgewidth=1.0) for k in range(4)]

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


def panel(ax, title, xs, series, xticks, xlabels, xlim, yticks, ylim, ylabel, xlabel=None):
    """One log2 panel: a line per implementation, and the band between MegaKernel+INC and its phased variant."""
    # the band, the lines and the points over them, under the spines and the ticks (zorder 2.5)
    ax.fill_between(xs, series[2], series[3], color=BAND, linewidth=0, zorder=1)
    for k, ys in enumerate(series):
        ax.plot(xs, ys, **{**STYLE[k], 'marker': 'none'}, zorder=2)
    for k, ys in enumerate(series):
        ax.plot(xs, ys, **{**STYLE[k], 'linestyle': 'none'}, zorder=2.1)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(xlabels)
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(['%g' % t for t in yticks])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(True, color='#b3b3b3', linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# the layout in px (1 inch = 72 units = 72 px): panels Wp x Hp, Gx apart (room for the y labels of the right column)
# and Gy apart (room for the tick labels of the top row and the titles of the bottom row)
Wp, Hp, Gx, Gy, L, B, R, T = 252, 170, 76, 58, 60, 52, 182, 30
FW, FH = L + 2 * Wp + Gx + R, B + 2 * Hp + Gy + T
fig = plt.figure(figsize=(FW / 72, FH / 72), dpi=72)


def axes(col, row):
    return fig.add_axes([(L + col * (Wp + Gx)) / FW, (B + (1 - row) * (Hp + Gy)) / FH, Wp / FW, Hp / FH])


PREFILL = dict(xticks=TOKENS, xlabels=['1k', '2k', '4k', '8k', '16k', '32k', '64k'], xlim=(870, 77000),
               ylabel='time per iteration (s)')
DECODE = dict(xticks=SEQS, xlabels=[str(s) for s in SEQS], xlim=(0.85, 75), ylabel='decode time (s)')
panel(axes(0, 0), 'MoE layers: prefill', TOKENS, MOE_PREFILL, yticks=[0.05, 0.1, 0.2, 0.5, 1, 2], ylim=(0.035, 2.8),
      **PREFILL)
moe_decode = axes(1, 0)
panel(moe_decode, 'MoE layers: decode', SEQS, MOE_DECODE, yticks=[0.3, 0.5, 1], ylim=(0.25, 1.5), **DECODE)
panel(axes(0, 1), 'whole model: prefill', TOKENS, MODEL_PREFILL, yticks=[0.1, 0.2, 0.5, 1, 2], ylim=(0.08, 3.4),
      xlabel='batch size (tokens)', **PREFILL)
panel(axes(1, 1), 'whole model: decode', SEQS, MODEL_DECODE, yticks=[1.4, 1.6, 1.8, 2], ylim=(1.45, 2.1),
      xlabel='concurrent sequences (1,024 tokens per seq)', **DECODE)
moe_decode.legend([Line2D([], [], **style) for style in STYLE], NAMES,   # one legend for the four panels
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
