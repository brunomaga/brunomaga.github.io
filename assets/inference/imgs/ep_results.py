"""Draw ep_results.svg: the runtime of Baseline, MegaKernel and MegaKernel+INC with the all-to-all backend, on log2 axes.

Left, prefill: the time per iteration, in s, against the batch size, in tokens per compute rank. Right, decode: the time
per output token, in ms, against the number of concurrent sequences; these values are placeholders, not measured yet,
and are drawn in grey. The data comes from the paper of the expert parallelism kernel fusion post. The picture follows
CLAUDE.md (Pictures): Source Sans 3 text at page size (1 unit = 1 px), drawn as glyph outlines.

usage: python3 ep_results.py   (needs matplotlib and the Source Sans 3 font)
"""
import os
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

NAMES = ['Baseline', 'MegaKernel', 'MegaKernel+INC']
PREFILL_X = [4096, 8192, 16384, 32768, 65536]
PREFILL = [[0.312, 0.497, 0.882, 1.605, 3.220],          # Baseline
           [0.254, 0.462, 0.839, 1.488, 2.837],          # MegaKernel
           [0.252, 0.454, 0.783, 1.385, 2.617]]          # MegaKernel+INC
PREFILL_LABELS = [['0.312', '0.497', '0.882', '1.61', '3.22'],      # the values printed next to the points
                  ['0.254', '0.462', '0.839', '1.49', '2.84'],
                  ['0.252', '0.454', '0.783', '1.39', '2.62']]
DECODE_X = [32, 64, 128, 256, 512]
DECODE = [[18.4, 19.1, 20.6, 23.8, 30.2],
          [16.9, 17.4, 18.5, 21.0, 26.1],
          [16.2, 16.6, 17.5, 19.6, 24.0]]

BLACK, GREY = '#000000', '#666666'
COLORS = ['#b85450', '#6c8ebf', '#82b366']               # red, blue and green accents
MARKERS = ['s', '^', 'o']
LABELS = [dict(xytext=(0, 5), ha='center', va='bottom'),  # Baseline above its points,
          dict(xytext=(6, 0), ha='left', va='center',     # MegaKernel right of them, over the lines,
               bbox=dict(boxstyle='square,pad=0.1', facecolor='#ffffff', edgecolor='none')),
          dict(xytext=(0, -6), ha='center', va='top')]    # MegaKernel+INC below them

plt.rcParams.update({
    'font.family': 'Source Sans 3', 'font.size': 12, 'svg.fonttype': 'path',
    'axes.linewidth': 1.4, 'axes.edgecolor': BLACK, 'axes.labelsize': 14, 'axes.labelcolor': BLACK,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'xtick.color': BLACK, 'ytick.color': BLACK,
    'xtick.major.width': 1.4, 'ytick.major.width': 1.4, 'xtick.major.size': 4, 'ytick.major.size': 4,
    'legend.fontsize': 14, 'svg.hashsalt': 'ep_results',   # the same ids on every run
})


def panel(ax, title, xs, series, labels, xticks, xlabels, yticks, xlim, ylim, xlabel, ylabel, colors, styles, color):
    """One log2 panel: a line per implementation, each point labelled with its value."""
    for k, (ys, texts, c, ls) in enumerate(zip(series, labels, colors, styles)):
        ax.plot(xs, ys, color=c, linestyle=ls, linewidth=1.4, marker=MARKERS[k], markersize=7, markerfacecolor=c,
                markeredgecolor=BLACK, markeredgewidth=1.0, label=NAMES[k], zorder=3)
        for x, y, t in zip(xs, ys, texts):
            ax.annotate(t, (x, y), textcoords='offset points', fontsize=12, color=color, **LABELS[k])
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
    ax.set_xlabel(xlabel, color=color)
    ax.set_ylabel(ylabel, color=color)
    ax.set_title(title, color=color, pad=8)
    ax.tick_params(colors=color)
    ax.grid(True, color='#b3b3b3', linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper left', frameon=False, labelcolor=color, handlelength=2.4, borderaxespad=0.2)


fig, (left, right) = plt.subplots(1, 2, figsize=(13.8, 4.4), dpi=72)     # 1 inch = 72 units = 72 px
panel(left, 'prefill', PREFILL_X, PREFILL, PREFILL_LABELS, PREFILL_X, ['4k', '8k', '16k', '32k', '64k'],
      [0.25, 0.5, 1, 2, 4], (3500, 86000), (0.185, 4.6), 'batch size (tokens)', 'time per iteration (s)', COLORS,
      ['-'] * 3, BLACK)
panel(right, 'decode', DECODE_X, DECODE, [['%.1f' % v for v in ys] for ys in DECODE], DECODE_X,
      [str(x) for x in DECODE_X], [16, 32], (27, 680), (13.8, 36), 'concurrent sequences', 'time per output token (ms)',
      [GREY] * 3, ['-', '--', ':'], GREY)
fig.subplots_adjust(wspace=0.26)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ep_results.svg')
fig.savefig(out, bbox_inches='tight', pad_inches=0.02, transparent=True, metadata={'Date': None})
svg = open(out).read()
head, rest = svg.split('<svg', 1)
head = head.split('<!DOCTYPE', 1)[0]                  # no DTD, as in the other SVGs of the site
attrs, body = rest.split('>', 1)
attrs = attrs.replace('pt"', '"', 2)                  # the width and height in px, so that 1 unit = 1 px
open(out, 'w').write(head + '<svg' + attrs + '>' + body)
