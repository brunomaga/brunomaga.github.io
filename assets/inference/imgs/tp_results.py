"""Draw tp_results.svg: the latency of the baseline and fused FFNs, and the improvement of the fused kernel.

The data is the mean and standard deviation of the latency, in ms, on 8 GPUs, from the paper "Towards Efficient LLM
Inference via Full-Stack Fusion of Computation, Communication, and In-Network Computing". The picture follows
CLAUDE.md (Pictures): Source Sans 3 text at page size (1 unit = 1 px), drawn as glyph outlines, 1.8 px axes, a 1.0 px
data line, and the legend in one column right of the plot.

usage: python3 tp_results.py   (needs matplotlib and the Source Sans 3 font)
"""
import os
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt

MODELS = ['DeepSeek V3\nMoE Top-2', 'DeepSeek V3\nPer-expert', 'Llama 3 8B\nSwiGLU', 'Scout\nPer-expert',
          'Scout\nShared MLP', 'DeepSeek V3\nDense/Shared', 'Llama 3 70B\nSwiGLU']
BASELINE = [(5.596, 0.311), (17.133, 0.758), (9.390, 1.106), (117.690, 3.302), (317.945, 8.114), (247.767, 4.188),
            (26.023, 1.322)]
FUSED = [(2.135, 0.011), (7.670, 0.040), (5.059, 0.029), (64.405, 0.188), (195.847, 0.505), (157.185, 0.274),
         (18.481, 0.058)]
IMPROVEMENT = [100 * (1 - f / b) for (b, _), (f, _) in zip(BASELINE, FUSED)]   # in %

plt.rcParams.update({
    'font.family': 'Source Sans 3', 'font.size': 12, 'svg.fonttype': 'path',
    'axes.linewidth': 1.8, 'axes.edgecolor': '#000000', 'axes.labelsize': 14, 'axes.labelcolor': '#000000',
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'xtick.color': '#000000', 'ytick.color': '#000000',
    'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'xtick.major.size': 0, 'ytick.major.size': 4,
    'legend.fontsize': 14, 'svg.hashsalt': 'tp_results',   # the same ids on every run
})

fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=72)     # 1 inch = 72 units = 72 px
x = range(len(MODELS))
bar = dict(width=0.36, edgecolor='#000000', linewidth=1.0,
           error_kw=dict(ecolor='#000000', elinewidth=1.0, capsize=2, capthick=1.0))
ax.bar([i - 0.18 for i in x], [m for m, _ in BASELINE], yerr=[s for _, s in BASELINE], color='#ffe6cc',
       label='Baseline', **bar)
ax.bar([i + 0.18 for i in x], [m for m, _ in FUSED], yerr=[s for _, s in FUSED], color='#dae8fc', label='Fused', **bar)
ax.set_xlim(-0.6, 6.6)
ax.set_ylim(0, 345)
ax.set_yticks([0, 100, 200, 300])
ax.set_ylabel('Latency (ms)')
ax.set_xticks(list(x))
ax.set_xticklabels(MODELS, rotation=40, ha='right', ma='right', rotation_mode='anchor', linespacing=1.0)
ax.yaxis.grid(True, color='#b3b3b3', linewidth=0.4)
ax.set_axisbelow(True)

ax2 = ax.twinx()
ax2.plot(list(x), IMPROVEMENT, color='#000000', linewidth=1.0, marker='o', markersize=6, markerfacecolor='#f8cecc',
         markeredgecolor='#000000', markeredgewidth=1.0, label='Improvement (%)')
for i, v in enumerate(IMPROVEMENT):
    ax2.annotate('%.1f' % v, (i, v), xytext=(4, 3), textcoords='offset points', ha='left', va='bottom', fontsize=12,
                 bbox=dict(boxstyle='square,pad=0.1', facecolor='#ffffff', edgecolor='none'))
ax2.set_ylim(25, 66)
ax2.set_yticks([30, 40, 50, 60])
ax2.set_ylabel('Improvement (%)')
for a in (ax, ax2):
    a.spines['top'].set_visible(False)

handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
fig.draw_without_rendering()                          # the legend goes 16 px right of the label of the right axis
right = ax2.yaxis.label.get_window_extent().x1 + 16
ax.legend(handles, ['Baseline', 'Fused', 'Improvement (%)'], ncol=1, loc='upper left',
          bbox_to_anchor=(ax.transAxes.inverted().transform((right, 0))[0], 1.0), frameon=False, handlelength=1.4,
          handleheight=0.85, borderaxespad=0.0, borderpad=0.0)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tp_results.svg')
fig.savefig(out, bbox_inches='tight', pad_inches=0.02, transparent=True, metadata={'Date': None})
svg = open(out).read()
head, rest = svg.split('<svg', 1)
head = head.split('<!DOCTYPE', 1)[0]                  # no DTD, as in the other SVGs of the site
attrs, body = rest.split('>', 1)
attrs = attrs.replace('pt"', '"', 2)                  # the width and height in px, so that 1 unit = 1 px
open(out, 'w').write(head + '<svg' + attrs + '>' + body)
