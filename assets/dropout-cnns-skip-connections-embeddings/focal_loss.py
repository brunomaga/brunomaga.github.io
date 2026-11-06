"""Draw focal_loss.svg: the focal loss -(1 - p_t)^gamma log(p_t), with alpha_t = 1, for gamma = 0 (cross-entropy) to 5.

The values of gamma are those of figure 1 of Focal Loss for Dense Object Detection (Lin et al., 2017).
usage: python3 focal_loss.py   (needs matplotlib, numpy and the fonts of CLAUDE.md, Pictures, section 4)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts
from matplotlib import transforms

GAMMAS = [(0, charts.BLUE), (0.5, charts.GREEN), (1, charts.ORANGE), (2, charts.RED), (5, charts.PURPLE)]

fig = charts.figure(470, 270, 'focal_loss')
ax = charts.axes(fig, 45, 45, 300, 200)
p = np.linspace(0.001, 1, 1000)
for gamma, color in GAMMAS:
    ax.plot(p, -(1 - p)**gamma * np.log(p), color=color, label='𝛾 = %g' % gamma)
ax.axvline(0.5, color=charts.GREY, lw=1.0, linestyle=(0, (3, 3)), zorder=1)
ax.text(0.5, 4.75, 'well-classified examples', size=12, ha='left', va='top',
        transform=transforms.offset_copy(ax.transData, fig, x=5, units='dots'))
ax.set(xlim=(0, 1), ylim=(0, 5), xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=range(6))
ax.set_ylabel('loss')
charts.tick_labels(ax)
charts.runs(ax, 0.5, 0, charts.sub('𝑝', '𝑡'), transform=transforms.offset_copy(ax.transAxes, fig, y=-38, units='dots'))
legend = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
for text in legend.get_texts():
    text.set_family(charts.MATH)
charts.save(fig, os.path.join(HERE, 'focal_loss.svg'))
