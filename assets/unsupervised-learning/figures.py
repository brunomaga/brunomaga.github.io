"""Draw the data pictures of the unsupervised learning post as SVG files next to this script.

Each picture runs its method on the post's example or on data drawn with a fixed seed. The SVGs follow CLAUDE.md
(Pictures): 1 unit = 1 px, text as glyph outlines, cropped to the drawn content plus 0.5 px.
usage: python3 figures.py [NAME ...]   (needs matplotlib, scikit-learn, scipy and the fonts of CLAUDE.md)
"""
import io, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

TEXT, MATH = 'Source Sans 3', 'NewComputerModernSansMath'
BLUE, GREEN, RED, ORANGE, GREY = '#6c8ebf', '#82b366', '#b85450', '#d79b00', '#666666'  # accents, for small marks
PASTEL = {BLUE: '#dae8fc', GREEN: '#d5e8d4', RED: '#f8cecc', ORANGE: '#ffe6cc'}
DASH = (0, (3, 3))  # draw.io's default dash pattern, scaled by the line weight
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': TEXT, 'svg.fonttype': 'path', 'svg.hashsalt': 'unsupervised-learning',
                     'axes.unicode_minus': True, 'lines.solid_capstyle': 'butt'})


class Canvas:
    """A W x H px picture; the overlay axes place shapes and text in px from the top-left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 72, h / 72), dpi=72)
        self.top = self.fig.add_axes([0, 0, 1, 1], zorder=50)
        self.top.set_xlim(0, w); self.top.set_ylim(h, 0); self.top.axis('off')

    def panel(self, x, y, w, h, xlim, ylim, frame=True):
        """Axes at (x, y) px of w x h px showing xlim x ylim, with a 1.8 px frame."""
        ax = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')
        if frame:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, lw=1.8, ec='k',
                                   joinstyle='miter', zorder=30, clip_on=False))
        return ax

    def text(self, x, y, s, size=14, font=TEXT, ha='center', va='baseline', color='k', **kw):
        return self.top.text(x, y, s, fontsize=size, fontfamily=font, ha=ha, va=va, color=color, **kw)

    def arrow(self, x, yc, length=30):
        """Block arrow pointing right from x: body 10 px, head 20 px wide and 10 px long."""
        b = x + length - 10
        pts = [(x, yc - 5), (b, yc - 5), (b, yc - 10), (x + length, yc), (b, yc + 10), (b, yc + 5), (x, yc + 5)]
        self.top.add_patch(Polygon(pts, closed=True, fc='#f5f5f5', ec='k', lw=1.0, joinstyle='miter'))

    def square(self, x, y, color, size=8):
        self.top.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, fc=color, ec='k', lw=1.0))

    def circle(self, x, y, color, size=10):
        self.top.add_patch(Ellipse((x, y), size, size, fc=color, ec='k', lw=1.0))

    def legend(self, xc, y, entries):
        """A row of (swatch, swatch width, label) centered at xc: label 4 px right of its swatch, 16 px apart."""
        widths = [sw + 4 + text_width(self.fig, label, 14) for _, sw, label in entries]
        x = xc - (sum(widths) + 16 * (len(entries) - 1)) / 2
        for (swatch, sw, label), w in zip(entries, widths):
            swatch(x + sw / 2, y)
            self.text(x + sw + 4, y, label, va='center_baseline', ha='left')
            x += w + 16

    def legend_column(self, x, yc, entries):
        """A column of (swatch, swatch width, label) from x, centered at yc, with rows 20 px apart."""
        sw_max = max(sw for _, sw, _ in entries)
        for r, (swatch, sw, label) in enumerate(entries):
            y = yc + 20 * (r - (len(entries) - 1) / 2)
            swatch(x + sw_max / 2, y)
            self.text(x + sw_max + 4, y, label, va='center_baseline', ha='left')

    def save(self, name):
        """Write <name>.svg cropped to its ink plus 0.5 px, and print its width P in % of the 880 px band."""
        png = io.BytesIO()
        self.fig.savefig(png, format='png', dpi=288, facecolor='white')
        ink = np.array(Image.open(png).convert('L')) < 245
        ys, xs = np.nonzero(ink)
        x0, y0 = xs.min() / 4 - 0.5, ys.min() / 4 - 0.5
        w, h = (xs.max() + 1) / 4 + 0.5 - x0, (ys.max() + 1) / 4 + 0.5 - y0
        svg = io.StringIO()
        self.fig.savefig(svg, format='svg', transparent=True, metadata={'Date': None})
        head, rest = svg.getvalue().split('<svg', 1)
        attrs, body = rest.split('>', 1)
        attrs = re.sub(r'width="[^"]*" height="[^"]*" viewBox="[^"]*"',
                       f'width="{w:.2f}" height="{h:.2f}" viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}"', attrs)
        with open(os.path.join(HERE, name + '.svg'), 'w') as f:
            f.write(head.split('<!DOCTYPE', 1)[0] + '<svg' + attrs + '>' + body)
        plt.close(self.fig)
        print(f'{name}.svg: {w:.1f} x {h:.1f} px, P = {w / 880 * 100:.1f}%')


def text_width(fig, s, size, font=TEXT):
    t = fig.text(0, 0, s, fontsize=size, fontfamily=font)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width
    t.remove()
    return w


def points(ax, xy, color, size=8, z=10):
    ax.plot(xy[:, 0], xy[:, 1], 's', ms=size, mfc=color, mec='k', mew=1.0, zorder=z)


def voronoi(ax, means, colors, box):
    """Fill the cell of each mean inside the box, with a 1 px black outline."""
    for i, (m, c) in enumerate(zip(means, colors)):
        cell = [(box[0], box[2]), (box[1], box[2]), (box[1], box[3]), (box[0], box[3])]
        for j, o in enumerate(means):
            if j != i:  # keep the half-plane closer to m than to o
                n, d = o - m, (o @ o - m @ m) / 2
                inside = lambda p: p @ n <= d
                out = []
                for p, q in zip(cell, cell[1:] + cell[:1]):
                    p, q = np.array(p), np.array(q)
                    if inside(p): out.append(p)
                    if inside(p) != inside(q): out.append(p + (q - p) * (d - p @ n) / ((q - p) @ n))
                cell = [tuple(p) for p in out]
        ax.add_patch(Polygon(cell, closed=True, fc=PASTEL[c], ec='k', lw=1.0, zorder=1))


def means(ax, centers, colors):
    for m, c in zip(centers, colors):
        ax.plot(*m, 'o', ms=10, mfc=c, mec='k', mew=1.0, zorder=20)


def knn():
    """The example of the post: A at 0, 1, 2 and B at 3.5, 6, 7 on a line, and the query 3."""
    A, B, q = np.array([0, 1, 2.]), np.array([3.5, 6, 7.]), 3.0
    c = Canvas(640, 220)
    sx = lambda v: 80 + (v + 0.5) * 56  # px of a value of the line, from -0.5 to 7.5
    rows = [('Data', TEXT, None), ('𝑘 = 1', MATH, 1), ('𝑘 = 3', MATH, 3)]
    for r, (label, font, k) in enumerate(rows):
        y = 30 + 52 * r
        c.text(66, y, label, font=font, ha='right', va='center_baseline')
        color = '#ffffff'
        if k:
            near = sorted(np.concatenate([A, B]), key=lambda v: abs(v - q))[:k]
            dist = abs(near[-1] - q)  # the band covers the k nearest neighbors
            c.top.add_patch(Rectangle((sx(q - dist) - 6, y - 11), sx(q + dist) - sx(q - dist) + 12, 22,
                                      fc='#fff2cc', ec='k', lw=1.0, zorder=1))
            vote = 'A' if sum(v in A for v in near) > k / 2 else 'B'
            color = BLUE if vote == 'A' else ORANGE
            c.text(sx(7.5) + 16, y, 'predicts ' + vote, ha='left', va='center_baseline')
        c.top.plot([sx(-0.5), sx(7.5)], [y, y], color='k', lw=1.4, zorder=2)
        for v in range(8):
            c.top.plot([sx(v), sx(v)], [y, y + 8], color='k', lw=1.4, zorder=2)
        for v in A: c.square(sx(v), y, BLUE)
        for v in B: c.square(sx(v), y, ORANGE)
        c.circle(sx(q), y, color)
    c.text(sx(q), 30 - 12, 'query', size=12)
    for v in range(8):
        c.text(sx(v), 30 + 52 * 2 + 23, str(v), size=12)
    band = lambda x, y: c.top.add_patch(Rectangle((x - 10, y - 6), 20, 12, fc='#fff2cc', ec='k', lw=1.0))
    c.legend(sx(3.5), 186, [(lambda x, y: c.square(x, y, BLUE), 8, 'Class A'),
                            (lambda x, y: c.square(x, y, ORANGE), 8, 'Class B'),
                            (lambda x, y: c.circle(x, y, '#ffffff'), 10, 'Query'),
                            (band, 20, 'Nearest neighbors')])
    c.save('knn')


def hierarchical_clustering():
    """The example of the post: the points 0, 1, 3, 4 and 8, with single and complete linkage, cut at 3."""
    x = np.array([0, 1, 3, 4, 8.])
    c = Canvas(790, 290)
    for p, method in enumerate(['single', 'complete']):
        Z = linkage(x[:, None], method)
        left = 70 + 380 * p
        ax = c.panel(left, 20, 300, 200, (-0.6, 8.6), (0, 9), frame=False)
        pos, top = {i: v for i, v in enumerate(x)}, {i: 0.0 for i in range(len(x))}
        for n, (a, b, h, _) in enumerate(Z, start=len(x)):  # each merge joins two clusters at height h
            a, b = int(a), int(b)
            ax.plot([pos[a], pos[a], pos[b], pos[b]], [top[a], h, h, top[b]], color='k', lw=1.8, zorder=5,
                    solid_joinstyle='miter')
            pos[n], top[n] = (pos[a] + pos[b]) / 2, h
        groups, label = {}, {}
        for i in range(len(x)):  # the clusters below the cut at 3
            root = i
            for n, (a, b, h, _) in enumerate(Z, start=len(x)):
                if h <= 3 and root in (a, b): root = n
            label[i] = groups.setdefault(root, [BLUE, GREEN, ORANGE][len(groups)])
        ax.plot([-0.6, 8.6], [3, 3], color=GREY, lw=1.0, ls=DASH, zorder=3)
        ax.text(8.72, 3, 'cut', fontsize=12, color=GREY, ha='left', va='center_baseline')
        ax.plot([-0.6, 8.6], [0, 0], color='k', lw=1.8, zorder=6)
        ax.plot([-0.6, -0.6], [0, 9], color='k', lw=1.8, zorder=6)
        for i, v in enumerate(x):
            ax.plot(v, 0, 's', ms=8, mfc=label[i], mec='k', mew=1.0, zorder=7, clip_on=False)
        for v in range(9):
            ax.text(v, -0.75, str(v), fontsize=12, ha='center', va='top')
        for v in range(0, 10, 2):
            ax.plot([-0.6, -0.8], [v, v], color='k', lw=1.8, zorder=6, clip_on=False)
            ax.text(-0.95, v, str(v), fontsize=12, ha='right', va='center_baseline')
        c.text(left + 150, 272, ['Single linkage', 'Complete linkage'][p])
    c.text(30, 120, 'Merge distance', rotation=90, va='center', ha='center')
    c.save('hierarchical_clustering')


def dbscan():
    """Two moons and uniform noise, clustered by DBSCAN with a radius of 0.25 and a minimum of 6 points."""
    rng = np.random.default_rng(0)
    X = np.vstack([make_moons(n_samples=100, noise=0.06, random_state=0)[0],
                   rng.uniform([-1.2, -0.7], [2.2, 1.2], size=(10, 2))])
    eps, m = 0.25, 6
    db = DBSCAN(eps=eps, min_samples=m).fit(X)
    core = np.zeros(len(X), bool); core[db.core_sample_indices_] = True
    box, colors = (-1.4, 2.4, -0.9, 1.3), [BLUE, GREEN]
    c = Canvas(830, 230)
    a = c.panel(10, 10, 311, 180, box[:2], box[2:])
    points(a, X, GREY)
    c.arrow(340, 100)
    b = c.panel(390, 10, 311, 180, box[:2], box[2:])
    for i, l in enumerate(db.labels_):
        col = GREY if l < 0 else (colors[l] if core[i] else PASTEL[colors[l]])
        points(b, X[i:i + 1], col)
    i = int(np.argmin(np.where(core, X[:, 0], np.inf)))  # the leftmost core point, with free space on its left
    b.add_patch(Ellipse(X[i], 2 * eps, 2 * eps, fill=False, ec='k', lw=1.4, ls=DASH, zorder=15))
    b.plot([X[i, 0] - eps, X[i, 0]], [X[i, 1], X[i, 1]], color='k', lw=1.4, zorder=15)
    b.text(X[i, 0] - eps / 2, X[i, 1] + 0.04, '𝜀', fontsize=12, fontfamily=MATH, ha='center', va='bottom', zorder=15)
    c.text(165.5, 211, 'Data')
    c.text(545.5, 211, 'DBSCAN')
    c.legend_column(717, 100, [(lambda x, y: c.square(x, y, BLUE), 8, 'Core point'),
                               (lambda x, y: c.square(x, y, PASTEL[BLUE]), 8, 'Border point'),
                               (lambda x, y: c.square(x, y, GREY), 8, 'Noise')])
    c.save('dbscan')
    print('  clusters:', db.labels_.max() + 1, 'core:', core.sum(), 'border:', ((db.labels_ >= 0) & ~core).sum(),
          'noise:', (db.labels_ < 0).sum(), 'points in the circle:', (np.linalg.norm(X - X[i], axis=1) <= eps).sum())


def match(labels, y):
    """The true cluster of y that each label of a clustering maps to, one to one, with the most points in common."""
    k = y.max() + 1
    common = np.array([[np.sum((labels == a) & (y == t)) for t in range(k)] for a in range(k)])
    return linear_sum_assignment(-common)[1]


def kmeans_panel(X, km, colors):
    """Draw the Voronoi cells of the k-means means, the points of each cluster and the means."""
    def draw(ax, box):
        voronoi(ax, km.cluster_centers_, colors, box)
        for k, col in enumerate(colors):
            points(ax, X[km.labels_ == k], col)
        means(ax, km.cluster_centers_, colors)
    return draw


def three_panels(name, X, results, labels):
    """The data, an arrow, then one panel per clustering, all over the same square box."""
    pad = 0.12 * (X.max(0) - X.min(0)).max()
    lo, hi = X.min(0) - pad, X.max(0) + pad
    span = (hi - lo).max()
    lo, hi = (lo + hi) / 2 - span / 2, (lo + hi) / 2 + span / 2
    box = (lo[0], hi[0], lo[1], hi[1])
    c = Canvas(760, 260)
    xs = [10, 280, 520]
    points(c.panel(xs[0], 10, 220, 220, box[:2], box[2:]), X, GREY)
    c.arrow(xs[0] + 230, 120)
    for x, draw in zip(xs[1:], results):
        draw(c.panel(x, 10, 220, 220, box[:2], box[2:]), box)
    for x, label in zip(xs, labels):
        c.text(x + 110, 251, label)
    c.save(name)


def spectral_clustering():
    """Two concentric rings: k-means cuts them in halves, spectral clustering separates the rings."""
    rng = np.random.default_rng(0)
    rings = []
    for n, r, s in ((50, 1.0, 0.05), (25, 0.4, 0.04)):  # points, radius and noise of each ring
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi / n)
        d = r + rng.normal(0, s, n)
        rings.append(np.c_[d * np.cos(t), d * np.sin(t)])
    X, y = np.vstack(rings), np.repeat([0, 1], [50, 25])
    colors = np.array([BLUE, GREEN])
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    sc = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=8, random_state=0).fit(X)
    def spectral_panel(ax, box):
        for k, col in enumerate(colors[match(sc.labels_, y)]):
            points(ax, X[sc.labels_ == k], col)
    three_panels('spectral_clustering', X, [kmeans_panel(X, km, colors[match(km.labels_, y)]), spectral_panel],
                 ['Data', 'k-means', 'Spectral clustering'])


def gmm():
    """Three elongated clusters: k-means cuts them with its Voronoi cells, a GMM fits their ellipses."""
    X, y = make_blobs(n_samples=90, random_state=57)
    X = X @ np.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])  # as in scikit-learn's k-means demo
    colors = np.array([BLUE, GREEN, RED])
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    gm = GaussianMixture(3, covariance_type='full', n_init=5, random_state=0).fit(X)
    labels = gm.predict(X)  # the component of largest responsibility
    gcol = colors[match(labels, y)]
    def gmm_panel(ax, box):
        for k, col in enumerate(gcol):  # the contour at two standard deviations of each Gaussian
            val, vec = np.linalg.eigh(gm.covariances_[k])
            deg = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
            ax.add_patch(Ellipse(gm.means_[k], 4 * np.sqrt(val[1]), 4 * np.sqrt(val[0]), angle=deg,
                                 fc=PASTEL[col], ec='k', lw=1.0, zorder=1))
            points(ax, X[labels == k], col)
        means(ax, gm.means_, gcol)
    three_panels('gmm', X, [kmeans_panel(X, km, colors[match(km.labels_, y)]), gmm_panel],
                 ['Data', 'k-means', 'Gaussian mixture model'])
    print('  ARI k-means:', round(adjusted_rand_score(y, km.labels_), 2),
          'GMM:', round(adjusted_rand_score(y, labels), 2))


FIGURES = {f.__name__: f for f in [knn, hierarchical_clustering, dbscan, spectral_clustering, gmm]}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
