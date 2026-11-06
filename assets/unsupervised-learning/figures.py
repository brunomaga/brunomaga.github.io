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
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

TEXT, MATH = 'Source Sans 3', 'NewComputerModernSansMath'
BLUE, GREEN, RED, GREY = '#6c8ebf', '#82b366', '#b85450', '#666666'  # accents, for small marks
PASTEL = {BLUE: '#dae8fc', GREEN: '#d5e8d4', RED: '#f8cecc'}
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


def square_box(X):
    """The (x0, x1, y0, y1) limits of a square around the points X, with a margin of 12% of their extent."""
    mid, span = (X.min(0) + X.max(0)) / 2, 1.24 * np.ptp(X, 0).max()
    return mid[0] - span / 2, mid[0] + span / 2, mid[1] - span / 2, mid[1] + span / 2


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
    """Two classes and a query next to a single point of B among the points of A: k = 1 predicts B, k = 3 predicts A."""
    rng = np.random.default_rng(0)
    q = np.array([-0.7, -0.4])
    at = lambda r, deg: q + r * np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    near = [at(0.3, 120), at(0.42, 300), at(0.46, 200)]  # the 3 nearest neighbors of the query: B, A, A
    A, B = [near[1], near[2]], []
    while len(A) < 12:  # the other points keep away from the query and from each other
        p = rng.normal([-0.8, -0.5], 0.55)
        if np.linalg.norm(p - q) > 0.62 and min(np.linalg.norm(p - o) for o in A + [near[0]]) > 0.2: A.append(p)
    while len(B) < 12:
        p = rng.normal([0.9, 0.6], 0.5)
        if min([np.linalg.norm(p - o) for o in A + B]) > 0.2: B.append(p)
    X, y = np.vstack(A + B + [near[0]]), np.repeat([0, 1], [12, 13])
    colors = np.array([BLUE, GREEN])
    def data_panel(ax, box):
        for k in range(2):
            points(ax, X[y == k], colors[k])
        ax.plot(*q, 'o', ms=10, mfc='#ffffff', mec='k', mew=1.0, zorder=20)
    def knn_panel(k):
        def draw(ax, box):
            model = KNeighborsClassifier(k).fit(X, y)
            gx, gy = np.meshgrid(np.linspace(box[0], box[1], 441), np.linspace(box[2], box[3], 441))
            grid = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)  # the class at each position
            ax.contourf(gx, gy, grid, levels=[-0.5, 0.5, 1.5], colors=[PASTEL[col] for col in colors], zorder=1)
            ax.contour(gx, gy, grid, levels=[0.5], colors='k', linewidths=1.0, zorder=2)
            for cls in range(2):
                points(ax, X[y == cls], colors[cls])
            r = np.sort(np.linalg.norm(X - q, axis=1))[k - 1]  # the radius of the k nearest neighbors
            ax.add_patch(Ellipse(q, 2 * r, 2 * r, fill=False, ec='k', lw=1.4, ls=DASH, zorder=15))
            ax.plot(*q, 'o', ms=10, mfc=colors[model.predict([q])[0]], mec='k', mew=1.0, zorder=20)
        return draw
    legend = [(lambda c, x, y: c.square(x, y, BLUE), 8, 'Class A'),
              (lambda c, x, y: c.square(x, y, GREEN), 8, 'Class B'),
              (lambda c, x, y: c.circle(x, y, '#ffffff'), 10, 'Query')]
    three_panels('knn', np.vstack([X, q]), [data_panel, knn_panel(1), knn_panel(3)],
                 ['Data', ('𝑘 = 1', MATH), ('𝑘 = 3', MATH)], legend)


def hierarchical_clustering():
    """Three groups of points: the single linkage dendrogram, cut at a height of 1.5, gives three clusters."""
    # each point, and the side of the point where its number goes
    data = [((1.1, 3.0), (-1, 0)), ((1.45, 3.15), (0, 1)), ((1.8, 2.6), (1, 0)),
            ((-0.3, 0.2), (-1, 0)), ((0.0, -0.05), (0, -1)), ((0.15, 0.95), (0, 1)),
            ((2.4, 0.3), (-1, 0)), ((2.95, 0.2), (1, 0)), ((2.7, 0.95), (0, 1))]
    X, side = np.array([p for p, _ in data]), np.array([o for _, o in data])
    Z, cut = linkage(X, 'single'), 1.5
    leaf = np.argsort(leaves_list(Z))  # the position of each point among the leaves, which also numbers it
    groups = fcluster(Z, cut, 'distance')
    col = {g: [BLUE, GREEN, RED][i] for i, g in enumerate(dict.fromkeys(groups[leaves_list(Z)]))}  # in leaf order
    c, box = Canvas(830, 260), square_box(X)
    a = c.panel(10, 10, 220, 220, box[:2], box[2:])
    points(a, X, GREY)
    px = (box[1] - box[0]) / 220  # data units per px
    for i, p in enumerate(X):
        a.text(*(p + side[i] * [10 * px, 11.5 * px]), str(leaf[i] + 1), fontsize=12, ha='center',
               va='center_baseline', zorder=20)
    c.arrow(240, 120)
    d = c.panel(322, 10, 180, 195, (-0.5, len(X) - 0.5), (0, 2.5), frame=False)
    pos, top = {i: float(leaf[i]) for i in range(len(X))}, {i: 0.0 for i in range(len(X))}
    for n, (i, j, h, _) in enumerate(Z, start=len(X)):  # each merge joins two clusters at height h
        i, j = int(i), int(j)
        d.plot([pos[i], pos[i], pos[j], pos[j]], [top[i], h, h, top[j]], color='k', lw=1.8, zorder=5,
               solid_joinstyle='miter')
        pos[n], top[n] = (pos[i] + pos[j]) / 2, h
    d.plot([-0.5, len(X) - 0.5], [cut, cut], color=GREY, lw=1.0, ls=DASH, zorder=3)
    d.text(len(X) - 0.4, cut, 'cut', fontsize=12, color=GREY, ha='left', va='center_baseline')
    d.plot([-0.5, len(X) - 0.5], [0, 0], color='k', lw=1.8, zorder=6)
    d.plot([-0.5, -0.5], [0, 2.5], color='k', lw=1.8, zorder=6)
    for i in range(len(X)):
        d.plot(leaf[i], 0, 's', ms=8, mfc=col[groups[i]], mec='k', mew=1.0, zorder=7, clip_on=False)
        d.text(leaf[i], -0.2, str(leaf[i] + 1), fontsize=12, ha='center', va='top')
    for v in range(3):
        d.plot([-0.5, -0.65], [v, v], color='k', lw=1.8, zorder=6, clip_on=False)
        d.text(-0.8, v, str(v), fontsize=12, ha='right', va='center_baseline')
    c.text(293, 107.5, 'Merge distance', rotation=90, va='center', ha='center')
    b = c.panel(540, 10, 220, 220, box[:2], box[2:])
    for g in np.unique(groups):  # each cluster in a hull 12 px away from its points
        P = X[groups == g]
        ring = np.vstack([P + 12 * px * np.c_[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 72)])
        hull = ring[ConvexHull(ring).vertices]
        b.add_patch(Polygon(hull, closed=True, fc=PASTEL[col[g]], ec='k', lw=1.0, zorder=1))
        points(b, P, col[g])
    for x, label in ((120, 'Data'), (412, 'Dendrogram'), (650, 'Clusters')):
        c.text(x, 251, label)
    c.save('hierarchical_clustering')
    print('  merge heights:', Z[:, 2].round(2))


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


def data_panel(X):
    """Draw the points of X in grey, as they are before clustering."""
    return lambda ax, box: points(ax, X, GREY)


def kmeans_panel(X, km, colors):
    """Draw the Voronoi cells of the k-means means, the points of each cluster and the means."""
    def draw(ax, box):
        voronoi(ax, km.cluster_centers_, colors, box)
        for k, col in enumerate(colors):
            points(ax, X[km.labels_ == k], col)
        means(ax, km.cluster_centers_, colors)
    return draw


def three_panels(name, X, draws, labels, legend=None):
    """Three panels over the same square box around X, an arrow after the first, and a legend column on the right."""
    c, box = Canvas(830, 260), square_box(X)
    xs = [10, 280, 520]
    for x, draw in zip(xs, draws):
        draw(c.panel(x, 10, 220, 220, box[:2], box[2:]), box)
    c.arrow(240, 120)
    for x, label in zip(xs, labels):
        text, font = label if isinstance(label, tuple) else (label, TEXT)
        c.text(x + 110, 251, text, font=font)
    if legend:
        c.legend_column(756, 120, [(lambda x, y, f=f: f(c, x, y), sw, text) for f, sw, text in legend])
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
    three_panels('spectral_clustering', X, [data_panel(X), kmeans_panel(X, km, colors[match(km.labels_, y)]),
                                            spectral_panel],
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
    three_panels('gmm', X, [data_panel(X), kmeans_panel(X, km, colors[match(km.labels_, y)]), gmm_panel],
                 ['Data', 'k-means', 'Gaussian mixture model'])
    print('  ARI k-means:', round(adjusted_rand_score(y, km.labels_), 2),
          'GMM:', round(adjusted_rand_score(y, labels), 2))


FIGURES = {f.__name__: f for f in [knn, hierarchical_clustering, dbscan, spectral_clustering, gmm]}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
