"""Draw the data pictures of the unsupervised learning post as SVG files next to this script.

Each picture runs its method on the post's example or on data drawn with a fixed seed. The SVGs take the shared
chart style of _tools/charts.py (1 unit = 1 px, text as glyph outlines), cropped to the drawn content plus 0.5 px.
usage: python3 figures.py [NAME ...]   (needs matplotlib, scikit-learn, scipy and the fonts of CLAUDE.md)
"""
import io, os, re, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '_tools'))
import charts
from charts import BLUE, GREEN, GREY, MATH, ORANGE, PURPLE, RED, YELLOW
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors

PASTEL = {BLUE: charts.BLUE_FILL, GREEN: charts.GREEN_FILL, ORANGE: charts.ORANGE_FILL, YELLOW: charts.YELLOW_FILL,
          RED: charts.RED_FILL, PURPLE: charts.PURPLE_FILL}  # the fill of each accent
DASH = (0, (3, 3))  # draw.io's default dash pattern, scaled by the line weight


class Canvas:
    """The w x h px picture name.svg; the overlay axes place shapes and text in px from the top-left corner."""

    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.fig = charts.figure(w, h, name)
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

    def text(self, x, y, s, size=14, font=None, ha='center', va='baseline', color='k', **kw):
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

    def save(self):
        """Write the SVG cropped to its ink plus 0.5 px, and print its width P in % of the 880 px band."""
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
        with open(os.path.join(HERE, self.name + '.svg'), 'w') as f:
            f.write(head.split('<!DOCTYPE', 1)[0] + '<svg' + attrs + '>' + body)
        plt.close(self.fig)
        print(f'{self.name}.svg: {w:.1f} x {h:.1f} px, P = {w / 880 * 100:.1f}%')


def square_box(X):
    """The (x0, x1, y0, y1) limits of a square around the points X, with a margin of 12% of their extent."""
    mid, span = (X.min(0) + X.max(0)) / 2, 1.24 * np.ptp(X, 0).max()
    return mid[0] - span / 2, mid[0] + span / 2, mid[1] - span / 2, mid[1] + span / 2


def points(ax, xy, color, size=8, z=10):
    ax.plot(xy[:, 0], xy[:, 1], 's', ms=size, mfc=color, mec='k', mew=1.0, zorder=z)


def voronoi_cells(ax, means, colors, box):
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


def voronoi():
    """The Voronoi cells of 20 prototypes drawn at random in a square, at least 0.12 apart, colored so that no two
    neighbors match."""
    rng, P = np.random.default_rng(0), []
    while len(P) < 20:
        p = rng.uniform(0.04, 0.96, 2)
        if all(np.linalg.norm(p - o) >= 0.12 for o in P): P.append(p)
    P = np.array(P)
    near = [set() for _ in P]  # the neighbors of each cell, from the Delaunay triangulation
    for tri in Delaunay(P).simplices:
        for i in tri:
            near[i] |= set(tri) - {i}
    order, left = [], set(range(len(P)))
    while left:  # smallest-last order, in which a greedy coloring of a planar graph takes at most 6 colors
        order.append(min(left, key=lambda i: len(near[i] & left)))
        left.remove(order[-1])
    col = {}
    for i in reversed(order):
        col[i] = next(c for c in PASTEL if c not in {col.get(j) for j in near[i]})
    c, box = Canvas('voronoi', 260, 260), (-0.05, 1.05, -0.05, 1.05)
    ax = c.panel(10, 10, 240, 240, box[:2], box[2:])
    voronoi_cells(ax, P, [col[i] for i in range(len(P))], box)
    ax.plot(*P.T, 'o', ms=6, mfc='k', mec='k', mew=1.0, zorder=20)
    c.save()


def kmeans_iterations():
    """Lloyd's algorithm with k = 3 on three Gaussian clusters, from three of the points as initial means, until the
    assignments no longer change; the lines trace the means of the previous iterations."""
    X, _ = make_blobs(n_samples=90, centers=3, center_box=(-5, 5), random_state=26)
    W, steps, prev = X[np.random.default_rng(4).choice(len(X), 3, replace=False)], [], None
    while True:
        labels = np.argmin(((X[:, None] - W) ** 2).sum(2), 1)  # assignment step
        steps.append((W, labels))
        if prev is not None and (labels == prev).all():
            break
        prev, W = labels, np.array([X[labels == i].mean(0) for i in range(3)])  # update step
    colors = [BLUE, GREEN, RED]
    def panel(t):
        def draw(ax, box):
            W, labels = steps[t]
            voronoi_cells(ax, W, colors, box)
            for k, col in enumerate(colors):
                points(ax, X[labels == k], col)
            trail = np.array([w for w, _ in steps[:t + 1]])  # the means of iterations 0 to t
            for k in range(3):
                ax.plot(*trail[:, k].T, 'o-', color='k', lw=1.4, ms=4, zorder=15)
            means(ax, W, colors)
        return draw
    row('kmeans_iterations', square_box(X), [panel(t) for t in range(len(steps))],
        [f'Iteration {t}' for t in range(len(steps))], size=185, arrow=False)
    print('  iterations:', len(steps) - 1, 'points that change cluster:',
          [int((steps[t][1] != steps[t - 1][1]).sum()) for t in range(1, len(steps))])


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
    def classes_panel(ax, box):
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
    row('knn', square_box(np.vstack([X, q])), [classes_panel, knn_panel(1), knn_panel(3)],
        ['Data', ('𝑘 = 1', MATH), ('𝑘 = 3', MATH)], legend=legend)


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
    c, box = Canvas('hierarchical_clustering', 830, 260), square_box(X)
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
    c.save()
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
    c = Canvas('dbscan', 830, 230)
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
    c.save()
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
        voronoi_cells(ax, km.cluster_centers_, colors, box)
        for k, col in enumerate(colors):
            points(ax, X[km.labels_ == k], col)
        means(ax, km.cluster_centers_, colors)
    return draw


def row(name, box, draws, labels, size=220, arrow=True, frame=True, legend=None):
    """Square panels of size px, 20 px apart, over one box (None: each draw sets its own), framed (frame), a block arrow
    after the first (arrow), a label under each (a text or a (text, font) pair) and a legend column on the right."""
    c = Canvas(name, 830, size + 40)
    xs = [10 + k * (size + 20) + (30 if arrow and k else 0) for k in range(len(draws))]
    lims = (box[:2], box[2:]) if box else ((0, 1), (0, 1))
    for x, draw in zip(xs, draws):
        draw(c.panel(x, 10, size, size, *lims, frame), box)
    if arrow:
        c.arrow(size + 20, size / 2 + 10)
    for x, label in zip(xs, labels):
        text, font = label if isinstance(label, tuple) else (label, None)
        c.text(x + size / 2, size + 31, text, font=font)
    if legend:
        c.legend_column(xs[-1] + size + 16, size / 2 + 10,
                        [(lambda x, y, f=f: f(c, x, y), sw, text) for f, sw, text in legend])
    c.save()


def guides(ax, box):
    """The coordinate axes through the origin, as grey dashed guide lines."""
    ax.plot(box[:2], [0, 0], color=GREY, lw=1.0, ls=DASH, zorder=2)
    ax.plot([0, 0], box[2:], color=GREY, lw=1.0, ls=DASH, zorder=2)


def rings():
    """Two concentric rings of 50 points of radius 1 and 25 points of radius 0.4, and the ring of each point."""
    rng, parts = np.random.default_rng(0), []
    for n, r, s in ((50, 1.0, 0.05), (25, 0.4, 0.04)):  # points, radius and noise of each ring
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi / n)
        d = r + rng.normal(0, s, n)
        parts.append(np.c_[d * np.cos(t), d * np.sin(t)])
    return np.vstack(parts), np.repeat([0, 1], [50, 25])


def spectral_clustering():
    """Two concentric rings: k-means cuts them in halves, spectral clustering separates the rings."""
    X, y = rings()
    colors = np.array([BLUE, GREEN])
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    sc = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=8, random_state=0).fit(X)
    def spectral_panel(ax, box):
        for k, col in enumerate(colors[match(sc.labels_, y)]):
            points(ax, X[sc.labels_ == k], col)
    row('spectral_clustering', square_box(X),
        [data_panel(X), kmeans_panel(X, km, colors[match(km.labels_, y)]), spectral_panel],
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
    row('gmm', square_box(X), [data_panel(X), kmeans_panel(X, km, colors[match(km.labels_, y)]), gmm_panel],
        ['Data', 'k-means', 'Gaussian mixture model'])
    print('  ARI k-means:', round(adjusted_rand_score(y, km.labels_), 2),
          'GMM:', round(adjusted_rand_score(y, labels), 2))


def som_training():
    """A SOM of 6 x 6 neurons trained on points drawn uniformly from the unit square, at four iterations."""
    rng = np.random.default_rng(0)
    n, T, shown = 6, 10000, [0, 500, 3000, 10000]
    grid = np.array([(i, j) for i in range(n) for j in range(n)], float)  # the position of each neuron in the grid
    W = 0.5 + 0.15 * rng.standard_normal((n * n, 2))  # the weights start in a small random cloud
    snaps = {0: W.copy()}
    for t in range(T):
        x = rng.uniform(0, 1, 2)
        j = np.argmin(((W - x) ** 2).sum(1))  # the winner
        eta, width = 0.5 * 0.02 ** (t / T), 3 * (1 / 6) ** (t / T)  # from 0.5 to 0.01, and from 3 to 0.5
        W += eta * np.exp(-((grid - grid[j]) ** 2).sum(1) / (2 * width ** 2))[:, None] * (x - W)
        if t + 1 in shown:
            snaps[t + 1] = W.copy()
    def lattice(w):
        def draw(ax, box):
            ax.add_patch(Rectangle((0, 0), 1, 1, fc=PASTEL[BLUE], ec='k', lw=1.0, zorder=1))
            g = w.reshape(n, n, 2)
            for line in [*g, *g.transpose(1, 0, 2)]:  # the rows and the columns of the grid
                ax.plot(*line.T, color='k', lw=1.4, zorder=5)
            ax.plot(*w.T, 'o', ms=5, mfc='#ffffff', mec='k', mew=1.0, zorder=10)
        return draw
    row('som_training', (-0.1, 1.1, -0.1, 1.1), [lattice(snaps[t]) for t in shown],
        [f'Iteration {t:,}' for t in shown], size=185, arrow=False)


def mnist():
    """The 5,000 MNIST digits of the SOM example, 500 of each digit, as rows of 784 pixels from 0 to 255, and their
    labels."""
    with np.load(os.path.join(HERE, 'mnist_digits.npz')) as d:
        return d['X'].astype(float), d['y'].astype(float)


def som_mnist():
    """SOMs of 6 x 6 neurons trained on the digits 0 to 3, as the som function of the post does: tuned, with a high
    learning rate, and with a large neighborhood width."""
    X, y = mnist()
    X, n, T = X[np.isin(y, [0, 1, 2, 3])], 6, 10000
    grid = np.array([(i, j) for i in range(n) for j in range(n)], float)
    def som(eta, width):
        rng = np.random.default_rng(0)
        W = rng.uniform(0, 255, (n * n, X.shape[1]))
        for i in rng.permutation(np.arange(T) % len(X)):  # each digit in turn, in a random order
            j = np.argmin(((W - X[i]) ** 2).sum(1))
            W += eta * np.exp(-((grid - grid[j]) ** 2).sum(1) / (2 * width ** 2))[:, None] * (X[i] - W)
        return W
    def tiles(W):
        def draw(ax, box):
            img = W.reshape(n, n, 28, 28).transpose(0, 2, 1, 3).reshape(n * 28, n * 28)  # neuron (i, j) at row i
            ax.imshow(255 - img, cmap='gray', vmin=0, vmax=255, interpolation='none', extent=(0, n, 0, n), zorder=1)
            for k in range(1, n):
                ax.plot([k, k], [0, n], color='#999999', lw=0.4, zorder=2)
                ax.plot([0, n], [k, k], color='#999999', lw=0.4, zorder=2)
            ax.add_patch(Rectangle((0, 0), n, n, fill=False, ec='k', lw=1.2, joinstyle='miter', zorder=3))
        return draw
    row('som_mnist', (0, n, 0, n), [tiles(som(0.2, 1)), tiles(som(0.9, 1)), tiles(som(0.2, 4))],
        ['Tuned', 'High learning rate', 'Large neighborhood width'], size=240, arrow=False, frame=False)


def pca():
    """A Gaussian centered at (1, 3), with standard deviations of 3 along (0.866, 0.5) and of 1 across, as in the PCA
    picture of Wikipedia: centered, rotated onto its eigenvectors, and reduced to its first principal component."""
    rng = np.random.default_rng(2)
    u = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    X = np.array([1, 3]) + rng.standard_normal((30, 2)) * [3, 1] @ np.array([u, [-u[1], u[0]]])
    Xc = X - X.mean(0)
    lam, F = np.linalg.eigh(np.cov(Xc.T))
    lam, v1 = lam[::-1], F[:, 1] * np.sign(F[0, 1])  # the eigenvalues in decreasing order; v1 points right
    F = np.c_[v1, [-v1[1], v1[0]]]  # v2 is v1 turned by 90 degrees, so that F is a rotation
    Dn = Xc @ F  # the columns of D_n = F^T D_c, as rows
    def scatter(P, arrows=False):
        def draw(ax, box):
            guides(ax, box)
            points(ax, P, GREY)
            for k, (dx, dy) in enumerate([(-2, 7), (-3, 6)] if arrows else []):  # each eigenvector, 3 std long
                tip = 3 * np.sqrt(lam[k]) * F[:, k]
                charts.arrow(ax, (0, 0), tip, zorder=25)
                at = transforms.offset_copy(ax.transData, ax.figure, dx, dy, units='points')  # dx, dy px from the tip
                charts.runs(ax, *tip, charts.sub('𝑣', str(k + 1), 13), 'right', at)
        return draw
    row('pca', square_box(np.vstack([X, Xc, Dn])), [scatter(X), scatter(Xc, True), scatter(Dn), scatter(Dn * [1, 0])],
        ['1. Data', '2. Centered', '3. Rotated', '4. First component'], size=185, arrow=False)
    print('  eigenvalues:', lam.round(2), 'variance of the first component:', round(lam[0] / lam.sum(), 3))


def classes(P, y, colors, size=8):
    """Draw the points P of each class of y in its color, size px wide, in a square box of their own, over guide
    lines."""
    def draw(ax, _):
        box = square_box(P)
        ax.set_xlim(*box[:2]); ax.set_ylim(*box[2:])
        guides(ax, box)
        for k, col in enumerate(colors):
            points(ax, P[y == k], col, size)
    return draw


def kernel_pca():
    """The two rings of spectral clustering, on their first two principal components: PCA only rotates them, and
    kernel PCA with a Gaussian kernel of width 0.25, computed as in the post, separates them along the first one."""
    X, y = rings()
    n, sigma = len(X), 0.25
    K = np.exp(-((X[:, None] - X) ** 2).sum(2) / (2 * sigma ** 2))
    one = np.full((n, n), 1 / n)
    Kt = K - one @ K - K @ one + one @ K @ one  # K centered in the feature space
    lam, A = np.linalg.eigh(Kt)
    lam, A = lam[:-3:-1], A[:, :-3:-1]  # the two largest eigenvalues and their unit eigenvectors
    Y = Kt @ (A / np.sqrt(lam))  # alpha scaled so that lambda alpha^T alpha = 1
    Xc = X - X.mean(0)
    P = Xc @ np.linalg.eigh(np.cov(Xc.T))[1][:, ::-1]
    colors = [BLUE, GREEN]
    row('kernel_pca', None, [classes(X, y, colors), classes(P, y, colors), classes(Y, y, colors)],
        ['Data', 'PCA', 'Kernel PCA'])
    print('  kernel PCA eigenvalues:', lam.round(2), 'gap between the rings on the first component:',
          round(Y[y == 1, 0].min() - Y[y == 0, 0].max(), 3))


def tsne():
    """t-SNE of 100 of each digit 0 to 3, perplexity 30, optimized as in its paper: exaggeration by 4 for 50 iterations,
    learning rate 100 with adaptive gains, momentum 0.5, then 0.8 from iteration 250."""
    X, y = mnist()
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(np.flatnonzero(y == d), 100, replace=False) for d in range(4)])
    X, y, n = X[keep] / 255, y[keep].astype(int), len(keep)
    D = ((X[:, None] - X) ** 2).sum(2)
    P = np.zeros((n, n))
    for i in range(n):  # the precision beta = 1 / (2 sigma_i^2) that gives p_{.|i} a perplexity of 30, by bisection
        d, lo, hi, beta = np.delete(D[i], i), 0.0, np.inf, 1.0
        for _ in range(100):
            p = np.exp(-(d - d.min()) * beta)
            p /= p.sum()
            H = -(p * np.log(np.maximum(p, 1e-300))).sum()
            if abs(H - np.log(30)) < 1e-5:
                break
            lo, hi = (beta, hi) if H > np.log(30) else (lo, beta)
            beta = beta * 2 if hi == np.inf else (lo + hi) / 2
        P[i, np.arange(n) != i] = p
    P = (P + P.T) / (2 * n)
    Y, V, gains, shown, snaps = rng.normal(0, 1e-2, (n, 2)), np.zeros((n, 2)), np.ones((n, 2)), (0, 50, 1000), {}
    for t in range(1001):
        if t in shown:
            snaps[t] = Y.copy()
        num = 1 / (1 + ((Y[:, None] - Y) ** 2).sum(2))
        np.fill_diagonal(num, 0)
        G = 4 * (((P * (4 if t < 50 else 1) - num / num.sum()) * num)[:, :, None] * (Y[:, None] - Y)).sum(1)
        gains = np.where((G > 0) != (V > 0), gains + 0.2, gains * 0.8).clip(0.01)  # the adaptive gains of Jacobs
        V = (0.5 if t < 250 else 0.8) * V - 100 * gains * G
        Y = Y + V - (Y + V).mean(0)
    colors = [BLUE, GREEN, ORANGE, RED]
    legend = [(lambda c, x, y, col=col: c.square(x, y, col), 8, f'Digit {d}') for d, col in enumerate(colors)]
    row('tsne', None, [classes(snaps[t], y, colors, 6) for t in shown], [f'Iteration {t:,}' for t in shown],
        arrow=False, legend=legend)


def ica():
    """Two independent uniform sources of unit variance, mixed, whitened, and separated by FastICA with the kurtosis,
    as in the post; the colors mark the quadrants of the sources."""
    rng = np.random.default_rng(0)
    cells = np.indices((9, 9)).reshape(2, -1).T  # one point in each cell of a 9 x 9 grid, for an even spread
    S = np.sqrt(3) * ((cells + rng.uniform(size=cells.shape)) * 2 / 9 - 1)
    X = S @ np.array([[1, 0.6], [0.4, 1]]).T  # x = A s
    Xc = X - X.mean(0)
    lam, F = np.linalg.eigh(np.cov(Xc.T))
    Z = Xc @ F / np.sqrt(lam)  # z = V x, with V = Λ^(-1/2) F^T
    U = np.zeros((2, 2))
    for k in range(2):  # each row w of U, by the fixed-point iteration of FastICA, orthogonal to the previous rows
        w = np.eye(2)[k]
        for _ in range(100):
            w = (Z * (Z @ w)[:, None] ** 3).mean(0) - 3 * w
            w -= U[:k].T @ (U[:k] @ w)
            w /= np.linalg.norm(w)
        U[k] = w
    Y = Z @ U.T  # s = U z
    C, R = np.corrcoef(Y.T, S.T)[:2, 2:], np.zeros_like(Y)  # the correlations of Y with the sources
    for k, j in zip(*linear_sum_assignment(-np.abs(C))):  # ICA cannot recover their order and signs: match them
        R[:, j] = np.sign(C[k, j]) * Y[:, k]
    col = np.array([BLUE, GREEN, ORANGE, RED])[(S[:, 0] < 0) + 2 * (S[:, 1] < 0)]  # by the quadrant of the source
    def quadrants(P):
        def draw(ax, box):
            guides(ax, box)
            for c in np.unique(col):
                points(ax, P[col == c], c)
        return draw
    r = 1.24 * np.abs(np.vstack([S, X, Z, R])).max()
    row('ica', (-r, r, -r, r), [quadrants(P) for P in (S, X, Z, R)],
        ['1. Sources', '2. Observed', '3. Whitened', '4. Recovered'], size=185, arrow=False)
    print('  correlations with the sources:', np.abs(C).max(1).round(3))


TRANSACTIONS = [{'bread', 'milk'}, {'bread', 'butter', 'eggs'}, {'milk', 'butter', 'eggs', 'cola'},
                {'bread', 'milk', 'butter', 'eggs'}, {'bread', 'milk', 'butter', 'cola'}]  # the example of the post
ITEMS, MIN_SUPPORT = ['bread', 'milk', 'butter', 'eggs', 'cola'], 3  # the items in the order of the post
FREQUENT, INFREQUENT, PRUNED = PASTEL[GREEN], PASTEL[RED], '#ffffff'  # the fill of an itemset's row


def text_width(c, s, **kw):
    t = c.text(0, 0, s, **kw)
    w = t.get_window_extent().width
    t.remove()
    return w


def table(c, x, y, header, rows, fills):
    """A table at (x, y) px of a grey header and rows of (text, number) cells, 22 px tall, with the fill of each row;
    returns its width."""
    widths = [max(text_width(c, r[k], fontweight='bold' if r is header else 'normal') for r in [header] + rows) + 16
              for k in range(2)]
    fills = ['#f5f5f5'] + fills
    for r, (cells, f) in enumerate(zip([header] + rows, fills)):
        c.top.add_patch(Rectangle((x, y + 22 * r), sum(widths), 22, fc=f, lw=0))
        for k, text in enumerate(cells):
            c.text(x + 8 if k == 0 else x + widths[0] + widths[1] / 2, y + 22 * r + 11, text, ha='left' if k == 0 else
                   'center', va='center_baseline', fontweight='bold' if r == 0 else 'normal')
    for r in range(1, len(fills)):  # a thin grey line between rows of one fill, a black boundary between two fills
        same = fills[r] == fills[r - 1]
        c.top.plot([x, x + sum(widths)], [y + 22 * r] * 2, color='#999999' if same else 'k', lw=0.4 if same else 0.6)
    c.top.plot([x + widths[0]] * 2, [y, y + 22 * len(fills)], color='#999999', lw=0.4)
    c.top.add_patch(Rectangle((x, y), sum(widths), 22 * len(fills), fill=False, ec='k', lw=1.2, joinstyle='miter'))
    return sum(widths)


def itemset(X):
    return '{' + ', '.join(sorted(X, key=ITEMS.index)) + '}'


def support(X):
    return sum(X <= t for t in TRANSACTIONS)


def swatch(c, x, y, fill):
    """The 16 x 12 px legend swatch of a fill, centered at (x, y) px."""
    c.top.add_patch(Rectangle((x - 8, y - 6), 16, 12, fc=fill, ec='k', lw=1.0))


def fill_legend(c, x, y, entries):
    """A legend column from x px, centered at y px, of (fill, label) entries."""
    c.legend_column(x, y, [(lambda x, y, f=f: swatch(c, x, y, f), 16, t) for f, t in entries])


def apriori():
    """The candidate itemsets of each size of the post's example, with the support of those that Apriori counts, and
    the candidates that it prunes, as one of their subsets is infrequent."""
    c, levels, frequent, C = Canvas('apriori', 830, 220), [], set(), [frozenset([i]) for i in ITEMS]
    while C:
        rows = [(X, None if len(X) > 1 and any(X - {i} not in frequent for i in X) else support(X))
                for X in sorted(C, key=lambda X: sorted(map(ITEMS.index, X)))]
        frequent = {X for X, n in rows if n is not None and n >= MIN_SUPPORT}
        levels.append(rows)
        C = {a | b for a in frequent for b in frequent if len(a | b) == len(a) + 1}  # pairs that differ in one item
    x, bottom = 10, 10 + 22 * (1 + max(map(len, levels)))  # the bottom of the tallest table
    for k, rows in enumerate(levels):
        fills = [PRUNED if n is None else FREQUENT if n >= MIN_SUPPORT else INFREQUENT for _, n in rows]
        w = table(c, x, 10, ('Itemset', 'Support'), [(itemset(X), '' if n is None else str(n)) for X, n in rows], fills)
        c.text(x + w / 2, bottom + 21, f'𝑘 = {k + 1}', font=MATH)
        x += w + 30
    fill_legend(c, x - 14, bottom / 2 + 5, [(FREQUENT, 'Frequent'), (INFREQUENT, 'Infrequent'), (PRUNED, 'Pruned')])
    c.save()


def fp_growth():
    """The FP-tree of the post's example, the paths of the tree that end at eggs (its conditional pattern base), and
    the support of the itemsets of eggs that they give."""
    count = Counter(i for t in TRANSACTIONS for i in t)
    order = sorted([i for i in ITEMS if count[i] >= MIN_SUPPORT], key=lambda i: -count[i])  # ties in the post's order
    root = {'item': 'root', 'count': None, 'kids': {}}
    for t in TRANSACTIONS:  # each transaction, its frequent items sorted by support, as a path from the root
        node = root
        for i in [i for i in order if i in t]:
            node = node['kids'].setdefault(i, {'item': i, 'count': 0, 'kids': {}, 'up': node})
            node['count'] += 1
    c, nodes, leaves = Canvas('fp_growth', 830, 300), [], []
    def place(node, depth):  # the leaves 90 px apart, each parent above the middle of its children
        kids = [place(k, depth + 1) for k in node['kids'].values()]
        node['x'] = np.mean([k['x'] for k in kids]) if kids else 90 * len(leaves)
        if not kids:
            leaves.append(node)
        node['y'] = 10 + 50 * depth
        nodes.append(node)
        return node
    place(root, 0)
    for n in nodes:
        label = n['item'] if n['count'] is None else f"{n['item']}: {n['count']}"
        c.top.add_patch(Rectangle((n['x'] + 10, n['y']), 72, 30, fc='#ffffff', ec='k', lw=1.0))
        c.text(n['x'] + 46, n['y'] + 15, label, va='center_baseline')
        for k in n['kids'].values():
            c.top.plot([n['x'] + 46, k['x'] + 46], [n['y'] + 30, k['y']], color='k', lw=1.4)
    paths = []  # the prefix path of each eggs node, with the count of that node
    for n in nodes:
        if n['item'] == 'eggs':
            up, path = n['up'], []
            while up['count'] is not None:
                path, up = [up['item']] + path, up['up']
            paths.append((path, n['count']))
    x0 = max(n['x'] for n in nodes) + 82 + 40
    w = table(c, x0, 10, ('Path', 'Count'), [(', '.join(p), str(n)) for p, n in paths], ['#ffffff'] * len(paths))
    base = Counter()  # the support of each item together with eggs
    for p, n in paths:
        base.update(dict.fromkeys(p, n))
    rows = [({i, 'eggs'}, base[i]) for i in order if i in base]
    w2 = table(c, x0 + w + 40, 10, ('Itemset', 'Support'), [(itemset(X), str(n)) for X, n in rows],
               [FREQUENT if n >= MIN_SUPPORT else INFREQUENT for _, n in rows])
    bottom = max(n['y'] for n in nodes) + 30
    for x, label in ((root['x'] + 46, '1. FP-tree'), (x0 + w / 2, '2. Paths to eggs'),
                     (x0 + w + 40 + w2 / 2, '3. Itemsets with eggs')):
        c.text(x, bottom + 21, label)
    fill_legend(c, x0 + w + 40 + w2 + 16, bottom / 2 + 5, [(FREQUENT, 'Frequent'), (INFREQUENT, 'Infrequent')])
    c.save()
    print('  FP-tree nodes:', len(nodes) - 1, 'items:', sum(len([i for i in order if i in t]) for t in TRANSACTIONS))


def isolation_forest():
    """60 points of a 2D Gaussian and 3 anomalies: the splits of random isolation trees until an anomaly, or a normal
    point, is alone, and the anomaly scores of a forest of 1,000 trees built on all 63 points."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 2)), [[3.2, 2.6], [-3.0, 2.4], [2.8, -3.0]]])
    box, anomaly, normal = (-4.5, 4.5, -4.5, 4.5), 60, int(np.argmin(np.linalg.norm(X[:60], axis=1)))
    def isolate(i, seed):
        """The splits (feature, value, cell) of a random isolation tree on the path of X[i], and its leaf's cell."""
        r, idx, cell, splits = np.random.default_rng(seed), np.arange(len(X)), list(box), []
        while len(idx) > 1:
            q = r.integers(2)
            v = r.uniform(X[idx, q].min(), X[idx, q].max())
            splits.append((q, v, cell.copy()))
            idx = idx[(X[idx, q] < v) == (X[i, q] < v)]
            cell[2 * q + (X[i, q] < v)] = v  # the side of the point: cell is [x0, x1, y0, y1]
        return splits, cell
    def path_panel(i, splits, cell, col):
        def draw(ax, _):
            ax.add_patch(Rectangle((cell[0], cell[2]), cell[1] - cell[0], cell[3] - cell[2], fc=PASTEL[col], lw=0))
            for q, v, c in splits:
                ax.plot([v, v] if q == 0 else c[:2], c[2:] if q == 0 else [v, v], color='k', lw=1.0, zorder=5)
            points(ax, np.delete(X, i, 0), GREY)
            points(ax, X[i:i + 1], col, z=12)
        return draw
    forest = IsolationForest(n_estimators=1000, max_samples=len(X), random_state=0).fit(X)
    levels, fills = [0.5, 0.65], ['#ffffff', PASTEL[YELLOW], PASTEL[RED]]
    def score_panel(ax, _):
        gx, gy = np.meshgrid(np.linspace(*box[:2], 361), np.linspace(*box[2:], 361))
        score = -forest.score_samples(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)  # s(x, m) of the post
        ax.contourf(gx, gy, score, levels=[0] + levels + [1], colors=fills, zorder=1)
        ax.contour(gx, gy, score, levels=levels, colors='k', linewidths=1.0, zorder=2)
        top = -forest.score_samples(X) >= levels[1]
        points(ax, X[~top], GREY)
        points(ax, X[top], RED)
    paths = {i: isolate(i, 18) for i in (anomaly, normal)}  # a tree whose two paths are close to their mean lengths
    labels = ['Score below 0.5', 'Score 0.5 to 0.65', 'Score above 0.65']
    legend = [(lambda c, x, y, f=f: swatch(c, x, y, f), 16, t) for f, t in zip(fills, labels)]
    row('isolation_forest', box, [path_panel(anomaly, *paths[anomaly], RED), path_panel(normal, *paths[normal], BLUE),
                                  score_panel],
        [f'Anomaly: {len(paths[anomaly][0])} splits', f'Normal point: {len(paths[normal][0])} splits', 'Anomaly score'],
        size=200, arrow=False, legend=legend)
    h = {i: np.mean([len(isolate(i, seed)[0]) for seed in range(100)]) for i in (anomaly, normal)}
    print('  mean splits over 100 trees: anomaly', h[anomaly], 'normal point', h[normal],
          'scores of the anomalies:', (-forest.score_samples(X[60:])).round(2),
          'highest score of the other points:', (-forest.score_samples(X[:60])).max().round(2))


def lof():
    """A sparse and a dense cluster, a global outlier and a local outlier next to the dense cluster: the two points of
    highest distance to their 5th nearest neighbor, and of highest LOF with k = 5."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([-1.5, 0], 0.8, (40, 2)), rng.normal([2.2, 1.2], 0.25, (30, 2)),
                   [[-1.0, 3.2], [2.2, -0.2]]])
    k = 5
    dist = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, k]  # neighbor 0 is the point itself
    factor = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
    def flagged(score):
        def draw(ax, _):
            top = np.argsort(-score)[:2]
            points(ax, np.delete(X, top, 0), GREY)
            points(ax, X[top], RED)
        return draw
    row('lof', square_box(X), [data_panel(X), flagged(dist), flagged(factor)],
        ['Data', 'Distance to the 5th neighbor', 'LOF'])
    print('  LOF of the global and the local outlier:', factor[-2:].round(2), 'highest LOF of the other points:',
          factor[:-2].max().round(2), 'rank of the local outlier by distance:', int((dist > dist[-1]).sum()) + 1)


def curse_of_dimensionality():
    """The distances from 100 random query points to 1,000 points drawn uniformly in the unit cube, each divided by the
    mean distance from its query point, in 2, 10, 100 and 1,000 dimensions."""
    rng, c = np.random.default_rng(0), Canvas('curse_of_dimensionality', 830, 200)
    for k, d in enumerate((2, 10, 100, 1000)):
        X, Q = rng.uniform(size=(1000, d)), rng.uniform(size=(100, d))
        D = np.sqrt(np.maximum((Q ** 2).sum(1)[:, None] + (X ** 2).sum(1) - 2 * Q @ X.T, 0))
        h, edges = np.histogram(D / D.mean(1, keepdims=True), bins=50, range=(0, 2.5))
        ax = c.panel(10 + 205 * k, 10, 185, 130, (0, 2.5), (0, 1.1), frame=False)
        ax.axis('on')
        ax.tick_params(left=False, labelleft=False)
        ax.set_xticks([0, 1, 2])
        ax.stairs(h / h.max(), edges, fill=True, fc=charts.BLUE_FILL, ec='k', lw=1.0)
        c.text(102.5 + 205 * k, 180, f'{d:,} dimensions')
        print(f'  {d} dimensions: median (d_max - d_min) / d_min =',
              np.median((D.max(1) - D.min(1)) / D.min(1)).round(2))
    c.save()


FIGURES = {f.__name__: f for f in [
    voronoi, kmeans_iterations, knn, hierarchical_clustering, dbscan, spectral_clustering, gmm, som_training,
    som_mnist, pca, kernel_pca, tsne, ica, apriori, fp_growth, isolation_forest, lof, curse_of_dimensionality]}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
