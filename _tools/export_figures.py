"""Export the pictures of a draw.io file to SVG, with the text as vector outlines.

usage: python3 _tools/export_figures.py assets/<folder>/<folder>.drawio [NAME ...]

Each picture is a group with id "<name>-figure" (CLAUDE.md, Pictures). Each one, or only the NAMEs given, is
copied to a temporary draw.io file of its own, at x=0, y=0 and without the tab's background color. draw.io
renders each file to a PDF cropped to the picture, which turns the text into glyph outlines (one PDF per
picture: in a PDF of several pages, draw.io can print parts of one page on the next), and pdftocairo converts
the PDF to SVG; JOBS pictures (default 4) are rendered at the same time. The SVG loses the white page background
that draw.io paints, and is cropped to its drawn content (draw.io's crop leaves room around arrowheads and text
boxes): its viewBox becomes the box of the page's non-white pixels, rendered at 288 dpi, plus 0.5 px. The SVG
is written over the file of the same name in the draw.io file's folder (or its subfolders), or to the folder
itself for a new picture.

draw.io prints 100 px per inch, so its PDF has 0.72 pt per draw.io px, and the SVG's width and height are set back
to px; its viewBox keeps the PDF's units. Chrome, which prints the PDF, fits the picture into a page that draw.io
sizes in hundredths of an inch, which can scale the picture by a few tenths of a percent: the scale is read from the
PDF and the width and height take it into account, so that 1 draw.io px is always 1 px on the page. The script
prints each picture's width and height in px and the width P = W / X * 100 to use in the Markdown,
<img width="P%" height="P%" ...>, where X is the band that the pictures of a post take, the width of the page
($content-width in assets/main.scss; CLAUDE.md, Pictures, section 3).

Requires the draw.io desktop app (DRAWIO, default "drawio"; on a headless Linux machine for example
DRAWIO="xvfb-run -a drawio --no-sandbox"), pdftocairo and pdftoppm (poppler-utils), and the fonts Source Sans 3
and New Computer Modern Sans Math (CLAUDE.md, Pictures, section 11).
"""
import collections, concurrent.futures, copy, os, re, shlex, shutil, subprocess, sys, tempfile, zlib
import xml.etree.ElementTree as ET


def band_width():
    """The width of the band of the pictures of a post in px: $content-width in assets/main.scss."""
    scss = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'assets', 'main.scss')
    with open(scss, encoding='utf-8') as f:
        return float(re.search(r'^\$content-width:\s*([\d.]+)px', f.read(), re.M).group(1))


PAGE = band_width()
PT_PER_PX = 0.72
DPI = 288
INK = re.compile(rb'[\x00-\xf5]')                  # a pixel darker than white


def cell(e):
    return e if e.tag == 'mxCell' else e.find('mxCell')


def parts(root, fid):
    """The cells inside a group, at any depth."""
    children = {}
    for e in root:
        children.setdefault(cell(e).get('parent'), []).append(e)
    out, todo = [], [fid]
    while todo:
        for e in children.get(todo.pop(), []):
            out.append(e); todo.append(e.get('id'))
    return out


def svg_path(drawio, name):
    folder = os.path.dirname(os.path.abspath(drawio))
    for d, _, files in os.walk(folder):
        if name + '.svg' in files:
            return os.path.join(d, name + '.svg')
    return os.path.join(folder, name + '.svg')


def ink_box(pdf, page):
    """The box of a PDF page's non-white pixels, in pt."""
    out = os.path.join(os.path.dirname(pdf), 'ink')
    subprocess.run(['pdftoppm', '-gray', '-r', str(DPI), '-f', str(page), '-l', str(page), '-singlefile', pdf, out],
                   check=True)
    data = open(out + '.pgm', 'rb').read()
    m = re.match(rb'P5\s+(\d+)\s+(\d+)\s+\d+\s', data)
    w, h, off = int(m.group(1)), int(m.group(2)), m.end()
    x0, y0, x1, y1 = w, h, -1, -1
    for y in range(h):
        row = data[off + y * w: off + (y + 1) * w]
        first = INK.search(row)
        if first:
            y0, y1 = min(y0, y), y
            x0 = min(x0, first.start())
            x1 = max(x1, w - 1 - INK.search(row[::-1]).start())
    if x1 < 0:
        return None
    k = 72 / DPI
    return x0 * k, y0 * k, (x1 + 1) * k, (y1 + 1) * k


def pdf_scale(pdf):
    """The scale of the picture in a draw.io PDF, 1 when 1 draw.io px is 0.72 pt. Chrome draws each vector element of
    the picture with a transform of 3 dots of 300 dpi per draw.io px times this scale; the page itself has a transform
    of 3.125 (300 dpi over 96 dpi), and images have their own."""
    data = open(pdf, 'rb').read()
    count = collections.Counter()
    for m in re.finditer(rb'stream\r?\n', data):
        try:
            z = zlib.decompress(data[m.end():data.find(b'endstream', m.end())])
        except zlib.error:
            continue
        for a, d in re.findall(rb'(?<![-\d.])([\d.]+) 0 0 ([\d.]+) -?[\d.]+ -?[\d.]+ cm', z):
            if a == d and 2.9 < float(a) < 3.1:
                count[float(a)] += 1
    return count.most_common(1)[0][0] / 3 if count else 1.0


def crop(svg, box, scale=1.0):
    """Drop the white page background, crop to the box (pt) plus 0.5 px, set width and height in px for a picture
    drawn at scale times 0.72 pt per px."""
    s = open(svg).read()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]

    def page_background(m):
        nums = [float(v) for v in re.findall(r'-?[\d.]+', m.group(1))]
        xs, ys = nums[0::2], nums[1::2]
        return '' if max(xs) - min(xs) >= 0.98 * vb[2] and max(ys) - min(ys) >= 0.95 * vb[3] else m.group(0)
    s = re.sub(r'<path fill-rule="nonzero" fill="rgb\(100%, 100%, 100%\)" fill-opacity="1" d="(M [^"]*)"/>',
               page_background, s)
    pt = PT_PER_PX * scale
    if box:
        x0, y0, x1, y1 = box[0] - 0.5 * pt, box[1] - 0.5 * pt, box[2] + 0.5 * pt, box[3] + 0.5 * pt
    else:
        x0, y0, x1, y1 = vb[0], vb[1], vb[0] + vb[2], vb[1] + vb[3]
    w, h = (x1 - x0) / pt, (y1 - y0) / pt
    head = re.search(r'<svg[^>]*>', s).group(0)
    new = re.sub(r'\swidth="[^"]*"', ' width="%.1f"' % w, head, count=1)
    new = re.sub(r'\sheight="[^"]*"', ' height="%.1f"' % h, new, count=1)
    new = re.sub(r'\sviewBox="[^"]*"', ' viewBox="%.3f %.3f %.3f %.3f"' % (x0, y0, x1 - x0, y1 - y0), new, count=1)
    open(svg, 'w').write(s.replace(head, new, 1))
    return w, h


def export(drawio, page, pdf):
    """Render a one-page draw.io file to a PDF cropped to its picture."""
    os.makedirs(os.path.dirname(pdf))
    src = os.path.join(os.path.dirname(pdf), 'page.drawio')
    doc = ET.Element('mxfile')
    doc.append(page)
    ET.ElementTree(doc).write(src, encoding='utf-8')
    subprocess.run(drawio + ['--export', '--format', 'pdf', '--crop', '--output', pdf, src],
                   check=True, stdout=subprocess.DEVNULL)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src, only = sys.argv[1], set(sys.argv[2:])
    pages = {}
    for diagram in ET.parse(src).getroot().iter('diagram'):
        model = diagram.find('mxGraphModel')
        if model is None:
            continue
        root = model.find('root')
        for fig in [e for e in root if (e.get('id') or '').endswith('-figure')]:
            name = fig.get('id')[:-len('-figure')]
            if only and name not in only:
                continue
            page = ET.Element('diagram', {'id': 'p0', 'name': name})
            m = ET.SubElement(page, 'mxGraphModel', {k: v for k, v in model.attrib.items() if k != 'background'})
            r = ET.SubElement(m, 'root')
            ET.SubElement(r, 'mxCell', {'id': '0'})
            ET.SubElement(r, 'mxCell', {'id': '1', 'parent': '0'})
            g = copy.deepcopy(fig)
            cell(g).set('parent', '1')
            geo = cell(g).find('mxGeometry')
            geo.set('x', '0'); geo.set('y', '0')
            r.append(g)
            r.extend(copy.deepcopy(e) for e in parts(root, fig.get('id')))
            pages[name] = page
    missing = only - set(pages)
    if missing:
        sys.exit('no such picture: ' + ', '.join(sorted(missing)))
    tmp = tempfile.mkdtemp()
    try:
        drawio = shlex.split(os.environ.get('DRAWIO', 'drawio'))
        pdfs = {name: os.path.join(tmp, str(i), 'page.pdf') for i, name in enumerate(pages)}
        with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('JOBS', '4'))) as pool:
            for job in [pool.submit(export, drawio, pages[name], pdfs[name]) for name in pages]:
                job.result()
        for name, pdf in pdfs.items():
            out = svg_path(src, name)
            subprocess.run(['pdftocairo', '-svg', pdf, out], check=True)
            w, h = crop(out, ink_box(pdf, 1), pdf_scale(pdf))
            print('%-40s %6.1f x %6.1f px   P = %.1f' % (os.path.relpath(out), w, h, min(100.0, w / PAGE * 100)))
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
