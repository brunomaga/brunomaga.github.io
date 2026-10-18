"""Export the pictures of a draw.io file to SVG, with the text as vector outlines.

usage: python3 _tools/export_figures.py assets/<folder>/<folder>.drawio [NAME ...]

Each picture is a group with id "<name>-figure" (CLAUDE.md, Pictures). Each one, or only the NAMEs given, is
copied to its own page of a temporary draw.io file, at x=0, y=0 and without the tab's background color.
draw.io renders all pages to one PDF, cropped to each picture, which turns the text into glyph outlines, and
pdftocairo converts each page to SVG. The SVG is written over the file of the same name in the draw.io file's
folder (or its subfolders), or to the folder itself for a new picture.

draw.io's PDF has 0.72 pt per draw.io px, so the SVG's width and height are set back to px; its viewBox keeps
the PDF's units. The script prints each picture's width and height in px and the width P = W / 880 * 100 to use
in the Markdown: <img width="P%" height="P%" ...>.

Requires the draw.io desktop app (DRAWIO, default "drawio"; on a headless Linux machine for example
DRAWIO="xvfb-run -a drawio --no-sandbox"), pdftocairo (poppler-utils) and the Times New Roman font.
"""
import copy, os, re, shlex, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET

PAGE = 880           # content width of a post, in px
PT_PER_PX = 0.72


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


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src, only = sys.argv[1], set(sys.argv[2:])
    doc = ET.Element('mxfile')
    names = []
    for diagram in ET.parse(src).getroot().iter('diagram'):
        model = diagram.find('mxGraphModel')
        if model is None:
            continue
        root = model.find('root')
        for fig in [e for e in root if (e.get('id') or '').endswith('-figure')]:
            name = fig.get('id')[:-len('-figure')]
            if only and name not in only:
                continue
            page = ET.SubElement(doc, 'diagram', {'id': 'p%d' % len(names), 'name': name})
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
            names.append(name)
    missing = only - set(names)
    if missing:
        sys.exit('no such picture: ' + ', '.join(sorted(missing)))
    tmp = tempfile.mkdtemp()
    try:
        pages, pdf = os.path.join(tmp, 'pages.drawio'), os.path.join(tmp, 'pages.pdf')
        ET.ElementTree(doc).write(pages, encoding='utf-8')
        drawio = shlex.split(os.environ.get('DRAWIO', 'drawio'))
        subprocess.run(drawio + ['--export', '--format', 'pdf', '--crop', '--all-pages', '--output', pdf, pages],
                       check=True, stdout=subprocess.DEVNULL)
        for i, name in enumerate(names, 1):
            out = svg_path(src, name)
            subprocess.run(['pdftocairo', '-svg', '-f', str(i), '-l', str(i), pdf, out], check=True)
            s = open(out).read()
            head = re.search(r'<svg[^>]*>', s).group(0)
            w = float(re.search(r'\swidth="([\d.]+)', head).group(1)) / PT_PER_PX
            h = float(re.search(r'\sheight="([\d.]+)', head).group(1)) / PT_PER_PX
            new = re.sub(r'\swidth="[^"]*"', ' width="%.1f"' % w, head, count=1)
            new = re.sub(r'\sheight="[^"]*"', ' height="%.1f"' % h, new, count=1)
            open(out, 'w').write(s.replace(head, new, 1))
            print('%-40s %6.1f x %6.1f px   P = %.1f' % (os.path.relpath(out), w, h, min(100.0, w / PAGE * 100)))
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
