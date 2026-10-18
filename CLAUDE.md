# Notes for Claude

## Writing rules

Read this section before any task that writes or edits text in `_posts`, `_drafts`, `publications.md` or
`index.md`, and follow it. When the user gives a new writing rule, add it here in the same change.

### Audience and voice

- The site reads like a college course: a professor teaching undergraduates, who speaks like an engineer. The
  readers are HPC and ML researchers, engineers and students. They come for clear information, and nobody likes
  long posts or boring sentences.
- Factually correct, direct, concise, technical, and relevant to HPC and ML. No business or marketing language,
  no anecdotes, no opinions, no hype, no motivation speeches, no calls to action. No "in this post we will"
  opening and no "in conclusion" ending.
- Third person only. Never "I", "we", "our", "us", "you", "your" or "my". The subject of a sentence is the
  algorithm, the model, the data or the hardware: "Each rank sends its block", not "We send the block". Quotes
  and paper titles keep their original words.

### Sentences and words

- Short sentences, one idea each. Short paragraphs. Direct and precise.
- Simple words that a non-native English speaker knows, plus the exact technical term ("redundant copies",
  "memory footprint", "latency-bound"). No colloquial wording ("wastes memory"), no fancy synonyms, no
  buzzwords.
- Few filler words ("very", "really", "basically", "simply", "just", "actually", "of course", "note that",
  "in order to"): use one only where it helps the sentence.
- Say each thing once. No repeated definitions, no paragraph that restates the previous one, no summary that
  only repeats the post.
- Short posts. Cut every sentence that adds no information. Keep every formula, algorithm step, number and
  result that does.
- When a post presents several methods, compare them where they appear, or in a table at the end of the post.
  A simple comparison is enough ("A needs less memory than B"). Say when to use each method, and the idea
  behind it.

### References

- Add the references a post needs, and only the important and relevant ones. Keep the existing references.
  Verify each new reference before adding it.
- Place each reference where it makes sense: a method links to the paper that introduced it, when needed. Always
  name the paper that introduced a recent method.
- The introduction cites only a background paper that the introduction needs. The end of a post can close with
  where to find more.
- Describe other work in a brief note, not at length.

### Technical correctness

- Check every formula, derivation, index, dimension, complexity, unit, number and factual claim, and fix the
  errors in the text. One symbol per quantity within a post, defined at its first use.
- Check that the text, the code and the pictures agree.
- Link a technical term to its English Wikipedia article at its first use in a post, when a reader may not
  know it. Link only articles that exist, and each term once.
- Code, commands and links stay unchanged unless they are wrong.
- A text task never adds, removes, changes or reorders pictures, their markup or their captions.
- Math is written in `$$...$$`, inline and on its own lines, as in the existing posts. In inline math, write
  `\vert` instead of `|`: kramdown reads a paragraph line that holds a `|` as a table row.

### Titles

- A post title, with the category that the home page puts before it ("Machine Learning: "), fits on one line of
  the home page on a desktop screen. The title column is 815 px wide at 16 px: keep the line under 750 px in the
  site's font, about 90 characters, so that it also fits with wider system fonts.
- A text task does not change titles: it proposes new ones in the title issues of `TODO.md`.

### TODO list

Problems that a task finds but does not fix, or that need a later look, go in `TODO.md` at the root of the
repository, with a proposed fix for each item. The file has four sections, with one subsection per post:

- Code issues: bugs in code and data files.
- Diagram issues: pictures that are wrong or disagree with the text. The picture tasks add their open items here
  too.
- Content issues: content problems that need the user's decision, such as overlap between posts.
- Title issues: proposed changes to post titles.

The user picks the items to fix before the next pass. A fixed item leaves the list.

### Process

- Rewrite one post at a time: read it in full, check its math and pictures, then rewrite it.
- Keep the line endings of each file (many posts use CRLF).
- Build the site with `bundle exec jekyll build --drafts` and check that every formula and picture renders.
- After each full pass over all posts, push a snapshot branch and show this file to the user.

## Pictures

These rules apply to every picture in _posts, _drafts and publications.md: drawing, restyling, exporting
to SVG, sizing it in the Markdown and writing its caption. Read this section before any such task, and add
each new picture rule the user gives to it, in the same change.

The two kernel fusion paper repositories (brunomaga/moe_paper_huawei and brunomaga/ffn_tp_paper_huawei)
are the style reference only. Never change them. Their imgs/*.svg figures are the example of the look to
follow (pastel draw.io fills, thin black outlines, thin black arrows, dark accent text).

### 1. Content

- A picture is correct, simple and factual, with a standard layout, like a figure in a college course or a
  textbook: one idea per picture, parts aligned on a grid, flow from left to right or from top to bottom, no
  decoration.
- Every label, dimension, number, arrow and step agrees with the original paper or another reliable source
  (the paper itself, university lecture notes, textbooks, Wikipedia), and with the post's text, code and
  formulas: the same names and symbols for the same quantities (for example "rank" and p, when the text says
  so). Check the source before drawing or changing a picture, and name the sources checked in the PR.
- The logic of a picture makes sense: each step follows from the previous one, arrows connect the right
  parts in the right direction, and shapes, data flow and results agree with the algorithm and the formulas.
  When they do not, the picture is fixed with facts from the sources.
- Worked examples (data, steps, results) are computed by running the algorithm, not by hand, and the picture
  shows the computed values.
- Third person, as the post text: no "I", "we", "our", "ours", "us", "you" or "your" in a picture. A paper's
  "Ours" becomes the method's name. Quoted inputs (prompts, chat turns, example sentences) keep their words.

### 2. Sources and files

- draw.io is the source of every diagram SVG. Change the draw.io file first, then export the SVG from it.
  Every SVG and its draw.io source are updated together, in the same commit, before a merge.
- One draw.io file per asset folder: assets/<folder>/<folder>.drawio. One tab per post, named after the
  post file without its date (for example "pipeline-parallelism"). The publications page uses
  assets/publications/publications.drawio.
- Each picture in a tab is a group with id "<name>-figure", where <name> is the SVG file name without its
  extension, plus a text cell "<name>-label" above it holding the file name. The label is not exported.
- The SVG keeps its path: assets/<folder>/[imgs/]<name>.svg. Delete the old PNG/JPG when it becomes an SVG,
  and fix every link to it.
- The kernel fusion posts show copies of the paper repositories' imgs/*.svg (the repositories are private,
  so the posts need copies). A copy that does not follow the sizing in section 3 is redrawn in this repo,
  in a tab of assets/inference/inference.drawio named after the post, from the paper repository's draw.io
  or TikZ source. The paper repositories stay unchanged.
- A picture used by a post and by the publications page is stored once, in the post's asset folder, and
  both pages link to that file. The same holds for a picture used by two posts (for example
  matrix-transpose/all_to_all.svg in the matrix transpose and mixture-of-experts posts).
- Charts drawn from data (plots, benchmarks, histograms) keep their data and the script or spreadsheet that
  draws them next to the SVG, and follow the same palette, font and sizes as the diagrams. A chart whose data
  or script is not in the repository is restyled in its SVG: its font, text sizes, colors and line weights
  change, its data and marks stay.
- Paper figures are redrawn in this style too, and their captions credit the paper (section 9).

### 3. Page width and scale

- The post content width is X = 880 px (minima's $content-width 940 px minus 2 x 30 px padding;
  assets/main.scss). The publications page shows its pictures in a 766 px column, but sizes them for
  X = 880 px too, so there they appear at 87% of their size in a post.
- Every picture is drawn at page scale: 1 draw.io px is 1 px on the page when the picture is shown at
  its natural size. No picture is wider than 880 px.
- The Markdown shows a picture of width W px at W/880*100 percent, rounded to one decimal:
  ```
  {: .picture}
  <img width="P%" height="P%" src="{{ site.assets }}/<folder>/<name>.svg"/>
  ```
  with P = W / 880 * 100 (W is the width of the exported SVG). So text, lines and arrowheads look the same
  size in every picture of the site.
- Pictures shown side by side in one row each keep their own P. When the P values of a row add up to more
  than 98 (the spaces between pictures take the rest), the pictures wrap onto the next line.
- A picture that does not fit 880 px at the text sizes below is redrawn: first narrow the empty space
  between its parts, then move parts (wrap a row into two rows, stack panels, break long labels over two
  lines). Never shrink the text to make it fit.

### 4. Font

- One font family in every picture: Times New Roman (fontFamily=Times New Roman), as in the kernel paper
  figures. Its metric twin Liberation Serif stands in where Times New Roman is not installed. Code
  identifiers may use Courier New. Formulas use draw.io's math or Unicode sub/superscripts in the same
  font.
- Text sizes, in page px (draw.io fontSize at page scale):
  ```
  title          16           the picture's title, and labels drawn larger than the regular text
  subtitle       14  bold     panel titles, headers, group names
  regular        14  normal   box labels, arrow labels, legends: the most common text of a picture
  small          12  normal   dimensions, axis ticks, annotations, subscripts in labels
  tiny           10  normal   only for dense labels inside small cells (pipeline micro-batch numbers,
                              matrix weights), where 12 does not fit
  ```
  A picture redrawn from an older one keeps the order of its text sizes: its most common size becomes 14,
  sizes at least 1.1 times larger become 16, sizes under 0.9 times smaller become 12.
  Italic is allowed for variables and for emphasis, bold only for titles, subtitles and names the post's
  text refers to.
- Charts use the same sizes: tick labels 12, axis labels and legends 14, panel titles 14 bold. A panel title
  that is a sentence goes into the caption; panel letters and short panel names stay. A legend with many
  entries in a small chart may use 10 (llama3_scaling_laws).
- Numbers: a true minus sign (−), a space between a number and its unit (25 μs), upright units, and commas
  between thousands (48,001).
- Text color: black #000000. Secondary text (faded, next iteration, notes): grey #666666. Accent text, for
  names the text refers to, uses the dark accent colors in section 5.
- Free text does not wrap inside a box narrower than the text (whiteSpace=nowrap, explicit <br> for
  line breaks).

### 5. Color palette

- Fills (draw.io pastel pairs). With black outlines, the second value is only used as an accent:
  ```
  blue     fill #dae8fc   accent #6c8ebf
  green    fill #d5e8d4   accent #82b366
  orange   fill #ffe6cc   accent #d79b00
  yellow   fill #fff2cc   accent #d6b656
  red      fill #f8cecc   accent #b85450
  purple   fill #e1d5e7   accent #9673a6
  grey     fill #f5f5f5   accent #666666
  white    fill #ffffff
  ```
  Dark grey #b3b3b3 fills masked, padding, NULL or inactive elements, so they stand out from the white and
  light grey ones. The accent colors may also fill small marks (dots of a scatter plot, nodes of a graph,
  heat map cells) where pastel fills are too faint.
- Dark accent text (from the kernel paper figures): blue #1e3a5f, green #456e36, orange #885e10,
  purple #6b4a80, red #b85450.
- Pipeline schedules and other pictures that need more distinct colors (all pipeline pictures share it):
  ```
  forward              #6c8ebf   second chunk / reverse direction  #385681
  backward             #82b366                                     #4f743a
  backward for inputs  #b85450                                     #74322f
  backward for weights #d79b00                                     #856000
  next iteration       #dae8fc with grey #666666 outline and text, forward passes only, then "..."
                       in the vertical centre
  idle                 hatched (section 7)
  ```
  Every pipeline picture uses 4 GPUs, except DualPipe (6), and shows no previous iteration. Section 8 gives
  the layout.
- The same thing has the same color within a post (one color per process, GPU, head, expert...), and the
  same meaning keeps its color across posts where possible (blue forward, green backward).
- New colors are only added when the palette above runs out, and must look like part of it (pastel fill,
  readable black text on top).
- Real photos keep their own colors (section 10).

### 6. Lines, outlines and arrows

- Every geometric element with a fill (rectangles, rounded boxes, circles, cubes, cylinders, filled table
  cells, block arrows, grey background panels and frames) has a black outline: strokeColor=#000000,
  strokeWidth=0.8, no stroke opacity. Dashed elements too (the dash keeps its pattern).
- Exceptions that keep their own look:
  - colored lines and frames that highlight part of a drawing (for example the red frames around modules
    in the GPT-lite pictures, the orange KV cache frames, the red frame of process A in all_to_all);
  - thicker lines that emphasise a part of a drawing (for example DualPipe's "overlapped forward &
    backward" frame): 2 px;
  - white masks that hide parts of a drawing, faded elements, curly brackets, red cross marks, and table
    grid lines without fill.
- Thin arrows and connectors: black #000000, 1 px, endArrow=block, endFill=1, endSize=6. Dashed arrows
  use the same head. Grey #666666 only for faded or "next iteration" content.
- Inhibition (one part blocks or breaks down another, as in a signalling pathway): 1 px black line with a
  flat bar head, endArrow=baseDash;endSize=10.
- Frames and lines without fill: black, 1 px (2 px for an emphasis). Lines thinner than 0.7 px at page scale
  (grid lines drawn over a photo or a matrix) keep their weight.
- No end-of-sequence cells or tokens in schedules: the last decode cell is the last token.

### 7. Shapes, blocks and sizes

- The same shape means the same thing in every picture:
  ```
  data (tensors, matrices, vectors, tokens, buffers)    rectangle with square corners
  operation, layer, module, kernel                       rounded rectangle
  device, process, group (GPU, rank, node, worker)       frame with a light fill, its name at the top left
  3D tensor                                              isometric cube (isoCube2); CNN feature map: cube
  storage (disk, database, cache)                        cylinder
  encoder, decoder                                       trapezoid
  decision                                               rhombus
  neuron, graph node                                     circle
  operator (+, x, concatenation)                         circle 20 px wide with the symbol inside
  span of several parts                                  curly bracket, 1 px
  transformation of a whole stage (a model into another) block arrow; thin arrows everywhere else
  ```
- Rounded rectangles have a 5 px corner radius (absoluteArcSize=1;arcSize=10). Capsules (fully round ends)
  keep their shape.
- Block arrows: body 10 px wide, head 20 px wide and 10 px long (flexArrow: width=10;endWidth=10;endSize=3.07;
  singleArrow: 20 px tall, arrowWidth=0.5). A block arrow with a label inside keeps a body taller than its
  label, with the same proportions. A block arrow thinner than 6 px is a thin arrow (section 6).
- Dashed lines use draw.io's default dash pattern; dotted lines use dashPattern=1 2.
- Arrows are straight or orthogonal (edgeStyle=orthogonalEdgeStyle). Curves only for pointers to a part of
  the drawing and for loops.
- Hatching (idle, masked or padding areas): black lines 0.6 px wide every 4 px at 45 degrees, drawn as vector
  paths: fillColor=#000000;fillStyle=hachure;sketch=1;jiggle=0;hachureAngle=45;hachureGap=4;fillWeight=0.6;
  disableMultiStroke=1;disableMultiStrokeFill=1. draw.io's fillStyle=hatch turns into a bitmap in the PDF
  export. A hatched area without outline lies 0.5 px inside the frame drawn over it, so the line ends stay
  under the frame.
- The pictures of one series in a post (the attention variants, the pipeline schedules...) use the same
  element sizes: the same cube, cell, row and box sizes in every picture of the series.
- Sizes for new and redrawn pictures: a one-line box 30 px tall, 20 px between boxes in a row, 40 px between
  rows (room for an arrow and its label), matrix and array cells 20 x 20 px with 12 px numbers (10 px when they
  do not fit).

### 8. Pipeline schedules

All pipeline schedules (the pipeline parallelism post and the publications page) share one layout:
- One row per GPU, labelled "GPU 0" to "GPU 3" ("Worker 1" to "Worker 4" where the post's text says workers,
  as for the PipeDream pictures), 14 px regular, right-aligned 4 px left of the first cell.
- Rows 20 px tall and 3 px apart. One time unit is 21.86 px wide: a forward pass takes 1 unit, a backward pass
  2 units, or 1 unit each for the backward for inputs (B) and the backward for weights (W). With two chunks per
  GPU (interleaved 1F1B, ZB-V, DualPipe, DualPipeV), each pass of a chunk takes half as long.
- Each row is a 1 px black frame as long as the schedule, hatched (section 7) behind the passes: the hatch is
  idle time. A pass is one cell with a 0.8 px black outline, in the pipeline colors of section 5; a backward pass
  is one cell of two units, not two cells.
- Micro-batch numbers are 10 px: black on the base colors, white on the dark ones, grey #666666 in the
  next-iteration cells.
- The schedule ends with the forward passes of the next iteration on every GPU, then "..." (16 px, grey),
  vertically centred to the right of the rows.
- A marker (for example "1F1B ends", the time at which the post's 1F1B schedule ends) is a dashed grey
  vertical line with a 12 px grey label right-aligned to it, above the rows.
- Overlapped forward and backward passes (DualPipe) have a 2 px black frame around them.
- The legend is centred under the rows, 12 px below them (or below the time axis): each entry is a 20 x 12 px
  swatch (two 10 x 12 px halves for the two chunks or directions) and its 14 px label 4 px to the right, with
  16 px between entries, on two rows when it is wider than the schedule. The labels are Forward, Backward,
  Backward for inputs, Backward for weights, Overlapped forward & backward, Idle, Next iteration.
- Schedules redrawn from a paper's grid of outlined cells over a hatched block (PipeDream) follow this layout
  too; their annotations ("Pipeline flush", "Startup State", the time axis) keep their place relative to
  the rows.
- No title inside a schedule (section 9). Pictures compared side by side in one row (pipeline_8_stages,
  pipeline_4_stages, pipeline_2_stages) are at most 287 px wide each, so that three fit one row.

### 9. Captions

- No caption inside a picture. The caption is Markdown below the picture, in the site's caption style
  (centred, small, black), one paragraph:
  ```
  {: .picture}
  <img ...>

  {: .caption}
  Caption text. Adapted from: [paper](link)
  ```
- A picture that comes from a source is credited in the last sentence of its caption, with a link to the
  paper (its arXiv abstract page when it has one), the Wikipedia article, the documentation or the lecture
  notes:
  - "Source: [title](link)" for the source's own figure or photo, shown as it is;
  - "Adapted from: [title](link)" for a picture redrawn from a figure of the source (same content,
    restyled, simplified or corrected);
  - "Inspired by: [title](link)" for a picture drawn for the post that loosely follows an idea or a figure
    of the source.
  A picture drawn from the post's own example, data or code has no credit.
- Panel letters such as "(a)" may stay inside the picture; the panel's text goes into the caption.
- Titles that describe a schedule or a variant (for example "ZB-H1 (same activation memory as 1F1B)") go
  into the caption, not the picture.
- Labels and legends stay inside the picture.

### 10. Real photos inside pictures

- Never destroy a real photo inside a figure (faces, the duck in Soft MoE, the dog in vDNN, satellite
  images, brains, stars...). Everything around the photo is redrawn as vector elements; the photo part
  is either kept as an embedded image cropped from the original, or replaced by vector elements that look
  very much like it.
- Photo parts keep their original colors and do not follow the palette.
- Tiles or patches cut from a photo must match the photo they come from (same crop, same detail).
- If a photo cannot be kept or redrawn faithfully, leave the picture as it is and say so.

### 11. Export

- Export with _tools/export_figures.py, which runs the draw.io desktop CLI:
  ```
  DRAWIO="xvfb-run -a drawio --no-sandbox" python3 _tools/export_figures.py assets/<folder>/<folder>.drawio [name ...]
  ```
  It renders each "<name>-figure" group (or the names given) to a cropped PDF, which turns the text into
  glyph outlines, converts it to SVG with pdftocairo, removes the white page background, crops the SVG to its
  drawn content (its non-white pixels plus 0.5 px), writes it over the SVG of the same name in the folder,
  and prints the width P to use in the Markdown. Outlined text looks the same on every device, whatever
  fonts it has, as in the kernel paper figures.
- Times New Roman must be installed where the export runs (for example the msttcorefonts package):
  without it draw.io draws another serif font, with other widths.
- Transparent background, no embedded fonts. Stencils from mxgraph.* libraries need the real draw.io CLI
  to render; do not export them with another renderer.
- Check every exported SVG at its page size (P% of 880 px) before committing: no overlapping text, no
  text outside its box, arrows ending on their targets.

### 12. Changes and review

- A picture task changes pictures, their draw.io sources, their Markdown (img tags, widths) and their
  captions. In the post's text it only makes tiny fixes that keep the text in agreement with a corrected
  picture (a word, a symbol, a number), never whole sentences or paragraphs; larger text issues are reported
  to the user.
- Work in iterations. After each one, show a before/after of the changed pictures, as images at their page
  size, and this file, then stop: the next iteration starts only when the user asks for it. Merge only after
  the user's OK.
- Problems that a picture task finds but does not fix, and items for a later look, go in the Diagram issues
  of TODO.md, as the TODO list section above says.
- Check the content against the sources (section 1). Fix content errors only when certainly wrong, and
  list them in the PR with the source that shows the right version.
- Keep useful earlier work: when a picture is redrawn, do not throw away a version that already followed
  these rules (for example the colored all_to_all of the matrix transpose post).
