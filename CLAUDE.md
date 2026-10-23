# Notes for Claude

## Writing rules

Read this section before any task that writes or edits text in `_posts`, `_drafts`, `publications.md` or
`index.md`, and follow it. When the user gives a new writing rule, add it here in the same change.

### Audience and voice

- The posts are written in the author's voice: an HPC and ML engineer who moved from writing a blog to teaching
  a university course, an enthusiastic young lecturer who wants students to read the posts. The text reads
  like a college lecture for HPC and ML readers, written as in a textbook: third person, sentences that flow and
  explain why each step is needed, Wikipedia links for technical terms at first use, and methods compared where
  they appear.
- The text reads as the author's own writing, not as an LLM's. Each page keeps the author's explanations,
  examples, derivations and order of ideas, and as much of the author's wording as the other rules allow: a
  reader who knows the original post recognizes it paragraph by paragraph.
- In practice, the rewrite writes in the third person, updates phrases with clearer and simpler synonyms, keeps
  the flow of the author's text, adds the missing sentences, and removes the redundant sentences and the
  redundant information.
- Third person only. Never "I", "we", "our", "us", "you", "your" or "my". The subject of a sentence is the
  algorithm, the model, the data or the hardware ("Each rank sends its block", not "We send the block"), or the
  sentence is passive ("the loss is computed as"). The imperatives of mathematics ("consider", "take", "imagine",
  "let $$x$$ be") and an impersonal "one" are fine. Quotes and paper titles keep their original words. The
  introduction of the home page (`index.md`) is the author's bio and keeps its first person.
- Factually correct, technical and relevant to HPC and ML. No business or marketing language, no hype, no calls
  to action, no "in conclusion" ending. An introduction may say what the post covers, as the author does, in the
  third person ("This post derives ... and implements ...").

### The author's style

This style was learned from the author's own text of the posts, as they were before any rewrite (commit 9c353c1).
The pipeline parallelism post was not written by the author alone, and the distributed matmul, faster attention
and both kernel fusion posts were written by Claude. They are rewritten in this style too, and the four posts
written by Claude read as if the author had written them from the beginning, not as edits of their earlier text.

- Openings start from the history or the context of the problem ("In 2017, the Transformer model was presented
  in the paper ...", "Sorting is probably the most common algorithm in computer science."), then say what the
  post covers.
- A new concept is named in **bold** where it is defined. *Italics* stress an idea or a word used in a special
  sense (*exposed*, *log-trick*, *compromise*, *how noisy*).
- Formulas are introduced with "is given by", "is computed as", "is defined as" or "as:", and followed by
  "where ..." to define their terms. Equations of a paper are cited inline: "(Eq. 4 in the paper)", "(MML book,
  section 8.3.1)".
- Pictures are announced before they appear ("The following picture illustrates ...", "An illustration of the
  principle is displayed below:"), and called pictures, diagrams or illustrations. The text then walks through
  what they show.
- The usual connectors: "In practice," to give the practical meaning of a formula or a claim, "Note that",
  "i.e." and "e.g." (with dots), "Therefore", "Thus", "However", "So", "Also", "Moreover", "As an example",
  "Similarly to", "Looking at the diagram above", "For the sake of comparison", "As a side note", "In brief",
  "To sum up".
- Parentheses hold short clarifications, abbreviations and alternative names ("(DDP)", "(or, equivalently, of
  the connectivity matrix of a graph)"). Semicolons join closely related clauses. Colons introduce a formula, a
  list or a picture.
- Lists enumerate methods, stages or properties, each item a name and a sentence or two (the sorting algorithms,
  the ZeRO stages). Explanations are prose.
- The questions that the author asks at turning points stay ("How to transpose it?", "Where's the caveat?"), and
  so do the key sentences of a paper, quoted as blockquotes ("The authors made an important observation:").
- Sentences have 20 to 25 words on average, long explanatory ones next to short ones. Paragraphs have two to
  four sentences.
- Implementation sections are hands-on: the code is introduced step by step ("First, ...", "Then, ...", "And
  that is all: ..."), and each snippet is preceded by what it does.
- Plain, precise words. The author's mild informal words stay ("trickier", "handy"), slang goes ("lingo"), and
  contractions are written out ("does not", "it is").

### From the author's first person to the third person

The words that carry the person change, and the rest of the author's sentence stays, unless a clearer or simpler
word helps:
- "Bayesian methods allow us to model the relationship ..." becomes "Bayesian methods model the relationship
  ...".
- "we compute the point of minimum loss as:" becomes "the point of minimum loss is computed as:".
- "In this post, we will look at the mathematical background behind diffusion models, and implement ..."
  becomes "This post looks at the mathematical background behind diffusion models, and implements ...".
- "We soon realized that the Transformer had much more to offer" becomes "It soon became clear that the
  Transformer had much more to offer".
- "Looking at the diagram above, we can see that each processor holds a copy of the model" becomes "The diagram
  above shows that each processor holds a copy of the model".
- "Now you can write your training loop normally, and torch will do all the communication" becomes "The training
  loop is then written as usual, and PyTorch does all the communication".
- "Imagine we have a model that is too large ..." becomes "Imagine a model that is too large ...".

### What the rewrite may change

- Clearer and simpler synonyms may replace the author's words. Redundant sentences are removed, and missing
  sentences and paragraphs relevant to the topic may be added. The flow of the author's ideas stays: no
  reordered ideas, and no merged or split paragraphs and sections without need.
- Phrases that help a student follow and understand the flow of logic are allowed: "This is where X comes in",
  "the key idea is", "crucial", a closing sentence that sums up the paragraph, triplets, em-dashes, "the figure
  shows", sentences that start with "With", "Since" or "Each", "Notice how", "Think of X as", "Note:" callouts.
- No rhetorical question answered right away ("Why? Because ...").
- Additions: a clause or a sentence that explains why a step is needed, a missing sentence, a paragraph relevant
  to the topic, a Wikipedia link, a one-sentence comparison of two methods, a reference to the paper that
  introduced a method. A comparison table goes at the end of a post only when the post compares three or more
  methods.

### Sentences and words

- Simple words that a non-native English speaker knows, plus the exact technical term ("redundant copies",
  "memory footprint", "latency-bound"). No fancy synonyms, no buzzwords.
- Few filler words ("really", "basically", "simply", "just", "actually", "of course", "in order to"). The
  author's "very" and "Note that" stay where they carry meaning.
- Say each thing once. No repeated definitions, no paragraph that restates the previous one, no summary that
  only repeats the post.
- A post is about as long as the author's original, plus the additions above and minus the redundant sentences.
  The author's explanations, examples, derivations, numbers and results stay, unless they are wrong or
  redundant.
- When a post presents several methods, compare them where they appear, or in a table at the end of the post.
  A simple comparison is enough ("A needs less memory than B"). Say when to use each method, and the idea
  behind it.

### References

- The author's pattern: relevant historical references at the beginning of a section, when needed; then the
  explanation of the topic; then, at its end, a paragraph such as "To find more about the topic ..." or "More
  advanced use cases can be found in ..." with further references. When each topic of a post comes from a
  different paper, each topic cites its paper.
- Add the references a post needs, and only the important and relevant ones. Keep the existing references.
  Verify each new reference before adding it.
- Place each reference where it makes sense: a method links to the paper that introduced it, when needed. Always
  name the paper that introduced a recent method.
- The introduction cites only a background paper that the introduction needs.
- Describe other work in a brief note, not at length.

### Technical correctness

- Check every formula, derivation, index, dimension, complexity, unit, number and factual claim, and fix the
  errors in the text. One symbol per quantity within a post, defined at its first use.
- Check that the text, the code and the pictures agree.
- Link a technical term to its English Wikipedia article at its first use in a post, when a reader may not
  know it. Link only articles that exist, and each term once.
- Commands and links stay unchanged unless they are wrong. A text task does not change code: bugs in the code
  go to the code issues of `TODO.md`, with a proposed fix.
- A text task never adds, removes, changes or reorders pictures, their markup or their captions. Problems in
  them go to the diagram issues of `TODO.md`.
- Math is written in `$$...$$`, inline and on its own lines, as in the existing posts. In inline math, write
  `\vert` instead of `|`: kramdown reads a paragraph line that holds a `|` as a table row.

### The author's papers

- The two kernel fusion posts, for tensor parallelism (the FFN) and for expert parallelism (MoE), summarize the
  author's own papers. For confidentiality, they hold no information that is specific to the hardware of the
  experiments, in the text, the captions or the pictures:
  - the only hardware fact is the number of GPUs ("8 GPUs in one node"). The devices are called GPUs, devices or
    ranks, with no vendor, device model, interconnect or switch name, and no number that points to a device model
    (an SM count, a bandwidth, a memory size);
  - libraries take generic names: CCL (collective communication library) for NCCL, HCCL and the like, and shmem
    for NVSHMEM and the like. Vendor programming models, instructions and APIs (CUDA, PTX, TMA, multimem,
    atomicAdd...) are described by what they do;
  - generic GPU terms stay (SMs, thread blocks, shared memory, HBM);
  - other work cited in these posts is described without hardware names either ("multicast and reduction inside
    the switch", "about 15% of the SMs").
- These two posts name no authors and have no acknowledgements, and nothing in them traces the work back to the
  company where it was developed.
- Only these two posts have these limits: the other posts, the drafts and the publications page name hardware,
  vendors and libraries as usual.
- Exception, at the user's request (2026-10-22): the expert parallelism post names the INC switches and deployments
  that it cites as other work (SHArP, SHARPv2, NVLink SHARP, Megatron-Core, vLLM, DeepSeek-V3 and DeepSeek-V4).
  The hardware and libraries of the post's own experiments stay unnamed.
- The expert parallelism post shows each communication model once, in one picture each followed by its table of
  operations, steps and relays (from the paper's appendix), in this order: AllGather-ReduceScatter; All-to-All,
  which sends each rank only the tokens it needs but needs a count exchange first; then in-network computing (INC)
  and the INC form of AllGather-ReduceScatter, which removes two steps (pack and global reduce); the INC
  All-to-All, as All-to-All-Unique and All-to-All-Reduce; and the partial collectives (One-to-Many-Unique and
  Many-to-One-Reduce), which lower the memory that INC All-to-All needs in the switch. No INC picture appears before
  INC is introduced. The two timelines of the collective and partial collective combine stand side by side.

### Titles

- A post title, with the category that the home page puts before it ("Machine Learning: "), fits on one line of
  the home page on a desktop screen. The title column is 815 px wide at 16 px: keep the line under 750 px in the
  site's font, about 90 characters, so that it also fits with wider system fonts.
- A text task does not change titles: it proposes new ones in the title issues of `TODO.md`. The user approved
  the titles of seven posts on 2026-10-18; they are applied.

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

This section is only for the current rewrite: it is removed from this file at the end of its last iteration.

- The rewrite runs in three iterations. Each iteration is a pass over all posts, drafts and the publications
  page, on top of the previous one, and ends with a pull request that the user reviews and merges before the
  next iteration starts. The first iteration makes small changes; each later one builds on the approved one.
- Iteration 1 starts from the post on master and returns to the author's wording. For each paragraph, the
  author's original text (commit 9c353c1) is the reference for the voice: where master changed only the style,
  the rewrite goes back to the original words; where master corrected a fact, a formula or a link, or added
  content that the user asked for, the rewrite keeps it, in the author's voice. Content of the original that
  master dropped comes back when it is correct and useful.
- Rewrite one post at a time: read the original and the master version in full, check the math and the
  pictures, then rewrite the post paragraph by paragraph.
- Keep the line endings of each file (many posts use CRLF).
- Build the site with `bundle exec jekyll build --drafts` and check that every formula and picture renders.
- At the end of each iteration, show the user a review page with each changed paragraph before (left) and after
  (right), and the formula fixes, together with this file. Then stop: the next iteration starts when the user
  asks for it.

---

## Pictures

These rules apply to every picture in _posts, _drafts and publications.md: drawing, restyling, exporting
to SVG, sizing it in the Markdown and writing its caption. Read this section before any such task, and add
each new picture rule the user gives to it, in the same change.

The two kernel fusion paper repositories (brunomaga/moe_paper_huawei and brunomaga/ffn_tp_paper_huawei)
are the style reference only. Never change them. Their imgs/*.svg figures are the example of the palette to
follow (pastel draw.io fills, black outlines and arrows, dark accent text). The line weights and corners follow
sections 6 and 7, and the font section 4.

### 1. Content

- A picture is correct, simple and factual, with a standard layout, like a figure in a college course, a
  conference paper or a textbook: direct to understand, one idea per picture, parts aligned on a grid, flow
  from left to right or from top to bottom, no decoration.
- Every label, dimension, number, arrow and step agrees with the original paper or another reliable source
  (the paper itself, university lecture notes, textbooks, Wikipedia), and with the post's text, code and
  formulas: the same names and symbols for the same quantities (for example "rank" and p, when the text says
  so). Check the source before drawing or changing a picture, and name the sources checked in the PR.
- A picture that comes from an external source (a paper, a book, a website) keeps the source's text, labels
  and content: only its style changes. Its content changes only after a check of the source shows an error;
  when the source cannot be reached, the content stays as it is.
- The logic of a picture makes sense: each step follows from the previous one, arrows connect the right
  parts in the right direction, and shapes, data flow and results agree with the algorithm and the formulas.
  When they do not, the picture is fixed with facts from the sources.
- Worked examples (data, steps, results) are computed by running the algorithm, not by hand, and the picture
  shows the computed values.
- The algorithm pictures of the distributed sort and the load balancing (slicing) posts describe the
  algorithm, not an implementation: their steps say send, receive, broadcast, scatter, gather, reduce, share
  or transpose, never the name of an MPI call. The MPI calls of an implementation go in the caption.
- Shapes that stand for sizes are drawn to scale: in the product of an M x N matrix and an N x O matrix, the
  three rectangles are M x N, N x O and M x O, with their splits (the Megatron-LM pictures).
- Third person, as the post text: no "I", "we", "our", "ours", "us", "you" or "your" in a picture. A paper's
  "Ours" becomes the method's name. Quoted inputs (prompts, chat turns, example sentences) keep their words.
- The pictures of the two kernel fusion posts follow "The author's papers" in the writing rules: no hardware
  names (CCL, shmem, GPUs or ranks), and nothing that traces the work back to the company where it was
  developed.

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
  so the posts need copies). A copy that does not follow the sizing in section 3 or the font in section 4
  is redrawn in this repo, in a tab of assets/inference/inference.drawio named after the post, from the
  paper repository's draw.io or TikZ source. The paper repositories stay unchanged. Two pictures are the
  exception: ep_stages.svg and ep_pipeline.svg of the expert parallelism post stay the paper's own figures,
  copied unchanged and shown at 100% width.
- A picture or an animation shown unchanged from Wikimedia Commons (the Newton's method and k-means GIFs)
  links to its original file on upload.wikimedia.org, with no copy in assets.
- A picture used by a post and by the publications page is stored once, in the post's asset folder, and
  both pages link to that file. The same holds for a picture used by two posts (for example
  matrix-transpose/all_to_all.svg in the matrix transpose and mixture-of-experts posts, and gptlite/gptlite.svg in
  the two GPT-lite posts).
- Charts drawn from data (plots, benchmarks, histograms) keep their data and the script or spreadsheet that
  draws them next to the SVG, and follow the same palette, font and sizes as the diagrams. A chart whose data
  or script is not in the repository is restyled in its SVG: its font, text sizes, colors and line weights
  change, its data and marks stay.
- Paper figures are redrawn in this style too, and their captions credit the paper (section 9).

### 3. Page width and scale

- The post content width is X = 880 px (minima's $content-width 940 px minus 2 x 30 px padding;
  assets/main.scss).
- Every picture is drawn for a post, also one that only the publications page shows: at page scale, where
  1 draw.io px is 1 px on the page when the picture is shown at its natural size, and no wider than 880 px.
  The publications page puts its pictures in a 766 px column, so there they appear at 87% of their size in
  a post, which is fine.
- The Markdown of every page, the publications page too, shows a picture of width W px at W/880*100
  percent, rounded to one decimal:
  ```
  {: .picture}
  <img width="P%" height="P%" src="{{ site.assets }}/<folder>/<name>.svg"/>
  ```
  with P = W / 880 * 100 (W is the width of the exported SVG). So text, lines and arrowheads look the same
  size in every picture of the posts. A picture with a title above it takes P on its span (section 9).
- Pictures shown side by side in one row each keep their own P. When the P values of a row add up to more
  than 98 (the spaces between pictures take the rest), the pictures wrap onto the next line. The two models at
  the top of the GPT-lite post are 92 px apart (twelve `&nbsp;` between their img tags), at the user's request.
- A picture that does not fit 880 px at the text sizes below is redrawn: first narrow the empty space
  between its parts, then move parts (wrap a row into two rows, stack panels, break long labels over two
  lines). Never shrink the text to make it fit.
- A picture whose arrows or labels are crowded grows, taller or wider up to 880 px, before any text or arrow
  gets smaller (the transformer picture is taller, the DualPipe schedule wider). Its elements keep their size as
  it grows: only the spaces between them widen, which gives the arrows and their labels more room (tora, at the
  user's request).

### 4. Font

- Text, labels and tensor dimensions: Source Sans 3 (fontFamily=Source Sans 3). Formulas: New Computer
  Modern Sans Math (fontFamily=NewComputerModernSansMath), the sans-serif math font of the Computer Modern
  family (no font is named "Latin Modern Sans Math"; Latin Modern Sans has no math symbols). Code
  identifiers may use Courier New. Indices use Unicode sub/superscripts or <sub>/<sup>.
- A formula is a variable with its indices, an expression or an equation (x, W_q, Δt = 25, softmax(QKᵀ)). A
  label that is only a formula has the math font in its style; a label that mixes words and formulas has it
  on the formulas only, as <span style="font-family: NewComputerModernSansMath;">. Words, numbers, units,
  tensor dimensions (S x H·h), tick labels and names with an index (TRM1, FFN2) keep the text font.
  Exception, at the user's request: the dimensions of sdpa and of the pictures of the pruning and quantization
  post are formulas in the math font, with the indices of the post's text (S<sub>q</sub> × h,
  𝑛<sub>heads</sub>⋅𝑑<sub>head</sub> × 𝐸). The math font has no bold or italic face: the browser slants and
  emboldens it. Formulas take the math font in every picture, also in one redrawn from a paper, whether or
  not the paper sets them in a math font. A product in a math-font formula is the dot operator ⋅ (U+22C5)
  with no spaces around it: the middle dot · with spaces leaves gaps of about 10 px in this font.
- In the pictures of the pruning and quantization post, at the user's request, a formula looks as LaTeX sets
  it: its variables are the italic letters of the math font (Unicode mathematical italic: 𝑊 for W, ℎ for h, 𝛼
  for α), and function names, words and indices of several letters stay upright (max, round, err, vocab,
  𝑑<sub>head</sub>). Inside brackets, a comma is followed by a thin space (U+2009), and an italic 𝑊 or 𝑈
  before a bracket or a superscript by a hair space (U+200A), its italic correction, so that the slanted letter
  does not touch them. The math font does not place combining accents, so a hat (𝑊̂) is a circumflex ˆ
  (U+02C6) that CSS sets over the letter: `<span style="position: relative; left: -0.566em; top: -0.244em;
  margin-right: -0.5em;">ˆ</span>` after 𝑊, with the offsets of the MATH table of the font.
- A quotient that is hard to read on one line (the update of gptq, the 𝑠<sub>𝑗</sub> of smoothquant) is a built-up
  fraction, as TeX sets it in display style, at the user's request; short ones stay inline (round(𝑤 / 𝑠),
  max|𝑋<sub>𝑗</sub> / 𝑠<sub>𝑗</sub>|). The label holds the numerator and the denominator, stacked in an inline-block
  around an empty block 0.06 em tall, with their baselines about 9 px above and under the main baseline, and more
  where a subscript or superscript would reach the rule. The fraction rule is a 0.78 px black line of the picture
  (the stroke of the minus sign) on the math axis, 0.25 em above the main baseline, as a browser rounds a CSS
  border or background to whole pixels.
- Text sizes, in page px (draw.io fontSize at page scale):
  ```
  title          16           the picture's title, and labels drawn larger than the regular text
  subtitle       14  bold     panel titles, headers, group names
  regular        14  normal   box labels, legends: the most common text of a picture
  arrow          13  normal   the text of an arrow, 1 px under the regular text
  small          12  normal   dimensions, axis ticks, annotations, subscripts in labels (11 in arrow text)
  tiny           10  normal   only for dense labels inside small cells (pipeline micro-batch numbers,
                              matrix weights), where 12 does not fit
  ```
  Arrow text is written on, above, below or beside one arrow, or a line that carries a flow, and names what
  it carries or does (wq, sum-reduce, the packets of the communication pictures, the weights of a network).
  The label inside a block arrow is arrow text too. The names at the start or the end of an arrow (Inputs,
  Output Probabilities), axis titles, and the labels of boxes, operators, brackets and legends are not arrow
  text. An arrow label that is a dimension stays small.
  Exception, at the user's request: the operators of an equation of matrices drawn to scale are sized to the
  matrices they join, larger than 16 (the "=" of mml_svd; softmax, the parentheses and "=" of
  context_parallelism).
  A picture redrawn from an older one keeps the order of its text sizes: its most common size becomes 14,
  sizes at least 1.1 times larger become 16, sizes under 0.9 times smaller become 12.
  Italic is allowed for variables and for emphasis, bold only for titles, subtitles and names the post's
  text refers to: the title of a frame or panel (its topmost label), N×, Encoder and Decoder, and the tensor
  name of a label "name: shape" or "name(a, b)" (only the name: **Q**: H x S x h). Bold changes the weight of a
  label, never its words, also in pictures from external sources.
- Charts use the same fonts and sizes: tick labels 12, axis labels and legends 14, panel titles 14 bold. A
  panel title that is a sentence goes into the caption; panel letters and short panel names stay. A legend
  with many entries in a small chart may use 10 (llama3_scaling_laws).
- Numbers: a true minus sign (−), a space between a number and its unit (25 μs), upright units, and commas
  between thousands (48,001).
- Text color: black #000000. Secondary text (faded, notes): grey #666666. Accent text, for
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
  Unused elements (the masked cells of an attention mask, padding tokens and cells, free slots of a cache,
  pruned weights, missing values, NULL cells) are white, with a solid grey #666666 outline (section 7), never
  grey: their text (NULL, 0) stays. The pad tokens of longvilla stay grey, as in the paper's figure, whose
  white cells are empty positions. Mask tokens are not unused elements, as the model reads them: the [M] tokens
  of dflash have a black outline, and the mask tokens of masked_autoencoders_cropped keep their grey #b3b3b3
  fill, at the user's request. The accent colors may also fill small marks (dots of a scatter plot, nodes of
  a graph, heat map cells, filled tokens) where pastel fills are too faint.
- Fills are flat: no gradients, shadows or glow, and the faces of a cube share one fill.
- A colored outline or frame always has a meaning that the text or the caption gives (a highlight, a group,
  the KV cache). No decorative accent frames.
- A picture redrawn from a paper figure takes, for each color of the figure, the closest color of the
  palette: a pastel fill for a box, an accent only for a small mark, as a large accent fill clashes with the
  pastel fills. When the closest fill already means something else in the picture, the box takes another
  fill (the dark green f and g blocks of the Megatron-LM figure are purple #e1d5e7, as green marks its Y and
  Z boxes).
- Dark accent text (from the kernel paper figures): blue #1e3a5f, green #456e36, orange #885e10,
  purple #6b4a80, red #b85450.
- Pipeline schedules and other pictures that need more distinct colors (all pipeline pictures share it):
  ```
  forward              #6c8ebf   second chunk / reverse direction  #385681
  backward             #82b366                                     #4f743a
  backward for inputs  #b85450                                     #74322f
  backward for weights #d79b00                                     #856000
  idle                 white, grey dashed outline (section 7)
  ```
  Every pipeline picture uses 4 GPUs, except DualPipe (8) and the 8 workers of pipeline_8_stages,
  pipeline_4_stages and pipeline_2_stages, and shows no previous or next iteration. Section 8 gives the layout.
- The same thing has the same color within a post (one color per process, GPU, head, expert...), and the
  same meaning keeps its color across posts where possible (blue forward, green backward).
- New colors are only added when the palette above runs out, and must look like part of it (pastel fill,
  readable black text on top).
- Real photos keep their own colors (section 10).

### 6. Lines, outlines and arrows

- Every geometric element with a fill (rectangles, rounded boxes, circles, cubes, cylinders, filled table
  cells, block arrows, grey background panels and frames) has a black outline: strokeColor=#000000, no stroke
  opacity. The cells of a block and the unused elements (section 7) are the exceptions: their outlines are grey,
  except where two fills meet (see the blocks below). Six
  weights, the same at page scale in every picture:
  ```
  thin       0.4 px   the cells of a matrix, an array, a table or a block of cubes, the lines that divide a
                      block into cells, the faces of a 3D tensor, pipeline passes, the black boundary between
                      two fills inside a block
  regular    1.0 px   operations, layers, modules, data boxes (tensors, vectors, tokens), neurons and graph
                      nodes, operators, single cubes, block arrows, square brackets, the blocks of a time line,
                      the grey outlines of idle areas, of unused single elements and of dropped neurons
  block      1.2 px   the outline of each block of cells (matrix, array, table, stack of objects), the
                      silhouette of each 3D tensor or block of cubes and the three edges at its front corner
  line       1.4 px   arrows and lines, axes of time lines, markers, brackets and number lines drawn as lines
  frame      1.8 px   frames of regions (devices, processes, groups, sections), colored or dashed ones too,
                      the axes and the data lines of charts, the frames of plots of a 2D domain or of an
                      image (k-means, voronoi, the slicing layouts, svd_example, coordinate_descent), lines
                      that divide a drawing into regions (the dashed lines between the panels of a picture)
  emphasis   3.6 px   red #b85450 outlines and frames, solid or dashed, that emphasise a part of a drawing
                      (the modules of the GPT-lite pictures, the MoE layers of moe_gshard, the red frames of
                      the first row and the first column in all_to_all), and black frames that emphasise a part
                      (DualPipe's overlapped forward and backward passes, the selected experts in
                      moe_switch_transformers)
  ```
  An emphasis outline or frame takes the corners of what it marks: square around data, matrices, tensors,
  tables and the regions of an XY map, 5 px around an operation and 10 px around a frame.
- A block (a matrix, an array, a table, a stack of objects, a block of cubes) has a black outline at the block
  weight around it and thin grey #666666 lines between its cells: the cells, the lines that divide the block
  and the faces of a 3D tensor are grey. Boxes glued into one matrix (the column and row splits of the
  Megatron-LM pictures) share one black block outline, with a thin grey line between two boxes of the same fill
  and a thin black boundary line (below) between two boxes of different fills. A single cell (a token, a K or V
  block, a single cube) has no block outline and keeps its black outline. A matrix or an array is one block,
  never a set of single cells, at the user's request: an attention mask is one S x S matrix with the fill of each
  cell and the labels of its rows and columns (omnigen), and a block of a paged KV cache is an array of a fixed
  number of slots, one 1 x 4 array (paged_attention, prefix_caching).
- Inside a block, a thin black line (0.4 px) marks the boundary between two regions of different fills, at the
  user's request: a colored region and a white one (unused, masked or empty cells, or the white background of
  the block), two colors, or a color and a picture. It is drawn over the grey lines of the cells, which stay
  between cells of the same fill, and it follows the fills, not what the cells stand for. The faces of a block of
  cubes, of a 3D tensor and of a stack of layers follow the same rule (transformer_attention_lbdl,
  group_normalization, ulysses_sequence_parallelism, graph_transformer_networks), while their silhouette and the
  three edges at their front corner keep the block weight. Where a black or colored line at least as thick
  already marks the boundary (a block outline, an emphasis frame), no line is added. Heat maps, whose cells each
  have their own shade, keep their grey lines (mixture_of_depths, transformer_attention).
- A time line (tp_timeline, tp_block_specialization, continuous_chunked_batching,
  local_variable_timestep_six_cell_example, the time row of prefill_decode) is made of intervals of time, not of
  the cells of a block: its blocks have square corners and regular black outlines, dashed ones too, and blocks
  stacked in one interval have no outline around them, as there is no block or region inside a time line. A
  square bracket over a time line points to its text. Pipeline schedules follow section 8: their passes keep thin
  black outlines. In continuous_chunked_batching, a forward pass lasts as long as its longest prefill: the
  requests that share it show their work first and padding until its end, as static batching pads the prompts of
  a group, and only empty slots are idle. Each decode is one block with a thin grey line between its tokens (one
  per step), and so is each prefill of the chunked panel, with a thin grey line between its chunks. The label of
  each prefill and decode (P1, D1) sits above it on a square bracket 3 px inside each end of its block, and each
  chunk has its own bracket and label, with the index of the chunk (P1<sub>1</sub>, P1<sub>2</sub>), at the user's
  request.
- Every table, array and stack of objects, in every picture, has black text, thin grey lines between its
  cells of the same fill, thin black lines between cells of different fills and a black block outline (the
  tables of the matrix transpose post keep their blue cells). Only a cell that the picture emphasizes has
  another outline (the emphasis weight above).
- Exceptions that keep their own look:
  - a box highlighted by a colored outline: the colored outline replaces its black one, with no second
    frame around the box (the red #b85450 outlines of the modules in the GPT-lite pictures);
  - other colored frames that highlight a group of parts: 1.8 px, as other frames (the orange KV cache
    frames, the red dashed windows of csa and hca). Red arrows and cross marks are not emphasis lines;
  - dashed outlines of elements that are not frames: 0.4 px on filled elements (1.0 px on idle boxes and
    dropped neurons, section 7), 1.4 px on lines without fill (a dashed frame is 1.8 px, as other frames; the
    dash keeps its pattern);
  - white masks that hide parts of a drawing, faded elements and red cross marks;
  - the arrows of the matrix transpose post, which are grey #666666;
  - the memory bars of deepspeed_stages and their legend swatches: thin black outlines, as their strips are
    4 px tall;
  - the dashed outlines of the tensor shards in efficiently_scaling_transformer_inference_multiquery: 1.2 px,
    at the user's request, so that they show next to the filled shards;
  - the grey #666666 1.4 px line on the left of the steps of a loop in an algorithm picture (odd_even_sort),
    at the user's request;
  - the arrays of the algorithm pictures of the distributed sort and slicing posts (sample_sort,
    odd_even_sort, balance, split, sbs): a 1.8 px black outline, at the user's request;
  - the lines between the samples (rows) of the token tables of variable_batch_lr and
    variable_batch_lr_pipeline: 1.2 px black, at the user's request, so that each sample stands apart;
  - the grey backgrounds of kohonen_1, kohonen_2 and kohonen_3: no outline, at the user's request;
  - the outlines of the Generator and Discriminator trapezoids of gan: 1.8 px, at the user's request, as in the
    original figure;
  - the curves of transformer_attention_lbdl, whose weights follow the attention scores, as in the original
    figure: the strongest one is 2.6 px, blue #6c8ebf on the left and red #b85450 on the right, thicker than the
    other lines and thinner than the emphasis weight, at the user's request;
  - the colored arrows and the red cross of roofline: 3.6 px, at the user's request, with the head of an arrow at
    the emphasis weight;
  - the cut lines of sbs: blue #6c8ebf, 3.6 px, in the middle of a 16 px gap between two arrays, with one red
    emphasis frame per slice in steps 2.1 and 3.1, at the user's request.
- Thin arrows: black #000000, 1.4 px, endArrow=classic, endFill=1, endSize=5: a head 6.4 px long and wide,
  endSize plus the line weight, with its notch at 3/4 of its length. Dashed arrows use the same head. An
  arrow at the emphasis weight has endSize=2.8, so that its head has the size of the others. Grey #666666 only
  for faded content. Lines without a head that join two parts of the flow (two boxes, or a box and an
  arrow): 1.4 px. The axis of a time line: 1.4 px, with the same head.
- Charts: the axes, the frame and the data lines of a plot are 1.8 px, in every chart, at the user's request: the
  data lines (the values, the fits, the roofs of a roofline chart, the line samples of the legend) are as thick as
  the axes. The axis line of a bar chart and the line in front of the bars of each category are 1.8 px too, and an
  axis drawn with a head takes the head of the thin arrows. The data (lines, markers, bars) is drawn behind the axes
  and the frame, and the data points are drawn over the lines of a plot (the observations over the sample lines of
  linear_bayesian_posterior). Exception, at the user's request: the x axes drawn as arrows in the charts of the
  variable timestep post (cvode_pulse_currents_table and the l5_neuron charts) are drawn behind the data that
  crosses them, as in the original figures, and the step markers of cvode_pulse_currents_table are red dots
  2.0 px wide. Bars and filled areas take a pastel fill of the palette other than grey,
  as #f5f5f5 hardly shows on white (the green #d5e8d4 of kde, at the user's request). The level lines of a contour
  plot (coordinate_descent, linear_bayesian_posterior) keep 1.0 px: they are the background of the path or the
  samples drawn over them, and their dense rings would merge at 1.8 px. Guide lines (a threshold, a projection on an
  axis) are grey #666666, dashed, 1.0 px (a lighter #999999 in roofline, at the user's request). The legend of a
  chart is one column outside the plot, on its right, and the chart grows wider to make room for it.
- The links of a dense network (each neuron to each neuron of the next layer) are thin arrows with heads, as
  the others; the heads that converge on one node may touch.
- Inhibition (one part blocks or breaks down another, as in a signalling pathway): 1.4 px black line with a
  flat bar head, endArrow=baseDash;endSize=10.
- Lines thinner than 0.4 px at page scale (grid lines drawn over a photo or a matrix) keep their weight.
- Arrow layout:
  - an arrow has at least 6 px of line before its head, and a bent arrow at least 6 px before its first turn;
    after its last turn, or after the line it leaves from (a fan-out), an arrow shows at least 10 px of line
    before its head. To get it, the turn or the fan-out point moves first; space inserted across a picture never
    cuts a block, an array or a shape, and keeps a row of equal parts evenly spaced;
  - arrowheads are at least 2 px apart: arrows into one box end at different points of its edge, or one
    joins the line of the other before the head;
  - two arrows share a segment only when they leave the same point (fan-out) or end at the same point
    (fan-in);
  - a single arrow into a shape points to its centre: a horizontal arrow runs at the height of the centre of the
    shape (the arrows into the latent volume and the latent sequence of sora_vae, at the user's request);
  - text never sits on, touches or crosses a line or an arrow, at the user's request: the label of an arrow
    sits above a horizontal segment or right of a vertical one, with no background
    (labelBackgroundColor=none);
  - text keeps at least 2.5 px from every line and shape that does not hold it. In a chart, a label wider than
    the spacing of the grid lines (optimum shifts in mutransfer) and a value label at the end of a bar may
    cross a grey grid line; they keep 2.5 px from the axes, the arrows and the other lines.
- No end-of-sequence cells or tokens in schedules: the last decode cell is the last token.

### 7. Shapes, blocks and sizes

- The same shape means the same thing in every picture:
  ```
  data (tensors, matrices, vectors, tokens, buffers)    rectangle with square corners
  operation, layer, module, kernel                       rectangle with 5 px round corners
  device, process, group (GPU, rank, node, worker),      frame with 10 px round corners and a light fill, its
  section                                                name at the top left
  3D tensor                                              isometric cube (isoCube2); CNN feature map: cube
  storage (disk, database, cache)                        cylinder
  encoder, decoder                                       trapezoid
  decision                                               rhombus
  neuron, graph node                                     circle
  operator (+, x, concatenation)                         circle 20 px wide with the symbol inside
  span of several parts                                  square bracket, 1.0 px, its tip toward its text
  idle area (schedules, time lines)                      white box with square corners, a grey dashed outline
                                                         and no text
  unused element (masked, padding, free, pruned or       white, with a solid grey outline
  missing cell, token or slot)
  dropped neuron (dropout)                               white circle, grey dashed outline, grey cross
  transformation of a whole stage (a model into another) block arrow; thin arrows everywhere else
  ```
- A square bracket is draw.io's curlyBracket with rounded=0: a straight line with square ends and a tip at
  its middle that points to its text. Its arms cover the parts it spans.
- Operations and frames have round corners: 5 px for an operation (rounded=1;absoluteArcSize=1;arcSize=10),
  10 px for a frame (arcSize=20). Every frame of a region has them, also a dashed or colored one
  (ring_attention_qkv, moe_gshard, jamba). Data, matrix and array cells, plot bars and the cells of a schedule
  or a time line have square corners (rounded=0). Capsules (fully round ends) keep their shape. The chat bubbles
  of llama2_gatt have 15 px round corners (arcSize=30), capsules for one line, as in the original figure, at the
  user's request. A frame that
  marks a region of an XY map (a 2D domain, a plot, an image) has square corners, as the region is an area
  of the map (str_steps).
- Block arrows: body 10 px wide, head 20 px wide and 10 px long (flexArrow: width=10;endWidth=10;endSize=3.07;
  singleArrow: 20 px tall, arrowWidth=0.5). A block arrow with a label inside keeps a body taller than its
  label, with the same proportions. A block arrow thinner than 6 px is a thin arrow (section 6).
- Dashed lines use draw.io's default dash pattern; dotted lines use dashPattern=1 2.
- Arrows are straight or orthogonal (edgeStyle=orthogonalEdgeStyle). Curves only for pointers to a part of
  the drawing and for loops.
- Idle areas (the idle time of a schedule or a time line): a white box with square corners and a grey
  #666666 dashed outline, 1.0 px (rounded=0;fillColor=#ffffff;strokeColor=#666666;strokeWidth=1;dashed=1),
  with no text inside, as its look says it is idle. A legend may name it Idle.
- Unused elements (a masked cell of an attention mask, a padding token or cell, a free slot of a cache, a
  pruned weight, a missing value, a NULL cell): a white fill and a solid grey #666666 outline, thin (0.4 px)
  for a cell inside a block and regular (1.0 px) for a single token, slot or cube, so that they differ from
  the dashed idle areas (ragged_batching, variable_batch_lr, paged_attention). Their text (NULL, 0) stays
  black. Inside a block, the thin black boundary line of section 6 replaces the grey outline of an unused cell on
  each side where it touches a cell of another fill.
- A dropped neuron (dropout) is a white circle with a grey #666666 dashed outline, 1.0 px, the weight of
  the other neurons, and a grey #666666 cross over it.
- A hatched mark that is not idle time (the new integration steps of local_variable_timestep_six_cell_example)
  is a hachure: black lines 0.6 px wide every 4 px at 45 degrees, drawn as vector paths: fillColor=#000000;
  strokeColor=none;fillStyle=hachure;sketch=1;jiggle=0;hachureAngle=45;hachureGap=4;fillWeight=0.6;
  disableMultiStroke=1;disableMultiStrokeFill=1. draw.io's fillStyle=hatch turns into a bitmap in the PDF
  export.
- The pictures of one series in a post (the attention variants, the pipeline schedules...) use the same
  element sizes: the same cube, cell, row and box sizes in every picture of the series.
- A legend is a row of entries, each a swatch with its label 4 px to its right, and 16 px between entries
  (24 px in the pictures of the prefill, decode, roofline and cache post; section 8 gives the pipeline
  legend). A chart's legend is one column (section 6).
- Sizes for new and redrawn pictures: a one-line box 30 px tall, 20 px between boxes in a row, 40 px between
  rows (room for an arrow and its label), matrix and array cells 20 x 20 px with 12 px numbers (10 px when they
  do not fit).

### 8. Pipeline schedules

All pipeline schedules (the pipeline parallelism post and the publications page) share one layout:
- One row per GPU, labelled "GPU 0" to "GPU 3" (to "GPU 7" for DualPipe; "Worker 1" to "Worker 4" where the
  post's text says workers, as for the PipeDream pictures), 14 px regular, right-aligned 4 px left of the
  first cell.
- Rows 20 px tall and 3 px apart. One time unit is 21.86 px wide (15.44 px for DualPipe, so that its 8 GPUs
  and 16 micro-batches fit 880 px): a forward pass takes 1 unit, a backward pass 2 units, or 1 unit each for
  the backward for inputs (B) and the backward for weights (W). With two chunks per GPU (interleaved 1F1B,
  ZB-V, DualPipeV), each pass of a chunk takes half as long. DualPipe runs one stage per direction on each
  GPU, so its passes take the full time.
- Each row holds the passes of its GPU and an idle box (section 7) in each gap between them, from the start
  of the schedule to its end, with no frame around the row. A pass is one cell with a 0.4 px black outline,
  in the pipeline colors of section 5; a backward pass is one cell of two units, not two cells.
- Every pass cell holds its micro-batch number, also the single micro-batch 1 of pipeline_8_stages,
  pipeline_4_stages and pipeline_2_stages. Micro-batch numbers are 10 px: black on the base colors, white on
  the dark ones.
- The schedule ends with its last pass: no passes of the next iteration and no "..." after the rows.
- No time markers over the rows (no "1F1B ends" line).
- Overlapped forward and backward passes (DualPipe) have a 3.6 px black frame around them.
- A time axis under the rows is 1.4 px, as the axes of section 6.
- The legend is centred under the rows, 12 px below them (or below the time axis): each entry is a 20 x 12 px
  swatch (two 10 x 12 px halves for the two chunks or directions) and its 14 px label 4 px to the right, with
  16 px between entries, on two rows when it is wider than the schedule. The labels are Forward, Backward,
  Backward for inputs, Backward for weights, Overlapped forward & backward, Idle; the Idle swatch is an idle
  box.
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
- Panel letters and step numbers such as "(a)" or "1." may stay inside the picture, with a short label of a
  few words when it helps to read the picture ("1. k initial means"); the rest of the panel's text goes into
  the caption.
- Titles that describe a schedule or a variant (for example "ZB-H1 (same activation memory as 1F1B)") go
  into the caption, not the picture.
- Pictures side by side in one row that the text names one by one (the four slicing layouts) may each have
  a short title above them, centred and bold, outside the picture. Each picture and its title go in a span of
  the picture's width P, with 4% of the width between the spans:
  ```
  {: .picture}
  <span style="display:inline-block; width:P%; vertical-align:bottom;"><b>Title</b><br><img width="100%" src="{{ site.assets }}/<folder>/<name>.svg"></span>
  <span style="display:inline-block; width:4%;"></span>
  ```
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
  It copies each "<name>-figure" group (or the names given) to a draw.io file of its own, renders it to a
  cropped PDF, which turns the text into glyph outlines, converts it to SVG with pdftocairo, removes the white
  page background, crops the SVG to its drawn content (its non-white pixels plus 0.5 px), writes it over the
  SVG of the same name in the folder, and prints the width P to use in the Markdown. Outlined text looks the
  same on every device, whatever fonts it has, as in the kernel paper figures.
- Source Sans 3 (Google Fonts) and New Computer Modern Sans Math (NewCMSansMath-Regular.otf, from the
  newcomputermodern package on CTAN) must be installed where the export runs: without them draw.io draws
  other fonts, with other widths.
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
