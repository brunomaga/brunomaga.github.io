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
- No filler: "very", "really", "basically", "simply", "just", "actually", "of course", "note that", "it is worth
  noting", "interestingly", "in order to".
- Say each thing once. No repeated definitions, no paragraph that restates the previous one, no summary that
  only repeats the post.
- Short posts. Cut every sentence that adds no information. Keep every formula, algorithm step, number and
  result that does.
- When a post presents several methods, compare them where they appear: the cost of each in runtime
  (computation, communication, number of steps) and in memory, and when each one is the better choice.

### Technical correctness

- Check every formula, derivation, index, dimension, complexity, unit, number and factual claim, and fix the
  errors in the text. One symbol per quantity within a post, defined at its first use.
- Add at most four references to a post, the most relevant papers, at the start or the end, and only when they
  help the reader. Keep the existing references. Verify each new reference before adding it.
- Check that the text, the code and the pictures agree.
- Link a technical term to its English Wikipedia article at its first use in a post, when a reader may not
  know it. Link only articles that exist, and each term once.
- Code, commands and links stay unchanged unless they are wrong.
- A text task never adds, removes, changes or reorders pictures, their markup or their captions. It lists the
  pictures that are wrong or disagree with the text, with a proposed fix, in
  `/mnt/project-files/blog-prose/diagram-issues.md`.
- Math is written in `$$...$$`, inline and on its own lines, as in the existing posts. In inline math, write
  `\vert` instead of `|`: kramdown reads a paragraph line that holds a `|` as a table row.

### Process

- Rewrite one post at a time: read it in full, check its math and pictures, then rewrite it.
- Keep the line endings of each file (many posts use CRLF).
- Build the site with `bundle exec jekyll build --drafts` and check that every formula and picture renders.
