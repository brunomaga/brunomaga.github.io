# Notes for Claude

## Writing rules

Read this section before any task that writes or edits text in `_posts`, `_drafts`, `publications.md` or
`index.md`, and follow it. When the user gives a new writing rule, add it here in the same change.

### Audience and voice

- The site reads like a college course: a professor teaching undergraduates, who speaks like an engineer. The
  readers are HPC and ML researchers and engineers. They come for clear information, and nobody likes long posts
  or boring sentences.
- Purely technical. No business or marketing language, no anecdotes, no opinions, no hype, no motivation
  speeches, no calls to action. No "in this post we will" opening and no "in conclusion" ending.
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

### Technical correctness

- Check every formula, derivation, index, dimension, complexity, unit and number, and fix the errors. One
  symbol per quantity within a post, defined at its first use.
- Check that the text, the code and the pictures agree.
- Link a technical term to its English Wikipedia article at its first use in a post, when a reader may not
  know it. Link only articles that exist, and each term once.
- Code, commands, links and picture markup stay unchanged unless they are wrong. A text task does not redraw
  pictures: it lists the pictures that are wrong or disagree with the text, for the user.
- Math is written in `$$...$$`, inline and on its own lines, as in the existing posts.

### Process

- Rewrite one post at a time: read it in full, check its math and pictures, then rewrite it.
- Build the site with `bundle exec jekyll build --drafts` and check that every formula and picture renders.
