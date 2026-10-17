# Notes for Claude

## Pictures

Before any task that draws, restyles, exports or sizes a picture (SVG, draw.io, PNG/JPG conversion) or writes
a picture caption, read `assets/rules.txt` and follow it. When the user gives a new picture rule, add it to
`assets/rules.txt` in the same change.

In short, every SVG in `_posts`, `_drafts` and `publications.md` follows one style:

- draw.io is the source of every diagram, and the draw.io file is updated with its SVG before a merge;
- drawn at page scale, at most 880 px wide (the post content width), and shown at width/880 % in the Markdown,
  `<img width="P%" height="P%" ...>`, so text and lines look the same size in every picture;
- Times New Roman: 16 bold titles, 14 bold subtitles, 14 regular text, 12 small text (10 only for dense
  numbers in small cells), black text;
- the draw.io pastel palette (plus the pipeline colors in `assets/rules.txt`);
- squares, rounded boxes, circles and every other filled shape have a 0.8 px black outline, dashed ones too;
- thin arrows are 1 px black with a filled block head;
- colored highlight frames and thicker emphasis lines are kept;
- captions are Markdown below the picture (`{: .caption}`), never inside it;
- real photos inside a figure are kept with their own colors.
