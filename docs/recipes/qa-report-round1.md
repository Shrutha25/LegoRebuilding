# Recipe QA report, round 1 (before tuning)

First Gemini vision run (gemini-3.1-flash-lite, 2 calls, 20 random realistic inventories per recipe, grids of 4). The recipes were tuned
afterwards (see `qa-report.md` for the current state): houses got taller walls and eaves, cars a long and low shape, towers a narrower
choice of bases, rockets a stripe and a minimum part count.

| recipe | builds from 20 inventories | yes | partly | no |
|---|---|---|---|---|
| car | 20 | 12 | 4 | 4 |
| house | 20 | 4 | 12 | 4 |
| tower | 20 | 19 | 0 | 1 |
| bridge | 20 | 20 | 0 | 0 |
| rocket | 20 | 17 | 0 | 3 |

What the model said about the weak ones:

- car "no" (4): short, tall cars (length 6, two cabin layers): "too blocky and vertical", "looks more like an abstract structure";
  "partly" (4): "lacks distinct wheel placement".
- house "no" (4): squat houses with 2-brick walls and a flat roof: "just a flat pile of bricks", "too flat to look like a house";
  "partly" (12): "a simple boxy structure".
- tower "no" (1): a 2 x 4 column with 2-brick bands on a wide base: "too wide at the base".
- rocket "no" (3): the smallest builds (10-18 parts): "does not look like a rocket".
