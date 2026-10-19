# Fit to pile

32 one-piece designs (bowl and mug excluded) against four piles, any colour. Cell = bricks the pile covers / bricks missing. Pile sizes (bricks of the 8 BrickGPT sizes): real_pile 11, small 12, medium 40, large 120.

| design | bricks needed | real_pile | small | medium | large |
|---|---:|---:|---:|---:|---:|
| train-s44 | 8 | 7 / 1 | 8 / 0 | 8 / 0 | 8 / 0 |
| guitar-s43 | 18 | 8 / 10 | 9 / 9 | 14 / 4 | 18 / 0 |
| guitar-s42 | 19 | 8 / 11 | 9 / 10 | 19 / 0 | 19 / 0 |
| guitar-s44 | 30 | 9 / 21 | 10 / 20 | 25 / 5 | 30 / 0 |
| bench-s42 | 35 | 9 / 26 | 9 / 26 | 26 / 9 | 35 / 0 |
| bus-s44 | 40 | 9 / 31 | 10 / 30 | 29 / 11 | 40 / 0 |
| bench-s43 | 53 | 10 / 43 | 11 / 42 | 31 / 22 | 47 / 6 |
| train-s43 | 56 | 8 / 48 | 8 / 48 | 26 / 30 | 44 / 12 |
| bus-s43 | 64 | 9 / 55 | 9 / 55 | 31 / 33 | 51 / 13 |
| tower-s44 | 68 | 6 / 62 | 6 / 62 | 13 / 55 | 33 / 35 |
| car-s44 | 69 | 11 / 58 | 11 / 58 | 33 / 36 | 52 / 17 |
| train-s42 | 69 | 11 / 58 | 12 / 57 | 36 / 33 | 58 / 11 |
| sofa-s44 | 78 | 10 / 68 | 10 / 68 | 33 / 45 | 63 / 15 |
| chair-s43 | 79 | 11 / 68 | 11 / 68 | 34 / 45 | 68 / 11 |
| chair-s44 | 79 | 6 / 73 | 7 / 72 | 23 / 56 | 48 / 31 |
| car-s43 | 81 | 11 / 70 | 12 / 69 | 35 / 46 | 60 / 21 |
| chair-s42 | 83 | 9 / 74 | 9 / 74 | 32 / 51 | 57 / 26 |
| table-s44 | 83 | 7 / 76 | 7 / 76 | 27 / 56 | 66 / 17 |
| piano-s42 | 85 | 11 / 74 | 12 / 73 | 37 / 48 | 61 / 24 |
| bed-s42 | 89 | 10 / 79 | 10 / 79 | 33 / 56 | 56 / 33 |
| table-s43 | 92 | 10 / 82 | 11 / 81 | 30 / 62 | 69 / 23 |
| tower-s43 | 99 | 3 / 96 | 3 / 96 | 13 / 86 | 35 / 64 |
| sofa-s42 | 100 | 11 / 89 | 12 / 88 | 36 / 64 | 69 / 31 |
| bookshelf-s42 | 110 | 9 / 101 | 10 / 100 | 28 / 82 | 91 / 19 |
| piano-s44 | 115 | 11 / 104 | 12 / 103 | 35 / 80 | 83 / 32 |
| birdhouse-s44 | 133 | 11 / 122 | 12 / 121 | 40 / 93 | 83 / 50 |
| tower-s42 | 138 | 11 / 127 | 12 / 126 | 33 / 105 | 64 / 74 |
| bookshelf-s44 | 147 | 7 / 140 | 8 / 139 | 24 / 123 | 64 / 83 |
| sofa-s43 | 157 | 11 / 146 | 12 / 145 | 39 / 118 | 95 / 62 |
| piano-s43 | 162 | 11 / 151 | 12 / 150 | 36 / 126 | 88 / 74 |
| table-s42 | 191 | 11 / 180 | 11 / 180 | 34 / 157 | 79 / 112 |
| bookshelf-s43 | 294 | 11 / 283 | 12 / 282 | 40 / 254 | 100 / 194 |

## How many designs can each pile build?

| pile | buildable (0 missing) | nearly (1-5 missing) | within 10 |
|---|---:|---:|---:|
| real_pile | 0 | 1 | 2 |
| small | 1 | 0 | 3 |
| medium | 2 | 2 | 5 |
| large | 6 | 0 | 7 |

Design sizes (bricks): min 8, median 83, max 294; 5 under 40, 8 at 60 or fewer, 25 at 120 or fewer.
