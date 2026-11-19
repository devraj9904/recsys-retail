# Phase 4: cold start, content vs collaborative filtering

## A. Same items, with and without click history

14,971 warm items (10%) were hidden from the content encoder's training. For 2,029 returning visitors whose test clicks include them, each method retrieves over the full catalogue with those items represented by:

| method | recall@20 | recall@100 | ndcg@20 |
|---|---:|---:|---:|
| CF: two-tower item vectors (learned from clicks) | 0.2092 | 0.2551 | 0.1612 |
| content: metadata encoder (cosine objective) | 0.1183 | 0.2417 | 0.0390 |
| content: metadata encoder (in-batch softmax objective) | 0.0086 | 0.0726 | 0.0025 |
| content: category centroid only | 0.0551 | 0.1822 | 0.0174 |

Mean cosine between a held-out item's content vector and its CF vector: encoder (cosine) 0.690, encoder (softmax) 0.475, category centroid 0.682.

## B. Genuinely new items

85,502 items are not in the trained catalogue. Returning visitors: 14,333, of whom 2,195 clicked at least one new item in the test window.

| method | recall@20 (all targets) | recall@20 (new-item targets) | share of slots on new items |
|---|---:|---:|---:|
| warm index only (Phase 1) | 0.2020 | 0.0000 | 0.0000 |
| warm + new-item content vectors in one index | 0.1991 | 0.0302 | 0.2054 |
| warm top 17 + 3 new-item slots | 0.2000 | 0.0216 | 0.1500 |

Ranking only among new items, for the 2,195 visitors who clicked one:

| method | recall@10 | recall@20 | ndcg@20 |
|---|---:|---:|---:|
| random new items | 0.0000 | 0.0000 | 0.0000 |
| content: TF-IDF kNN of the visitor's last 5 items | 0.0545 | 0.0733 | 0.0381 |
| content: new items in the visitor's categories | 0.0140 | 0.0219 | 0.0094 |
| hybrid: user tower x content encoder | 0.0505 | 0.0683 | 0.0321 |

## C. New visitors after one click

20,715 visitors with no history before the test window; predict the distinct items they click afterwards (context items excluded).

| method | recall@10 | recall@20 | ndcg@20 | coverage@20 | recall@20 (first click warm) | recall@20 (first click new item) |
|---|---:|---:|---:|---:|---:|---:|
| CF: popularity (last 7 days) | 0.0068 | 0.0124 | 0.0050 | 0.0001 | 0.0135 | 0.0032 |
| content: popular in clicked categories | 0.1601 | 0.2270 | 0.1141 | 0.0473 | 0.2389 | 0.1260 |
| content: TF-IDF item kNN | 0.1929 | 0.2549 | 0.1382 | 0.4133 | 0.2532 | 0.2697 |
| CF: co-view item kNN | 0.2158 | 0.2660 | 0.1576 | 0.2291 | 0.2929 | 0.0376 |
| CF: ALS fold-in | 0.1199 | 0.1595 | 0.0881 | 0.0157 | 0.1779 | 0.0032 |
| CF: two-tower user tower (warm index) | 0.2559 | 0.3187 | 0.1878 | 0.3245 | 0.3559 | 0.0032 |
| hybrid: two-tower + content encoder (full index) | 0.2583 | 0.3196 | 0.1914 | 0.4198 | 0.3422 | 0.1277 |
| hybrid router: two-tower if a click is warm, else TF-IDF kNN | 0.2788 | 0.3468 | 0.2038 | 0.3950 | 0.3559 | 0.2697 |

10.5% of these visitors' first click is on a new item.

## C. New visitors after three clicks

3,647 visitors with no history before the test window; predict the distinct items they click afterwards (context items excluded).

| method | recall@10 | recall@20 | ndcg@20 | coverage@20 |
|---|---:|---:|---:|---:|
| CF: popularity (last 7 days) | 0.0049 | 0.0094 | 0.0045 | 0.0001 |
| content: popular in clicked categories | 0.1086 | 0.1656 | 0.0865 | 0.0382 |
| content: TF-IDF item kNN | 0.1380 | 0.2007 | 0.1073 | 0.1498 |
| CF: co-view item kNN | 0.1619 | 0.2286 | 0.1281 | 0.1242 |
| CF: ALS fold-in | 0.0927 | 0.1305 | 0.0740 | 0.0126 |
| CF: two-tower user tower (warm index) | 0.1903 | 0.2612 | 0.1527 | 0.1194 |
| hybrid: two-tower + content encoder (full index) | 0.1916 | 0.2589 | 0.1508 | 0.1301 |
| hybrid router: two-tower if a click is warm, else TF-IDF kNN | 0.1964 | 0.2684 | 0.1573 | 0.1237 |

## Notes

- Item metadata is RetailRocket's latest property snapshot. Category and price are
  known when an item is listed, but the snapshot can include later edits, a mild
  optimism for content methods. Property values are hashed: opaque tokens.
- CF models (popularity, co-view, ALS, two-tower) are fitted on events before the
  test window; the content encoder is trained on warm items only.
- Gaps in a list are filled with popular items, so every method shows 20 items.
- Missing targets count as misses (no target is dropped for being cold).
- The router hybrid was added after seeing the first-click breakdown (content wins
  when the first click is a new item, CF wins otherwise), so its gain is measured
  on the same data that suggested it.
- The encoder objective was chosen on held-out items' cosine to their CF vectors,
  which uses no test-window clicks.
