## Ranking results

Trained on 1,007,603 candidate rows from 3,018 tune-week visitors with a positive; evaluated on 14,333 test visitors (1,725,578 candidates). Candidate recall ceiling: 0.254.

| ranking | ndcg@10 | ndcg@20 | recall@20 | purchase_recall@20 | ild@20 | categories@20 |
|---|---|---|---|---|---|---|
| recently viewed + trending | 0.1663 | 0.1674 | 0.1799 | 0.2782 | 0.5530 | 12.7971 |
| two-tower order | 0.1669 | 0.1707 | 0.2020 | 0.2368 | 0.3250 | 3.5619 |
| LambdaRank (engagement) | 0.1797 | 0.1839 | 0.2169 | 0.2980 | 0.4257 | 5.4533 |
| conversion model only | 0.1333 | 0.1349 | 0.1504 | 0.1678 | 0.5685 | 8.9588 |
| LambdaRank + conversion (w=0.2) | 0.1757 | 0.1801 | 0.2164 | 0.2966 | 0.4274 | 5.4926 |
| LambdaRank + conversion (w=0.4) | 0.1631 | 0.1678 | 0.2052 | 0.2852 | 0.4303 | 5.5519 |
| LambdaRank + conversion (w=0.6) | 0.1562 | 0.1606 | 0.1971 | 0.2669 | 0.4319 | 5.5833 |
| + MMR λ=0.1 | 0.1756 | 0.1799 | 0.2160 | 0.2966 | 0.4427 | 6.0864 |
| + MMR λ=0.2 | 0.1752 | 0.1795 | 0.2151 | 0.3018 | 0.4641 | 6.8820 |
| + MMR λ=0.3 | 0.1743 | 0.1786 | 0.2139 | 0.2966 | 0.4870 | 7.7808 |

Top features (LambdaRank gain share):

- `hist_pos`: 48.9%
- `ui_events`: 23.1%
- `tt_rank`: 6.9%
- `ui_max_weight`: 3.5%
- `item_pop_7d`: 3.3%
- `from_hist`: 2.8%
- `tt_score`: 2.1%
- `u_hours_since`: 1.6%
- `leaf_share`: 1.3%
- `item_pop`: 1.2%

### Notes

- The candidate set caps recall@20 at 0.254. LambdaRank reaches 0.217, about 85% of that ceiling.
- Graded LambdaRank already learns purchase intent: purchase recall@20 is 0.298, versus 0.237 for two-tower order. Blending in the separate conversion model trades NDCG for little purchase gain, so the default serving config uses a small weight (w=0.2).
- MMR with λ=0.2 adds about 1.4 extra categories per list (+26%) and +9% intra-list diversity for a 2.4% NDCG@20 cost, and slightly raises purchase recall.
- The ranker is trained on a single tune week (3,018 visitors with a positive). Training on several rolling cutoffs would add labels and is the clearest next improvement.
