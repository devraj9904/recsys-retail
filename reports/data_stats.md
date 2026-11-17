## Dataset summary

| metric | value |
|---|---|
| events (after cleaning) | 2,688,604 |
| visitors | 1,407,548 |
| sessions | 1,757,251 |
| items with events | 233,597 |
| items with category / price | 184,242 / 184,242 |
| view events | 2,606,847 (97.0%) |
| addtocart events | 62,743 (2.3%) |
| transaction events | 19,014 (0.7%) |
| visitors with a single event | 71.2% |
| warm users / warm items (train) | 1,214,890 / 148,095 |
| test events from new visitors | 87.8% |
| test events on cold items | 11.6% |

| split | start | end | events | visitors | items | sessions |
|---|---|---|---|---|---|---|
| train | 2015-05-03 | 2015-08-27 | 2,325,239 | 1,214,890 | 218,673 | 1,510,304 |
| tune | 2015-08-28 | 2015-09-03 | 119,435 | 72,467 | 46,879 | 81,755 |
| test | 2015-09-04 | 2015-09-18 | 243,930 | 143,675 | 72,391 | 165,228 |

### Notes

- 71% of visitors have a single event and 88% of test events come from visitors never seen in train, so session-based and popularity/content signals matter more than per-user CF for most traffic.
- Only 17% of tune events come from warm users; the ranker needs session features, not just user-history features.
- 21% of items with events have no property snapshot (no category, price or tokens); 12% of test events land on items too rare to enter the warm catalogue.
- Purchases are 0.7% of events (19k), so the purchase objective needs graded labels and careful weighting.
