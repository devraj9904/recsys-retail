## Retrieval results

14,333 test visitors with pre-test history · 148,095 warm items · two-tower trained on 740,815 next-item examples

| model | recall@20 | recall@100 | ndcg@20 | discovery_recall@20 | discovery_recall@100 | purchase_recall@100 | coverage@100 |
|---|---|---|---|---|---|---|---|
| popularity (all-time) | 0.0082 | 0.0280 | 0.0042 | 0.0065 | 0.0246 | 0.0661 | 0.0007 |
| popularity (last 7d) | 0.0124 | 0.0408 | 0.0058 | 0.0109 | 0.0404 | 0.0856 | 0.0007 |
| recently viewed + trending | 0.1799 | 0.2023 | 0.1674 | 0.0109 | 0.0404 | 0.3225 | 0.1762 |
| ALS | 0.0795 | 0.1264 | 0.0492 | 0.0376 | 0.0761 | 0.2350 | 0.0579 |
| two-tower + FAISS flat | 0.2023 | 0.2415 | 0.1708 | 0.0614 | 0.1113 | 0.2915 | 0.8848 |
| two-tower + FAISS hnsw | 0.2020 | 0.2416 | 0.1707 | 0.0613 | 0.1118 | 0.2915 | 0.8851 |

| timing | value |
|---|---|
| ALS train+eval (s) | 72.6 |
| two-tower train (s) | 274.8 |
| FAISS flat search, ms/query (batch) | 0.2439 |
| FAISS hnsw search, ms/query (batch) | 0.0373 |
| FAISS hnsw single query p50/p99 (ms) | [0.189, 0.318] |
