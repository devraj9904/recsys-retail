# recsys-retail

A real-time, two-stage e-commerce recommender on the [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) dataset: two-tower retrieval with FAISS, LightGBM LambdaRank ranking for clicks and purchases, Kafka → Redis streaming features, content-based cold start, and a FastAPI serving stack with Prometheus/Grafana.



## Build phases

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Data pipeline: cleaning, sessionization, temporal split, item metadata ([stats](reports/data_stats.md)) | ✅ |
| 1 | Retrieval: two-tower + FAISS vs ALS and popularity baselines ([results](reports/retrieval.md)) | ✅ |
| 2 | Ranking: multi-objective LambdaRank, NDCG@K / Recall@K, diversity reranking | ⏳ |
| 3 | Real-time features: Kafka ingestion, Redis online store, live preference updates | ⏳ |
| 4 | Cold start: content features for new users and items vs CF | ⏳ |
| 5 | Serving: FastAPI, Docker, Redis cache, load tests, Prometheus/Grafana | ⏳ |

## Quickstart

```bash
pip install -e ".[dev,download]"
export PYTHONPATH=src

# Real data (needs KAGGLE_USERNAME / KAGGLE_KEY, or pass --zip path/to/archive.zip)
python -m recsys.data.download
# ...or a small synthetic dataset in the same format, no download needed
python -m recsys.data.synthetic --out data/retailrocket/raw

python -m recsys.data.prepare       # -> data/retailrocket/processed/*.parquet
python -m recsys.data.report        # -> reports/data_stats.md
python -m recsys.train.train_retrieval   # -> reports/retrieval.md, artifacts/retrieval/
pytest                              # runs on synthetic data, no downloads
```

Set `OPENBLAS_NUM_THREADS=1` before training ALS; `implicit` runs much slower with OpenBLAS threading.

## Data pipeline (Phase 0)

`recsys.data.prepare` turns the raw CSVs into model-ready Parquet:

1. **Clean.** Drop duplicate rows, collapse page-reload bursts of the same (visitor, item, event), and remove bot-like visitors (more than 1,000 events overall or 300 in a day).
2. **Sessionize.** A new session starts after 30 minutes of inactivity.
3. **Temporal split.** A global time split: train | tune (7 days) | test (last 14 days). Unlike leave-last-out, the test window contains genuinely new visitors and newly listed items, which is exactly the cold-start population Phase 4 has to serve. `tune` provides the ranker's labels, so the ranker never trains on interactions the retrievers saw.
4. **Item metadata.** The latest snapshot of each item property: category and its root-first path through the category tree, availability, a price-like numeric property (`790`), and hashed categorical property values as description tokens. RetailRocket hashes nearly all property text, so these tokens stand in for product descriptions.
5. **Index.** Contiguous ids for warm users (seen in train) and warm items (at least 2 train events). Everything else is cold and handled by the content path.

Event types carry graded relevance (`view=1`, `addtocart=2`, `transaction=3`), used later as ALS confidence weights and LambdaRank labels.

The loader returns normalised frames (`ts`, `visitor_id`, `item_id`, `event`), so another dataset such as Coveo SIGIR-21 only needs a new loader.

## Retrieval (Phase 1)

All models learn from the train split only. At test time each visitor is represented by their latest pre-test history (train + tune), and the targets are the distinct items they touch in the 14-day test window. Items outside the warm catalogue count as misses.

| model | recall@20 | recall@100 | NDCG@20 | discovery recall@100 | purchase recall@100 |
|---|---|---|---|---|---|
| popularity (all-time) | 0.008 | 0.028 | 0.004 | 0.025 | 0.066 |
| popularity (last 7 days) | 0.012 | 0.041 | 0.006 | 0.040 | 0.086 |
| recently viewed + trending | 0.180 | 0.202 | 0.167 | 0.040 | **0.323** |
| ALS (64 factors) | 0.080 | 0.126 | 0.049 | 0.076 | 0.235 |
| **two-tower + FAISS HNSW** | **0.202** | **0.242** | **0.171** | **0.112** | 0.292 |

14,333 test visitors with history, 148k warm items. *Discovery* drops items the visitor already saw from both the ranking and the targets, which measures whether a model finds new products rather than echoing history.

What the numbers say:

- **Two-tower is the best retriever overall**: 2.5x ALS on recall@20, and 1.5x ALS on discovery recall@100.
- **Re-engagement is a big share of e-commerce traffic.** Simply echoing a visitor's own recent items gets recall@20 of 0.18 and the best purchase recall, because people buy what they viewed. The ranker in Phase 2 will merge both candidate sources rather than pick one.
- **Popularity is weak here** because the catalogue is long-tailed (148k items) and covers only 0.07% of it.
- **HNSW matches exact search** (recall within 0.001) at about 0.19 ms p50 / 0.32 ms p99 per single query, 6.5x faster than flat search in batch.

Model details:

- **Two-tower** (`models/two_tower.py`): the item tower combines item-id, leaf-category, top-category, price-bucket and availability embeddings through an MLP. The user tower attention-pools the visitor's last 30 items (weighted by event type and recency) and adds the most recent item. There is no user-id embedding, so any visitor with at least one event can be encoded without retraining, which Phase 3 uses for live updates. It is trained on 741k next-item examples with in-batch softmax and logQ correction for popularity bias, which takes about 5 minutes on 4 CPU cores.
- **ALS** (`models/als.py`): `implicit` ALS with confidence 1 + 10 x (summed event weights). User factors are re-solved from each visitor's pre-test history, so it sees the same information as the two-tower.
- **FAISS** (`index/ann.py`): exact `IndexFlatIP` or `IndexHNSWFlat` over L2-normalised item vectors, with a numpy fallback.

## Layout

```
src/recsys/
  config.py        paths, event weights, PrepareConfig
  data/            download, loaders, prepare, synthetic, report
  retrieval/       history streams, training examples, eval set
  models/          popularity, ALS, two-tower
  index/           FAISS item index
  eval/            Recall@K, NDCG@K, hit rate, coverage
  train/           train_retrieval (trains and compares all retrievers)
tests/             unit + end-to-end tests on synthetic data
reports/           generated metric tables
```
