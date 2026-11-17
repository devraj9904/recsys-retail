# recsys-retail

A real-time, two-stage e-commerce recommender on the [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) dataset: two-tower retrieval with FAISS, LightGBM LambdaRank ranking for clicks and purchases, Kafka → Redis streaming features, content-based cold start, and a FastAPI serving stack with Prometheus/Grafana.

Structure is inspired by [realtime-recsys](https://github.com/mohamed-abo-taha/realtime-recsys) and rebuilt for implicit e-commerce events (views, add-to-carts, purchases) instead of explicit ratings.

## Build phases

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Data pipeline: cleaning, sessionization, temporal split, item metadata | ✅ |
| 1 | Retrieval: two-tower + FAISS vs ALS and popularity baselines | ⏳ |
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
pytest                              # runs on synthetic data, no downloads
```

## Data pipeline (Phase 0)

`recsys.data.prepare` turns the raw CSVs into model-ready Parquet:

1. **Clean.** Drop duplicate rows, collapse page-reload bursts of the same (visitor, item, event), and remove bot-like visitors (more than 1,000 events overall or 300 in a day).
2. **Sessionize.** A new session starts after 30 minutes of inactivity.
3. **Temporal split.** A global time split: train | tune (7 days) | test (last 14 days). Unlike leave-last-out, the test window contains genuinely new visitors and newly listed items, which is exactly the cold-start population Phase 4 has to serve. `tune` provides the ranker's labels, so the ranker never trains on interactions the retrievers saw.
4. **Item metadata.** The latest snapshot of each item property: category and its root-first path through the category tree, availability, a price-like numeric property (`790`), and hashed categorical property values as description tokens. RetailRocket hashes nearly all property text, so these tokens stand in for product descriptions.
5. **Index.** Contiguous ids for warm users (seen in train) and warm items (at least 2 train events). Everything else is cold and handled by the content path.

Event types carry graded relevance (`view=1`, `addtocart=2`, `transaction=3`), used later as ALS confidence weights and LambdaRank labels.

The loader returns normalised frames (`ts`, `visitor_id`, `item_id`, `event`), so another dataset such as Coveo SIGIR-21 only needs a new loader.

## Layout

```
src/recsys/
  config.py        paths, event weights, PrepareConfig
  data/            download, loaders, prepare, synthetic, report
tests/             unit + end-to-end tests on synthetic data
reports/           generated metric tables
```
