# recsys-retail

A real-time, two-stage e-commerce recommender on the [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) dataset: two-tower retrieval with FAISS, LightGBM LambdaRank ranking for clicks and purchases, Kafka → Redis streaming features, content-based cold start, and a FastAPI serving stack with Prometheus/Grafana.



## Build phases

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Data pipeline: cleaning, sessionization, temporal split, item metadata ([stats](reports/data_stats.md)) | ✅ |
| 1 | Retrieval: two-tower + FAISS vs ALS and popularity baselines ([results](reports/retrieval.md)) | ✅ |
| 2 | Ranking: multi-objective LambdaRank, NDCG@K / Recall@K, diversity reranking ([results](reports/ranking.md)) | ✅ |
| 3 | Real-time features: Kafka ingestion, Redis online store, live preference updates ([results](reports/streaming.md)) | ✅ |
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
python -m recsys.train.train_ranker      # -> reports/ranking.md, artifacts/ranking/
docker compose up -d                     # Kafka :9092 + Redis :6379
python -m recsys.train.run_streaming     # -> reports/streaming.md, artifacts/streaming/
pytest                              # runs on synthetic data, no downloads
```

**macOS (Apple silicon):** install PyTorch, FAISS and LightGBM from conda-forge so they share one OpenMP runtime. The pip wheels each bundle their own, and mixing them crashes with a segfault.

```bash
conda create -n recsys -c conda-forge python=3.12 pytorch faiss-cpu lightgbm -y
conda activate recsys
pip install -e ".[dev]" --no-deps
pip install implicit scikit-learn pyarrow pandas scipy redis confluent-kafka "fakeredis[lua]" pytest ruff
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

## Ranking (Phase 2)

Two-stage: retrieval proposes about 120 candidates per visitor, and a LightGBM ranker orders them.

- **Candidates** (`ranking/candidates.py`): two-tower + FAISS top-100, plus the visitor's own recent items, plus the 20 items trending in the week before the cutoff. Every candidate carries its two-tower score.
- **Features** (`ranking/features.py`), 24 in total, all computed strictly before the cutoff:
  - retrieval scores and ranks
  - user-item history: count, strongest event, hours since
  - item popularity (all-time and 7-day), cart rate and buy rate
  - user activity
  - category affinity (share of the visitor's events in the item's leaf and top category)
  - price bucket and gap to the visitor's usual price
  - availability
- **Training**: rows are built at the start of the tune week from train-only history, and labelled with each visitor's strongest tune-week event (view=1, cart=2, purchase=3).
- **Models** (`ranking/ranker.py`): LambdaRank on the graded labels (label gains 0/1/3/7), plus a conversion classifier (add-to-cart or purchase, with purchases weighted 3x). They are blended per visitor on standardised scores: `(1 - w) * z(engagement) + w * z(logit p_conversion)`.
- **Diversity** (`ranking/diversity.py`): MMR reranking over the top 50, using two-tower item embeddings as the similarity space.

Evaluated on 14,333 test visitors over the same 14-day window as Phase 1:

| ranking | NDCG@10 | NDCG@20 | recall@20 | purchase recall@20 | categories in top 20 |
|---|---|---|---|---|---|
| recently viewed + trending | 0.166 | 0.167 | 0.180 | 0.278 | 12.8 |
| two-tower order (Phase 1) | 0.167 | 0.171 | 0.202 | 0.237 | 3.6 |
| **LambdaRank** | **0.180** | **0.184** | **0.217** | 0.298 | 5.5 |
| LambdaRank + conversion (w=0.2) | 0.176 | 0.180 | 0.216 | 0.297 | 5.5 |
| **+ MMR λ=0.2** (serving default) | 0.175 | 0.180 | 0.215 | **0.302** | 6.9 |

- LambdaRank improves NDCG@20 by 8% and purchase recall by 26% over the two-tower order, and reaches 85% of the candidate-set ceiling (recall@20 0.254).
- History features dominate the learned ranking: `hist_pos` and `ui_events` carry about 72% of the gain. The two-tower score decides among items the visitor has not seen.
- MMR λ=0.2 adds 1.4 categories per list (+26%) for a 2.4% NDCG cost. Full trade-off curve in [reports/ranking.md](reports/ranking.md).

## Real-time features (Phase 3)

```
events ─▶ Kafka topic `events` ─▶ consumer ─▶ Redis online store ─▶ user tower ─▶ u:{v}:vec ─▶ FAISS
         (keyed by visitor)       (batch poll,  (recent items, counts,     (re-encode touched
                                   one pipeline) session, category affinity, visitors per batch)
                                                 trending)
```

- **Producer** (`streaming/producer.py`) replays any split into Kafka in event-time order, keyed by visitor so each visitor's events stay ordered within a partition. It can cap the rate or compress the original gaps.
- **Online store** (`features/online_store.py`): per visitor, the last 90 raw events, event counts, the current session (a new one after 30 minutes idle) and a 7-day half-life category affinity; per item, event counts; and a trending sorted set. Session and affinity updates depend on the previous state, so they run as one Lua script per event, which keeps them atomic with several consumers.
- **Consumer** (`streaming/consumer.py`) polls up to 1,000 messages, writes them in one Redis pipeline, re-encodes the user vector of every visitor in the batch with the Phase 1 user tower, then commits offsets. The next recommendation request already reflects the click.
- **Warm start**: before the stream starts, each visitor's last 30 batch events are loaded into Redis, the hand-off a nightly job would do in production.

Replaying the RetailRocket test window through local Kafka and Redis (single broker, single consumer, CPU):

| replay rate | p50 freshness | p95 | p99 |
|---|---|---|---|
| 500 events/s | 46 ms | 74 ms | 80 ms |
| 2,000 events/s | 80 ms | 121 ms | 132 ms |

Freshness is measured per event, from the producer's send to the moment the visitor's new vector is readable in Redis. One consumer sustains about 5,700 events/s at full load. RetailRocket's real traffic averages about 0.25 events/s, so these rates are far above what the dataset needs.

**Does freshness matter?** For 23,591 test events by 11,376 returning visitors, the item each visitor interacted with next was predicted from two states: the vector frozen at the start of the test window, as a nightly batch job would serve it, and the vector updated after every earlier event.

| user state | Hit@20 | MRR@20 | Hit@20, item not in last 30 steps |
|---|---|---|---|
| popularity | 0.008 | 0.002 | 0.007 |
| batch vector (stale) | 0.116 | 0.048 | 0.053 |
| **streaming vector (live)** | **0.216** | **0.069** | **0.125** |

Live updates nearly double next-event Hit@20 (+86%), and more than double it for items the visitor had not touched recently (+136%). Details and caveats are in [reports/streaming.md](reports/streaming.md).

## Layout

```
src/recsys/
  config.py        paths, event weights, PrepareConfig
  data/            download, loaders, prepare, synthetic, report
  retrieval/       history streams, training examples, eval set
  models/          popularity, ALS, two-tower
  index/           FAISS item index
  eval/            Recall@K, NDCG@K, hit rate, coverage
  ranking/         candidates, features, LambdaRank + conversion models, MMR diversity
  features/        Redis online feature store
  streaming/       event schema, Kafka producer/consumer, live recommender, stale-vs-live eval
  train/           train_retrieval, train_ranker, run_streaming
tests/             unit + end-to-end tests on synthetic data
reports/           generated metric tables
```
