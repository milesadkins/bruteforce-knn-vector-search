# Brute-Force KNN Vector Search in Databricks SQL

Tag financial news headlines to the most relevant stock tickers using embedding similarity — no vector search index required.

## How It Works

This demo uses a **brute-force K-nearest neighbors (KNN)** approach to match query rows against a base table using vector embeddings. Unlike approximate nearest neighbor (ANN) indexes like those in Databricks Vector Search, brute-force KNN computes the exact similarity between every query-base pair via a cross join and returns the true top-K results.

### The Approach

```
┌─────────────────────┐       ┌─────────────────────┐
│   Headlines Table    │       │    Tickers Table     │
│   (query rows)       │       │    (base rows)       │
│                      │       │                      │
│  "iPhone 17 sales    │       │  AAPL: "Consumer     │
│   shatter records"   │  ───► │   electronics..."    │
│                      │ CROSS │  NVDA: "Graphics     │
│  "Oil prices surge   │ JOIN  │   processing..."     │
│   past $100..."      │       │  XOM:  "Oil and      │
│                      │       │   natural gas..."    │
└─────────────────────┘       └─────────────────────┘
                    │
                    ▼
        ┌───────────────────┐
        │ vector_cosine_     │
        │ similarity(q, b)   │
        │ per every pair     │
        └───────────────────┘
                    │
                    ▼
        ┌───────────────────┐
        │ GROUP BY query     │
        │ max_by(*, score, K)│
        │ → top-K per query  │
        └───────────────────┘
```

1. **Embed both tables** — Each row's text column is embedded using `ai_query('databricks-gte-large-en', text)`, a Foundation Model endpoint available in any Databricks workspace.
2. **Cross join** — Every query row is paired with every base row.
3. **Score** — `vector_cosine_similarity()` computes the similarity between each pair's embedding vectors.
4. **Top-K selection** — `max_by(struct(*), score, K)` returns the K highest-scoring base rows per query row in a single aggregation — no window functions needed.
5. **Flatten** — `inline()` unnests the array of structs into individual result rows.

### When to Use Brute-Force KNN

| Scenario | Recommendation |
|---|---|
| Base table < 100K rows | Brute-force KNN works well |
| Need exact results, not approximate | Brute-force is the only option |
| One-off analysis or ad-hoc matching | Brute-force — no index to build |
| Base table > 1M rows | Use [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) indexes |
| Real-time serving at low latency | Use Vector Search with a serving endpoint |

Brute-force KNN is ideal when you need **exact results on moderate-sized tables without the overhead of creating and managing a vector search index**.

## Demo: Headline-to-Ticker Tagging

The included notebook demonstrates this by tagging 15 financial news headlines to their most relevant stock tickers (from a set of 30 major US equities).

### Example Results

| Headline | #1 Match | #2 Match | #3 Match |
|---|---|---|---|
| iPhone 17 sales shatter records... | AAPL | DIS | AMZN |
| Oil prices surge past $100... | XOM | CVX | BA |
| NVIDIA unveils next-gen AI chip... | NVDA | AMD | MSFT |
| New weight-loss drug shows 25% more effectiveness... | LLY | PFE | MRK |

## Getting Started

### Requirements

- Databricks workspace with serverless compute or a DBR 18.1+ cluster
- Access to the `databricks-gte-large-en` Foundation Model endpoint (enabled by default)
- Unity Catalog with permissions to create a schema

### Run the Demo

1. Import `notebooks/validate_knn_search.py` into your Databricks workspace
2. Update the catalog/schema names if needed (defaults to `madkins2_catalog.tagging`)
3. Attach to serverless compute and **Run All**

The notebook will:
- Create the schema and both tables
- Generate embeddings via `ai_query()`
- Execute the brute-force KNN search
- Display top-3 ticker matches per headline with similarity scores

## Adapting for Your Data

To use this pattern with your own tables, set these variables in the notebook:

```python
query_table_name = "your_catalog.your_schema.queries"
query_embedding_column_name = "embedding"
base_table_name = "your_catalog.your_schema.base_items"
base_embedding_column_name = "embedding"
similarity_distance_function = "vector_cosine_similarity"  # or vector_l2_distance, vector_inner_product
num_results = 10
```

Both tables must have a pre-computed embedding column (ARRAY&lt;FLOAT&gt;). You can generate embeddings with:

```sql
SELECT *, ai_query('databricks-gte-large-en', text_column) AS embedding
FROM your_table
```
