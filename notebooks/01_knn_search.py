# Databricks notebook source
# MAGIC %md
# MAGIC # Brute-Force KNN Scale Test: 40M × 10K
# MAGIC
# MAGIC Stress test the brute-force KNN vector search at production scale.
# MAGIC
# MAGIC **Prereq:** Run `00_setup_data` first to create the tables.
# MAGIC
# MAGIC **Requirements:** Serverless compute or DBR 18.1+ with Photon

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import time

catalog = "madkins2_catalog"
schema = "tagging"

query_table_name = f"{catalog}.{schema}.transactions_40m"
base_table_name = f"{catalog}.{schema}.tickers_10k"
query_embedding_column_name = "embedding"
base_embedding_column_name = "embedding"
similarity_distance_function = "vector_cosine_similarity"
num_results = 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify tables exist

# COMMAND ----------

q_count = spark.sql(f"SELECT count(*) FROM {query_table_name}").collect()[0][0]
b_count = spark.sql(f"SELECT count(*) FROM {base_table_name}").collect()[0][0]
print(f"Query table: {q_count:,} rows")
print(f"Base table:  {b_count:,} rows")
print(f"Cross join:  {q_count * b_count:,} pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build column aliases

# COMMAND ----------

query_cols = spark.table(query_table_name).columns
base_cols = spark.table(base_table_name).columns

query_alias = ", ".join(
    f"{c} AS query_{c}"
    for c in query_cols if c != query_embedding_column_name
)
base_alias = ", ".join(
    f"{c} AS base_{c}"
    for c in base_cols if c != base_embedding_column_name
)
query_ref = ", ".join(
    f"query_{c}"
    for c in query_cols if c != query_embedding_column_name
)
base_ref = ", ".join(
    f"base_{c}"
    for c in base_cols if c != base_embedding_column_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run brute-force KNN (full 40M × 10K)
# MAGIC
# MAGIC **400 billion pairs** with cosine similarity on 1024-dim vectors.
# MAGIC Expect this to either take a very long time or fail — that's the point.

# COMMAND ----------

start = time.time()

df = spark.sql(
    f"""
    WITH query_table AS (
        SELECT
            {query_alias},
            {query_embedding_column_name} AS query_embedding,
            monotonically_increasing_id() AS __query_row_id
        FROM {query_table_name}
        DISTRIBUTE BY __query_row_id
    ),
    base_table AS (
        SELECT
            {base_alias},
            {base_embedding_column_name} AS base_embedding
        FROM {base_table_name}
    ),
    with_score AS (
        SELECT
            {query_ref},
            {base_ref},
            {similarity_distance_function}(query_embedding, base_embedding) AS search_score,
            __query_row_id
        FROM query_table
        JOIN base_table
    )
    SELECT inline(
        max_by(
            struct(* EXCEPT (__query_row_id)),
            search_score,
            {num_results}
        )
    )
    FROM with_score
    GROUP BY __query_row_id
    """
)

result_count = df.count()
elapsed = time.time() - start

print(f"KNN search completed: {result_count:,} result rows in {elapsed:.1f}s")
print(f"Pairs computed: {q_count:,} × {b_count:,} = {q_count * b_count:,}")

# COMMAND ----------

display(df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smaller scale checkpoints
# MAGIC
# MAGIC If the full run fails, use these to find the practical limit.
# MAGIC Uncomment one at a time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1K × 10K = 10M pairs

# COMMAND ----------

# start = time.time()
# df_1k = spark.sql(f"""
#     WITH q AS (
#         SELECT *, embedding AS query_embedding, monotonically_increasing_id() AS __rid
#         FROM {query_table_name} TABLESAMPLE (1000 ROWS) DISTRIBUTE BY __rid
#     ),
#     b AS (SELECT *, embedding AS base_embedding FROM {base_table_name}),
#     scored AS (
#         SELECT q.transaction_id AS query_txn, b.ticker AS base_ticker,
#                {similarity_distance_function}(query_embedding, base_embedding) AS score, __rid
#         FROM q JOIN b
#     )
#     SELECT inline(max_by(struct(* EXCEPT (__rid)), score, {num_results}))
#     FROM scored GROUP BY __rid
# """)
# print(f"1K × 10K: {df_1k.count():,} rows in {time.time() - start:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10K × 10K = 100M pairs

# COMMAND ----------

# start = time.time()
# df_10k = spark.sql(f"""
#     WITH q AS (
#         SELECT *, embedding AS query_embedding, monotonically_increasing_id() AS __rid
#         FROM {query_table_name} TABLESAMPLE (10000 ROWS) DISTRIBUTE BY __rid
#     ),
#     b AS (SELECT *, embedding AS base_embedding FROM {base_table_name}),
#     scored AS (
#         SELECT q.transaction_id AS query_txn, b.ticker AS base_ticker,
#                {similarity_distance_function}(query_embedding, base_embedding) AS score, __rid
#         FROM q JOIN b
#     )
#     SELECT inline(max_by(struct(* EXCEPT (__rid)), score, {num_results}))
#     FROM scored GROUP BY __rid
# """)
# print(f"10K × 10K: {df_10k.count():,} rows in {time.time() - start:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 100K × 10K = 1B pairs

# COMMAND ----------

# start = time.time()
# df_100k = spark.sql(f"""
#     WITH q AS (
#         SELECT *, embedding AS query_embedding, monotonically_increasing_id() AS __rid
#         FROM {query_table_name} TABLESAMPLE (100000 ROWS) DISTRIBUTE BY __rid
#     ),
#     b AS (SELECT *, embedding AS base_embedding FROM {base_table_name}),
#     scored AS (
#         SELECT q.transaction_id AS query_txn, b.ticker AS base_ticker,
#                {similarity_distance_function}(query_embedding, base_embedding) AS score, __rid
#         FROM q JOIN b
#     )
#     SELECT inline(max_by(struct(* EXCEPT (__rid)), score, {num_results}))
#     FROM scored GROUP BY __rid
# """)
# print(f"100K × 10K: {df_100k.count():,} rows in {time.time() - start:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1M × 10K = 10B pairs

# COMMAND ----------

# start = time.time()
# df_1m = spark.sql(f"""
#     WITH q AS (
#         SELECT *, embedding AS query_embedding, monotonically_increasing_id() AS __rid
#         FROM {query_table_name} TABLESAMPLE (1000000 ROWS) DISTRIBUTE BY __rid
#     ),
#     b AS (SELECT *, embedding AS base_embedding FROM {base_table_name}),
#     scored AS (
#         SELECT q.transaction_id AS query_txn, b.ticker AS base_ticker,
#                {similarity_distance_function}(query_embedding, base_embedding) AS score, __rid
#         FROM q JOIN b
#     )
#     SELECT inline(max_by(struct(* EXCEPT (__rid)), score, {num_results}))
#     FROM scored GROUP BY __rid
# """)
# print(f"1M × 10K: {df_1m.count():,} rows in {time.time() - start:.1f}s")
