# Databricks notebook source
# MAGIC %md
# MAGIC # Brute-Force KNN Scale Test: 40M × 10K
# MAGIC
# MAGIC Stress test the brute-force KNN vector search at production scale:
# MAGIC - **Query table:** 40M synthetic ecommerce purchase transactions
# MAGIC - **Base table:** 10K synthetic stock tickers
# MAGIC - **Cross join:** 400 billion pairs with cosine similarity on 1024-dim vectors
# MAGIC
# MAGIC Embeddings are synthetic (random float arrays) to isolate cross-join performance
# MAGIC from embedding generation cost.
# MAGIC
# MAGIC **Requirements:** Serverless compute or DBR 18.1+ with Photon

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

catalog = "madkins2_catalog"
schema = "tagging"

num_transactions = 40_000_000
num_tickers = 10_000
embedding_dim = 1024

query_table_name = f"{catalog}.{schema}.transactions_40m"
base_table_name = f"{catalog}.{schema}.tickers_10k"
query_embedding_column_name = "embedding"
base_embedding_column_name = "embedding"
similarity_distance_function = "vector_cosine_similarity"
num_results = 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Create schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Generate 10K synthetic tickers (base table)
# MAGIC
# MAGIC Each ticker gets a random 1024-dim embedding vector.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
import time

start = time.time()

tickers_df = (
    spark.range(1, num_tickers + 1)
    .withColumnRenamed("id", "ticker_id")
    .withColumn("ticker", F.concat(F.lit("TKR"), F.lpad(F.col("ticker_id").cast("string"), 5, "0")))
    .withColumn("company_name", F.concat(F.lit("Company "), F.col("ticker_id").cast("string"), F.lit(" Inc.")))
    .withColumn("sector", F.element_at(
        F.array(*[F.lit(s) for s in [
            "Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary",
            "Consumer Staples", "Industrials", "Materials", "Utilities", "Real Estate",
            "Communication Services", "Aerospace & Defense"
        ]]),
        (F.col("ticker_id") % 12 + 1).cast("int")
    ))
    .withColumn("description", F.concat(
        F.col("company_name"), F.lit(" operates in the "), F.col("sector"),
        F.lit(" sector providing products and services to global markets.")
    ))
    .withColumn("embedding", F.expr(f"transform(sequence(1, {embedding_dim}), x -> cast(rand() as float))"))
)

tickers_df.write.mode("overwrite").saveAsTable(base_table_name)

elapsed = time.time() - start
print(f"Created {num_tickers:,} tickers in {elapsed:.1f}s")

# COMMAND ----------

spark.sql(f"SELECT count(*) AS ticker_count, size(embedding) AS embedding_dim FROM {base_table_name}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Generate 40M synthetic ecommerce transactions (query table)
# MAGIC
# MAGIC Each transaction has a product description and a random 1024-dim embedding.
# MAGIC This is the large table — generation itself will take a few minutes.

# COMMAND ----------

categories = [
    "Electronics", "Clothing", "Home & Garden", "Sports & Outdoors", "Books",
    "Toys & Games", "Automotive", "Health & Beauty", "Grocery", "Pet Supplies",
    "Office Supplies", "Jewelry", "Music & Instruments", "Software", "Industrial"
]

products_per_cat = [
    "wireless headphones", "laptop stand", "USB-C hub", "mechanical keyboard", "webcam",
    "running shoes", "winter jacket", "yoga pants", "hiking backpack", "sunglasses",
    "air purifier", "coffee maker", "desk lamp", "throw blanket", "wall shelf",
    "basketball", "camping tent", "fishing rod", "dumbbells", "bicycle helmet",
    "sci-fi novel", "cookbook", "travel guide", "self-help book", "graphic novel",
    "board game", "action figure", "puzzle set", "building blocks", "remote control car",
    "floor mats", "phone mount", "dash cam", "tire inflator", "car vacuum",
    "vitamin supplements", "face moisturizer", "electric toothbrush", "hair dryer", "first aid kit",
    "organic pasta", "protein bars", "olive oil", "ground coffee", "sparkling water",
    "dog treats", "cat litter", "fish tank filter", "bird feeder", "pet bed",
    "printer paper", "whiteboard markers", "desk organizer", "label maker", "ergonomic chair",
    "diamond ring", "gold necklace", "silver bracelet", "pearl earrings", "watch band",
    "electric guitar", "drum sticks", "piano keyboard", "violin strings", "microphone",
    "antivirus software", "project management tool", "VPN subscription", "cloud storage", "photo editor",
    "power drill", "safety goggles", "welding gloves", "tool belt", "work boots"
]

start = time.time()

transactions_df = (
    spark.range(1, num_transactions + 1)
    .withColumnRenamed("id", "transaction_id")
    .withColumn("customer_id", (F.col("transaction_id") % 2_000_000 + 1).cast("int"))
    .withColumn("product_idx", (F.rand() * len(products_per_cat)).cast("int"))
    .withColumn("product_name", F.element_at(
        F.array(*[F.lit(p) for p in products_per_cat]),
        F.col("product_idx") + 1
    ))
    .withColumn("category", F.element_at(
        F.array(*[F.lit(c) for c in categories]),
        (F.col("product_idx") / 5).cast("int") + 1
    ))
    .withColumn("price", F.round(F.rand() * 500 + 5, 2))
    .withColumn("quantity", (F.rand() * 5 + 1).cast("int"))
    .withColumn("purchase_date", F.date_add(F.lit("2025-01-01"), (F.rand() * 365).cast("int")))
    .withColumn("description", F.concat(
        F.lit("Purchase of "), F.col("product_name"),
        F.lit(" in "), F.col("category"),
        F.lit(" category for $"), F.col("price").cast("string"),
        F.lit(" x"), F.col("quantity").cast("string")
    ))
    .withColumn("embedding", F.expr(f"transform(sequence(1, {embedding_dim}), x -> cast(rand() as float))"))
    .drop("product_idx")
)

transactions_df.write.mode("overwrite").saveAsTable(query_table_name)

elapsed = time.time() - start
print(f"Created {num_transactions:,} transactions in {elapsed:.1f}s")

# COMMAND ----------

spark.sql(f"SELECT count(*) AS transaction_count, size(embedding) AS embedding_dim FROM {query_table_name}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Run brute-force KNN search (400B pairs)
# MAGIC
# MAGIC This is the real test. The cross join produces **40M × 10K = 400 billion** pairs,
# MAGIC each requiring a cosine similarity computation on 1024-dimensional vectors.
# MAGIC
# MAGIC **Expect this to either take a very long time or fail.** That's the point —
# MAGIC this establishes the practical boundary of brute-force KNN.

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
print(f"Pairs computed: {num_transactions:,} × {num_tickers:,} = {num_transactions * num_tickers:,}")

# COMMAND ----------

display(df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Smaller scale checkpoints
# MAGIC
# MAGIC If the full 40M × 10K fails or is too slow, run these smaller slices to find the
# MAGIC practical limit. Uncomment and run one at a time.

# COMMAND ----------

# -- 100K × 10K = 1B pairs
# start = time.time()
# small_df = spark.sql(f"""
#     WITH query_table AS (
#         SELECT *, embedding AS query_embedding, monotonically_increasing_id() AS __query_row_id
#         FROM {query_table_name} TABLESAMPLE (100000 ROWS)
#         DISTRIBUTE BY __query_row_id
#     ),
#     base_table AS (
#         SELECT *, embedding AS base_embedding FROM {base_table_name}
#     ),
#     with_score AS (
#         SELECT query_table.transaction_id AS query_transaction_id,
#                base_table.ticker AS base_ticker,
#                {similarity_distance_function}(query_embedding, base_embedding) AS search_score,
#                __query_row_id
#         FROM query_table JOIN base_table
#     )
#     SELECT inline(max_by(struct(* EXCEPT (__query_row_id)), search_score, {num_results}))
#     FROM with_score GROUP BY __query_row_id
# """)
# count = small_df.count()
# elapsed = time.time() - start
# print(f"100K × 10K: {count:,} rows in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (optional)

# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS {query_table_name}")
# spark.sql(f"DROP TABLE IF EXISTS {base_table_name}")
