# Databricks notebook source
# MAGIC %md
# MAGIC # FTS Index Scale Test: 40M Rows
# MAGIC
# MAGIC Benchmark Full-Text Search index on the 40M ecommerce transaction table.
# MAGIC
# MAGIC **Prereq:** Run `00_setup_data` first to create the tables.
# MAGIC
# MAGIC **Requirements:** Serverless compute or DBR 18.0+ with FTS Private Preview enabled

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import time

catalog = "madkins2_catalog"
schema = "tagging"
table_name = f"{catalog}.{schema}.transactions_40m"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify table exists

# COMMAND ----------

row_count = spark.sql(f"SELECT count(*) FROM {table_name}").collect()[0][0]
print(f"Table: {table_name}")
print(f"Rows:  {row_count:,}")

# COMMAND ----------

spark.sql(f"SELECT * FROM {table_name} LIMIT 5").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline: SEARCH without an index (full table scan)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search 1: rare error string (needle in haystack)

# COMMAND ----------

start = time.time()
count = spark.sql(f"""
    SELECT count(*) AS matches
    FROM {table_name}
    WHERE SEARCH(search_text, 'NullPointerException', mode => 'substring')
""").collect()[0][0]
elapsed = time.time() - start
print(f"Baseline — 'NullPointerException': {count:,} matches in {elapsed:.1f}s (full scan)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search 2: non-existent pattern

# COMMAND ----------

start = time.time()
count = spark.sql(f"""
    SELECT count(*) AS matches
    FROM {table_name}
    WHERE SEARCH(search_text, 'SegmentationFault', mode => 'substring')
""").collect()[0][0]
elapsed = time.time() - start
print(f"Baseline — 'SegmentationFault': {count:,} matches in {elapsed:.1f}s (full scan)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search 3: moderately common pattern

# COMMAND ----------

start = time.time()
count = spark.sql(f"""
    SELECT count(*) AS matches
    FROM {table_name}
    WHERE SEARCH(search_text, 'warranty', mode => 'substring')
""").collect()[0][0]
elapsed = time.time() - start
print(f"Baseline — 'warranty': {count:,} matches in {elapsed:.1f}s (full scan)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create FTS index (ngram tokenizer)

# COMMAND ----------

start = time.time()
spark.sql(f"CREATE SEARCH INDEX idx_transactions_ngram ON {table_name}(search_text)")
elapsed = time.time() - start
print(f"Index created in {elapsed:.1f}s")

# COMMAND ----------

spark.sql("DESCRIBE SEARCH INDEX idx_transactions_ngram").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## SEARCH with the index
# MAGIC
# MAGIC Same queries, now with file pruning enabled.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search 1: rare error string (with index)

# COMMAND ----------

start = time.time()
count = spark.sql(f"""
    SELECT count(*) AS matches
    FROM {table_name}
    WHERE SEARCH(search_text, 'NullPointerException', mode => 'substring')
""").collect()[0][0]
elapsed = time.time() - start
print(f"Indexed — 'NullPointerException': {count:,} matches in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search 2: non-existent pattern (with index)

# COMMAND ----------

start = time.time()
count = spark.sql(f"""
    SELECT count(*) AS matches
    FROM {table_name}
    WHERE SEARCH(search_text, 'SegmentationFault', mode => 'substring')
""").collect()[0][0]
elapsed = time.time() - start
print(f"Indexed — 'SegmentationFault': {count:,} matches in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search 3: moderately common pattern (with index)

# COMMAND ----------

start = time.time()
count = spark.sql(f"""
    SELECT count(*) AS matches
    FROM {table_name}
    WHERE SEARCH(search_text, 'warranty', mode => 'substring')
""").collect()[0][0]
elapsed = time.time() - start
print(f"Indexed — 'warranty': {count:,} matches in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search 4: combined patterns (with index)

# COMMAND ----------

start = time.time()
count = spark.sql(f"""
    SELECT count(*) AS matches
    FROM {table_name}
    WHERE SEARCH(search_text, 'refund', mode => 'substring')
      AND SEARCH(search_text, 'credit card', mode => 'substring')
""").collect()[0][0]
elapsed = time.time() - start
print(f"Indexed — 'refund' AND 'credit card': {count:,} matches in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (optional)

# COMMAND ----------

# spark.sql("DROP SEARCH INDEX idx_transactions_ngram")
