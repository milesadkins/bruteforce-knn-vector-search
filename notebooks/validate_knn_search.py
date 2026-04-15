# Databricks notebook source
# MAGIC %md
# MAGIC # Validate Brute-Force KNN Vector Search
# MAGIC **Use case:** Tag financial news headlines to the most relevant stock tickers using embedding similarity.
# MAGIC
# MAGIC **Requirements:** Serverless compute or DBR 18.1+ cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Setup schema

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS madkins2_catalog.tagging;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Create base table: stock tickers with embeddings

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE madkins2_catalog.tagging.tickers AS
# MAGIC WITH raw_tickers AS (
# MAGIC   SELECT * FROM VALUES
# MAGIC     ('AAPL',  'Apple Inc.',                'Consumer electronics, iPhones, iPads, Mac computers, and services like the App Store and Apple Music.'),
# MAGIC     ('MSFT',  'Microsoft Corp.',           'Enterprise software, cloud computing via Azure, Windows operating system, and Office productivity suite.'),
# MAGIC     ('GOOGL', 'Alphabet Inc.',             'Internet search engine Google, online advertising, YouTube video platform, and cloud computing services.'),
# MAGIC     ('AMZN',  'Amazon.com Inc.',           'E-commerce retail marketplace, Amazon Web Services cloud platform, and digital streaming.'),
# MAGIC     ('TSLA',  'Tesla Inc.',                'Electric vehicles, battery energy storage, solar panels, and autonomous driving technology.'),
# MAGIC     ('NVDA',  'NVIDIA Corp.',              'Graphics processing units, AI accelerator chips, data center GPUs, and gaming hardware.'),
# MAGIC     ('META',  'Meta Platforms Inc.',       'Social media platforms Facebook and Instagram, virtual reality headsets, and digital advertising.'),
# MAGIC     ('JPM',   'JPMorgan Chase & Co.',      'Investment banking, commercial banking, asset management, and consumer financial services.'),
# MAGIC     ('V',     'Visa Inc.',                 'Global payments technology, credit and debit card processing network, and digital payment solutions.'),
# MAGIC     ('JNJ',   'Johnson & Johnson',         'Pharmaceuticals, medical devices, and consumer health products including over-the-counter medicines.'),
# MAGIC     ('UNH',   'UnitedHealth Group Inc.',   'Health insurance plans, healthcare services, pharmacy benefits, and health data analytics.'),
# MAGIC     ('PFE',   'Pfizer Inc.',               'Pharmaceutical company developing vaccines, oncology drugs, and antiviral treatments.'),
# MAGIC     ('XOM',   'Exxon Mobil Corp.',         'Oil and natural gas exploration, production, refining, and petrochemical manufacturing.'),
# MAGIC     ('CVX',   'Chevron Corp.',             'Integrated energy company with oil and gas production, refining, and renewable energy investments.'),
# MAGIC     ('WMT',   'Walmart Inc.',              'Discount retail stores, grocery supercenters, and e-commerce platform for consumer goods.'),
# MAGIC     ('COST',  'Costco Wholesale Corp.',    'Membership-based warehouse club offering bulk groceries, electronics, and household goods.'),
# MAGIC     ('DIS',   'Walt Disney Co.',           'Entertainment conglomerate with theme parks, Disney+ streaming, film studios, and media networks.'),
# MAGIC     ('NFLX',  'Netflix Inc.',              'Subscription streaming service for movies and TV shows, with original content production.'),
# MAGIC     ('BA',    'Boeing Co.',                'Commercial aircraft manufacturer, defense systems, space exploration, and aviation services.'),
# MAGIC     ('LMT',   'Lockheed Martin Corp.',     'Defense contractor producing fighter jets, missiles, satellites, and cybersecurity solutions.'),
# MAGIC     ('GS',    'Goldman Sachs Group Inc.',  'Investment banking, securities trading, asset management, and wealth management services.'),
# MAGIC     ('AMD',   'Advanced Micro Devices',    'Semiconductor company producing CPUs, GPUs, and data center processors for computing.'),
# MAGIC     ('CRM',   'Salesforce Inc.',           'Cloud-based customer relationship management software and enterprise business applications.'),
# MAGIC     ('UBER',  'Uber Technologies Inc.',    'Ride-hailing, food delivery via Uber Eats, and freight logistics technology platform.'),
# MAGIC     ('LLY',   'Eli Lilly and Co.',         'Pharmaceutical company specializing in diabetes treatments, oncology, and weight loss drugs.'),
# MAGIC     ('MRK',   'Merck & Co. Inc.',          'Pharmaceutical company producing vaccines, oncology therapies, and animal health products.'),
# MAGIC     ('NKE',   'Nike Inc.',                 'Athletic footwear, apparel, and sports equipment brand with global retail and e-commerce.'),
# MAGIC     ('SBUX',  'Starbucks Corp.',           'Global coffeehouse chain operating retail stores and selling packaged coffee products.'),
# MAGIC     ('F',     'Ford Motor Co.',            'Automobile manufacturer producing cars, trucks, SUVs, and electric vehicles.'),
# MAGIC     ('GM',    'General Motors Co.',        'Automobile manufacturer producing vehicles, electric cars, and autonomous driving technology.')
# MAGIC   AS t(ticker, company_name, description)
# MAGIC )
# MAGIC SELECT
# MAGIC   ticker,
# MAGIC   company_name,
# MAGIC   description,
# MAGIC   ai_query('databricks-gte-large-en', description) AS embedding
# MAGIC FROM raw_tickers;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, company_name, description, size(embedding) AS embedding_dim
# MAGIC FROM madkins2_catalog.tagging.tickers;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Create query table: news headlines with embeddings

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE madkins2_catalog.tagging.headlines AS
# MAGIC WITH raw_headlines AS (
# MAGIC   SELECT * FROM VALUES
# MAGIC     (1,  'iPhone 17 sales shatter records in the first weekend after launch'),
# MAGIC     (2,  'Oil prices surge past $100 per barrel amid Middle East tensions'),
# MAGIC     (3,  'NVIDIA unveils next-gen AI chip that doubles data center performance'),
# MAGIC     (4,  'Federal Reserve raises interest rates, bank stocks rally on wider margins'),
# MAGIC     (5,  'New weight-loss drug shows 25% more effectiveness in late-stage clinical trial'),
# MAGIC     (6,  'Electric vehicle demand accelerates as new EV models hit showroom floors'),
# MAGIC     (7,  'Disney+ subscriber count surges after streaming bundle price cut'),
# MAGIC     (8,  'Amazon expands same-day delivery to 50 new metro areas across the US'),
# MAGIC     (9,  'Pentagon awards $10 billion defense contract for next-generation fighter jets'),
# MAGIC     (10, 'Microsoft Azure revenue jumps 35% as enterprises shift to cloud computing'),
# MAGIC     (11, 'Global coffee prices hit five-year high due to drought in Brazil'),
# MAGIC     (12, 'Uber reports first full year of profitability driven by ride-hailing growth'),
# MAGIC     (13, 'Retail giant opens 50 new supercenters focused on grocery and household goods'),
# MAGIC     (14, 'Social media advertising revenue climbs as Instagram Reels gains traction'),
# MAGIC     (15, 'Semiconductor shortage eases as AMD ramps up production of server CPUs')
# MAGIC   AS t(headline_id, headline)
# MAGIC )
# MAGIC SELECT
# MAGIC   headline_id,
# MAGIC   headline,
# MAGIC   ai_query('databricks-gte-large-en', headline) AS embedding
# MAGIC FROM raw_headlines;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT headline_id, headline, size(embedding) AS embedding_dim
# MAGIC FROM madkins2_catalog.tagging.headlines;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Run brute-force KNN vector search
# MAGIC This is the core logic from the original notebook, parameterized for our ticker-tagging use case.

# COMMAND ----------

query_table_name = "madkins2_catalog.tagging.headlines"
query_embedding_column_name = "embedding"
base_table_name = "madkins2_catalog.tagging.tickers"
base_embedding_column_name = "embedding"
similarity_distance_function = "vector_cosine_similarity"
num_results = 3

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Results: top-3 ticker matches per headline

# COMMAND ----------

display(
  df.select("query_headline_id", "query_headline", "base_ticker", "base_company_name", "search_score")
    .orderBy("query_headline_id", ascending=True)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save results (optional)

# COMMAND ----------

# df.write.mode("overwrite").saveAsTable("madkins2_catalog.tagging.headline_ticker_matches")
