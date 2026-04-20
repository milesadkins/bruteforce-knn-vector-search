# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: Generate Scale Test Data
# MAGIC
# MAGIC **Run this once.** Creates the 40M ecommerce transaction table and 10K ticker table
# MAGIC with synthetic 1024-dim embeddings. Subsequent notebooks read from these tables
# MAGIC without regenerating.
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

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate 10K synthetic tickers (base table)

# COMMAND ----------

from pyspark.sql import functions as F
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

tickers_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.tickers_10k")

elapsed = time.time() - start
print(f"Created {num_tickers:,} tickers in {elapsed:.1f}s")

# COMMAND ----------

spark.sql(f"SELECT count(*) AS ticker_count, size(embedding) AS embedding_dim FROM {catalog}.{schema}.tickers_10k").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate 40M synthetic ecommerce transactions (query table)

# COMMAND ----------

categories = [
    "Electronics", "Clothing", "Home & Garden", "Sports & Outdoors", "Books",
    "Toys & Games", "Automotive", "Health & Beauty", "Grocery", "Pet Supplies",
    "Office Supplies", "Jewelry", "Music & Instruments", "Software", "Industrial"
]

products = [
    ("wireless headphones", "Sony WH-1000XM5 noise cancelling bluetooth headphones"),
    ("laptop stand", "Adjustable aluminum laptop riser for MacBook Pro"),
    ("USB-C hub", "Anker 7-in-1 USB-C hub with HDMI and ethernet"),
    ("mechanical keyboard", "Keychron K2 wireless mechanical keyboard cherry MX brown"),
    ("webcam", "Logitech C920 HD Pro webcam 1080p autofocus"),
    ("running shoes", "Nike Air Zoom Pegasus 41 mens running shoes"),
    ("winter jacket", "North Face ThermoBall insulated winter puffer jacket"),
    ("yoga pants", "Lululemon Align high-rise leggings breathable fabric"),
    ("hiking backpack", "Osprey Atmos AG 65L backpack with anti-gravity suspension"),
    ("camping tent", "REI Co-op Half Dome 2 Plus lightweight backpacking tent"),
    ("air purifier", "Dyson Pure Cool TP07 HEPA air purifier tower fan"),
    ("coffee maker", "Breville Barista Express espresso machine stainless steel"),
    ("desk lamp", "BenQ ScreenBar LED monitor desk lamp auto-dimming"),
    ("throw blanket", "Barefoot Dreams CozyChic ribbed throw blanket"),
    ("vitamin supplements", "Garden of Life multivitamin organic whole food"),
    ("face moisturizer", "CeraVe daily moisturizing lotion hyaluronic acid"),
    ("electric toothbrush", "Oral-B iO Series 9 rechargeable electric toothbrush"),
    ("protein bars", "RXBAR chocolate sea salt protein bars 12 pack"),
    ("olive oil", "California Olive Ranch extra virgin olive oil cold pressed"),
    ("ground coffee", "Lavazza Super Crema whole bean espresso coffee"),
    ("dog treats", "Blue Buffalo wilderness trail treats grain free"),
    ("cat litter", "Dr Elseys precious cat ultra clumping litter"),
    ("printer paper", "HP premium choice LaserJet 32lb bright white paper"),
    ("ergonomic chair", "Herman Miller Aeron remastered size B graphite"),
    ("diamond ring", "14K white gold round cut solitaire diamond engagement ring"),
    ("electric guitar", "Fender Player Stratocaster maple fingerboard sunburst"),
    ("antivirus software", "Norton 360 Deluxe antivirus VPN dark web monitoring"),
    ("power drill", "DeWalt 20V MAX brushless cordless hammer drill kit"),
    ("safety goggles", "3M SecureFit protective eyewear anti-fog coating"),
    ("work boots", "Timberland PRO steel toe waterproof work boots")
]

order_notes = [
    "Please deliver before 5pm, doorbell is broken so knock loudly",
    "Gift wrap requested — this is a birthday present do not include receipt",
    "Shipping address updated: use the back entrance of building B",
    "Customer called to add expedited shipping — needs by Friday",
    "FRAGILE — handle with care, previous order arrived damaged",
    "Apply promo code SUMMER25 at checkout for free shipping",
    "Loyalty member since 2019 — earned 500 bonus points this order",
    "Return authorization #RA-48291 approved — refund pending",
    "Backordered item — expected restock date March 15 2026",
    "Price match guarantee applied — competitor had $10 lower price",
    "Duplicate charge reported on credit card ending 4829 — investigation open",
    "Exchange request: wrong size received, need size Medium not Large",
    "Cancel order per customer request — payment reversal initiated",
    "International shipment to Germany — customs declaration required",
    "Warehouse pick confirmed — item located in aisle 14 shelf B3",
    "Quality inspection flagged: item SKU-7291 has cosmetic defect on surface",
    "Customer complaint: NullPointerException error on checkout page",
    "VIP customer — priority handling and complimentary next-day air",
    "Subscription renewal processed — next billing date April 2026",
    "Package returned to sender — addressee unknown at delivery location",
    "Bulk order for corporate account ACME-Corp — net 30 payment terms",
    "Dropship order routed to third-party fulfillment center",
    "Flash sale purchase — inventory reserved for 24 hours only",
    "Tax exempt order — reseller certificate #TX-2026-8834 on file",
    "Out of stock notification sent — customer opted for email alert",
    "Cart abandoned recovery email triggered — 10% discount code sent",
    "PayPal payment pending verification — hold shipment until cleared",
    "Warranty claim submitted — product failed within 90-day coverage",
    "Same-day delivery zone confirmed — dispatched via local courier",
    "Split shipment: items 1-3 ship today, item 4 ships next week from Ohio warehouse"
]

start = time.time()

products_array = F.array(*[F.struct(
    F.lit(p[0]).alias("name"),
    F.lit(p[1]).alias("full_description")
) for p in products])

notes_array = F.array(*[F.lit(n) for n in order_notes])

transactions_df = (
    spark.range(1, num_transactions + 1)
    .withColumnRenamed("id", "transaction_id")
    .withColumn("customer_id", (F.col("transaction_id") % 2_000_000 + 1).cast("int"))
    .withColumn("product_idx", (F.rand() * len(products)).cast("int"))
    .withColumn("product", F.element_at(products_array, F.col("product_idx") + 1))
    .withColumn("product_name", F.col("product.name"))
    .withColumn("category", F.element_at(
        F.array(*[F.lit(c) for c in categories]),
        (F.col("product_idx") / 2).cast("int") % len(categories) + 1
    ))
    .withColumn("product_description", F.col("product.full_description"))
    .withColumn("price", F.round(F.rand() * 500 + 5, 2))
    .withColumn("quantity", (F.rand() * 5 + 1).cast("int"))
    .withColumn("purchase_date", F.date_add(F.lit("2025-01-01"), (F.rand() * 365).cast("int")))
    .withColumn("order_notes", F.element_at(notes_array, (F.rand() * len(order_notes)).cast("int") + 1))
    .withColumn("search_text", F.concat_ws(" | ",
        F.col("product_name"),
        F.col("product_description"),
        F.col("order_notes")
    ))
    .withColumn("embedding", F.expr(f"transform(sequence(1, {embedding_dim}), x -> cast(rand() as float))"))
    .drop("product_idx", "product")
)

transactions_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.transactions_40m")

elapsed = time.time() - start
print(f"Created {num_transactions:,} transactions in {elapsed:.1f}s")

# COMMAND ----------

spark.sql(f"SELECT count(*) AS total_rows FROM {catalog}.{schema}.transactions_40m").display()

# COMMAND ----------

spark.sql(f"SELECT * FROM {catalog}.{schema}.transactions_40m LIMIT 5").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data is ready
# MAGIC
# MAGIC Tables created:
# MAGIC - `madkins2_catalog.tagging.tickers_10k` — 10K tickers with 1024-dim embeddings
# MAGIC - `madkins2_catalog.tagging.transactions_40m` — 40M transactions with text + 1024-dim embeddings
# MAGIC
# MAGIC You can now run the search notebooks without regenerating data:
# MAGIC - **01_knn_search** — Brute-force KNN vector search
# MAGIC - **01_fts_search** — Full-text search index
