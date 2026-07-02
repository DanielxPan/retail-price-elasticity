# -*- coding: utf-8 -*-
"""
Price Elasticity by Category — Step 1: Download & Aggregate Raw Data

Extracts weekly transaction data (Store x Product x Week grain) from a
retail POS database, month by month, and saves each month to CSV. Weekly
aggregation is used to de-noise cent-level price variation on weighed items.

Expected source table (generic names — adapt to your own schema):
    transactions
        store_title        text    -- store name
        department_title   text    -- e.g. "01 - DAIRY"
        space_title        text    -- category, e.g. "MILK PRODUCTS"
        product_title      text    -- product name
        transacted_at      timestamptz
        total              numeric -- line revenue (tax-inclusive)
        quantity           numeric -- units (fractional for weighed items)
        price_current      numeric -- listed unit price
        is_returned / is_void / is_removed  boolean
    stores
        id, title

NOTE: Fill in your own DB credentials, output path, timezone, and any
business-specific exclusion values before running. All values below are
placeholders.
"""

import pandas as pd
import numpy as np
import psycopg2
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------------
# Configuration — replace all placeholders with your own values
# ---------------------------------------------------------------------------
db_params = {
    'host':     "<DB_HOST>",
    'database': "<DB_NAME>",
    'user':     "<DB_USER>",
    'password': "<DB_PASSWORD>",
}

### Output folder for the monthly CSV extracts
path_dataset = r"<OUTPUT_FOLDER>\\"

### Analysis window (one fiscal year, adjust to your own)
range_start = pd.Timestamp("2025-06-23")
range_end   = pd.Timestamp("2026-06-21")

### Local timezone for week bucketing (adjust to your own)
LOCAL_TZ = "<LOCAL_TIMEZONE>"          # e.g. "Australia/Melbourne"

### Non-sales departments to exclude (adjust to your own data)
EXCLUDED_DEPARTMENTS = ('<EXCLUDED_DEPT_1>', '<EXCLUDED_DEPT_2>')

print(f'Start Date: {range_start}, End Date: {range_end}')


# ---------------------------------------------------------------------------
# SQL template — weekly aggregation at Store x Product x Week grain
# ---------------------------------------------------------------------------
sql_template = """
    SELECT
         s.title AS Store
        ,REGEXP_REPLACE(ti.department_title, '^\\d+\\s*-\\s*', '') AS Dept
        ,REGEXP_REPLACE(ti.space_title, '^\\d+\\s*-\\s*', '')      AS Category
        ,ti.product_title AS Product
        ,DATE_TRUNC('week', (ti.transacted_at AT TIME ZONE 'UTC') AT TIME ZONE '{local_tz}')::date AS Week
        ,SUM(ti.total)    AS Total_Sales
        ,SUM(ti.quantity) AS Total_Qty
        ,SUM(ti.total) / NULLIF(SUM(ti.quantity), 0) AS Weekly_Price
        ,COUNT(DISTINCT DATE((ti.transacted_at AT TIME ZONE 'UTC') AT TIME ZONE '{local_tz}')) AS Days_Sold
        ,AVG(ti.price_current) AS Avg_Listed_Price
    FROM transactions ti
    LEFT JOIN stores s ON ti.shop_id = s.id
    WHERE
        ti.transacted_at BETWEEN ((('{date_start_str} 00:00:00'::timestamptz) AT TIME ZONE 'UTC') AT TIME ZONE '{local_tz}')::timestamp
                             AND ((('{date_end_str} 23:59:59'::timestamptz) AT TIME ZONE 'UTC') AT TIME ZONE '{local_tz}')::timestamp
        AND is_returned <> TRUE AND is_void <> TRUE AND is_removed <> TRUE
        AND ti.quantity > 0 AND ti.total > 0
        AND ti.product_title NOT ILIKE '%NON SCAN%'
        AND REGEXP_REPLACE(ti.department_title, '^\\d+\\s*-\\s*', '') NOT IN {excluded_departments}
    GROUP BY
         s.title, ti.department_title, ti.space_title, ti.product_title
        ,DATE_TRUNC('week', (ti.transacted_at AT TIME ZONE 'UTC') AT TIME ZONE '{local_tz}')::date
"""


# ---------------------------------------------------------------------------
# Month-by-month extraction (one query per month keeps memory & runtime sane)
# ---------------------------------------------------------------------------
current = range_start
while current <= range_end:
    seg_end = min(current + relativedelta(months=1) - pd.Timedelta(days=1), range_end)
    date_start_str = current.strftime("%Y-%m-%d")
    date_end_str   = seg_end.strftime("%Y-%m-%d")

    sql = sql_template.format(
        date_start_str=date_start_str,
        date_end_str=date_end_str,
        local_tz=LOCAL_TZ,
        excluded_departments=EXCLUDED_DEPARTMENTS,
    )

    conn = psycopg2.connect(**db_params)
    try:
        df_m = pd.read_sql(sql, conn)        # runs ONCE per month
    finally:
        conn.close()

    out = path_dataset + f"weekly_{date_start_str}.csv"
    df_m.to_csv(out, index=False)
    print(f"{date_start_str} -> {date_end_str}: {len(df_m):,} rows -> {out}")

    current = current + relativedelta(months=1)
