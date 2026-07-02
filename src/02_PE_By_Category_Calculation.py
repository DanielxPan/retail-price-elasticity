# -*- coding: utf-8 -*-
"""
Price Elasticity by Category — Step 2: Estimate PE & Build Decision Matrix

Reads the weekly extracts from Step 1, estimates a price elasticity (PE) for
each product category via a log-log fixed-effects regression (pyfixest, which
absorbs Store / Product / Month effects by demeaning), flags reliability, and
joins gross-profit data to produce a price-increase decision matrix.

Method summary
--------------
- PE = slope of  ln(qty) ~ ln(price), estimated per category.
- Fixed effects (Store, Product, Month) remove store-size, cross-product, and
  seasonal confounding, isolating the within-product price response.
- Reliability flags combine significance, sample size, and whether the
  confidence interval straddles the -1 (unit-elastic) decision threshold.

All numbers shown in the commented "Result" blocks are ILLUSTRATIVE
placeholders, not real figures. Paths and DB credentials are placeholders.

Expected input columns (from Step 1 CSVs):
    Store, Dept, Category, Product, Week,
    Total_Sales, Total_Qty, Weekly_Price, Days_Sold, Avg_Listed_Price
"""

import glob
import numpy as np
import pandas as pd
import pyfixest as pf

pd.set_option('display.float_format', lambda x: f'{x:,.2f}')


# ===========================================================================
# Block 1: Read & combine the monthly weekly-extract files
# ===========================================================================
path_dataset = r"<INPUT_FOLDER>\\"

##### Read all monthly files and append
files = sorted(glob.glob(path_dataset + "weekly_*.csv"))
df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

##### Standardise column names to the downstream convention
df_all = df_all.rename(columns={
    'store':            'Store',
    'dept':             'Dept',
    'category':         'Category',
    'product':          'Product',
    'week':             'Week',
    'total_sales':      'Total_Sales',
    'total_qty':        'Total_Qty',
    'weekly_price':     'Price',          # weekly unit price = the clean price
    'days_sold':        'Days_Sold',
    'avg_listed_price': 'Listed_Price',
})

##### Combine any weeks split across month-boundary files
df = (df_all.groupby(['Store', 'Dept', 'Category', 'Product', 'Week'], as_index=False)
            .agg(Total_Sales=('Total_Sales', 'sum'),
                 Total_Qty=('Total_Qty', 'sum'),
                 Days_Sold=('Days_Sold', 'sum')))


# ===========================================================================
# Block 2: Derived variables
# ===========================================================================
##### Weekly unit price = revenue / qty (averages out cent-level noise)
df['Price'] = df['Total_Sales'] / df['Total_Qty'].replace(0, np.nan)

df['Week'] = pd.to_datetime(df['Week'])

##### Demand quantity = within-week daily average
### total_qty / days_sold normalises weeks where the item sold on fewer days
df['AVG_QTY'] = df['Total_Qty'] / df['Days_Sold']

##### (Optional) tidy any inconsistent store spellings before joining a ref table
### e.g. df.loc[df['Store'] == '<VARIANT_SPELLING>', 'Store'] = '<CANONICAL_SPELLING>'

### (Optional) join a store reference table to attach a store code for FE
# file_store_ref = r"<STORE_REF_FILE>.csv"
# df_ref_store = pd.read_csv(file_store_ref)
# df = pd.merge(df, df_ref_store, how='inner', on='Store')

##### Keep only sales-generating departments (adjust to your own data)
list_sales_dept = ['<DEPT_1>', '<DEPT_2>', '<DEPT_3>']   # ... list your departments
if list_sales_dept and not list_sales_dept[0].startswith('<'):
    df = df[df['Dept'].isin(list_sales_dept)]

print("Shape       :", df.shape)
print("Stores      :", df['Store'].nunique())
print("Departments :", df['Dept'].nunique())
print("Categories  :", df['Category'].nunique())
print("Products    :", df['Product'].nunique())
print("Week range  :", df['Week'].min(), "->", df['Week'].max())

### Result (illustrative placeholders — not real figures)
# Shape       : (N_rows, 11)
# Stores      : N_stores
# Departments : N_departments
# Categories  : N_categories
# Products    : N_products


# ===========================================================================
# Block 3: Inspect weekly-observation quality (decide sparse-week filter)
# ===========================================================================
### Each row is a Store x Product x Week observation. Weeks where an item sold
### on very few days / very low qty are weak, noisy observations.

print("\n===== Days_Sold per weekly observation =====")
print(df['Days_Sold'].describe(percentiles=[.05, .10, .25, .50, .75, .90, .95]).round(2))

print("\n===== Total_Qty per weekly observation =====")
print(df['Total_Qty'].describe(percentiles=[.05, .10, .25, .50, .75, .90, .95]).round(2))

print("\n===== Rows surviving each (MIN_DAYS, MIN_QTY) cutoff =====")
total = len(df)
for d, q in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (3, 4), (3, 5)]:
    survive = ((df['Days_Sold'] >= d) & (df['Total_Qty'] >= q)).sum()
    print(f"  Days>={d}, Qty>={q} : {survive:7,} kept ({survive/total*100:4.1f}%)")

##### Confirm each product keeps enough weeks to estimate a slope
for label, (md, mq) in [('Days>=1,Qty>=3', (1, 3)),
                        ('Days>=1,Qty>=2', (1, 2)),
                        ('Days>=2,Qty>=2', (2, 2))]:
    tmp = df[(df['Days_Sold'] >= md) & (df['Total_Qty'] >= mq)]
    wpp = tmp.groupby('Product')['Week'].nunique()
    print(f"\n===== {label} =====")
    print(f"  Products retained  : {tmp['Product'].nunique():,}")
    print(f"  %Products retained : {(tmp['Product'].nunique()/df['Product'].nunique())*100:,.1f}%")
    print(wpp.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95]).round(1))

### Decision: Qty per week matters more than days (long purchase-cycle items
### legitimately sell on few days). Days>=1 & Qty>=2 keeps the healthiest tail.


# ===========================================================================
# Block 4: Weekly observation filter
# ===========================================================================
### Drop sparse weeks. A large drop is EXPECTED under weekly aggregation
### because the long tail of low-turnover products has many thin weeks.

df = df[df['Total_Qty'] > 0].copy()          # log safety

MIN_DAYS_PER_WEEK = 1
MIN_QTY_PER_WEEK  = 2

df_clean = df[(df['Days_Sold'] >= MIN_DAYS_PER_WEEK) &
              (df['Total_Qty'] >= MIN_QTY_PER_WEEK)].copy()

print(f"Weekly obs before : {len(df):,}")
print(f"Weekly obs after  : {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}%)")
print(f"Products before   : {df['Product'].nunique():,}")
print(f"Products after    : {df_clean['Product'].nunique():,}")


# ===========================================================================
# Block 5: Category PE via pyfixest (fixed effects + reliability + signal)
# ===========================================================================
##### Modelling frame + log variables
df_model = df_clean.copy()
df_model['Week']     = pd.to_datetime(df_model['Week'])
df_model['Month']    = df_model['Week'].dt.month.astype(str)
df_model['ln_qty']   = np.log(df_model['AVG_QTY'])
df_model['ln_price'] = np.log(df_model['Price'])
df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna(subset=['ln_qty', 'ln_price'])

##### One regression per category; FE list built dynamically (only vary-ing FEs)
results = []
for cat, sub in df_model.groupby('Category'):
    n_rows, n_products = len(sub), sub['Product'].nunique()
    n_stores, dept     = sub['Store'].nunique(), sub['Dept'].iloc[0]

    if n_rows < 30 or n_products < 2:
        results.append({'Category': cat, 'Dept': dept, 'PE': np.nan, 'p_value': np.nan,
                        'CI_low': np.nan, 'CI_high': np.nan, 'CI_range': np.nan,
                        'within_R2': np.nan, 'Variables_used': 'none', 'N_Rows': n_rows,
                        'N_Products': n_products, 'N_Stores': n_stores,
                        'REASON': 'insufficient_data'})
        continue
    try:
        fe = []
        if n_stores > 1:               fe.append('Store')
        if n_products > 1:             fe.append('Product')
        if sub['Month'].nunique() > 1: fe.append('Month')

        formula = "ln_qty ~ ln_price" + (" | " + " + ".join(fe) if fe else "")
        m  = pf.feols(formula, data=sub)
        ci = m.confint().loc['ln_price']
        results.append({
            'Category': cat, 'Dept': dept,
            'PE'       : m.coef()['ln_price'],
            'p_value'  : m.pvalue()['ln_price'],
            'CI_low'   : ci.iloc[0], 'CI_high': ci.iloc[1],
            'CI_range' : ci.iloc[1] - ci.iloc[0],
            'within_R2': m._r2_within,
            'Variables_used': ' + '.join(fe) if fe else 'none',
            'N_Rows'   : n_rows, 'N_Products': n_products, 'N_Stores': n_stores,
            'REASON'   : 'ok'})
    except Exception as e:
        results.append({'Category': cat, 'Dept': dept, 'PE': np.nan, 'p_value': np.nan,
                        'CI_low': np.nan, 'CI_high': np.nan, 'CI_range': np.nan,
                        'within_R2': np.nan, 'Variables_used': 'none', 'N_Rows': n_rows,
                        'N_Products': n_products, 'N_Stores': n_stores,
                        'REASON': f'error: {str(e)[:30]}'})

category_pe = pd.DataFrame(results)

##### Reliability flag (significance + sample + CI straddling -1)
def reliability(r):
    if r['REASON'] != 'ok':               return 'insufficient_data'
    if r['p_value'] >= 0.05:              return 'not_significant'
    if r['N_Products'] < 10:             return 'low_confidence_few_products'
    if r['CI_low'] < -1 < r['CI_high']:  return 'uncertain_elastic_or_not'
    return 'reliable'
category_pe['RELIABILITY'] = category_pe.apply(reliability, axis=1)

##### Price-increase signal from PE magnitude
def raise_signal(pe):
    if pd.isna(pe):   return 'n/a'
    if pe <= -1.0:    return 'DO NOT RAISE (elastic)'
    if pe <= -0.75:   return 'WEAK (near unit-elastic)'
    if pe <= -0.50:   return 'MODERATE'
    return 'STRONG (very inelastic)'      # PE between 0 and -0.5
category_pe['RAISE_SIGNAL'] = category_pe['PE'].apply(raise_signal)

##### Trust the signal only if the estimate is reliable
def final_call(r):
    if r['RELIABILITY'] != 'reliable':   return 'review (' + r['RELIABILITY'] + ')'
    return r['RAISE_SIGNAL']
category_pe['DECISION'] = category_pe.apply(final_call, axis=1)

cols = ['Category', 'Dept', 'PE', 'CI_low', 'CI_high', 'CI_range', 'p_value',
        'within_R2', 'N_Products', 'Variables_used', 'RAISE_SIGNAL', 'RELIABILITY', 'DECISION']
print(category_pe.sort_values('PE', ascending=False)[cols].to_string(index=False))


# ===========================================================================
# Block 6: Export category PE to Excel
# ===========================================================================
cols = ['Dept', 'Category', 'PE', 'CI_low', 'CI_high', 'CI_range',
        'p_value', 'within_R2', 'N_Products', 'N_Stores',
        'Variables_used', 'RAISE_SIGNAL', 'RELIABILITY', 'DECISION']
out = category_pe[cols].sort_values(by=['Dept', 'PE'],
                                    ascending=[True, False]).reset_index(drop=True)

path_out = r"<OUTPUT_FOLDER>\Category_Price_Elasticity.xlsx"
with pd.ExcelWriter(path_out, engine='xlsxwriter') as writer:
    out.to_excel(writer, sheet_name='Category_PE', index=False)
    wb, ws = writer.book, writer.sheets['Category_PE']

    hdr = wb.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white',
                         'border': 1, 'align': 'center', 'valign': 'vcenter'})
    for col_num, name in enumerate(out.columns):
        ws.write(0, col_num, name, hdr)

    fmt_2dp = wb.add_format({'num_format': '0.00'})
    fmt_3dp = wb.add_format({'num_format': '0.000'})
    for col in ['PE', 'CI_low', 'CI_high', 'CI_range', 'within_R2']:
        c = out.columns.get_loc(col)
        ws.set_column(c, c, 10, fmt_2dp)
    c = out.columns.get_loc('p_value')
    ws.set_column(c, c, 10, fmt_3dp)

    ws.set_column(out.columns.get_loc('Category'), out.columns.get_loc('Category'), 22)
    ws.set_column(out.columns.get_loc('Variables_used'), out.columns.get_loc('Variables_used'), 22)
    ws.set_column(out.columns.get_loc('RAISE_SIGNAL'), out.columns.get_loc('RAISE_SIGNAL'), 24)
    ws.set_column(out.columns.get_loc('DECISION'), out.columns.get_loc('DECISION'), 30)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(out), len(out.columns) - 1)

print(f"Saved: {path_out}")


# ===========================================================================
# Diagnostics: understanding what drives a category's PE
# ===========================================================================
### These compare model specifications on a single department to show how
### seasonal control and product fixed effects each affect the estimate.
### (Useful when a new estimate disagrees with a previous one.)

dept_to_check = '<DEPARTMENT_NAME>'
sub_dept = df_model[df_model['Dept'] == dept_to_check]
if len(sub_dept) > 0:
    ##### Effect of seasonal (Month) control
    m_no_season   = pf.feols("ln_qty ~ ln_price | Store + Product", data=sub_dept)
    m_with_season = pf.feols("ln_qty ~ ln_price | Store + Product + Month", data=sub_dept)
    print(f"PE without season control : {m_no_season.coef()['ln_price']:.3f}")
    print(f"PE with    season control : {m_with_season.coef()['ln_price']:.3f}")

    ##### Effect of product fixed effects (removes cross-product confounding)
    m_no_prod   = pf.feols("ln_qty ~ ln_price | Store + Month", data=sub_dept)
    m_with_prod = pf.feols("ln_qty ~ ln_price | Store + Product + Month", data=sub_dept)
    print(f"PE without product FE : {m_no_prod.coef()['ln_price']:.3f}")
    print(f"PE with    product FE : {m_with_prod.coef()['ln_price']:.3f}")

### Insight: a large gap between volume-weighted and unweighted (pooled)
### estimates reveals within-category heterogeneity — e.g. a few high-volume,
### high-elasticity products can dominate a volume-weighted departmental figure
### while the typical product is far less elastic. The two answer different
### business questions (total-volume response vs typical-product response).


# ===========================================================================
# Block 7: Final decision matrix — PE x GP
# ===========================================================================
### Join category PE with category gross profit, derive a safe price increase
### from PE, compute GP uplift, and aggregate against a GP growth target.

import psycopg2

db_params = {
    'host':     "<DB_HOST>",
    'database': "<DB_NAME>",
    'user':     "<DB_USER>",
    'password': "<DB_PASSWORD>",
}

LOCAL_TZ = "<LOCAL_TIMEZONE>"
EXCLUDED_DEPARTMENTS = ('<EXCLUDED_DEPT_1>', '<EXCLUDED_DEPT_2>')

##### Step 0: GP matrix from DB (same period as PE)
gp_sql = f"""
SELECT
     REGEXP_REPLACE(ti.space_title, '^\\d+\\s*-\\s*', '')      AS "Category"
    ,REGEXP_REPLACE(ti.department_title, '^\\d+\\s*-\\s*', '') AS "Dept"
    ,SUM(ti.quantity)     AS "Total_Qty"
    ,SUM(ti.total)        AS "Total_Revenue"
    ,SUM(ti.total_margin) AS "Total_GP"
    ,SUM(ti.total_margin) / NULLIF(SUM(ti.total), 0) AS "GP_pct"
    -- Non-promotional revenue (only non-promo prices can be raised in a
    -- normal price adjustment; promo periods are set by head office)
    ,SUM(CASE WHEN ti.is_promotion = FALSE THEN ti.total END)        AS "Revenue_NonPromo"
    ,SUM(CASE WHEN ti.is_promotion = FALSE THEN ti.total_margin END) AS "GP_NonPromo"
FROM transactions ti
WHERE
    ti.transacted_at BETWEEN ((('2025-06-23 00:00:00'::timestamptz) AT TIME ZONE 'UTC') AT TIME ZONE '{LOCAL_TZ}')::timestamp
                         AND ((('2026-06-21 23:59:59'::timestamptz) AT TIME ZONE 'UTC') AT TIME ZONE '{LOCAL_TZ}')::timestamp
    AND ti.is_returned <> TRUE
    AND ti.is_void     <> TRUE
    AND ti.is_removed  <> TRUE
    AND ti.quantity    > 0
    AND ti.product_title NOT ILIKE '%NON SCAN%'
    AND REGEXP_REPLACE(ti.department_title, '^\\d+\\s*-\\s*', '') NOT IN {EXCLUDED_DEPARTMENTS}
GROUP BY
     REGEXP_REPLACE(ti.space_title, '^\\d+\\s*-\\s*', '')
    ,REGEXP_REPLACE(ti.department_title, '^\\d+\\s*-\\s*', '')
"""

conn = psycopg2.connect(**db_params)
try:
    gp_matrix = pd.read_sql(gp_sql, conn)
finally:
    conn.close()

print(f"GP matrix loaded: {len(gp_matrix)} categories")

##### Step 1: Join PE and GP on Category
matrix = category_pe.merge(
    gp_matrix[['Category', 'Total_Revenue', 'Revenue_NonPromo',
               'Total_GP', 'GP_pct', 'Total_Qty']],
    on='Category', how='left'
)

##### Step 2: Safe price-increase % from PE (only reliable, inelastic categories)
def price_increase(row):
    pe, rel = row['PE'], row['RELIABILITY']
    if pd.isna(pe) or rel != 'reliable':   return 0.0
    if pe <= -1.0:   return 0.0      # elastic -> no raise
    if pe <= -0.75:  return 0.005    # WEAK
    if pe <= -0.50:  return 0.015    # MODERATE
    if pe < 0:       return 0.025    # STRONG (inelastic)
    return 0.0                        # positive PE (anomalous) -> no raise
matrix['PRICE_INCREASE'] = matrix.apply(price_increase, axis=1)

##### Step 3: Quantity response + GP uplift
### Uplift = revenue x [ (1 + price_increase) x (1 + qty_change) - 1 ]
### i.e. new revenue minus old revenue; the increase flows to margin because
### cost is unchanged. Base is NON-PROMO revenue (only non-promo can be raised).
matrix['QTY_CHANGE'] = matrix['PE'] * matrix['PRICE_INCREASE']
matrix['REV_MULT']   = (1 + matrix['PRICE_INCREASE']) * (1 + matrix['QTY_CHANGE'])
matrix['GP_UPLIFT']  = matrix['Revenue_NonPromo'] * (matrix['REV_MULT'] - 1)

##### Step 4: New GP
matrix['NEW_GP'] = matrix['Total_GP'] + matrix['GP_UPLIFT']

##### Step 5: Show the decision, largest contribution first
cols = ['Category', 'Dept', 'PE', 'RELIABILITY', 'PRICE_INCREASE',
        'Total_Revenue', 'Revenue_NonPromo', 'Total_GP', 'GP_pct',
        'GP_UPLIFT', 'NEW_GP']
out = matrix.sort_values('GP_UPLIFT', ascending=False)
print(out[cols].to_string(index=False))

##### Step 6: Total uplift vs the GP growth target
total_uplift = matrix['GP_UPLIFT'].sum()
TARGET = 4_400_000          # replace with your own GP growth target
print("\n===== SUMMARY =====")
print(f"Total GP uplift from safe increases : ${total_uplift:,.0f}")
print(f"GP target                           : ${TARGET:,.0f}")
print(f"Coverage of target                  : {total_uplift/TARGET*100:.1f}%")
print(f"Remaining gap                       : ${TARGET - total_uplift:,.0f}")
