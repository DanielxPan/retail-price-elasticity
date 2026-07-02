# Retail Price Elasticity & Pricing Decision Engine

Estimates category-level price elasticity (PE) for a multi-store retailer from
POS transaction data, then turns those elasticities into an actionable
price-increase decision matrix that projects gross-profit (GP) uplift against a
target.

> All data is sourced from a private POS database; this repo contains **code
> and methodology only**. Credentials, paths, and figures are placeholders.

## Problem

Rising labour costs required a GP-margin improvement. Rather than raising prices
across the board, the goal was to identify **which categories can absorb a price
increase with minimal volume loss**, and quantify how much GP that could deliver.

## Method

1. **Weekly aggregation** (`Store × Product × Week`) to de-noise cent-level
   price variation on weighed/fresh items.
2. **Log-log fixed-effects regression** per category: `ln(qty) ~ ln(price)`,
   estimated with [`pyfixest`](https://github.com/py-econometrics/pyfixest),
   which absorbs **Store, Product, and Month** fixed effects by demeaning —
   removing store-size, cross-product, and seasonal confounding to isolate the
   true within-product price response. The `ln_price` coefficient *is* the
   elasticity.
3. **Reliability flags** combining statistical significance, sample size, and
   whether the confidence interval straddles the −1 (unit-elastic) threshold.
4. **Decision matrix**: map each reliable, inelastic category to a safe price
   increase, join gross-profit data, and compute GP uplift on non-promotional
   revenue, aggregated against the GP target.

## Key learnings

- **Weighted vs unweighted estimates answer different questions.** A
  volume-weighted departmental PE can be dominated by a few high-volume,
  high-elasticity products, masking a far less elastic typical product. Both are
  valid — one measures total-volume response, the other the typical product.
- **Finer ≠ better.** Splitting to store or single-product level fragments the
  sample and inflates confidence intervals; category level is the sweet spot
  between granularity and statistical reliability.
- **Pricing alone has a ceiling.** Reaching an aggressive GP target through
  price increases would require raises well beyond the range the elasticities
  were validated on — a diagnostic finding that argues for non-pricing levers.

## Repo structure

```
retail-price-elasticity/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_download_weekly_data.py     # extract weekly data, month by month
│   └── 02_estimate_pe_decision.py     # PE estimation + decision matrix
└── docs/
    └── methodology.md                 # (optional) extended write-up
```

## Tech stack

Python · pandas · pyfixest (high-dimensional fixed effects) · psycopg2 ·
PostgreSQL · xlsxwriter

## Usage

```bash
pip install -r requirements.txt
# 1. Fill in DB credentials, paths, timezone, and department lists in each script
python src/01_download_weekly_data.py      # writes monthly weekly_*.csv extracts
python src/02_estimate_pe_decision.py      # estimates PE, exports the decision matrix
```

## Notes

All numeric results in code comments are illustrative placeholders, not real
figures. This project demonstrates methodology; it does not expose any
proprietary data.
