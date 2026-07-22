from pathlib import Path

import pandas as pd

VALUATION_YEAR = 2007

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv'

INTERIM_DIRECTORY = PROJECT_ROOT / 'data' / 'interim'


CLEAN_CSV_PATH = (INTERIM_DIRECTORY / 'ppauto_loss_development_clean.csv')
BASE_COLUMNS = [
    "company_code",
    "company_name",
    "is_single_entity",
    "accident_year",
    "development_year",
    "development_lag",
    "cumulative_paid",
    "reported_incurred",
    "bulk_ibnr_reserve",
    "earned_premium_direct",
    "earned_premium_ceded",
    "earned_premium_net",
    "posted_reserve_2007",
]

raw = pd.read_csv(RAW_CSV_PATH)
clean = raw.copy()

COLUMN_RENAME_MAP = {
    "GRCODE": "company_code",
    "GRNAME": "company_name",
    "AccidentYear": "accident_year",
    "DevelopmentYear": "development_year",
    "DevelopmentLag": "development_lag",
    "IncurredLosses": "reported_incurred",
    "CumPaidLoss": "cumulative_paid",
    "BulkLoss": "bulk_ibnr_reserve",
    "EarnedPremDIR": "earned_premium_direct",
    "EarnedPremCeded": "earned_premium_ceded",
    "EarnedPremNet": "earned_premium_net",
    "Single": "is_single_entity",
    "PostedReserves2007": "posted_reserve_2007",
}

clean = clean.rename(
    columns=COLUMN_RENAME_MAP,
    errors='raise'
)

clean = clean.loc[:,BASE_COLUMNS].copy()


ROW_ORDER = [
    'company_code',
    'accident_year',
    'development_lag'
]

clean = (
    clean.sort_values(ROW_ORDER)
    .reset_index(drop=True)
)



clean['is_observed_at_2007'] = (
    clean['development_year'] <= VALUATION_YEAR
)

clean['reported_unpaid_reserve'] = (
    clean['reported_incurred'] - clean['cumulative_paid']
)

clean['reported_case_reserve'] = (
    clean['reported_unpaid_reserve']
    - clean['bulk_ibnr_reserve']
)

DEVELOPMENT_GROUP_COLUMNS = [
    'company_code',
    'accident_year'
]

clean['incremental_paid'] = (
    clean
    .groupby(DEVELOPMENT_GROUP_COLUMNS)[
        'cumulative_paid'
    ]
    .diff()
)

clean.loc[
    clean['development_lag'] == 1, 
    'incremental_paid', 
    ] = clean.loc[
        clean['development_lag'] == 1,
        'cumulative_paid'
    ]

clean["has_negative_incremental_paid"] = (
    clean["incremental_paid"] < 0
)

clean["has_negative_reported_unpaid"] = (
    clean["reported_unpaid_reserve"] < 0
)

clean["has_negative_reported_case_reserve"] = (
    clean["reported_case_reserve"] < 0
)

clean.to_csv(
    CLEAN_CSV_PATH,
    index=False
)

print('\nClean data saved to:')
print(CLEAN_CSV_PATH)

#raw -> clean
