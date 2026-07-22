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

clean = clean.loc[
    clean['is_observed_at_2007']
]

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




# EXAMPLE_COMPANY_CODE = 43
# EXAMPLE_ACCIDENT_YEAR = 2005

# EXAMPLE_COLUMNS = [
#     "company_code",
#     "company_name",
#     "accident_year",
#     "development_year",
#     "development_lag",
#     "cumulative_paid",
#     "incremental_paid",
#     "reported_incurred",
#     "reported_unpaid_reserve",
#     "reported_case_reserve",
#     "bulk_ibnr_reserve",
#     "is_observed_at_2007",
# ]

# example_cohort_mask = (
#     (
#         clean['company_code']
#         == EXAMPLE_COMPANY_CODE
#     )
#     & (
#         clean['accident_year']
#         == EXAMPLE_ACCIDENT_YEAR
#     )
# )

# example_cohort = clean.loc[example_cohort_mask, EXAMPLE_COLUMNS]

# print("\n=== EXAMPLE CLEAN COHORT ===")
# print(
#     example_cohort.to_string(index=False)
# )

# print('\nHW1')
# clean['is_future_holdout'] = (
#     clean['development_year'] > 2007
# )

# print(clean.loc[:,'is_future_holdout'])


# print('\nHW2')
# same = (
#     clean['development_year'] == clean['is_observed_at_2007']
# )
# print(int(same.sum()))

# print('\nHW3')
# clean = clean.sort_values(by='development_lag').copy()
# print(clean)

# homework_cohort_mask = (
#     (clean['company_code == 43'])
#     & (clean['accident_year=2005'])
# )

# homework_columns = [
#     'development_year',
#     'development_lag',
#     'cumulative_paid',
#     'incremental_paid',
#     'reported_unpaid_reserve',
#     'is_observed_at_2007',
# ]

# homework_cohort = (
#     clean.loc[
#     homework_cohort_mask,
#     homework_columns
#     ]
#     .sort_values('development_lag')
# )
