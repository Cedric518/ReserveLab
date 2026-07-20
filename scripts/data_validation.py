from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv'

raw = pd.read_csv(RAW_CSV_PATH)

print("\n=== DATAFRAME SUMMARY ===")
raw.info()

print("\n=== COLUMN DATA TYPES ===")
print(raw.dtypes)

# data validation
EXPECTED_COLUMNS = {
    "GRCODE",
    "GRNAME",
    "AccidentYear",
    "DevelopmentYear",
    "DevelopmentLag",
    "IncurredLosses",
    "CumPaidLoss",
    "BulkLoss",
    "EarnedPremDIR",
    "EarnedPremCeded",
    "EarnedPremNet",
    "Single",
    "PostedReserves2007",
}

actual_columns = set(raw.columns)

missing_columns = EXPECTED_COLUMNS - actual_columns
extra_columns = actual_columns - EXPECTED_COLUMNS

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

# print('\nAll expected 2007 columns are present.')
# print('Extra columns:', sorted(extra_columns))


print('\n===TEST===')
# print(sorted(raw["AccidentYear"].dropna().unique()))

TIME_COLUMNS = [
    'AccidentYear',
    'DevelopmentYear',
    'DevelopmentLag'
]

time_check = (
    raw[TIME_COLUMNS]
    .dropna()
    .copy()
)


time_check['ExpectedDevelopmentYear'] = (
    time_check['AccidentYear'] 
    + time_check['DevelopmentLag']
    - 1
)

time_mismatch_mask = (
    time_check['DevelopmentYear'] !=
    time_check['ExpectedDevelopmentYear'] +1
)

time_mismatch_count = int(time_mismatch_mask.sum())


if time_mismatch_count > 0:
    time_problem_rows = time_check.loc[time_mismatch_mask]

# print('\n===TIME RELATIONSHIP===')
# print(time_problem_rows.head(20))
