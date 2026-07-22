from pathlib import Path 
import pandas as pd

VALUATION_YEAR = 2007
COMPANY_CODE = 43

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INTERIM_CSV = PROJECT_ROOT / 'data' / 'interim' / 'ppauto_loss_development_clean.csv'
PROCESS_TRIANGLE_DIRECTORY = PROJECT_ROOT / 'processed' / 'triangles'
RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv'
PAID_TRIANGLE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'triangles' / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'
# IDS Property Cas Ins Co


clean = pd.read_csv(CLEAN_INTERIM_CSV)

company_mask = (
    clean['company_code'] == COMPANY_CODE
)

company_cleaned_data = clean.loc[company_mask, :].copy()

if company_cleaned_data['company_name'].empty:
    raise ValueError(
        f'No company found from code: {COMPANY_CODE}'
    )


NEEDED_COLUMNS = [
    'accident_year',
    'development_year',
    'development_lag',
    'cumulative_paid'
]

company_needed_data = company_cleaned_data.loc[:, NEEDED_COLUMNS].copy()

#forming the triangle
paid_triangle = company_needed_data.pivot(
    index='accident_year',
    columns='development_lag',
    values='cumulative_paid'
)

# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#    print(paid_triangle)

#check if number of NaN fields are as expected
if (paid_triangle.index - 1998 != paid_triangle.isna().sum(axis=1).values).all() :
    raise ValueError(
        f'Unexpected data behaivor: {paid_triangle.isna().sum(axis=1)}, \n(should be 0 to 9, or inspect paid_triangle.index)'
    )


print(paid_triangle.shape)
paid_triangle.to_csv(
    PAID_TRIANGLE_PATH,
    index=True,
)
print(f'Company Code {COMPANY_CODE} of valudation year {VALUATION_YEAR} saved to PAID_TRIANGLE_PATH')