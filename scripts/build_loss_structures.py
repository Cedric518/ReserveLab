from pathlib import Path 
import pandas as pd

VALUATION_YEAR = 2007
COMPANY_CODE = 43

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INTERIM_CSV = PROJECT_ROOT / 'data' / 'interim' / 'ppauto_loss_development_clean.csv'

RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv'

#loss triangle
PROCESS_TRIANGLE_DIRECTORY = PROJECT_ROOT / 'data' /'processed' / 'triangles'
LOSS_TRIANGLE_PATH = PROCESS_TRIANGLE_DIRECTORY / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'

#future included
PROCESS_RECTANGLE_DIRECTORY = PROJECT_ROOT / 'data' / 'processed' / 'rectangles'
PROCESS_RECTANGLE_PATH = PROCESS_RECTANGLE_DIRECTORY / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'


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
    'development_lag',
    'cumulative_paid',
    'is_observed_at_2007'
]


company_needed_data = company_cleaned_data.loc[
    :
    , NEEDED_COLUMNS
    ].copy()

#forming the triangle
loss_triangle = company_needed_data[company_needed_data['is_observed_at_2007']].pivot(
    index='accident_year',
    columns='development_lag',
    values='cumulative_paid'
)

#forming full data
loss_rectangle = company_needed_data.pivot(
    index='accident_year',
    columns='development_lag',
    values='cumulative_paid'
)

#check if number of NaN fields are as expected
if (loss_triangle.index - 1998 != loss_triangle.isna().sum(axis=1).values).all() :
    raise ValueError(
        f'Unexpected data behaivor: {loss_triangle.isna().sum(axis=1)}, \n(should be 0 to 9, or inspect loss_triangle.index)'
    )

PROCESS_TRIANGLE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

loss_triangle.to_csv(
    LOSS_TRIANGLE_PATH,
    index=True,
)
print(f'Loss triangle of {COMPANY_CODE} of valudation year {VALUATION_YEAR} saved to: \n{LOSS_TRIANGLE_PATH}')

#print('\n===TEST====')
#print(loss_triangle.index)

PROCESS_RECTANGLE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

loss_rectangle.to_csv(
    PROCESS_RECTANGLE_PATH,
    index=True
)
print(f'Loss rectangle of {COMPANY_CODE} of valudation year {VALUATION_YEAR} saved to: \n{PROCESS_RECTANGLE_PATH}')
print("Triangle shape:", loss_triangle.shape)
print("Triangle index:", loss_triangle.index.tolist())
print(loss_triangle.tail(3).to_string())

#clean -> company_cleaned_data (company specific data)-> company_needed_data (company specific & needed data for current analysis)-> loss_triangle (final result) & loss_rectangle (for future comparison)