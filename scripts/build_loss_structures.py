from pathlib import Path 
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INTERIM_CSV = PROJECT_ROOT / 'data' / 'interim' / 'ppauto_loss_development_clean.csv'
PROCESS_TRIANGLE_DIRECTORY = PROJECT_ROOT / 'data' /'processed' / 'triangles'
PROCESS_RECTANGLE_DIRECTORY = PROJECT_ROOT / 'data' / 'processed' / 'rectangles'

def create_industry_structure(data,VALUATION_YEAR):
    aggregated_industry_data = data.groupby(
        [
            'accident_year',
            'development_lag'
        ]
    )['cumulative_paid'].sum().copy()
    aggregated_industry_data = aggregated_industry_data.reset_index()
    aggregated_industry_data[f'is_observed_at_{VALUATION_YEAR}'] = aggregated_industry_data['development_lag'] + aggregated_industry_data['accident_year'] -1 <= VALUATION_YEAR
    loss_triangle = aggregated_industry_data[aggregated_industry_data[f'is_observed_at_{VALUATION_YEAR}']].pivot(
        index='accident_year',
        columns='development_lag',
        values='cumulative_paid'
    )
    loss_rectangle = aggregated_industry_data.pivot(
        index='accident_year',
        columns='development_lag',
        values='cumulative_paid'
    )
    
    return loss_triangle, loss_rectangle

#main

def build_strcture(COMPANY_CODE, VALUATION_YEAR):


    clean = pd.read_csv(CLEAN_INTERIM_CSV)

    see = f'is_observed_at_{VALUATION_YEAR}'

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
        f'is_observed_at_{VALUATION_YEAR}'
    ]

    PROCESS_INDUSTRY_TRIANGLE_PATH = PROCESS_TRIANGLE_DIRECTORY / 'industry_2007.csv'
    PROCESS_INDUSTRY_RECTANGLE_PATH = PROCESS_RECTANGLE_DIRECTORY / f'industry_{VALUATION_YEAR}.csv'

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

    #loss triangle
    LOSS_TRIANGLE_PATH = PROCESS_TRIANGLE_DIRECTORY / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'

    #future included
    PROCESS_RECTANGLE_PATH = PROCESS_RECTANGLE_DIRECTORY / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'

    #industry wide data
    PROCESS_INDUSTRY_RECTANGLE_PATH = PROCESS_RECTANGLE_DIRECTORY / f'industry_{VALUATION_YEAR}.csv'

    # IDS Property Cas Ins Co
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
        f'is_observed_at_{VALUATION_YEAR}'
    ]


    company_needed_data = company_cleaned_data.loc[
        :
        , NEEDED_COLUMNS
        ].copy()

    #forming the triangle
    loss_triangle = company_needed_data[company_needed_data[f'is_observed_at_{VALUATION_YEAR}']].pivot(
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
    print(f'\nLoss triangle of company_{COMPANY_CODE} of valudation year {VALUATION_YEAR} saved to: \n{LOSS_TRIANGLE_PATH}')

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
    print(f'\nLoss rectangle of company_{COMPANY_CODE} of valudation year {VALUATION_YEAR} saved to: \n{PROCESS_RECTANGLE_PATH}')


    #INDUSTRY
    industry_needed_data = clean.loc[
    :
    , NEEDED_COLUMNS
    ].copy()

    industry_loss_triangle, industry_loss_rectangle = create_industry_structure(industry_needed_data, VALUATION_YEAR)

    industry_loss_triangle.to_csv(
        PROCESS_INDUSTRY_TRIANGLE_PATH,
        index=True
    )
    print(f'\nLoss triangle of whole industry of valudation year {VALUATION_YEAR} saved to: \n{PROCESS_INDUSTRY_TRIANGLE_PATH}')

    industry_loss_rectangle.to_csv(
        PROCESS_INDUSTRY_RECTANGLE_PATH,
        index=True
    )
    print(f'\nLoss Rectangle of whole industry of valudation year {VALUATION_YEAR} saved to: \n{PROCESS_INDUSTRY_RECTANGLE_PATH}')
