from pathlib import Path 
import pandas as pd
from . import utilities as ut

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_INTERIM_CSV = PROJECT_ROOT / 'data' / 'interim' / 'ppauto_loss_development_clean.csv'

#persist=False skips writing anything to disk and just hands back the four
#structures in memory - used for the walk-forward backtest in analysis.py,
#which builds a triangle for many historical valuation years purely as
#scratch work and only needs to keep the final one on disk.
def build_strcture(company_code, valuation_year, persist=True):
    #paths
    clean = pd.read_csv(CLEAN_INTERIM_CSV)

    #columns needed for calculation
    needed_columns = [
        'accident_year',
        'development_lag',
        'cumulative_paid',
        'development_year',
    ]

    #forming company data
    company_data = get_company_data(clean, company_code, needed_columns)
    company_triangle, company_square = create_company_structure(company_data, valuation_year)

    #forming industry data
    industry_data = clean.loc[:, needed_columns].copy()
    industry_triangle, industry_square = create_industry_structure(industry_data, valuation_year)

    if persist:
        triangle_path, triangle_industry_path = ut.get_triangle_paths(company_code, valuation_year)
        square_path, square_industry_path = ut.get_square_paths(company_code)

        #save triangles
        ut.save_data(
        company_data=company_triangle,
        company_path=triangle_path,
        industry_data=industry_triangle,
        industry_path=triangle_industry_path,
        )
        #save squares (same content regardless of valuation_year - see
        #get_square_paths)
        ut.save_data(
        company_data=company_square,
        company_path=square_path,
        industry_data=industry_square,
        industry_path=square_industry_path,
        )

    return company_triangle, company_square, industry_triangle, industry_square


def create_industry_structure(data,valuation_year):
    aggregated_industry_data = data.groupby(
        [
            'accident_year',
            'development_lag'
        ]
    )['cumulative_paid'].sum().copy()
    aggregated_industry_data = aggregated_industry_data.reset_index()
    aggregated_industry_data[f'is_observed_at_{valuation_year}'] = aggregated_industry_data['development_lag'] + aggregated_industry_data['accident_year'] -1 <= valuation_year
    loss_triangle = aggregated_industry_data[aggregated_industry_data[f'is_observed_at_{valuation_year}']].pivot(
        index='accident_year',
        columns='development_lag',
        values='cumulative_paid'
    )
    loss_square = aggregated_industry_data.pivot(
        index='accident_year',
        columns='development_lag',
        values='cumulative_paid'
    )
    return loss_triangle, loss_square

def create_company_structure(data, valuation_year):
    data[f'is_observed_at_{valuation_year}'] = data['development_lag'] + data['accident_year'] -1 <= valuation_year
    loss_triangle = data[data[f'is_observed_at_{valuation_year}']].pivot(
        index='accident_year',
        columns='development_lag',
        values='cumulative_paid'
    )
    loss_square = data.pivot(
        index='accident_year',
        columns='development_lag',
        values='cumulative_paid'
    )
    return loss_triangle, loss_square


def get_company_data(clean, company_code, needed_columns):
    company_data = clean.loc[clean['company_code'] == company_code,:].copy()
    if company_data.empty:
        raise ValueError(f'No company found from code: {company_code}')
    return company_data.loc[:, needed_columns]