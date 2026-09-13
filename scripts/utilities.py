from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#GET CLEAN DATA PATH
def get_clean_data_path():
    return PROJECT_ROOT / 'data' / 'interim' / 'ppauto_loss_development_clean.csv'

#GET PATHS------------------------------------------------------------------------------------------------------------------------------------------------
def get_results_paths(company_code, method_name, factor_average, valuation_year):
    company_specific_est_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / f'company_{company_code}_{method_name}_{factor_average}_company_as_of_{valuation_year}.csv'
    company_industry_est_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / f'company_{company_code}_{method_name}_{factor_average}_industry_as_of_{valuation_year}.csv'
    return company_specific_est_path, company_industry_est_path
#company_specific_est_path, company_industry_est_path = ut.get_results_paths(company_code, method_name, factor_average, valuation_year)
def get_rectangle_paths(company_code, valuation_year):
    rectangle_path = PROJECT_ROOT / 'data' / 'processed' / 'rectangles' / f'{company_code}_{valuation_year}.csv'
    rectangle_industry_path = PROJECT_ROOT / 'data' / 'processed' / 'rectangles' / f'industry_{valuation_year}.csv'
    return rectangle_path, rectangle_industry_path 
# rectangle_path, rectangle_industry_path = ut.get_rectangle_paths(company_code, valuation_year)

def get_triangle_paths(company_code, valuation_year):
    triangle_path = PROJECT_ROOT / 'data' / 'processed' / 'triangles' / f'{company_code}_{valuation_year}.csv'
    triangle_industry_path = PROJECT_ROOT / 'data' / 'processed' / 'triangles' / f'industry_{valuation_year}.csv'
    return triangle_path, triangle_industry_path 
# triangle_path, triangle_industry_path = ut.get_triangle_paths(company_code, valuation_year)

#GET DATA------------------------------------------------------------------------------------------------------------------------------------------------
def get_results(company_code, method_name, factor_average, valuation_year):
    company_specific_est, company_industry_est  = get_results_paths(company_code, method_name, factor_average, valuation_year)
    company_specific_est = pd.read_csv(company_specific_est, index_col=0)
    company_industry_est = pd.read_csv(company_industry_est, index_col=0)
    return company_specific_est, company_industry_est
# company_specific_est, company_industry_est = ut.get_results(company_code, method_name, factor_average, valuation_year)

def get_rectangles(company_code, valuation_year):
    rectangle_path, rectangle_industry_path = get_rectangle_paths(company_code, valuation_year)
    rectangle_specific = pd.read_csv(rectangle_path, index_col=0)
    rectangle_industry = pd.read_csv(rectangle_industry_path, index_col=0)
    return rectangle_specific, rectangle_industry
# rectangle_specific, rectangle_industry = ut.get_rectangles(company_code, valuation_year)

def get_triangles(company_code, valuation_year):
    triangle_path, triangle_industry_path = get_triangle_paths(company_code, valuation_year)
    triangle_specific = pd.read_csv(triangle_path, index_col=0)
    triangle_industry = pd.read_csv(triangle_industry_path, index_col=0)
    return triangle_specific, triangle_industry
#triangle_specific, triangle_industry = ut.get_triangles(company_code, valuation_year)

def add_data(col, col_name, data, company_est=None, industry_est=None):
    if company_est is not None:
        if col_name in company_est.columns:
            company_est[col_name] = data
        else:
            company_est.insert(col, col_name, data)

    if industry_est is not None:
        if col_name in industry_est.columns:
            industry_est[col_name] = data
        else:
            industry_est.insert(col, col_name, data)

    return company_est, industry_est

def save_data(company_data=None, company_path=None, industry_data=None, industry_path=None):
    if company_data is not None and company_path is not None:
        company_data.to_csv(company_path, index_label='accident_year')

    if industry_data is not None and industry_path is not None:
        industry_data.to_csv(industry_path, index_label='accident_year')

