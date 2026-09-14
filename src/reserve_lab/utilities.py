from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#GET CLEAN DATA PATH
def get_clean_data_path():
    return PROJECT_ROOT / 'data' / 'interim' / 'ppauto_loss_development_clean.csv'

#every company code present in the clean data - used to loop the whole
#pipeline over every firm instead of one hardcoded company_code.
def get_all_company_codes():
    clean = pd.read_csv(get_clean_data_path())
    return sorted(clean['company_code'].unique().tolist())

#GET PATHS------------------------------------------------------------------------------------------------------------------------------------------------
# squares, triangles and results are all company-specific (results doubly
# so - even the "industry pattern" result is still this company's own
# paid-to-date projected with industry factors), so each company gets its
# own subfolder under processed/{squares,triangles,results}/ instead of
# every firm's files sitting side by side in one flat directory. The
# industry-wide square/triangle themselves are the one genuine exception -
# they're the same aggregate regardless of which company you're analyzing
# - so those stay as shared files at the top level, not nested under any
# particular company's folder.
def get_results_paths(company_code, method_name, factor_average, valuation_year):
    company_dir = PROJECT_ROOT / 'data' / 'processed' / 'results' / str(company_code)
    company_specific_est_path = company_dir / f'{method_name}_{factor_average}_company_as_of_{valuation_year}.csv'
    company_industry_est_path = company_dir / f'{method_name}_{factor_average}_industry_as_of_{valuation_year}.csv'
    return company_specific_est_path, company_industry_est_path
#company_specific_est_path, company_industry_est_path = ut.get_results_paths(company_code, method_name, factor_average, valuation_year)
def get_square_paths(company_code, valuation_year=None):
    # a "square" is the standard actuarial term for a fully-populated
    # loss triangle - every accident-year/development-lag cell filled in,
    # not just the ones known as of some valuation date (that's what a
    # "triangle" holds - see get_triangle_paths below). We can build a full
    # square here because this is historical backtesting data: every
    # accident year has already fully run off, so nothing is unknown.
    # It never depends on valuation_year, so there is one square file per
    # company/industry, not one per valuation year. valuation_year is
    # accepted (and ignored) so existing callers that still pass it don't
    # need to change.
    square_path = PROJECT_ROOT / 'data' / 'processed' / 'squares' / str(company_code) / 'square.csv'
    square_industry_path = PROJECT_ROOT / 'data' / 'processed' / 'squares' / 'industry.csv'
    return square_path, square_industry_path
# square_path, square_industry_path = ut.get_square_paths(company_code)

def get_weighting_path(company_code, method_name, factor_average, valuation_year):
    company_dir = PROJECT_ROOT / 'data' / 'processed' / 'results' / str(company_code)
    return company_dir / f'{method_name}_{factor_average}_weighting_by_lag_as_of_{valuation_year}.csv'
# weighting_path = ut.get_weighting_path(company_code, method_name, factor_average, valuation_year)

def get_triangle_paths(company_code, valuation_year):
    triangle_path = PROJECT_ROOT / 'data' / 'processed' / 'triangles' / str(company_code) / f'{valuation_year}.csv'
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

def get_squares(company_code, valuation_year):
    square_path, square_industry_path = get_square_paths(company_code, valuation_year)
    square_specific = pd.read_csv(square_path, index_col=0)
    square_industry = pd.read_csv(square_industry_path, index_col=0)
    return square_specific, square_industry
# square_specific, square_industry = ut.get_squares(company_code, valuation_year)

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
    # company_path is now often a per-company subfolder (e.g.
    # results/<company_code>/...) that won't exist yet the first time a
    # given company is processed, so make sure it's there before writing.
    if company_data is not None and company_path is not None:
        company_path.parent.mkdir(parents=True, exist_ok=True)
        company_data.to_csv(company_path, index_label='accident_year')

    if industry_data is not None and industry_path is not None:
        industry_path.parent.mkdir(parents=True, exist_ok=True)
        industry_data.to_csv(industry_path, index_label='accident_year')

