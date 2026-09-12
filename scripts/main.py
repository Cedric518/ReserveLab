import pandas as pd
from pathlib import Path

#modules
import initial_data_validation as initial_data_validation
import clean_table as clean
import build_loss_structures as build
import manual_chain_ladder as mcl
import backtesting as test
import compare as compare
def main():
    #important pathsß
    #PROJECT_ROOT = Path(__file__).resolve().parents[1]
    #RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv'

    #key metrics for choosing data for analysis
    VALUATION_YEAR = 2007
    COMPANY_CODE = 43
    FINAL_DEVELOPMENT_LAG = 10
    PATTERN_SOURCE = 'industry'
    METHOD_NAME = 'paid_chain_ladder'
    FACTOR_AVERAGE = 'volume'
    visualization_style = 'classical'

    #from initial_data_validation.py
    initial_data_validation.start_validation()
    #clean, validate, build basic loss development triangle and rectangle
    clean.start_cleaning(2007)
    build.build_strcture(COMPANY_CODE, VALUATION_YEAR)

    #from manual_chain_ladder.py here
    #getting all the files, since file paths depend on key metrics at top
    LOSS_TRIANGLE_PATH, ESTIMATE_OUTPUT_PATH, LOSS_RECTANGLE_PATH = mcl.get_final_paths(COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE)
    #company_triangle is the company that we want to do analysis on
    #calculation_source depends on whether this calculation is based on industry data or own company
    company_triangle, calculation_source = mcl.get_loss_triangle(LOSS_TRIANGLE_PATH, VALUATION_YEAR, PATTERN_SOURCE)
    #calculate and output the final estimates.csv
    estimates = mcl.start_calculating(calculation_source, company_triangle, COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE, ESTIMATE_OUTPUT_PATH)

    #from backtesting.py
    test.backtesting(COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE, METHOD_NAME, FACTOR_AVERAGE)
    compare.start_visualization(visualization_style, COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
    

    #from compare.py
    # getting fully processed data
    company_specific_data, company_industry_data = compare.get_data(COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
    print(company_specific_data, company_industry_data)

if __name__ == '__main__':
    main()