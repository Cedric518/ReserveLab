import pandas as pd
from pathlib import Path

#modules
import initial_data_validation as initial_data_validation
import clean_table as clean
import build_loss_structures as build
import manual_chain_ladder as mcl
import backtesting as test
import compare as compare
import analysis as analysis

def main():
    #important pathsß
    #PROJECT_ROOT = Path(__file__).resolve().parents[1]
    #RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv'

    #key metrics for choosing data for analysis
    VALUATION_YEAR = 2007
    COMPANY_CODE = 43
    FINAL_DEVELOPMENT_LAG = 10
    PATTERN_SOURCE = 'company'
    METHOD_NAME = 'paid_chain_ladder'
    FACTOR_AVERAGE = 'volume'
    visualization_style = 'classical'

    #from initial_data_validation.py
    initial_data_validation.start_validation()
    #clean, validate, build basic loss development triangle and rectangle
    clean.start_cleaning(2007)
    build.build_strcture(COMPANY_CODE, VALUATION_YEAR)

    #from manual_chain_ladder.py here
    #calculate and output the final estimates.csv
    estimates = mcl.start_calculating(COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE)

    #from backtesting.py
    test.backtesting(COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, PATTERN_SOURCE, VALUATION_YEAR)
    compare.start_visualization(visualization_style, COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
    
    #analyze the best ratio of using company vs industry
    analysis.start_analysis(COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)



if __name__ == '__main__':
    main()