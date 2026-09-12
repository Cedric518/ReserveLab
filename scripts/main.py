import pandas as pd
from pathlib import Path

#modules
import clean_table as clean
import build_loss_structures as build
import manual_chain_ladder as mcl

def main():
    #key metrics for choosing data for analysis
    VALUATION_YEAR = 2007
    COMPANY_CODE = 43
    FINAL_DEVELOPMENT_LAG = 10
    PATTERN_SOURCE = 'company'

    #clean, validate, build basic loss development triangle and rectangle
    clean.start_cleaning(2007)
    build.build_strcture(COMPANY_CODE, VALUATION_YEAR)

    #chain_ladder_method here
    LOSS_TRIANGLE_PATH, ESTIMATE_OUTPUT_PATH = mcl.get_final_paths(COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE)
    company_triangle, calculation_source = mcl.get_loss_triangle(LOSS_TRIANGLE_PATH, VALUATION_YEAR, PATTERN_SOURCE)
    mcl.start_calculating(calculation_source, company_triangle, COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE, ESTIMATE_OUTPUT_PATH)

if __name__ == '__main__':
    main()