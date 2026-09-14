import pandas as pd
from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from reserve_lab import initial_data_validation as initial_data_validation
from reserve_lab import clean_table as clean
from reserve_lab import build_loss_structures as build
from reserve_lab import manual_chain_ladder as mcl
from reserve_lab import backtesting as test
from reserve_lab import compare as compare  
from reserve_lab import analysis as analysis
from reserve_lab import utilities as ut

def main():
    #key metrics for choosing data for analysis
    VALUATION_YEAR = 2007
    METHOD_NAME = 'paid_chain_ladder'
    FACTOR_AVERAGE = 'volume'

    #from initial_data_validation.py
    initial_data_validation.start_validation()
    #clean, validate, build basic loss development triangle and square -
    #this is company-agnostic (the whole raw file), so it only runs once,
    #before the per-company loop below
    clean.start_cleaning(2007)

    company_codes = ut.get_all_company_codes()
    print(f'\n=== running the full pipeline for {len(company_codes)} companies ===')

    # restored to loop over every company (was temporarily narrowed to just
    # company 43 for testing) - the credibility-vs-size correlation below
    # only means something once many companies have been processed, so it
    # needs this loop back.
    for company_code in company_codes:
        try:
            run_one_company(company_code, VALUATION_YEAR, METHOD_NAME, FACTOR_AVERAGE)
        except Exception as exc:
            print(f'WARNING: skipping company_code={company_code}, failed with: {exc}')

    #does a company that represents a bigger share of the industry's paid
    #losses get a higher fitted credibility weight? pools every company
    #processed above (reads what's already on disk, no recomputation) and
    #saves the correlation/binned-curve csvs
    analysis.analyze_credibility_vs_size(METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
    #chart exactly the csvs analyze_credibility_vs_size just saved - these
    #only use fig.savefig (no plt.show()), so unlike
    #compare.start_visualization they don't block and are safe to call
    #unconditionally here
    compare.plot_credibility_scatter(METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
    compare.plot_credibility_curve(METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)

def run_one_company(company_code, valuation_year, method_name, factor_average):
    print(f'\n--- company_code={company_code} ---')
    build.build_strcture(company_code, valuation_year)

    mcl.start_calculating(company_code, valuation_year, 'company')
    mcl.start_calculating(company_code, valuation_year, 'industry')

    #from backtesting.py
    test.backtesting(company_code, method_name, factor_average, 'company', valuation_year)

    #analyze the best ratio of using company vs industry - this adds
    #blended_estimated_cumulative_paid_* / blended_estimated_reserve_* onto
    #the results csv
    analysis.start_analysis(company_code, method_name, factor_average, valuation_year)


if __name__ == '__main__':
    main()
