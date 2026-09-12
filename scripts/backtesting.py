from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# VALUATION_YEAR = 2007
# COMPANY_CODE = 43
# FINAL_DEVELOPMENT_LAG = 10
# METHOD_NAME = 'paid_chain_ladder'
# FACTOR_AVERAGE = 'volume'
# PATTERN_SOURCE = 'company'

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#reading paths


PROCESSED_DIRECTORY= (
    PROJECT_ROOT
    / 'data'
    / 'processed'
   / 'reserve_estimates'
)


def backtesting(COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE, METHOD_NAME, FACTOR_AVERAGE):
    estimates = pd.read_csv(PROCESSED_DIRECTORY / f'company_{COMPANY_CODE}_{METHOD_NAME}_{FACTOR_AVERAGE}_{PATTERN_SOURCE}_as_of_{VALUATION_YEAR}.csv', index_col=0)
    #retriving data
    RECTANGLE_PATH = (PROJECT_ROOT / 'data' / 'processed' / 'rectangles' / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv')

    #compare the estimations to the actual result and returns the error and error percentage
    error = calculate_error(estimates, RECTANGLE_PATH)
    #this joins the freshly calculated error into the original data file 
    add_backtesting_results(estimates, error, COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE, METHOD_NAME, FACTOR_AVERAGE)

#the function that is getting called at main
#compare the estimations to the actual result
def calculate_error(estimates, RECTANGLE_PATH):
    #get rectangle
    loss_rectangle = pd.read_csv(RECTANGLE_PATH, index_col=0)

    error = pd.DataFrame()
    error['error'] = estimates.loc[:, 'estimated_cumulative_paid_lag_10'] - loss_rectangle.loc[:, '10']
    error['error_percentage'] = error['error'] / loss_rectangle.loc[:, '10'] * 100
    print('\nERROR CALCULATION')
    print(error)

    return error

def add_backtesting_results(estimates, error, COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE, METHOD_NAME, FACTOR_AVERAGE):
    estimates.insert(loc=7, column='error_percentage', value=error.loc[:,'error_percentage'])
    estimates.insert(loc=7, column='error', value=error.loc[:,'error'])
    estimates.to_csv(
        PROCESSED_DIRECTORY / f'company_{COMPANY_CODE}_{METHOD_NAME}_{FACTOR_AVERAGE}_{PATTERN_SOURCE}_as_of_{VALUATION_YEAR}.csv',
        index='True',
        index_label='accident_year'
    )
    return estimates

# print('\nthe final estimation file with error included')
# with pd.option_context('display.max_rows', None, 'display.max_columns',None):
#     print(estimates)