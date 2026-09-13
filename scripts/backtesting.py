import pandas as pd
import utilities as ut

#the function that is getting called at main
#compare the estimations to the actual result
def backtesting(company_code, method_name, factor_average, pattern_source, valuation_year):
    estimates, estimates_industry = ut.get_results(company_code, method_name, factor_average, valuation_year)
    rectangle, _ = ut.get_rectangles(company_code, valuation_year)
    company_specific_est_path, company_industry_est_path = ut.get_results_paths(company_code, method_name, factor_average, valuation_year)

    #compare the estimations to the actual result and returns the error and error percentage
    error = calculate_error(estimates, rectangle)
    error_industry = calculate_error(estimates_industry, rectangle)

    ut.add_data(9, 'error_percentage', error['error_percentage'], estimates, None)
    ut.add_data(9, 'error', error['error'], estimates, None)
    ut.add_data(9, 'error_percentage', error_industry['error_percentage'], industry_est=estimates_industry)
    ut.add_data(9, 'error', error_industry['error'], industry_est=estimates_industry)

    ut.save_data(estimates, company_specific_est_path, estimates_industry, company_industry_est_path)


def calculate_error(estimates, rectangle):
    error = pd.DataFrame()
    error['error'] = estimates['estimated_cumulative_paid'] - rectangle['10']
    error['error_percentage'] = error['error'] / rectangle['10'] * 100
    print('\nERROR CALCULATION')
    print(error)
    return error