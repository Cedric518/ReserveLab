import math
import utilities as ut

#this function is called by main to execute this whole thing
def start_analysis(company_code, method_name, factor_average, valuation_year):
    print('START ANALYSIS WAS RUN')
    company_specific_est, company_industry_est = ut.get_results_paths(company_code, method_name, factor_average, valuation_year)
    company_specific_data, company_industry_data = ut.get_results(company_code, method_name, factor_average, valuation_year)

    best_r_w_mae, best_r_w_rmse = calculate_best_ratio(company_specific_data, company_industry_data)
    
    ut.add_data(11, 'best_r_w_mae', best_r_w_mae, company_specific_data, company_industry_data)
    ut.add_data(11, 'best_r_w_rmse', best_r_w_rmse, company_specific_data, company_industry_data)
    #get new data
    ut.save_data(company_specific_data, company_specific_est, company_industry_data, company_industry_est)


# col, col_name, company_code, data, valuation_year, method_name, factor_average
def calculate_best_ratio(company, industry):
    company_factor = company['age_to_lag_factor']
    industry_factor = industry['age_to_lag_factor']
    last_observed_cumulative_paid = company['last_observed_paid']

    best_mae = math.inf
    best_rmse = math.inf
    best_r_w_mae = 0
    best_r_w_rmse = 0
    for r in range(0,101):

        company_ratio = r/100
        weighted_cumulative_paid_lag_10 = (last_observed_cumulative_paid * (company_ratio * company_factor + (1-company_ratio) * industry_factor))

        #squared error strongly punish large errors,
        #absolute error minimizes the error
        #signed error allows cancelation of over + less
        #MAE = mean absolut error
        #RMSE = root mean square error
        errors = weighted_cumulative_paid_lag_10 - company['actual_paid']
        mae = errors.abs().mean()
        rmse = (errors ** 2).mean() ** 0.5
        if mae < best_mae:
            best_r_w_mae = company_ratio
            best_mae = mae
        if rmse < best_rmse:
            best_r_w_rmse = company_ratio
            best_rmse = rmse

    return best_r_w_mae, best_r_w_rmse