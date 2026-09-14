from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import utilities as ut


method_name = 'paid_chain_ladder'
factor_average = 'volume'

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# most companies in this dataset (121 of 143) have the full 10 accident
# years (1998-2007) of history. A handful exited the line, were acquired,
# or stopped filing early - e.g. company 41700 only has 1998 and 1999, 2
# accident years total. Below this many accident years, both the fitted
# age-to-lag factors and (especially) analysis.py's backtested best_r are
# fit from so little data that they should be treated as unreliable rather
# than a genuine pattern - see thin_history_flag below.
THIN_HISTORY_ACCIDENT_YEARS = 5

#triangles=(company_triangle, industry_triangle) and square let a caller
#that already built these in memory (see analysis.py's walk-forward
#backtest) skip re-reading them from disk. persist=False skips writing the
#estimates csv - used for the same backtest, which only needs to keep the
#final valuation year's results on disk.
def start_calculating(company_code, valuation_year, pattern_source, triangles=None, square=None, persist=True):
    #get paths
    output_path = None
    if persist:
        company_specific_est_path, company_industry_est_path = ut.get_results_paths(company_code, method_name, factor_average, valuation_year)
        output_path = company_specific_est_path if pattern_source == 'company' else company_industry_est_path

    if triangles is not None:
        company_triangle, industry_triangle = triangles
    else:
        company_triangle, industry_triangle = ut.get_triangles(company_code, valuation_year)

    if square is not None:
        loss_square = square
    else:
        loss_square, _ = ut.get_squares(company_code, valuation_year)
    # column labels are read back as strings from csv (e.g. '10'), but are
    # still plain ints when passed in-memory straight from build_strcture -
    # normalize so 'actual_paid': actuals['10'] below works either way.
    loss_square.columns = loss_square.columns.astype(str)
    print('\nHERE IS LOSS SQUARE')
    print(loss_square)


    if pattern_source == 'industry':
        calculation_source = industry_triangle
    else:
        calculation_source = company_triangle
    company_triangle.columns = company_triangle.columns.astype(int)
    calculation_source.columns = calculation_source.columns.astype(int)

    # return the difference in claim size in each lag
    print('\n===calculate how much more is claimed each year')
    added_claims = calculate_added_claims(company_triangle)
    print(added_claims)
    #return one individual factor for each available accident year and development period (age to age).
    print('\n===individual factors====')
    individual_factors = calculate_individual_factors(calculation_source)
    print(individual_factors)

    # return one selected factor for every two consecutive lags
    print('\n===selected factors====')
    selected_factors = calculate_selected_factors(calculation_source)
    print(selected_factors)

    #return age to lage factors for each available development period
    print('c\n===calculate_age_to_lag_factors===')
    age_to_lag_factors = calculate_age_to_lag_factors(selected_factors)
    print(age_to_lag_factors)

    #return estimated reserves and estimated claims
    print('c\n===calculate_reserves===')
    estimates = calculate_reserves(company_triangle, age_to_lag_factors, valuation_year)
    print(estimates)

    #return estimated value at each lag and development year as a full square
    print('c\n===project_loss_triangle===')
    print(project_loss_triangle(company_triangle, calculate_selected_factors(calculation_source)))

    # ---------------------------------------------------------
    # Find the latest observed position for each accident year
    # ---------------------------------------------------------
    latest_observed_lag = pd.Series(
        valuation_year - company_triangle.index.astype(int) + 1,
        index=company_triangle.index,
        name='latest_observed_lag',
    )
    print('\n===latest observed lag for each accident year===')
    print(latest_observed_lag)
    print(f'\n===cumulative paid observed at {valuation_year}===')
    print(company_triangle.ffill(axis=1).iloc[:, -1])

    # ---------------------------------------------------------
    # Estimate cumulative paid and reserve through lag 10
    # ---------------------------------------------------------
    # loss_square holds every accident year in the raw data (e.g. 2007),
    # regardless of valuation_year. company_triangle only holds accident
    # years that have actually incepted by valuation_year (e.g. 1998-2006
    # when valuation_year=2006). Restrict the actuals to that same set of
    # accident years so all the pieces below line up.
    actuals = loss_square.reindex(company_triangle.index)
    print('\nHERE IS ACTUAL PAY')
    print(actuals)

    # how many accident years does this company actually have any data
    # for, at all - a flag every row of this company's results carries, so
    # it survives into the saved csv (the "data" side of the flag) and is
    # available to compare.py to warn on (the "visualization" side).
    num_accident_years = len(company_triangle.index)
    thin_history_flag = num_accident_years < THIN_HISTORY_ACCIDENT_YEARS

    estimates = pd.DataFrame(
        {
            'valuation_year': valuation_year,
            'latest_observed_lag': latest_observed_lag,
            'pattern_source': pattern_source,
            'last_observed_paid': get_last_observed_paid(company_triangle, valuation_year),
            'age_to_lag_factor': latest_observed_lag.map(age_to_lag_factors.loc[:,'age_to_lag_factor']),
            'estimated_cumulative_paid': estimates.loc[:,'estimated_claims'].values,
            'estimated_reserve': estimates.loc[:,'reserve_estimate'].values,
            'actual_paid': actuals['10'],
            'actual_reserve': actuals['10'] - get_last_observed_paid(company_triangle, valuation_year),
            'company_code': company_code,
            'method': method_name,
            'factor_average': factor_average,
            'num_accident_years': num_accident_years,
            'thin_history_flag': thin_history_flag,

        }

    )
    print('\nfinal fil that is about to be exported into .csv')
    print(estimates)

    # ---------------------------------------------------------
    # Save and display the results
    # ---------------------------------------------------------
    if persist:
        ut.save_data(company_data=estimates, company_path=output_path)
        print(f'\n!!! company_{company_code}_{method_name}_{factor_average}_{pattern_source}_as_of_{valuation_year} successfully saved to:')
        print(output_path)

    return estimates

#this function returns the amount of claims added each lag
def calculate_added_claims(loss_triangle):
    added_claims = loss_triangle.copy()
    added_claims = added_claims.diff(axis=1)
    added_claims.loc[:,1] = loss_triangle.iloc[:,0].values
    return added_claims

#this functions takes processed loss triangle as input and gives individual ldfs
#return one individual factor for each available accident year and development period.
def calculate_individual_factors(loss_triangle: pd.DataFrame)-> pd.DataFrame:

    individual_factors = pd.DataFrame(
        index = loss_triangle.index
    )

    final_development_lag = loss_triangle.columns[-1]

    for current_lag in range(1, final_development_lag):
        next_lag = current_lag + 1
        matched_pairs = (
            loss_triangle
            .loc[:, [current_lag, next_lag]]
            .dropna()
        )

        denominator = matched_pairs[current_lag]
        numerator = matched_pairs[next_lag]
        
        if denominator.eq(0).any():
            raise ValueError('Cannot calculate the ' f'{current_lag}-to-{next_lag}', 'because denominator is zero')
        
        age_to_age_factor = numerator / denominator
        
        individual_factors[f'{current_lag}_to{next_lag}'] = age_to_age_factor

    return individual_factors
    

  


#this function takes processed loss_triangle as inputs and give back volume weighted selected factors
#return one selected factor for every two consecutive lags
def calculate_selected_factors(loss_triangle: pd.DataFrame) -> pd.DataFrame:
    factor_records = []
    
    final_development_lag = loss_triangle.columns[-1]
    
    for current_lag in range(1, final_development_lag):
        next_lag = current_lag + 1
        
        matched_pairs = (
            loss_triangle
            .loc[:, [current_lag, next_lag]]
            .dropna()
        )

        denominator = matched_pairs[current_lag].sum()
        numerator = matched_pairs[next_lag].sum()

        if denominator == 0:
            raise ValueError('Cannot calculate the ' f'{current_lag}-to-{next_lag}', 'because denominator is zero')

        age_to_age_factor = numerator / denominator

        factor_records.append({
            'current_lag': current_lag,
            'next_lag': next_lag,
            'denominator': denominator,
            'numerator': numerator,
            'age_to_age_factor': age_to_age_factor,
        })
    return pd.DataFrame(factor_records).set_index('current_lag')

#return age to lage factors for each available development period
def calculate_age_to_lag_factors(selected_factors):
    final_lag = selected_factors['next_lag'].iloc[-1]
    running_factor = 1.0
    records = [{
        'current_lag': final_lag,
        'age_to_lag_factor': running_factor
        }]
    
    for current_lag in selected_factors.index.tolist()[::-1]:
        selected_factor = selected_factors.loc[current_lag, 'age_to_age_factor']
        running_factor *= selected_factor
        records.append({
            'current_lag': current_lag,
            'age_to_lag_factor': running_factor
        })

    return pd.DataFrame(records).sort_values('current_lag').set_index('current_lag')

#this function takes in the company specific loss_triangle and the age the lag factor to calculate the estimated claims and reserves for each accident year
#the loss triangle parameter here should be the specific company, it will be different than all previous one if we are considering industry wide data for the factor calculation.
def calculate_reserves(loss_triangle, age_to_lag_factors, valuation_year):
    records = []
    #last observed by valuation_year    
    last_observed_claims = get_last_observed_paid(loss_triangle, valuation_year)

    #lags of these
    current_lags = valuation_year - loss_triangle.index.astype(int) + 1
    current_lags = pd.Series(current_lags,index=loss_triangle.index).astype(int)

    #development_year
    development_years = loss_triangle.index.tolist()

    records = pd.DataFrame(
        {
            'current_lag': current_lags,
            'last_observed_claims': last_observed_claims,
        },
        index = loss_triangle.index)
    #estimate
    records['factor'] = records['current_lag'].map(age_to_lag_factors['age_to_lag_factor'])
    records['estimated_claims'] = records['last_observed_claims'] * records['factor']
    records['reserve_estimate'] = records['estimated_claims'] - records['last_observed_claims']
    return records

#this function gives a full square of estimated datas
#this function takes in the company specific loss_triangle and the age the lag factor to calculate the estimated claims and reserves for each accident year
def project_loss_triangle(loss_triangle, selected_factors):
    projected_triangle = loss_triangle.copy()
    development_lags = projected_triangle.columns.tolist()
    for i in projected_triangle.index:
        for e in development_lags:
            inspect = projected_triangle.loc[i, e]
            if pd.isna(inspect):
                last_value = projected_triangle.loc[i, e-1]
                projected_triangle.loc[i, e] = last_value * selected_factors.loc[e-1, 'age_to_age_factor']
    assert not projected_triangle.isna().any().any(), 'Projected triangle still has NaN values'
    return projected_triangle

def get_last_observed_paid(loss_triangle, valuation_year):
    current_lags = valuation_year - loss_triangle.index.astype(int) + 1

    return pd.Series(
        [
            loss_triangle.loc[ay, lag]
            if lag in loss_triangle.columns
            else float('nan')
            for ay, lag in zip(loss_triangle.index, current_lags)
        ],
        index=loss_triangle.index
    )