from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


METHOD_NAME = 'paid_chain_ladder'
FACTOR_AVERAGE = 'volume'

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_final_paths(COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE):
    LOSS_TRIANGLE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'triangles' / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'
    ESTIMATE_OUTPUT_PATH = (PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates' / f'company_{COMPANY_CODE}_{METHOD_NAME}_{FACTOR_AVERAGE}_{PATTERN_SOURCE}_as_of_{VALUATION_YEAR}.csv')

    return LOSS_TRIANGLE_PATH, ESTIMATE_OUTPUT_PATH



def get_loss_triangle(LOSS_TRIANGLE_PATH, VALUATION_YEAR, PATTERN_SOURCE):
    loss_triangle = pd.read_csv(
        LOSS_TRIANGLE_PATH,
        index_col=0,
    )

    #company triangle is the loss development triangle of the specific company that we want to estimate
    company_triangle = loss_triangle.copy()
    company_triangle.columns = (
        company_triangle.columns.astype(int)
    )

    if PATTERN_SOURCE == 'industry':
        calculation_source = pd.read_csv(
            PROJECT_ROOT
            / 'data'
            / 'processed'
            / 'triangles'
            / f'industry_{VALUATION_YEAR}.csv',
            index_col=0,
        )
        calculation_source.columns = (
        calculation_source.columns.astype(int)
    )
    else:
        calculation_source = company_triangle

    #calculation_source is the data that we will use to calculate the age-to-age factors. It can be either the industry-wide data or the company-specific data, depending on the PATTERN_SOURCE variable.

    print('\n===company triangle===')
    print(company_triangle)
    print('\n===calculation source===')
    print(calculation_source)

    return company_triangle, calculation_source

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
def calculate_reserves(loss_triangle, age_to_lag_factors):
    records = []
    #last observed by 2007
    last_observed_claims = loss_triangle.ffill(axis=1).iloc[:, -1].copy()
    #lags of these
    current_lags = loss_triangle.apply(
        lambda cl: cl.last_valid_index(), axis=1
    )
    current_lags = current_lags.astype(int)
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

#this function gives a full rectangle of estimated datas
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

def start_calculating(calculation_source, company_triangle, COMPANY_CODE, VALUATION_YEAR, PATTERN_SOURCE, ESTIMATE_OUTPUT_PATH):
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
    estimates = calculate_reserves(company_triangle, age_to_lag_factors)
    print(estimates)

    #return estimated value at each lag and development year as a full square
    print('c\n===project_loss_triangle===')
    print(project_loss_triangle(company_triangle, calculate_selected_factors(calculation_source)))
    company_data = []
    company_data = pd.DataFrame()
    
    # ---------------------------------------------------------
    # Find the latest observed position for each accident year
    # ---------------------------------------------------------

    latest_observed_lag = (
        company_triangle
        .notna()
        .sum(axis=1)
        .astype(int)
    )

    print('\n===latest observed lag for each accident year===')
    print(latest_observed_lag)

    print(f'\n===cumulative paid observed at {VALUATION_YEAR}===')
    print(company_triangle.ffill(axis=1).iloc[:, -1])

    # ---------------------------------------------------------
    # Estimate cumulative paid and reserve through lag 10
    # ---------------------------------------------------------
    
    estimates = pd.DataFrame(
        {
            'valuation_year': VALUATION_YEAR,
            'latest_observed_lag': latest_observed_lag,
            'pattern_source': PATTERN_SOURCE,
            'cumulative_paid': company_triangle.ffill(axis=1).iloc[:, -1],
            'age_to_lag_factor': latest_observed_lag.map(age_to_lag_factors.loc[:,'age_to_lag_factor']),
            'estimated_cumulative_pay_at_10': estimates.loc[:,'estimated_claims'].values,
            'company_code': COMPANY_CODE,
            'method': METHOD_NAME,
            'factor_average': FACTOR_AVERAGE,

        }
        
    )
    print('\nfinal fil that is about to be exported into .csv')
    print(estimates)

    # ---------------------------------------------------------
    # Save and display the results
    # ---------------------------------------------------------

    estimates.to_csv(
        ESTIMATE_OUTPUT_PATH,
        index=True,
        index_label='accident_year',
    )
    print(f'\n!!! company_{COMPANY_CODE}_{METHOD_NAME}_{FACTOR_AVERAGE}_as_of_{VALUATION_YEAR} successfully saved to:')
    print(ESTIMATE_OUTPUT_PATH)





# def visualization(projected_triangle, loss_triangle, reserve_estimates):
#     assert not projected_triangle.isna().any().any()
#     if projected_triangle.isna().any().any():
#         missing_locations = projected_triangle.isna().stack()
#         print(missing_locations[missing_locations].index.tolist())
#     validation_table = pd.DataFrame()
#     validation_table['current_claim']= loss_triangle.ffill(axis=1).iloc[:,-1].values
#     validation_table['projected'] = projected_triangle.iloc[:,-1].values
#     validation_table['estimated_claims'] = reserve_estimates['estimated_claims'].values
#     validation_table['difference'] = validation_table['projected'] - validation_table['estimated_claims']
#     validation_table['reserve'] = reserve_estimates['reserve_estimate'].values
#     print(reserve_estimates['reserve_estimate'])
#     print(validation_table['estimated_claims'])
#     assert np.allclose(projected_triangle.iloc[:,-1].values, reserve_estimates['estimated_claims'].values), 'Projected triangle last column does not match estimated claims from reserve estimates'
#     total_current_claims = validation_table['current_claim'].sum()
#     total_projected_claims = validation_table['estimated_claims'].sum()
#     total_reserve = validation_table['reserve'].sum()
#     print(total_current_claims)
#     portfolio_summary = pd.DataFrame({
#         'current_claims': [total_current_claims],
#         'projected_claims': [total_projected_claims],
#         'reserve': [total_reserve],
#     },
#     index=['total']
#     )
#     print(portfolio_summary.round(2))
#     reserve_report = validation_table[
#         [
#             'current_claim',
#             'estimated_claims',
#             'reserve'
#         ]
#     ].copy()
#     print(reserve_report.round(2))

#     plot_data = reserve_report[
#         [
#             'current_claim',
#             'reserve',
#         ]
#     ].copy()

#     ax = plot_data.plot(
#         kind= 'bar',
#         stacked=True,
#         figsize= (10,6)
#     )

#     ax.set_title( 'Current Claims and Estimated Reserve by Accident Year' )
#     ax.set_xlabel('Accident Year')
#     ax.set_ylabel('Claim Amoun')
#     ax.legend(['Current Claims', 'Estimated Reserve'])
#     plt.tight_layout()
#     plt.show()

# cal results
# selected_factors = calculate_selected_factors(chain_ladder)
# age_to_lag_factors = calculate_age_to_lag_factors(selected_factors)
# reserve_estimates = calculate_reserves(chain_ladder, age_to_lag_factors)
# projected_triangle = project_loss_triangle(chain_ladder, selected_factors)
# validate_and_summary(chain_ladder, projected_triangle, reserve_estimates)

#========================================================================================================================
#====TESTING========TESTING========TESTING========TESTING========TESTING========TESTING========TESTING========TESTING====
#========================================================================================================================
#REMEMBER TO PASS IN chain_ladder NOT loss_triangle, chain_ladder is a processed loss triangle where columns are integers and not strings, loss_triangle is a processed loss triangle where columns are strings.




# ---------------------------------------------------------
# 1. Calculate volume-weighted age-to-age factors
# ---------------------------------------------------------

# factor_records = []

# for current_lag in range(1, FINAL_DEVELOPMENT_LAG):
#     next_lag = current_lag + 1
    
#     matched_pairs = (
#         calculation_source
#         .loc[:, [current_lag, next_lag]]
#         .dropna()
#     )

#     denominator = matched_pairs[current_lag].sum()
#     numerator = matched_pairs[next_lag].sum()

#     if denominator == 0:
#         raise ValueError(
#             'Cannot calculate the '
#             f'{current_lag}-to-{next_lag} factor '
#             'because its denominator sum is zero.'
#         )

#     age_to_age_factor = numerator / denominator

#     factor_records.append(
#         {
#             'current_lag': current_lag,
#             'next_lag': next_lag,
#             'matched_accident_years': len(
#                 matched_pairs
#             ),
#             'denominator': denominator,
#             'numerator': numerator,
#             'age_to_age_factor': (
#                 age_to_age_factor
#             ),
#         }
#     )


# age_to_age_table = (
#     pd.DataFrame(factor_records)
#     .set_index('current_lag')
# )


# ---------------------------------------------------------
# 2. Convert age-to-age factors to age-to-lag-10 factors
# ---------------------------------------------------------

# cumulative_development_factor = (
#     age_to_age_table[
#         'age_to_age_factor'
#     ]
#     .sort_index(ascending=False)
#     .cumprod()
#     .sort_index()
# )

# cumulative_development_factor.loc[
#     FINAL_DEVELOPMENT_LAG
# ] = 1.0

# cumulative_development_factor = (
#     cumulative_development_factor
#     .sort_index()
# )

# cumulative_development_factor.name = (
#     'age_to_lag_10_factor'
# )




# ---------------------------------------------------------
# 5. Prepare the factor output table
# ---------------------------------------------------------

# factor_output = age_to_age_table.reindex(
#     range(
#         1,
#         FINAL_DEVELOPMENT_LAG + 1,
#     )
# )

# factor_output[
#     'age_to_lag_10_factor'
# ] = cumulative_development_factor

# factor_output.insert(
#     0,
#     'company_code',
#     COMPANY_CODE,
# )

# factor_output.insert(
#     1,
#     'valuation_year',
#     VALUATION_YEAR,
# )

# factor_output.insert(
#     2,
#     'method',
#     METHOD_NAME,
# )

# factor_output.insert(
#     3,
#     'factor_average',
#     FACTOR_AVERAGE,
# )

# factor_output.insert(
#     4,
#     'pattern_source',
#     PATTERN_SOURCE,
# )





    
