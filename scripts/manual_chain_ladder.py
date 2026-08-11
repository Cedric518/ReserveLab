from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VALUATION_YEAR = 2007
COMPANY_CODE = 43
FINAL_DEVELOPMENT_LAG = 10

METHOD_NAME = 'paid_chain_ladder'
FACTOR_AVERAGE = 'volume'
PATTERN_SOURCE = 'company_specific'

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOSS_TRIANGLE_PATH = (
    PROJECT_ROOT
    / 'data'
    / 'processed'
    / 'triangles'
    / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / 'data'
    / 'processed'
   / 'reserve_estimates'
)

FACTOR_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / (
        f'company_{COMPANY_CODE}_'
        f'paid_chain_ladder_volume_factors_'
        f'as_of_{VALUATION_YEAR}.csv'
    )
)

ESTIMATE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / (
        f'company_{COMPANY_CODE}_'
        f'paid_chain_ladder_volume_estimates_'
        f'as_of_{VALUATION_YEAR}.csv'
    )
)


loss_triangle = pd.read_csv(
    LOSS_TRIANGLE_PATH,
    index_col=0,
)

chain_ladder = loss_triangle.copy()

chain_ladder.columns = (
    chain_ladder.columns.astype(int)
)

# ---------------------------------------------------------
# 1. Calculate volume-weighted age-to-age factors
# ---------------------------------------------------------

factor_records = []

for current_lag in range(1, FINAL_DEVELOPMENT_LAG):
    next_lag = current_lag + 1
    
    matched_pairs = (
        chain_ladder
        .loc[:, [current_lag, next_lag]]
        .dropna()
    )

    denominator = matched_pairs[current_lag].sum()
    numerator = matched_pairs[next_lag].sum()

    if denominator == 0:
        raise ValueError(
            'Cannot calculate the '
            f'{current_lag}-to-{next_lag} factor '
            'because its denominator sum is zero.'
        )

    age_to_age_factor = numerator / denominator

    factor_records.append(
        {
            'current_lag': current_lag,
            'next_lag': next_lag,
            'matched_accident_years': len(
                matched_pairs
            ),
            'denominator': denominator,
            'numerator': numerator,
            'age_to_age_factor': (
                age_to_age_factor
            ),
        }
    )


age_to_age_table = (
    pd.DataFrame(factor_records)
    .set_index('current_lag')
)


# ---------------------------------------------------------
# 2. Convert age-to-age factors to age-to-lag-10 factors
# ---------------------------------------------------------

cumulative_development_factor = (
    age_to_age_table[
        'age_to_age_factor'
    ]
    .sort_index(ascending=False)
    .cumprod()
    .sort_index()
)

cumulative_development_factor.loc[
    FINAL_DEVELOPMENT_LAG
] = 1.0

cumulative_development_factor = (
    cumulative_development_factor
    .sort_index()
)

cumulative_development_factor.name = (
    'age_to_lag_10_factor'
)


# ---------------------------------------------------------
# 3. Find the latest observed position for each accident year
# ---------------------------------------------------------

latest_observed_lag = (
    chain_ladder
    .notna()
    .sum(axis=1)
    .astype(int)
)

cumulative_paid_at_valuation = (
    chain_ladder
    .ffill(axis=1)
    .iloc[:, -1]
)

selected_cumulative_factor = (
    latest_observed_lag.map(
        cumulative_development_factor
    )
)


# ---------------------------------------------------------
# 4. Estimate cumulative paid and reserve through lag 10
# ---------------------------------------------------------

estimates = pd.DataFrame(
    {
        'latest_observed_lag': (
            latest_observed_lag
        ),
        'cumulative_paid_at_valuation': (
            cumulative_paid_at_valuation
        ),
        'age_to_lag_10_factor': (
            selected_cumulative_factor
        ),
    }
)

estimates[
    'estimated_cumulative_paid_lag_10'
] = (
    estimates[
        'cumulative_paid_at_valuation'
    ]
    * estimates[
        'age_to_lag_10_factor'
    ]
)

estimates[
    'estimated_reserve_to_lag_10'
] = (
    estimates[
        'estimated_cumulative_paid_lag_10'
    ]
    - estimates[
        'cumulative_paid_at_valuation'
    ]
)

estimates.index.name = 'accident_year'

estimates.insert(
    0,
    'company_code',
    COMPANY_CODE,
)

estimates.insert(
    1,
    'valuation_year',
    VALUATION_YEAR,
)

estimates.insert(
    2,
    'method',
    METHOD_NAME,
)

estimates.insert(
    3,
    'factor_average',
    FACTOR_AVERAGE,
)

estimates.insert(
    4,
    'pattern_source',
    PATTERN_SOURCE,
)

# ---------------------------------------------------------
# 5. Prepare the factor output table
# ---------------------------------------------------------

factor_output = age_to_age_table.reindex(
    range(
        1,
        FINAL_DEVELOPMENT_LAG + 1,
    )
)

factor_output[
    'age_to_lag_10_factor'
] = cumulative_development_factor

factor_output.insert(
    0,
    'company_code',
    COMPANY_CODE,
)

factor_output.insert(
    1,
    'valuation_year',
    VALUATION_YEAR,
)

factor_output.insert(
    2,
    'method',
    METHOD_NAME,
)

factor_output.insert(
    3,
    'factor_average',
    FACTOR_AVERAGE,
)

factor_output.insert(
    4,
    'pattern_source',
    PATTERN_SOURCE,
)


# ---------------------------------------------------------
# 6. Save and display the results
# ---------------------------------------------------------

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

factor_output.to_csv(
    FACTOR_OUTPUT_PATH,
    index=True,
    index_label='current_lag',
)

estimates.to_csv(
    ESTIMATE_OUTPUT_PATH,
    index=True,
    index_label='accident_year',
)

total_estimated_reserve = (
    estimates[
        'estimated_reserve_to_lag_10'
    ].sum()
)

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


def validate_and_summary(loss_triangle, projected_triangle, reserve_estimates):
    projected_cells = projected_triangle.where(loss_triangle.notna()) 
    pd.testing.assert_frame_equal(
        projected_cells, loss_triangle.where(loss_triangle.notna()),
        check_dtype=False,
    )
    assert not projected_triangle.isna().any().any()
    if projected_triangle.isna().any().any():
        missing_locations = projected_triangle.isna().stack()
        print(missing_locations[missing_locations].index.tolist())
    validation_table = pd.DataFrame()
    validation_table['current_claim']= loss_triangle.ffill(axis=1).iloc[:,-1].values
    validation_table['projected'] = projected_triangle.iloc[:,-1].values
    validation_table['estimated_claims'] = reserve_estimates['estimated_claims'].values
    validation_table['difference'] = validation_table['projected'] - validation_table['estimated_claims']
    validation_table['reserve'] = reserve_estimates['reserve_estimate'].values
    print(reserve_estimates['reserve_estimate'])
    print(validation_table['estimated_claims'])
    assert np.allclose(projected_triangle.iloc[:,-1].values, reserve_estimates['estimated_claims'].values), 'Projected triangle last column does not match estimated claims from reserve estimates'
    total_current_claims = validation_table['current_claim'].sum()
    total_projected_claims = validation_table['estimated_claims'].sum()
    total_reserve = validation_table['reserve'].sum()
    print(total_current_claims)
    portfolio_summary = pd.DataFrame({
        'current_claims': [total_current_claims],
        'projected_claims': [total_projected_claims],
        'reserve': [total_reserve],
    },
    index=['total']
    )
    print(portfolio_summary.round(2))
    reserve_report = validation_table[
        [
            'current_claim',
            'estimated_claims',
            'reserve'
        ]
    ].copy()
    print(reserve_report.round(2))

    plot_data = reserve_report[
        [
            'current_claim',
            'reserve',
        ]
    ].copy()

    ax = plot_data.plot(
        kind= 'bar',
        stacked=True,
        figsize= (10,6)
    )

    ax.set_title( 'Current Claims and Estimated Reserve by Accident Yeaer' )
    ax.set_xlabel('Accident Year')
    ax.set_ylabel('Claim Amoun')
    ax.legend(['Current Claims', 'Estimated Reserve'])
    plt.tight_layout()
    plt.show()

#cal results
selected_factors = calculate_selected_factors(chain_ladder)
age_to_lag_factors = calculate_age_to_lag_factors(selected_factors)
reserve_estimates = calculate_reserves(chain_ladder, age_to_lag_factors)
projected_triangle = project_loss_triangle(chain_ladder, selected_factors)
validate_and_summary(chain_ladder, projected_triangle, reserve_estimates)

#========================================================================================================================
#====TESTING========TESTING========TESTING========TESTING========TESTING========TESTING========TESTING========TESTING====
#========================================================================================================================
#REMEMBER TO PASS IN chain_ladder NOT loss_triangle, chain_ladder is a processed loss triangle where columns are integers and not strings, loss_triangle is a processed loss triangle where columns are strings.
# #return one individual factor for each available accident year and development period.

# #return one individual factor for each available accident year and development period.
# print('\n===individual factors====')
# print(calculate_individual_factors(chain_ladder))

# # return one selected factor for every two consecutive lags
# print('\n===selected factors====')
# print(calculate_selected_factors(chain_ladder))

# #return age to lage factors for each available development period
# print('c\n===calculate_age_to_lag_factors===')
# print(calculate_age_to_lag_factors(selected_factors=calculate_selected_factors(chain_ladder)))

# #return estimated reserves and estimated claims
# print('c\n===calculate_reserves===')
# print(calculate_reserves(loss_triangle, calculate_age_to_lag_factors(calculate_selected_factors(chain_ladder))))

# #return estimated value at each lag and development year as a full square
# print('c\n===project_loss_triangle===')
# print(project_loss_triangle(chain_ladder, calculate_selected_factors(chain_ladder)))
