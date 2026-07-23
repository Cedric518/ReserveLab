from pathlib import Path
import pandas as pd

VALUATION_YEAR = 2007
COMPANY_CODE = 43

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSS_TRIANGLE_PATH = PROJECT_ROOT / 'data' /'processed' / 'triangles' / f'{COMPANY_CODE}_{VALUATION_YEAR}.csv'
OUTPUT_DIRECTORY = PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates'
FACTOR_OUTPUT_PATH = OUTPUT_DIRECTORY / f'company_{COMPANY_CODE}_selected_factors_as_of_{VALUATION_YEAR}.csv'
ESTIMATE_OUTPUT_PATH = OUTPUT_DIRECTORY / f'company_{COMPANY_CODE}_chain_ladder_estimates_as_of_{VALUATION_YEAR}.csv'

loss_triangle = pd.read_csv(LOSS_TRIANGLE_PATH, index_col=0)
chain_ladder = loss_triangle.copy()
chain_ladder.columns = chain_ladder.columns.astype(int)

chain_ladder_age_to_age = (chain_ladder.pct_change(axis=1).drop(columns=[1])+1).copy()


#getting CDF and indexing right
cumulative_development_factor = pd.DataFrame()
cumulative_development_factor = chain_ladder_age_to_age.mean().iloc[::-1].cumprod()
cumulative_development_factor.index = range(9,0,-1)
cumulative_development_factor = pd.concat([pd.Series({10: 1.0}),cumulative_development_factor])

#getting latest known value by 2007 and the age
age = chain_ladder.notna().sum(axis=1)
cumulative_pay_by_2007 = chain_ladder.ffill(axis=1).iloc[:,-1]

estimated_cumulative_paid_lag_10 = cumulative_pay_by_2007 * age.map(cumulative_development_factor)
estimated_reserve_to_lag_10 = estimated_cumulative_paid_lag_10 - cumulative_pay_by_2007

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)
cumulative_development_factor.to_csv(
    FACTOR_OUTPUT_PATH,
    index=True,
    index_label='current_lag'
)
estimated_reserve_to_lag_10.to_csv(
    ESTIMATE_OUTPUT_PATH,
    index=True,
    index_label='accident_year'
)



#print(estimated_cumulative_paid_lag_10)
#print(cumulative_pay_by_2007)
#print(estimated_reserve_to_lag_10)

#This is WRONG, but I leave it here since I learned many from coding this
#chain_ladder_result = pd.DataFrame()
#for i,v in enumerate(range(1, len(chain_ladder.columns))):
#    #print('\n===TARGET===')
#    #print(chain_ladder.iloc[:,0:v].dropna().sum(axis=1).iloc[:-1])
#    #print('\n===SUM===')
#    print(chain_ladder.iloc[:,0:v+1].dropna().sum(axis=1))
#    
#    factor = chain_ladder.iloc[:,0:v+1].dropna().sum(axis=1) / chain_ladder.iloc[:,0:v].dropna().sum(axis=1).iloc[:-1]
#    #print('\n===result===')
#    #print(factor)
#    chain_ladder_result[v]= factor
#    chain_ladder_result.to_csv(CHAIN_LADDER_PATH, index=True)
