from pathlib import Path
import utilities as ut
import pandas as pd

PROJECT_ROOT = ut.PROJECT_ROOT
RAW_CSV_PATH = (PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv')

def start_cleaning(valuation_year):
    #load raw file
    raw = pd.read_csv(RAW_CSV_PATH)

    #clean colum names and select required columns
    clean = prepare_columns(raw)

    #sort data
    clean = sort_data(clean)

    #create calculated columns
    clean = add_observation_flag(clean, valuation_year)
    clean = add_reserve_columns(clean)
    clean = add_incremental_paid(clean)
    clean = add_data_quality_flags(clean)

    #save
    clean_path = ut.get_clean_data_path()
    clean.to_csv(clean_path, index=False)

#rename raw columns and retain only the columns needed for analysis
def prepare_columns(raw):

    column_rename_map = {
        "GRCODE": "company_code",
        "GRNAME": "company_name",
        "AccidentYear": "accident_year",
        "DevelopmentYear": "development_year",
        "DevelopmentLag": "development_lag",
        "IncurredLosses": "reported_incurred",
        "CumPaidLoss": "cumulative_paid",
        "BulkLoss": "bulk_ibnr_reserve",
        "EarnedPremDIR": "earned_premium_direct",
        "EarnedPremCeded": "earned_premium_ceded",
        "EarnedPremNet": "earned_premium_net",
        "Single": "is_single_entity",
        "PostedReserves2007": "posted_reserve_2007",
    }

    base_columns = [
        'company_code',
        'company_name',
        'is_single_entity',
        'accident_year',
        'development_year',
        'development_lag',
        'cumulative_paid',
        'reported_incurred',
        'bulk_ibnr_reserve',
        'earned_premium_direct',
        'earned_premium_ceded',
        'earned_premium_net',
        'posted_reserve_2007',
    ]

    clean = raw.rename(
        columns=column_rename_map,
        errors='raise'
    )

    return clean.loc[:, base_columns].copy()

    


#sort observations into accident-year/development order
def sort_data(data):
    sort_columns = [
        'company_code',
        'accident_year',
        'development_lag'
    ]

    return data.sort_values(sort_columns).reset_index(drop=True)

#mark whether each observation is available at the requested valuation year
def add_observation_flag(data, valuation_year):
    column_name = f'is_observed_at_{valuation_year}'

    data[column_name] = (
        data['development_year'] <= valuation_year
    )

    return data

#calculate incremental paid losses within each company/accident-year development sequence
def add_incremental_paid(data):
    group_columns = ['company_code', 'accident_year']

    data['incremental_paid'] = data.groupby(group_columns)['cumulative_paid'].diff()

    first_lag_mask = data['development_lag'] == 1

    data.loc[first_lag_mask, 'incremental_paid'] = data.loc[first_lag_mask, 'cumulative_paid']

    return data

#add flags for negative values that mya require investigation
def add_data_quality_flags(data):
    data['has_negative_incremental_paid'] = data['incremental_paid'] < 0
    data['has_negative_reported_unpaid'] = data['reported_unpaid_reserve'] < 0
    data['has_negative_reported_case_reserve'] = data['reported_case_reserve'] < 0

    return data

#calculate unpaid, case, and related reserve measures
def add_reserve_columns(data):
    data['reported_unpaid_reserve'] = data['reported_incurred'] - data['cumulative_paid']
    data['reported_case_reserve'] = data['reported_unpaid_reserve'] - data['bulk_ibnr_reserve']

    return data