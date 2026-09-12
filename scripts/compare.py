from pathlib import Path

import pandas as pd
from manual_chain_ladder import get_final_paths
import matplotlib.pyplot as plt
import streamlit as st
import io
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ppauto_pos.csv'
raw = pd.read_csv(RAW_CSV_PATH)

# for testing
VALUATION_YEAR = 2007
COMPANY_CODE = 43
FINAL_DEVELOPMENT_LAG = 10
PATTERN_SOURCE = 'company'
METHOD_NAME = 'paid_chain_ladder'
FACTOR_AVERAGE = 'volume'

metrics = {
    'age_to_lag_factor': 'Age-to-Lag Factor',
    'estimated_cumulative_paid_lag_10': 'Cumulative Paid at Lag 10',
    'estimated_reserve': 'Estimated Reserve',
    'error': 'Net Error',
    'error': 'Error Percentage'
}

def start_visualization(style, COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR):
    company_specific_data, company_industry_data = get_data(COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
    print('hi')
    if style == 'classical':
        classic_visualization(company_specific_data, company_industry_data, metrics)
    elif style == 'interactive':
        interactive_visualization(company_specific_data, company_industry_data, COMPANY_CODE)
        

def get_data(COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR):
    company_specific_data = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates' / f'company_{COMPANY_CODE}_{METHOD_NAME}_{FACTOR_AVERAGE}_company_as_of_{VALUATION_YEAR}.csv', index_col=0)
    company_industry_data = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates' / f'company_{COMPANY_CODE}_{METHOD_NAME}_{FACTOR_AVERAGE}_industry_as_of_{VALUATION_YEAR}.csv', index_col=0)

    return company_specific_data, company_industry_data


# company_specific_data, company_industry_data = get_data(COMPANY_CODE, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
# visualization(company_specific_data, company_industry_data)


def classic_visualization(company, industry, metrics):
    n =len(metrics) #number of graphs
    rows = min(2,n)
    cols = math.ceil(n/rows)
    fig, axes = plt.subplots(rows,cols , figsize=(7 * cols, 5 * rows))

    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(
            company.index,
            company[metric],
            marker='o',
            label='Company-specific'
        )
        ax.plot(industry.index,
                industry[metric],
                marker='o',
                label='Company-wide'
        )

        ax.set_xlabel('Accident Year')
        ax.set_ylabel(metric)
        ax.set_title(f'Company vs Industry: {metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# classic_visualization(company_specific_data, company_industry_data, ['age_to_lag_factor', 'estimated_cumulative_paid_lag_10', 'estimated_reserve', 'error'])


def interactive_visualization(company, industry, COMPANY_CODE):

    #user selection
    metric = st.selectbox(
        'Select metric',
        ['age_to_lag_factor', 'estimated_cumulative_paid_lag_10', 'estimated_reserve', 'error', 'error_percentage']
        ) 

    # #save results
    # result = company[[metric]].copy()
    # result.index.name = 'Accident Year'

    # csv = result.to_csv().encode('utf-8')
    # st.download_button(
    #     label='Download company results',
    #     data=csv,
    #     file_name=f'{COMPANY_CODE},{metric}.csv',
    #     mime='text/csv'
    # )



    #create graph
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(
        company.index,
        company[metric],
        marker='o',
        label='Company-specific'
    )
    ax.plot(industry.index,
            industry[metric],
            marker='o',
            label='industry-wide'
    )
    ax.set_xlabel("Accident Year")
    ax.set_ylabel(metric)
    ax.set_title(f"Company vs Industry: {metric}")

    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    img_buffer.seek(0)

    st.download_button(
        label="Download graph",
        data=img_buffer,
        file_name=f"{COMPANY_CODE}_{metric}.png",
        mime="image/png"
    )