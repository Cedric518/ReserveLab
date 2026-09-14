from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import math
from scipy import stats
from . import utilities as ut
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# for testing
# VALUATION_YEAR = 2007
# COMPANY_CODE = 43
# FINAL_DEVELOPMENT_LAG = 10
# PATTERN_SOURCE = 'company'
# METHOD_NAME = 'paid_chain_ladder'
# FACTOR_AVERAGE = 'volume'

metrics = {
    'age_to_lag_factor': 'Age-to-Lag Factor',
    'estimated_cumulative_paid': 'Cumulative Paid at Lag 10',
    'estimated_reserve': 'Estimated Reserve',
    'error': 'Net Error',
    'error_percentage': 'Error Percentage',  # was a duplicate 'error' key before - silently
}                                            # dropped the Net Error panel since a dict can't
                                             # hold the same key twice; the second value just
                                             # overwrote the first.
def start_visualization(style, company_code, method_name, fator_average, valuation_year):
    company_specific_data, company_industry_data = ut.get_results(company_code, method_name, fator_average, valuation_year)
    print('company_specific_data')
    print(company_specific_data)
    print(company_specific_data.dtypes)
    print(company_specific_data.index)
    print(company_specific_data.columns)

    print("\n=== INDUSTRY ===")
    print(company_industry_data)
    print(company_industry_data.dtypes)
    print(company_industry_data.index)
    print(company_industry_data.columns)
    if style == 'classical':
        classic_visualization(company_specific_data, company_industry_data, metrics)
    elif style == 'interactive':
        interactive_visualization(company_specific_data, company_industry_data, company_code)
        

# def get_data(company_code, method_name, factor_average, valuation_year):
#     company_specific_data = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates' / f'company_{company_code}_{method_name}_{factor_average}_company_as_of_{valuation_year}.csv', index_col=0)
#     company_industry_data = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates' / f'company_{company_code}_{method_name}_{factor_average}_industry_as_of_{valuation_year}.csv', index_col=0)

#     return company_specific_data, company_industry_data


# company_specific_data, company_industry_data = get_data(company_code, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
# visualization(company_specific_data, company_industry_data)


def classic_visualization(company, industry, metrics):
    n =len(metrics) #number of graphs
    rows = min(2,n)
    cols = math.ceil(n/rows)
    fig, axes = plt.subplots(rows,cols , figsize=(7 * cols, 5 * rows))

    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # thin_history_flag/num_accident_years come from manual_chain_ladder's
    # estimates dataframe (constant across every row of one company), so
    # .iloc[0] is enough - only missing for results saved before this flag
    # existed, hence the column check.
    if 'thin_history_flag' in company.columns and bool(company['thin_history_flag'].iloc[0]):
        num_years = int(company['num_accident_years'].iloc[0])
        fig.suptitle(
            f'⚠ THIN DATA: only {num_years} accident year(s) observed for this company '
            '- estimates and best_r below may be unreliable',
            color='firebrick', fontsize=12, fontweight='bold',
        )

    for ax, metric in zip(axes, metrics):
        print(company[metric])
        ax.plot(
            company.index,
            company[metric],
            marker='o',
            label='Company-specific'
        )
        ax.plot(industry.index,
                industry[metric],
                marker='o',
                label='Industry-wide'
        )
        if metric == 'estimated_cumulative_paid':
            ax.plot(company.index,
                company['actual_paid'],
                marker='o',
                label='Actual-paid'
            )
            #blended_estimated_cumulative_paid_rmse only exists once
            #analysis.py has run (it applies the backtested best r on top
            #of the company/industry factors already plotted above) - guard
            #so this still works if analysis hasn't run yet.
            if 'blended_estimated_cumulative_paid_rmse' in company.columns:
                ax.plot(company.index,
                    company['blended_estimated_cumulative_paid_rmse'],
                    marker='o',
                    linestyle='--',
                    label='Blended (best r, RMSE)'
                )
        if metric == 'estimated_reserve':
            ax.plot(company.index,
                    company['actual_reserve'],
                    marker='o',
                    label='Actual-reserve'
            )
            if 'blended_estimated_reserve_rmse' in company.columns:
                ax.plot(company.index,
                    company['blended_estimated_reserve_rmse'],
                    marker='o',
                    linestyle='--',
                    label='Blended (best r, RMSE)'
                )



        ax.set_xlabel('Accident Year')
        ax.set_ylabel(metric)
        ax.set_title(f'Company vs Industry: {metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    if 'thin_history_flag' in company.columns and bool(company['thin_history_flag'].iloc[0]):
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # leave room for the suptitle warning above
    else:
        plt.tight_layout()
    plt.show()

# classic_visualization(company_specific_data, company_industry_data, ['age_to_lag_factor', 'estimated_cumulative_paid', 'estimated_reserve', 'error'])


def interactive_visualization(company, industry, COMPANY_CODE):

    if 'thin_history_flag' in company.columns and bool(company['thin_history_flag'].iloc[0]):
        num_years = int(company['num_accident_years'].iloc[0])
        st.warning(
            f'⚠ Only {num_years} accident year(s) of data observed for this company - '
            'estimates and best_r may be unreliable.'
        )

    #user selection
    metric = st.selectbox(
        'Select metric',
        ['age_to_lag_factor', 'estimated_cumulative_paid', 'estimated_reserve', 'error', 'error_percentage']
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


# ---------------------------------------------------------------------
# cross-company credibility-vs-size visualizations. Both of these read
# csvs that analysis.analyze_credibility_vs_size already saved to disk -
# neither one recomputes anything, so they're cheap to re-run while
# tweaking the chart, and work even if the analysis was run in a
# different session. This split (analysis.py computes and saves csvs,
# compare.py is the only module that turns them into a chart) keeps the
# heavy stats logic usable and testable without matplotlib ever entering
# the picture.
# ---------------------------------------------------------------------

_CREDIBILITY_LABELS = {'best_r_w_mae': 'best r (MAE-optimal)', 'best_r_w_rmse': 'best r (RMSE-optimal)'}


#scatter of company_share_of_industry vs best_r for every pooled
#(company, development_lag) pair, one panel per metric, with the
#Spearman rank correlation (see analysis.analyze_credibility_vs_size for
#why Spearman rather than Pearson is the more informative number here)
#annotated in each panel's title.
def plot_credibility_scatter(method_name, factor_average, valuation_year, min_observations=2):
    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    pooled_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / f'credibility_vs_share_pooled_{tag}.csv'
    if not pooled_path.exists():
        raise FileNotFoundError(f'{pooled_path} does not exist yet - run analysis.analyze_credibility_vs_size first.')
    pooled = pd.read_csv(pooled_path)
    reliable = pooled.loc[pooled['num_observations'] >= min_observations]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, ratio_column in zip(axes, ['best_r_w_mae', 'best_r_w_rmse']):
        valid = reliable.dropna(subset=['company_share_of_industry', ratio_column])
        valid = valid.loc[valid['company_share_of_industry'] > 0]
        spearman_r, spearman_p = stats.spearmanr(valid['company_share_of_industry'], valid[ratio_column])

        ax.scatter(valid['company_share_of_industry'], valid[ratio_column], alpha=0.35, s=18)
        ax.set_xscale('log')
        ax.set_xlabel('company share of industry paid losses (log scale)')
        ax.set_ylabel(_CREDIBILITY_LABELS[ratio_column])
        ax.set_title(f'{_CREDIBILITY_LABELS[ratio_column]}\nSpearman rho={spearman_r:.3f} (p={spearman_p:.3g}), n={len(valid)}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'credibility_vs_share_scatter_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'!!! scatter plot saved to:\n{figure_path}')
    return figure_path


#the "credibility curve": mean AND median best_r per company-size decile,
#with the gap between them shaded. A decile where the median sits at 1.0
#but the mean sits well below it means most companies in that group are
#fully trusted, but a real minority are fully distrusted, averaging out
#to something in between - very different from a decile where mean and
#median sit close together (a genuinely middling, unpolarized group).
#Each point is also annotated with its count, since a decile's average is
#only as trustworthy as how many (company, lag) pairs back it.
def plot_credibility_curve(method_name, factor_average, valuation_year):
    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    binned_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / f'credibility_vs_share_binned_{tag}.csv'
    if not binned_path.exists():
        raise FileNotFoundError(f'{binned_path} does not exist yet - run analysis.analyze_credibility_vs_size first.')
    binned = pd.read_csv(binned_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, ratio_column in zip(axes, ['best_r_w_mae', 'best_r_w_rmse']):
        subset = binned.loc[binned['ratio_column'] == ratio_column].sort_values('decile')

        ax.plot(subset['decile'], subset['mean'], marker='o', label='mean', color='tab:blue')
        ax.plot(subset['decile'], subset['median'], marker='s', linestyle='--', label='median', color='tab:orange')
        ax.fill_between(subset['decile'], subset['mean'], subset['median'], color='gray', alpha=0.15)

        for _, row in subset.iterrows():
            ax.annotate(
                f"n={int(row['count'])}", (row['decile'], 1.03),
                ha='center', fontsize=8, color='dimgray',
            )

        ax.set_ylim(0, 1.12)
        ax.set_xticks(subset['decile'])
        ax.set_xlabel('company-size decile (1=smallest 10% of company/lag pairs, 10=biggest 10%)')
        ax.set_ylabel(_CREDIBILITY_LABELS[ratio_column])
        ax.set_title(_CREDIBILITY_LABELS[ratio_column])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Credibility curve: mean vs. median best_r by company-size decile')
    plt.tight_layout()

    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'credibility_vs_share_curve_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'!!! credibility curve plot saved to:\n{figure_path}')
    return figure_path