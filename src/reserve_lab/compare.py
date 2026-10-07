from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st
import io
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

def start_visualization(style, company_code, method_name, fator_average, valuation_year, download=False):
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
        #both figures are built (and saved to disk) before either window
        #opens - matplotlib's plt.show() blocks until every currently-open
        #figure's window is closed, so calling it once here after both
        #exist pops both windows up together, instead of showing the first
        #and blocking until it's closed before the second is even created.
        classic_visualization(company_specific_data, company_industry_data, company_code, method_name, fator_average, valuation_year, download=download)
        plot_company_vs_median_paid(company_specific_data, company_code, method_name, fator_average, valuation_year, download=download)
        plt.show()
    elif style == 'interactive':
        interactive_visualization(company_specific_data, company_industry_data, company_code)
        

# def get_data(company_code, method_name, factor_average, valuation_year):
#     company_specific_data = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates' / f'company_{company_code}_{method_name}_{factor_average}_company_as_of_{valuation_year}.csv', index_col=0)
#     company_industry_data = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reserve_estimates' / f'company_{company_code}_{method_name}_{factor_average}_industry_as_of_{valuation_year}.csv', index_col=0)

#     return company_specific_data, company_industry_data


# company_specific_data, company_industry_data = get_data(company_code, METHOD_NAME, FACTOR_AVERAGE, VALUATION_YEAR)
# visualization(company_specific_data, company_industry_data)


#draws the cumulative-paid-loss lines (Company-specific / Industry-wide /
#Actual-paid / Blended) onto a given ax - split out from
#classic_visualization so the exact same drawing code can be reused for
#one company's own standalone chart and for one company's panel inside
#generate_all_companies_pdf's combined PDF, without duplicating it.
def _draw_cumulative_paid_loss(ax, company, industry, company_code):
    ax.plot(company.index, company['estimated_cumulative_paid'], marker='o', label='Company-specific')
    ax.plot(industry.index, industry['estimated_cumulative_paid'], marker='o', label='Industry-wide')
    ax.plot(company.index, company['actual_paid'], marker='o', label='Actual-paid')
    #blended_estimated_cumulative_paid_rmse only exists once analysis.py
    #has run (it applies the backtested best r on top of the
    #company/industry factors already plotted above) - guard so this
    #still works if analysis hasn't run yet.
    if 'blended_estimated_cumulative_paid_rmse' in company.columns:
        ax.plot(
            company.index, company['blended_estimated_cumulative_paid_rmse'],
            marker='o', linestyle='--', label='Blended (best r, RMSE)',
        )

    ax.set_xlabel('Accident Year')
    ax.set_ylabel('Cumulative Paid Loss ($)')
    ax.set_title(f'Company {company_code} vs Industry: Cumulative Paid Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)


#used to build the whole 5-panel grid (age_to_lag_factor,
#estimated_cumulative_paid, estimated_reserve, error, error_percentage);
#now only the cumulative-paid-loss panel is worth saving, so this is the
#one chart classic_visualization draws.
#
#download=False (the default) only shows the chart - nothing is written
#to reports/figures/. Pass download=True (scripts/visualize.py's
#--download flag) to also save the PNG - saving is opt-in so exploring
#companies one at a time doesn't silently litter reports/figures/ with a
#PNG per company looked at.
def classic_visualization(company, industry, company_code, method_name, factor_average, valuation_year, download=False):
    fig, ax = plt.subplots(figsize=(6.3, 4.2))

    # thin_history_flag/num_accident_years come from manual_chain_ladder's
    # estimates dataframe (constant across every row of one company), so
    # .iloc[0] is enough - only missing for results saved before this flag
    # existed, hence the column check.
    if 'thin_history_flag' in company.columns and bool(company['thin_history_flag'].iloc[0]):
        num_years = int(company['num_accident_years'].iloc[0])
        fig.suptitle(
            f'⚠ THIN DATA: only {num_years} accident year(s) observed for this company '
            '- estimates below may be unreliable',
            color='firebrick', fontsize=12, fontweight='bold',
        )

    _draw_cumulative_paid_loss(ax, company, industry, company_code)

    if 'thin_history_flag' in company.columns and bool(company['thin_history_flag'].iloc[0]):
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # leave room for the suptitle warning above
    else:
        plt.tight_layout()

    if not download:
        return None

    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'company_{company_code}_cumulative_paid_loss_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f'!!! cumulative paid loss plot saved to:\n{figure_path}')

    return figure_path


#draws the indexed company-vs-median trend lines onto a given ax - split
#out from plot_company_vs_median_paid for the same reason
#_draw_cumulative_paid_loss was: the same drawing code backs both that
#function's standalone chart and this company's panel inside
#generate_all_companies_pdf's combined PDF.
#
#the two series can differ by orders of magnitude (a small company vs.
#hundreds of companies' median - see compute_industry_paid_benchmark),
#so plotting both in raw dollars on one axis flattens the smaller line
#into an unreadable streak near zero, and a second y-axis would be worse:
#a dual-axis chart's two scales line up arbitrarily, which invents
#a correlation that isn't really there (see the dataviz skill's
#anti-patterns doc). Instead, both series are indexed to 100 at the
#first accident year they share, so what's plotted is each one's %
#movement from that common starting point - which is what this chart is
#for. The raw-dollar comparison is a different question, already covered
#by _draw_cumulative_paid_loss's chart.
def _draw_median_trend(ax, company, company_code, benchmark):
    common_years = company.index.intersection(benchmark.index)
    if common_years.empty:
        raise ValueError(
            f'Company {company_code} and the industry benchmark share no accident year in common - cannot index.'
        )
    base_year = common_years.min()

    company_indexed = company['actual_paid'] / company['actual_paid'].loc[base_year] * 100
    median_indexed = benchmark['median_ultimate_paid'] / benchmark['median_ultimate_paid'].loc[base_year] * 100

    ax.plot(company_indexed.index, company_indexed, marker='o', label=f'Company {company_code} (actual paid)')
    ax.plot(
        median_indexed.index, median_indexed,
        marker='s', linestyle='--', color='tab:orange', label='Median across all companies',
    )
    ax.axhline(100, color='gray', linewidth=1, linestyle=':', alpha=0.6)
    ax.set_xlabel('Accident Year')
    ax.set_ylabel(f'Cumulative Paid Loss, indexed ({base_year} = 100)')
    ax.set_title(f'Company {company_code} vs. Industry Median: Cumulative Paid Loss Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)


#this company's own actual ultimate paid loss (already sitting in
#company_specific results from ut.get_results) against the industry
#benchmark analysis.compute_industry_paid_benchmark saved - literal
#dollars a "typical" company paid, not a chain-ladder projection (that's
#what the industry-wide line in classic_visualization's chart is for).
#Median, not mean: company size is heavily right-skewed (see
#compute_industry_paid_benchmark's docstring), so the mean line is left
#out entirely rather than included and likely to mislead.
#
#download=False (the default) only shows the chart - see
#classic_visualization's docstring for why saving is opt-in.
def plot_company_vs_median_paid(company, company_code, method_name, factor_average, valuation_year, download=False):
    benchmark_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / 'industry_paid_benchmark_by_accident_year.csv'
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f'{benchmark_path} does not exist yet - run analysis.compute_industry_paid_benchmark first.'
        )
    benchmark = pd.read_csv(benchmark_path, index_col=0)

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    _draw_median_trend(ax, company, company_code, benchmark)
    plt.tight_layout()

    if not download:
        return None

    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'company_{company_code}_vs_median_paid_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f'!!! company vs median paid plot saved to:\n{figure_path}')

    return figure_path


#one combined PDF, one page per company that has saved results, each page
#holding both of that company's charts (cumulative paid loss, and the
#indexed trend vs. the industry median) side by side, headed by the
#company code and name (ut.get_company_names) - the two charts
#classic_visualization/plot_company_vs_median_paid draw for one company
#at a time, batched across every company for a single document to skim
#or hand off. Reads only what scripts/run_pipeline.py already saved to
#disk (per-company results csvs, plus the industry_paid_benchmark_by_
#accident_year.csv from analysis.compute_industry_paid_benchmark) -
#doesn't recompute anything, so it's cheap to regenerate.
def generate_all_companies_pdf(method_name, factor_average, valuation_year, output_path=None):
    benchmark_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / 'industry_paid_benchmark_by_accident_year.csv'
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f'{benchmark_path} does not exist yet - run analysis.compute_industry_paid_benchmark first.'
        )
    benchmark = pd.read_csv(benchmark_path, index_col=0)
    company_names = ut.get_company_names()

    results_dir = PROJECT_ROOT / 'data' / 'processed' / 'results'
    company_codes = sorted(
        (int(p.name) for p in results_dir.iterdir() if p.is_dir()),
    )
    if not company_codes:
        raise RuntimeError('No company has saved results yet - run scripts/run_pipeline.py first.')

    if output_path is None:
        tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
        output_path = PROJECT_ROOT / 'reports' / 'figures' / f'all_companies_cumulative_paid_and_trend_{tag}.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, []
    with PdfPages(output_path) as pdf:
        for company_code in company_codes:
            try:
                company, industry = ut.get_results(company_code, method_name, factor_average, valuation_year)
            except FileNotFoundError:
                # this company was processed with a different
                # method/factor/valuation_year than the one asked for here
                skipped.append(company_code)
                continue

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.2, 4.2))
            name = company_names.get(company_code, 'Unknown company name')
            fig.suptitle(f'Company {company_code} — {name}', fontsize=14, fontweight='bold')

            if 'thin_history_flag' in company.columns and bool(company['thin_history_flag'].iloc[0]):
                num_years = int(company['num_accident_years'].iloc[0])
                fig.text(
                    0.5, 0.925,
                    f'⚠ THIN DATA: only {num_years} accident year(s) observed - estimates below may be unreliable',
                    color='firebrick', fontsize=10, fontweight='bold', ha='center',
                )

            _draw_cumulative_paid_loss(ax_left, company, industry, company_code)
            try:
                _draw_median_trend(ax_right, company, company_code, benchmark)
            except ValueError as exc:
                ax_right.axis('off')
                ax_right.text(0.5, 0.5, str(exc), ha='center', va='center', wrap=True)

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig)
            plt.close(fig)
            written += 1

    if skipped:
        print(f'WARNING: skipped {len(skipped)} company/companies with no results for this method/factor_average/valuation_year: {skipped}')
    print(f'\n!!! {written} companies\' charts saved to:\n{output_path}')
    return output_path


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
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
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

#the "credibility curve": mean AND median best_r (RMSE-optimal) per
#company-size decile, with the gap between them shaded. A decile where
#the median sits at 1.0 but the mean sits well below it means most
#companies in that group are fully trusted, but a real minority are
#fully distrusted, averaging out to something in between - very
#different from a decile where mean and median sit close together (a
#genuinely middling, unpolarized group). Each point is also annotated
#with its count, since a decile's average is only as trustworthy as how
#many (company, lag) pairs back it.
#
#RMSE-optimal only, not MAE - the two rarely disagreed enough to be
#worth a second panel, and compare.plot_share_vs_r (the --noise dot
#chart) already covers the same "does size predict best_r" question at
#a different, complementary granularity (one point per company there vs.
#one point per (company, lag) pair here) - no need for a third,
#overlapping scatter view (see plot_credibility_scatter's removal).
#
#download=False (the default) only shows the chart - see
#classic_visualization's docstring for why saving is opt-in everywhere
#in this module (this used to be the one holdout that always saved
#unconditionally - now consistent with the rest).
def plot_credibility_curve(method_name, factor_average, valuation_year, download=False):
    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    binned_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / f'credibility_vs_share_binned_{tag}.csv'
    if not binned_path.exists():
        raise FileNotFoundError(f'{binned_path} does not exist yet - run analysis.analyze_credibility_vs_size first.')
    binned = pd.read_csv(binned_path)
    subset = binned.loc[binned['ratio_column'] == 'best_r_w_rmse'].sort_values('decile')

    fig, ax = plt.subplots(figsize=(6.3, 4.2))

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
    ax.set_ylabel('best r (RMSE-optimal)')
    ax.set_title('Credibility curve: mean vs. median best_r (RMSE-optimal) by company-size decile')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if not download:
        return None

    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'credibility_vs_share_curve_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f'!!! credibility curve plot saved to:\n{figure_path}')
    return figure_path


# ---------------------------------------------------------------------
# noise/accuracy visualizations - all read
# noise_and_accuracy_by_company_{tag}.csv, which
# analysis.compute_noise_and_accuracy saves (one row per company: how
# noisy its own growth trend is, how accurate its own-pattern estimate
# is, its size, and its fitted credibility weight). Same
# compute-in-analysis/visualize-in-compare split as the credibility
# charts above, for the same reason (heavy stats stay usable/testable
# without matplotlib ever entering the picture).
# ---------------------------------------------------------------------

def _read_noise_and_accuracy(method_name, factor_average, valuation_year):
    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    path = PROJECT_ROOT / 'data' / 'processed' / 'results' / f'noise_and_accuracy_by_company_{tag}.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist yet - run analysis.compute_noise_and_accuracy first.')
    return pd.read_csv(path), tag


#one dot per company, on a given ax, with the Spearman rank correlation
#(not Pearson - company size/error/share here are heavily skewed over
#several orders of magnitude, so a rank-based measure is far less
#sensitive to a few extreme companies distorting the picture; see
#analysis.py's own note on this same choice for the credibility-vs-size
#analysis) annotated in the title. Log axes where the underlying
#quantity spans orders of magnitude (dollars, share) - a linear axis
#would crush the bulk of companies into a corner while a handful of
#much bigger/noisier ones stretch the rest of the scale flat.
def _dot_chart(ax, data, x_col, y_col, xlabel, ylabel, title, logx=False, logy=False):
    valid = data.dropna(subset=[x_col, y_col])
    if logx:
        valid = valid.loc[valid[x_col] > 0]
    if logy:
        valid = valid.loc[valid[y_col] > 0]
    rho, p = stats.spearmanr(valid[x_col], valid[y_col])

    ax.scatter(valid[x_col], valid[y_col], alpha=0.55, s=32, color='tab:blue', edgecolor='white', linewidth=0.4)

    # a straight best-fit line is a stronger, DIFFERENT claim than
    # Spearman's rho above - rho only says the two move together
    # monotonically, not that the relationship is a straight (or
    # log-straight) line. Drawn anyway, but fit in whatever space the
    # axis actually displays (log10 of a column that's log-scaled, raw
    # otherwise) so the line is honestly straight on the chart - fitting
    # on raw dollars and only log-scaling the axis for display draws a
    # curve, not the line its own slope/R^2 would describe. R^2 here is
    # a second, complementary number: how much of the pattern a single
    # straight/log-linear line explains, which can run well below rho
    # when the real relationship is there but a different shape
    # (outlier-driven, saturating, etc).
    x_fit = np.log10(valid[x_col]) if logx else valid[x_col]
    y_fit = np.log10(valid[y_col]) if logy else valid[y_col]
    slope, intercept, r_value, _, _ = stats.linregress(x_fit, y_fit)
    x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
    y_line = slope * x_line + intercept
    plot_x = 10 ** x_line if logx else x_line
    plot_y = 10 ** y_line if logy else y_line
    ax.plot(plot_x, plot_y, color='tab:orange', linewidth=1.8, zorder=4)

    # the fit was done in log10-space wherever an axis is log-scaled, so
    # "y = slope*x + intercept" is the equation of that fit, not of the
    # curve drawn above in real units - translate it back:
    #   neither logged  -> y = m*x + b               (ordinary line)
    #   only y logged   -> y = A * 10^(m*x)           (exponential)
    #   only x logged   -> y = m*log10(x) + b         (logarithmic)
    #   both logged     -> y = A * x^m                (power law)
    # where A = 10^intercept in the two exponential-form cases.
    if logx and logy:
        equation = f'y = {10**intercept:.3g} · x^{slope:.3f}'
    elif logy:
        equation = f'y = {10**intercept:.3g} · 10^({slope:.3f}·x)'
    elif logx:
        equation = f'y = {slope:.3f}·log10(x) + {intercept:.3f}'
    else:
        equation = f'y = {slope:.3f}·x + {intercept:.3f}'

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f'{title}\n'
        f'Spearman rho={rho:.3f} (p={p:.2e}), n={len(valid)}  |  best-fit line R²={r_value**2:.3f}',
        fontsize=10,
    )
    ax.text(
        0.03, 0.96, equation, transform=ax.transAxes, ha='left', va='top', fontsize=9.5, color='tab:orange',
        bbox=dict(facecolor='white', edgecolor='tab:orange', alpha=0.8, boxstyle='round,pad=0.3'),
    )
    ax.grid(True, alpha=0.3, which='both' if (logx or logy) else 'major')


#does a noisier company (see analysis.compute_noise_and_accuracy for the
#exact log-linear-trend-residual definition) have a less accurate
#own-pattern estimate? Answers this directly - see the walkthrough in
#the conversation that led here for why this correlation held up
#(rho~0.57-0.60) even though best_r itself barely responds to noise at
#all (noise inflates the company-only AND industry-only error together,
#which cancels out of the company-vs-industry comparison best_r is
#actually fit on).
def plot_noise_vs_error(method_name, factor_average, valuation_year, download=False):
    data, tag = _read_noise_and_accuracy(method_name, factor_average, valuation_year)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    _dot_chart(
        ax, data, 'residual_noise_std', 'company_only_rmse_pct',
        'Company noise (residual std around own trend, log-dollar space)',
        'Company-only prediction error, walk-forward RMSE % (log scale)',
        'Company noise vs. company-only prediction error',
        logx=False, logy=True,
    )
    plt.tight_layout()

    if not download:
        return None
    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'noise_vs_error_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f'!!! noise vs error plot saved to:\n{figure_path}')
    return figure_path


#does a bigger company's share of industry paid losses predict a higher
#fitted credibility weight? (one dot per company here - mean best_r and
#mean share averaged across that company's own lags - a coarser, more
#readable companion to plot_credibility_scatter's per-(company,lag)
#version, matching the granularity of plot_noise_vs_error/
#plot_share_vs_error above and below so all three are directly
#comparable.)
def plot_share_vs_r(method_name, factor_average, valuation_year, download=False):
    data, tag = _read_noise_and_accuracy(method_name, factor_average, valuation_year)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    _dot_chart(
        ax, data, 'mean_company_share', 'mean_best_r_rmse',
        "Company's mean share of industry paid losses (log scale)",
        "Mean fitted best_r (RMSE-optimal), across that company's lags",
        'Company share vs. credibility weight (best_r)',
        logx=True, logy=False,
    )
    plt.tight_layout()

    if not download:
        return None
    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'share_vs_r_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f'!!! share vs credibility-weight plot saved to:\n{figure_path}')
    return figure_path


#does a bigger company's share of industry paid losses predict a more
#accurate own-pattern estimate? (the same size-vs-error relationship
#that first surfaced as the raw "deviation from median" analysis,
#rebuilt directly from company_only_rmse_pct for a cleaner comparison
#against plot_noise_vs_error above.)
def plot_share_vs_error(method_name, factor_average, valuation_year, download=False):
    data, tag = _read_noise_and_accuracy(method_name, factor_average, valuation_year)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    _dot_chart(
        ax, data, 'mean_company_share', 'company_only_rmse_pct',
        "Company's mean share of industry paid losses (log scale)",
        'Company-only prediction error, walk-forward RMSE % (log scale)',
        'Company share vs. company-only prediction error',
        logx=True, logy=True,
    )
    plt.tight_layout()

    if not download:
        return None
    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'share_vs_error_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f'!!! share vs error plot saved to:\n{figure_path}')
    return figure_path


#the "noise curve": mean AND median residual_noise_std per company-size
#decile - mirrors plot_credibility_curve's design exactly (same
#mean/median-plus-shaded-gap, same per-decile n annotation) but reads
#analysis.analyze_noise_vs_size's csv and only needs one panel (noise
#has no MAE/RMSE-optimal split the way best_r does).
def plot_noise_curve(method_name, factor_average, valuation_year, download=False):
    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    binned_path = PROJECT_ROOT / 'data' / 'processed' / 'results' / f'noise_vs_size_binned_{tag}.csv'
    if not binned_path.exists():
        raise FileNotFoundError(f'{binned_path} does not exist yet - run analysis.analyze_noise_vs_size first.')
    binned = pd.read_csv(binned_path).sort_values('decile')

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(binned['decile'], binned['mean'], marker='o', label='mean', color='tab:blue')
    ax.plot(binned['decile'], binned['median'], marker='s', linestyle='--', label='median', color='tab:orange')
    ax.fill_between(binned['decile'], binned['mean'], binned['median'], color='gray', alpha=0.15)
    for _, row in binned.iterrows():
        ax.annotate(
            f"n={int(row['count'])}", (row['decile'], max(row['mean'], row['median']) + 0.03),
            ha='center', fontsize=8, color='dimgray',
        )

    ax.set_xticks(binned['decile'])
    ax.set_xlabel('company-size decile (1=smallest 10% of companies, 10=biggest 10%)')
    ax.set_ylabel('residual noise (log-dollar space)')
    ax.set_title('Noise curve: mean vs. median noise-around-own-trend by company-size decile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if not download:
        return None
    figure_path = PROJECT_ROOT / 'reports' / 'figures' / f'noise_curve_{tag}.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f'!!! noise curve plot saved to:\n{figure_path}')
    return figure_path