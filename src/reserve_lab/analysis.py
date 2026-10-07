import numpy as np
import pandas as pd
from scipy import stats
from . import utilities as ut
from . import build_loss_structures as build
from . import manual_chain_ladder as mcl

#this function is called by main to execute this whole thing.
#Rather than backtesting a single valuation date - which pins every
#development lag to exactly one accident year, see the note on
#run_walk_forward_backtests below - this walks the valuation date forward
#one year at a time and pools the results, so each development lag ends up
#backed by several independent observations instead of one.
def start_analysis(company_code, method_name, factor_average, valuation_year):
    print('START ANALYSIS WAS RUN')
    pooled_company, pooled_industry = run_walk_forward_backtests(company_code, valuation_year)

    weighting_by_lag = calculate_best_ratio(pooled_company, pooled_industry)
    print(weighting_by_lag)

    #this is a per-development-lag summary, not a per-accident-year table,
    #so it gets its own output file instead of being merged into the
    #company/industry estimates csvs.
    weighting_path = ut.get_weighting_path(company_code, method_name, factor_average, valuation_year)
    weighting_path.parent.mkdir(parents=True, exist_ok=True)
    weighting_by_lag.to_csv(weighting_path, index_label='development_lag')
    print(f'\n!!! company vs industry weighting by lag saved to:\n{weighting_path}')

    #now actually use the freshly-fit ratios: blend the target valuation
    #year's own company/industry factors and add the resulting estimated
    #cumulative paid and reserve onto the existing results file, so both
    #the results csv and the visualization can show it.
    target_company_path, _ = ut.get_results_paths(company_code, method_name, factor_average, valuation_year)
    target_company, target_industry = ut.get_results(company_code, method_name, factor_average, valuation_year)
    target_company = apply_best_ratio(target_company, target_industry, weighting_by_lag)
    ut.save_data(company_data=target_company, company_path=target_company_path)
    print(f'\n!!! blended estimates (best r applied) added to:\n{target_company_path}')

    return weighting_by_lag


#blend the target valuation year's own company and industry factors using
#the ratio backtested for each accident year's current development lag, and
#project the resulting cumulative paid and reserve - the same chain-ladder
#formula used everywhere else in this project, just with a credibility-
#weighted factor instead of a pure company or pure industry one:
#   blended_factor           = r * company_factor + (1-r) * industry_factor
#   blended_cumulative_paid  = last_observed_paid * blended_factor
#   blended_reserve          = blended_cumulative_paid - last_observed_paid
#done once using the MAE-optimal r, once using the RMSE-optimal r, since
#weighting_by_lag carries both and they can disagree.
def apply_best_ratio(target_company, target_industry, weighting_by_lag):
    target_company = target_company.copy()

    for metric_name, ratio_column in [('mae', 'best_r_w_mae'), ('rmse', 'best_r_w_rmse')]:
        r = target_company['latest_observed_lag'].map(weighting_by_lag[ratio_column])
        # a lag with no informative observations (company and industry
        # factor always agreed - see calculate_best_ratio's filtering,
        # which always excludes the terminal lag: at the last development
        # lag there's nothing left to develop into, so both factors are
        # trivially 1.0) has no row in weighting_by_lag at all, so r comes
        # back NaN for it. r is genuinely meaningless there - blending is
        # a no-op when company_factor == industry_factor, so ANY r gives
        # the same blended_factor - a neutral 0.5 avoids propagating NaN
        # into the saved estimate without pretending r means anything.
        r = r.fillna(0.5)
        blended_factor = r * target_company['age_to_lag_factor'] + (1 - r) * target_industry['age_to_lag_factor']
        blended_cumulative_paid = target_company['last_observed_paid'] * blended_factor
        blended_reserve = blended_cumulative_paid - target_company['last_observed_paid']

        target_company[f'blended_r_{metric_name}'] = r
        target_company[f'blended_estimated_cumulative_paid_{metric_name}'] = blended_cumulative_paid
        target_company[f'blended_estimated_reserve_{metric_name}'] = blended_reserve

    return target_company


#regenerate the triangles/squares/estimates at every valuation year from
#the earliest one with any development history through `valuation_year`,
#and pool the resulting per-accident-year estimate tables into two long
#tables (one company-pattern, one industry-pattern), each row tagged with
#the valuation year that produced it.
#
#why this is necessary: with a single valuation year, latest_observed_lag =
#valuation_year - accident_year + 1 is a strict one-to-one mapping, so every
#development lag is backed by exactly one accident year - not enough to fit
#anything (see the walk-through in the previous analysis.py write-up).
#Walking the valuation date forward lets the SAME accident year be observed
#at many different lags over time (once per valuation year it lives
#through), and lets DIFFERENT accident years land on the SAME lag as of
#different valuation years - e.g. lag 1 is "1999 as seen at the end of
#1999", "2000 as seen at the end of 2000", etc. - so each lag bucket ends
#up with multiple, genuinely different, historical data points. This is the
#standard "walk-forward" / rolling-origin backtest used to validate a
#time-indexed model out of sample: at each stop, only information available
#as of that valuation date is used to predict, and the real eventual paid
#amount (always fully known in this historical dataset) is used purely to
#grade that prediction afterward.
def run_walk_forward_backtests(company_code, valuation_year):
    clean = pd.read_csv(ut.get_clean_data_path())
    company_accident_years = clean.loc[clean['company_code'] == company_code, 'accident_year']
    first_accident_year = int(company_accident_years.min())

    #need at least two development lags observed to fit even one age-to-age
    #factor, so the earliest usable valuation year is one year after the
    #company's first accident year
    first_valuation_year = first_accident_year + 1
    if first_valuation_year > valuation_year:
        raise ValueError(
            f'valuation_year={valuation_year} is too early for company {company_code}: '
            f'need at least valuation_year={first_valuation_year} to observe two development lags.'
        )

    company_frames = []
    industry_frames = []
    for vy in range(first_valuation_year, valuation_year + 1):
        #only the final, real valuation year's triangle/square/estimates
        #are an actual deliverable worth keeping on disk - every earlier
        #year in this loop is pure backtest scratch work, so skip writing
        #(and re-reading) those to/from csv entirely.
        is_target_year = (vy == valuation_year)
        try:
            company_triangle, company_square, industry_triangle, _ = build.build_strcture(
                company_code, vy, persist=is_target_year
            )
            # industry_triangle columns are still whatever development_lag's
            # native dtype is (int) at this point, straight from
            # build_strcture - get_last_observed_paid needs that to look up
            # each accident year's current lag as a column key.
            industry_last_observed_paid = mcl.get_last_observed_paid(industry_triangle, vy)
            # persist=False always here, even for the target year: main.py
            # (or whoever called start_analysis) already computed and saved
            # the target year's raw estimates before analysis ran, and
            # backtesting.py adds its own error/error_percentage columns on
            # top of that same file. Re-saving raw estimates from here would
            # silently wipe out those columns. apply_best_ratio(), below,
            # is the only thing in this module that touches that file, and
            # it does so by reading-modifying-saving, same as backtesting.py.
            company_est = mcl.start_calculating(
                company_code, vy, 'company',
                triangles=(company_triangle, industry_triangle), square=company_square,
                persist=False,
            )
            industry_est = mcl.start_calculating(
                company_code, vy, 'industry',
                triangles=(company_triangle, industry_triangle), square=company_square,
                persist=False,
            )
        except Exception as exc:
            #one thin/unlucky valuation year (e.g. a zero denominator in an
            #early, sparse triangle) shouldn't take down the whole backtest
            print(f'WARNING: skipping valuation_year={vy}, failed with: {exc}')
            continue
        # tag on how much the whole industry (every company combined) had
        # paid at this same accident year/lag, so later we can measure what
        # share of that total this one company represents. Assigning a
        # Series aligns by index (accident_year) automatically.
        company_est = company_est.copy()
        company_est['industry_last_observed_paid'] = industry_last_observed_paid

        company_frames.append(company_est.reset_index())
        industry_frames.append(industry_est.reset_index())

    if not company_frames:
        raise RuntimeError('No valuation year produced usable estimates - nothing to analyze.')

    pooled_company = pd.concat(company_frames, ignore_index=True)
    pooled_industry = pd.concat(industry_frames, ignore_index=True)
    return pooled_company, pooled_industry


#for one development lag, solve for the blend ratio r that minimizes MAE and
#the ratio that minimizes RMSE, using every observation available at that
#lag (now one per accident-year/valuation-year pair that reached it - see
#run_walk_forward_backtests).
#
#the blended lag-10 estimate for a company ratio r (industry ratio 1-r) is:
#   estimate(r) = last_observed_paid * (r * company_factor + (1-r) * industry_factor)
#   error(r)    = estimate(r) - actual_paid
#               = b + r * x
#   where  x = last_observed_paid * (company_factor - industry_factor)
#          b = last_observed_paid * industry_factor - actual_paid   (error at r=0)
#
#RMSE^2 is proportional to sum((b + r*x)^2), an upward parabola in r, so it
#has exactly one minimum, solvable with calculus instead of a search grid:
#   r* = -sum(x*b) / sum(x^2), clipped to [0, 1] since r must be a valid weight
#
#MAE = mean(|b + r*x|) is convex and piecewise-linear in r. A sum of "kinked"
#absolute-value terms like this always has its minimum at r=0, r=1, or at one
#of the kinks -b_i/x_i where a single observation's error crosses zero, so
#checking those candidates finds the exact minimizer without scanning
#thousands of r values.
def _fit_ratio(x, b):
    # RMSE: exact closed-form minimizer of a convex quadratic in r
    sum_x2 = np.sum(x ** 2)
    if sum_x2 > 0:
        r_rmse = float(np.clip(-np.sum(x * b) / sum_x2, 0.0, 1.0))
    else:
        # company and industry factors give identical predictions here,
        # so RMSE does not depend on r at all
        r_rmse = np.nan
        #error **2
    best_rmse = float(np.sqrt(np.mean((b + (0.0 if np.isnan(r_rmse) else r_rmse) * x) ** 2)))

    # MAE: exact minimizer of a convex, piecewise-linear function of r
    candidates = {0.0, 1.0}
    nonzero = x != 0
    candidates.update(np.clip(-b[nonzero] / x[nonzero], 0.0, 1.0).tolist())
    candidates = np.array(sorted(candidates))
    mae_at_candidates = np.abs(b[None, :] + candidates[:, None] * x[None, :]).mean(axis=1)
    best_idx = np.argmin(mae_at_candidates)
    r_mae = float(candidates[best_idx])
    best_mae = float(mae_at_candidates[best_idx])

    return r_mae, best_mae, r_rmse, best_rmse


def calculate_best_ratio(company, industry):
    # match company-pattern and industry-pattern rows for the same accident
    # year observed as of the same valuation year. Now that several
    # valuation years are pooled together, accident_year alone is no
    # longer a unique key (each accident year appears once per valuation
    # year it lived through), so join on the full (accident_year,
    # valuation_year) pair instead of aligning on the bare index.
    merged = pd.merge(
        company,
        industry,
        on=['accident_year', 'valuation_year', 'latest_observed_lag'],
        suffixes=('_company', '_industry'),
    )

    results = []
    development_lags = sorted(merged['latest_observed_lag'].dropna().unique())

    for lag in development_lags:
        subset = merged.loc[merged['latest_observed_lag'] == lag]
        if subset.empty:
            continue

        # a row where the company's own factor and the industry's factor
        # are (essentially) identical can't tell us anything about which
        # one to trust more: blending them with ANY ratio r gives the
        # exact same prediction, so that row's error is fixed no matter
        # what r ends up fitted. This isn't rare - it happens whenever
        # neither the company nor the industry has yet lived past the lag
        # being projected (see run_walk_forward_backtests' docstring):
        # both factors fall back to the same "assume no more development"
        # value, not because the company and industry patterns genuinely
        # agree. Keeping such a row would only pad the observation count
        # and inflate the reported error with a mismatch r can't fix, so
        # it's dropped before fitting rather than left in the dataset.
        company_factor = subset['age_to_lag_factor_company'].to_numpy(dtype=float)
        industry_factor = subset['age_to_lag_factor_industry'].to_numpy(dtype=float)
        informative = ~np.isclose(company_factor, industry_factor)
        num_excluded = int((~informative).sum())
        subset = subset[informative]
        if subset.empty:
            print(
                f'WARNING: development lag {lag} has no informative observations left '
                '(company and industry factor always agreed, likely from truncated early '
                'valuation years) - skipping this lag entirely.'
            )
            continue

        last_observed = subset['last_observed_paid_company'].to_numpy(dtype=float)
        company_factor = subset['age_to_lag_factor_company'].to_numpy(dtype=float)
        industry_factor = subset['age_to_lag_factor_industry'].to_numpy(dtype=float)
        actual = subset['actual_paid_company'].to_numpy(dtype=float)

        x = last_observed * (company_factor - industry_factor)
        b = last_observed * industry_factor - actual

        r_mae, best_mae, r_rmse, best_rmse = _fit_ratio(x, b)

        # what fraction of the WHOLE industry's paid losses (every company
        # combined, at this same accident year/lag) does this one company
        # represent? Averaged across the observations backing this lag -
        # a volume/credibility proxy, for correlating against best_r later
        # (see analyze_credibility_vs_size).
        industry_paid = subset['industry_last_observed_paid'].to_numpy(dtype=float)
        company_share_of_industry = float(np.mean(last_observed / industry_paid))

        # baselines for context: is blending actually better than picking
        # one side outright?
        mae_industry_only = float(np.abs(b).mean())
        mae_company_only = float(np.abs(b + x).mean())
        rmse_industry_only = float(np.sqrt(np.mean(b ** 2)))
        rmse_company_only = float(np.sqrt(np.mean((b + x) ** 2)))

        results.append({
            'latest_observed_lag': lag,
            'num_observations': len(subset),
            'num_excluded_uninformative': num_excluded,
            'accident_years': sorted(subset['accident_year'].unique().tolist()),
            'company_share_of_industry': company_share_of_industry,
            'best_r_w_mae': r_mae,
            'best_mae': best_mae,
            'mae_industry_only': mae_industry_only,
            'mae_company_only': mae_company_only,
            'best_r_w_rmse': r_rmse,
            'best_rmse': best_rmse,
            'rmse_industry_only': rmse_industry_only,
            'rmse_company_only': rmse_company_only,
        })

    result_df = pd.DataFrame(results).set_index('latest_observed_lag')

    low_n = result_df['num_observations'] < 2
    if low_n.any():
        print(
            'WARNING: development lag(s) '
            f'{result_df.index[low_n].tolist()} still have fewer than 2 observations backing them, '
            'even after pooling valuation years. Treat their fitted ratio as unreliable - it exactly '
            'fits a single historical point rather than reflecting a genuine pattern.'
        )

    return result_df


#the credibility-theory hypothesis this project has been building toward:
#does a company that represents a bigger share of the whole industry's paid
#losses (company_share_of_industry, from calculate_best_ratio above) also
#get a higher fitted credibility weight (best_r)? Pools every (company,
#development_lag) pair from every company that run_pipeline.py has already
#produced a weighting-by-lag file for - it does NOT recompute anything, it
#only reads what's already on disk, so run the pipeline for however many
#companies you want included before calling this.
#
#reports Pearson (linear correlation), Spearman (rank correlation - doesn't
#assume a straight line, but its tie-correction isn't built for as many
#exact ties as best_r has), and Kendall's tau (also rank-based, but its
#"tau-b" variant - scipy's default - is specifically built to handle heavy
#ties correctly) for both the MAE-optimal and RMSE-optimal ratio. Also
#produces a "credibility curve" table: companies grouped into size
#deciles, with the mean/median best_r per decile - a view that sidesteps
#correlation coefficients entirely and is closer to how a credibility
#table is normally read.
#
#this function only computes and saves csvs - it draws no charts. See
#compare.plot_credibility_curve for the visualization of exactly the csv
#this saves; keeping that
#separate means this stays reusable (and testable) without matplotlib
#ever entering the picture, matching the compute/vs/visualize split the
#rest of this project follows (manual_chain_ladder.py, backtesting.py etc.
#compute; compare.py is the only module that touches matplotlib).
def analyze_credibility_vs_size(method_name, factor_average, valuation_year, min_observations=2):
    company_codes = ut.get_all_company_codes()

    rows = []
    for company_code in company_codes:
        weighting_path = ut.get_weighting_path(company_code, method_name, factor_average, valuation_year)
        if not weighting_path.exists():
            continue
        weighting_by_lag = pd.read_csv(weighting_path)
        weighting_by_lag['company_code'] = company_code
        rows.append(weighting_by_lag)

    if not rows:
        raise RuntimeError(
            'No company has a saved weighting-by-lag file yet - run analysis.start_analysis '
            '(e.g. via scripts/run_pipeline.py) for at least one company first.'
        )

    pooled = pd.concat(rows, ignore_index=True)

    # a ratio fit from too few observations exactly matches a single
    # historical point by construction (see the warning in
    # calculate_best_ratio) - excluding those keeps the correlation from
    # being driven by degenerate, meaningless "perfect fits"
    reliable = pooled.loc[pooled['num_observations'] >= min_observations].copy()
    reliable = reliable.replace([np.inf, -np.inf], np.nan)

    summary_rows = []
    binned_rows = []

    for ratio_column in ['best_r_w_mae', 'best_r_w_rmse']:
        valid = reliable.dropna(subset=['company_share_of_industry', ratio_column])
        valid = valid.loc[valid['company_share_of_industry'] > 0]

        pearson_r, pearson_p = stats.pearsonr(valid['company_share_of_industry'], valid[ratio_column])
        spearman_r, spearman_p = stats.spearmanr(valid['company_share_of_industry'], valid[ratio_column])
        # Kendall's tau: like Spearman, a rank-based (not straight-line)
        # measure, but built to handle heavy ties correctly (scipy defaults
        # to "tau-b", which corrects for exactly this) - worth checking
        # alongside Spearman whenever a column is dominated by repeated
        # values, as best_r is here (over half the rows sit at exactly 0.0
        # or 1.0 - see the corner-clustering discussed earlier).
        kendall_tau, kendall_p = stats.kendalltau(valid['company_share_of_industry'], valid[ratio_column])
        summary_rows.append({
            'ratio_column': ratio_column,
            'n': len(valid),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
        })

        # a "credibility curve" table: group companies into deciles by size
        # and look at the mean/median best_r within each decile. This
        # sidesteps the correlation-coefficient machinery entirely - it
        # doesn't care that best_r is corner-heavy or that company size
        # spans many orders of magnitude, it's just "the average trust
        # level for the smallest 10% of companies, vs the next 10%, ... vs
        # the biggest 10%" - closer to how actuaries actually read a
        # credibility table.
        decile = pd.qcut(valid['company_share_of_industry'], q=10, duplicates='drop')
        binned = valid.groupby(decile, observed=True)[ratio_column].agg(['mean', 'median', 'count'])
        binned = binned.reset_index().rename(columns={'company_share_of_industry': 'share_decile'})
        # share_decile is a text range like "(0.00119, 0.00345]" - sorting
        # by that column in a spreadsheet sorts it alphabetically, not
        # numerically, which scrambles the order. This plain 1-10 number
        # is what to actually sort/read by.
        binned.insert(0, 'decile', range(1, len(binned) + 1))
        binned.insert(0, 'ratio_column', ratio_column)
        binned_rows.append(binned)

    summary = pd.DataFrame(summary_rows).set_index('ratio_column')
    binned_summary = pd.concat(binned_rows, ignore_index=True)
    # force the row order explicitly (all MAE deciles 1->10, then all RMSE
    # deciles 1->10) rather than relying on whatever order concat happened
    # to produce - makes the saved csv correct to read top-to-bottom with
    # no sorting needed, in any tool.
    ratio_order = ['best_r_w_mae', 'best_r_w_rmse']
    binned_summary['ratio_column'] = pd.Categorical(binned_summary['ratio_column'], categories=ratio_order, ordered=True)
    binned_summary = binned_summary.sort_values(['ratio_column', 'decile']).reset_index(drop=True)
    binned_summary['ratio_column'] = binned_summary['ratio_column'].astype(str)
    print('\n=== correlation: company_share_of_industry vs best_r ===')
    print(summary)
    print('\n=== average best_r by company-size decile (credibility curve) ===')
    print(binned_summary)

    results_dir = ut.PROJECT_ROOT / 'data' / 'processed' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    pooled_path = results_dir / f'credibility_vs_share_pooled_{tag}.csv'
    summary_path = results_dir / f'credibility_vs_share_correlation_{tag}.csv'
    binned_path = results_dir / f'credibility_vs_share_binned_{tag}.csv'
    pooled.to_csv(pooled_path, index=False)
    summary.to_csv(summary_path)
    binned_summary.to_csv(binned_path, index=False)
    print(f'\n!!! pooled (company, lag) data saved to:\n{pooled_path}')
    print(f'!!! correlation summary saved to:\n{summary_path}')
    print(f'!!! binned credibility curve saved to:\n{binned_path}')

    return pooled, summary, binned_summary

#cross-company benchmark: what did a "typical" company actually pay (in
#raw dollars) by accident year, across every company that has a saved
#square? Squares hold the true, fully-run-off cumulative paid amount at
#every development lag (build_loss_structures.create_company_structure) -
#the last column is each company's real ultimate paid loss for that
#accident year, no chain-ladder projection involved. This answers "is
#company X a lot bigger/smaller than a typical company", which is a
#different question from the industry-wide chain-ladder line already in
#the cumulative-paid chart (that one asks "how well does the industry's
#own factor pattern predict company X's future").
#
#company size is heavily right-skewed (see the population section of the
#report - a handful of companies are enormous next to the rest), so the
#mean ultimate paid is saved here too but is dragged toward those few
#giants; compare.py's benchmark chart plots the median, not the mean, for
#exactly that reason.
#
#doesn't depend on method_name/factor_average/valuation_year (squares
#don't - see get_square_paths), so unlike the credibility csvs this one
#saves under a single untagged filename.
def compute_industry_paid_benchmark():
    company_codes = ut.get_all_company_codes()

    ultimate_by_year = {}
    for company_code in company_codes:
        square_path, _ = ut.get_square_paths(company_code)
        if not square_path.exists():
            continue
        square = pd.read_csv(square_path, index_col=0)
        ultimate = square[square.columns[-1]]  # last column = final development lag = ultimate
        for accident_year, value in ultimate.items():
            ultimate_by_year.setdefault(int(accident_year), []).append(float(value))

    if not ultimate_by_year:
        raise RuntimeError(
            'No company has a saved square yet - run build_loss_structures.build_strcture '
            '(e.g. via scripts/run_pipeline.py) for at least one company first.'
        )

    rows = []
    for accident_year, values in sorted(ultimate_by_year.items()):
        values = np.array(values, dtype=float)
        rows.append({
            'accident_year': accident_year,
            'num_companies': len(values),
            'median_ultimate_paid': float(np.median(values)),
            'mean_ultimate_paid': float(np.mean(values)),
        })
    benchmark = pd.DataFrame(rows).set_index('accident_year')

    benchmark_path = ut.PROJECT_ROOT / 'data' / 'processed' / 'results' / 'industry_paid_benchmark_by_accident_year.csv'
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(benchmark_path, index_label='accident_year')
    print(f'\n!!! industry paid benchmark (median/mean ultimate paid by accident year) saved to:\n{benchmark_path}')

    return benchmark


#per-company "how noisy is this company, and how well does that predict
#things" table - the compute half of the noise/accuracy exploration from
#this session (see compare.py's plot_noise_vs_error/plot_share_vs_r/
#plot_share_vs_error/plot_noise_curve for the visualize half). Three
#pieces per company:
#
#  1. residual_noise_std / annual_growth_rate_pct / trend_r_squared: fit
#     a straight line to log(actual_paid) vs accident_year (ordinary
#     least squares) - the SLOPE is this company's smooth compounding
#     growth rate, the leftover residuals are noise around that trend,
#     line, not around a flat average (see plot_noise_vs_error's
#     docstring for why a flat average would wrongly count fast, smooth
#     growth as "noisy"). Only fit for companies with >=4 nonzero-paid
#     accident years - fewer than that and a 2-parameter line fit isn't
#     meaningful.
#  2. company_only_rmse_pct: this company's own age-to-lag pattern's
#     walk-forward prediction error, in percentage terms, using ONLY the
#     company's own factor (no industry blending) - reuses
#     run_walk_forward_backtests, the same pooled data calculate_best_ratio
#     fits r from, just aggregated as one percentage-RMSE number per
#     company instead of split by lag.
#  3. mean_company_share / mean_best_r_rmse: averaged across that
#     company's own lags from the already-saved weighting_by_lag file -
#     one number per company instead of one per (company, lag) pair, so
#     they're directly comparable to (1) and (2) above (also one-per-
#     company).
def compute_noise_and_accuracy(method_name, factor_average, valuation_year):
    company_codes = [
        int(p.name) for p in (ut.PROJECT_ROOT / 'data' / 'processed' / 'results').iterdir() if p.is_dir()
    ]
    company_codes.sort()
    company_names = ut.get_company_names()

    rows = []
    for company_code in company_codes:
        try:
            company, _ = ut.get_results(company_code, method_name, factor_average, valuation_year)
        except FileNotFoundError:
            continue

        paid = company['actual_paid']
        nonzero = paid[paid > 0]
        residual_noise_std = annual_growth_rate_pct = trend_r_squared = np.nan
        if len(nonzero) >= 4:
            years = nonzero.index.to_numpy(dtype=float)
            log_paid = np.log(nonzero.to_numpy(dtype=float))
            slope, intercept, r_value, _, _ = stats.linregress(years, log_paid)
            fitted = slope * years + intercept
            residual_noise_std = float(np.std(log_paid - fitted))
            annual_growth_rate_pct = float((np.exp(slope) - 1) * 100)
            trend_r_squared = float(r_value ** 2)

        company_only_rmse_pct = np.nan
        try:
            pooled_company, _ = run_walk_forward_backtests(company_code, valuation_year)
            company_only_estimate = pooled_company['last_observed_paid'] * pooled_company['age_to_lag_factor']
            pct_error = (company_only_estimate - pooled_company['actual_paid']) / pooled_company['actual_paid'] * 100
            pct_error = pct_error.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pct_error) >= 4:
                company_only_rmse_pct = float(np.sqrt(np.mean(pct_error ** 2)))
        except Exception as exc:
            print(f'WARNING: could not compute company-only accuracy for company_code={company_code}, failed with: {exc}')

        mean_company_share = mean_best_r_rmse = np.nan
        weighting_path = ut.get_weighting_path(company_code, method_name, factor_average, valuation_year)
        if weighting_path.exists():
            weighting = pd.read_csv(weighting_path)
            mean_company_share = float(weighting['company_share_of_industry'].mean())
            mean_best_r_rmse = float(weighting['best_r_w_rmse'].mean())

        rows.append({
            'company_code': company_code,
            'company_name': company_names.get(company_code, 'Unknown company name'),
            'total_ultimate_paid': float(paid.sum()),
            'residual_noise_std': residual_noise_std,
            'annual_growth_rate_pct': annual_growth_rate_pct,
            'trend_r_squared': trend_r_squared,
            'company_only_rmse_pct': company_only_rmse_pct,
            'mean_company_share': mean_company_share,
            'mean_best_r_rmse': mean_best_r_rmse,
        })

    result = pd.DataFrame(rows).set_index('company_code')

    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    result_path = ut.PROJECT_ROOT / 'data' / 'processed' / 'results' / f'noise_and_accuracy_by_company_{tag}.csv'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(result_path, index_label='company_code')
    print(f'\n!!! per-company noise/accuracy table saved to:\n{result_path}')

    return result


#the "noise curve": mean AND median residual_noise_std per company-size
#decile, mirroring calculate_best_ratio's credibility curve exactly (same
#qcut-into-10-equal-count-groups approach, same mean/median-side-by-side
#reasoning - see analyze_credibility_vs_size's docstring for why both are
#shown rather than just one) but for noise instead of best_r, and binned
#by company (one row per company) rather than by (company, lag) pair.
#reads the csv compute_noise_and_accuracy already saved rather than
#recomputing it - that function reruns the walk-forward backtest for
#every company, expensive enough to be worth caching to disk once.
def analyze_noise_vs_size(method_name, factor_average, valuation_year):
    tag = f'{method_name}_{factor_average}_as_of_{valuation_year}'
    noise_path = ut.PROJECT_ROOT / 'data' / 'processed' / 'results' / f'noise_and_accuracy_by_company_{tag}.csv'
    if not noise_path.exists():
        raise FileNotFoundError(f'{noise_path} does not exist yet - run analysis.compute_noise_and_accuracy first.')
    noise_and_accuracy = pd.read_csv(noise_path)

    valid = noise_and_accuracy.dropna(subset=['total_ultimate_paid', 'residual_noise_std'])
    valid = valid.loc[valid['total_ultimate_paid'] > 0]

    decile = pd.qcut(valid['total_ultimate_paid'], q=10, duplicates='drop')
    binned = valid.groupby(decile, observed=True)['residual_noise_std'].agg(['mean', 'median', 'count'])
    binned = binned.reset_index().rename(columns={'total_ultimate_paid': 'size_decile_range'})
    binned.insert(0, 'decile', range(1, len(binned) + 1))

    binned_path = ut.PROJECT_ROOT / 'data' / 'processed' / 'results' / f'noise_vs_size_binned_{tag}.csv'
    binned.to_csv(binned_path, index=False)
    print(f'\n!!! noise-by-size-decile table saved to:\n{binned_path}')

    return binned


# company_code = 43
# valuation_year = 2006
# pooled_company, pooled_industry = run_walk_forward_backtests(company_code, valuation_year)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(calculate_best_ratio(pooled_company, pooled_industry))
