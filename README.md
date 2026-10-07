# reserveLab

How much should an auto insurer trust its own loss history, and how much should it lean on the industry's?

reserveLab is a paid chain-ladder reserving study on U.S. private passenger auto data (NAIC Schedule P, from the CAS Loss Reserving Database). For each company and each development lag, it fits the best blend of the company's own development factor and the industry factor, using walk-forward backtesting. It then tests whether bigger, steadier companies earn more credibility for their own data.

The full method, derivations and results are in [reports/technical_report.tex](reports/technical_report.tex).

## Key findings

From the technical report (valuation year 2007, volume-weighted paid chain ladder):

- **Coverage.** 112 of the 143 companies run end to end. The other 31 stop at a data-validity check during triangle construction (a zero denominator in an age-to-age factor).
- **Size and credibility are only weakly linked.** Pearson correlation finds nothing (r = 0.024, p = 0.45). Rank-based measures find a small but significant effect (Spearman ρ = 0.129, Kendall τ_b = 0.093, both p < 0.001). Pearson misses it because 57% of fitted blend ratios sit exactly at 0 or 1.
- **Volatility matters more than size.** Companies whose paid losses swing more from year to year have larger prediction errors (Spearman ρ = 0.479), and larger companies are much steadier (ρ = −0.683).
- **Volatility does not predict the blend ratio directly.** The fitted ratio varies a lot from lag to lag within a single company, so one volatility number per company explains little of it.

## Method

1. **Validate and clean** the raw Schedule P extract.
2. **Build loss structures** for each company:
   - the *triangle*: cumulative paid losses that were known as of the valuation year
   - the *square*: the full accident-year × lag matrix, including outcomes disclosed later. It is used only to grade predictions, never as a model input.

   Industry versions are the cell-by-cell sum across all companies.
3. **Run the paid chain ladder twice** per company: once with its own volume-weighted age-to-age factors and once with the industry's.
4. **Walk the valuation year forward** one year at a time (a rolling-origin backtest). With a single valuation date, each lag has exactly one accident year behind it; walking forward gives each lag several out-of-sample observations.
5. **Fit a blend ratio r ∈ [0, 1] per lag**:

   ```
   blended_factor = r × company_factor + (1 − r) × industry_factor
   ```

   r is solved exactly, with no grid search: closed form for RMSE, and a finite candidate set for MAE.
6. **Analyze across companies.** All companies are pooled to correlate the fitted r against each company's share of industry paid losses and against a per-company noise measure.

## Data

| | |
|---|---|
| Name | PP Auto Data Set (private passenger auto) |
| Source | [CAS: Loss Reserving Data Pulled from NAIC Schedule P](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p) |
| Format | CSV |
| Coverage | 143 companies, accident years 1998–2007, development lags 1–10 |

Download the private passenger auto file and save it as `data/raw/ppauto_pos.csv`. Everything under `data/raw/`, `data/interim/` and `data/processed/` is gitignored.

## Setup

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Both scripts add `src/` to `sys.path` themselves, so they still run if the editable install doesn't resolve.

## Usage

### 1. Run the pipeline

```bash
python scripts/run_pipeline.py
```

This validates and cleans the raw data, processes every company, and then runs the cross-company analyses. If a company fails, it is skipped with a warning and the run continues. Each step prints its intermediate tables, so expect a lot of console output.

The valuation year (2007), method (`paid_chain_ladder`) and factor average (`volume`) are set at the top of `main()` in [scripts/run_pipeline.py](scripts/run_pipeline.py).

### 2. Look at the results

[scripts/visualize.py](scripts/visualize.py) only reads results that are already saved; it recomputes nothing. By default it just displays charts. Add `--download` to also save them to `reports/figures/`.

| Command | What it shows |
|---|---|
| `python scripts/visualize.py --list` | Companies with saved results (code and name) |
| `python scripts/visualize.py 43` | Company 43: estimated vs. actual cumulative paid loss, and its trend against the industry median |
| `python scripts/visualize.py --credibility` | Credibility curve: mean and median fitted r by company-size decile |
| `python scripts/visualize.py --noise` | Noise vs. error, share vs. r, share vs. error, and the noise-by-size curve |
| `python scripts/visualize.py --pdf --download` | One combined PDF with a page per company |

The interactive view uses Streamlit, so launch it through `streamlit run`:

```bash
streamlit run scripts/visualize.py -- 43 --style interactive
```

`--valuation-year`, `--method-name` and `--factor-average` override the defaults if a run used different settings.

## Outputs

```
data/
├── interim/ppauto_loss_development_clean.csv
└── processed/
    ├── triangles/<company_code>/<valuation_year>.csv   # plus industry_<valuation_year>.csv
    ├── squares/<company_code>/square.csv               # plus industry.csv
    └── results/
        ├── <company_code>/
        │   ├── <method>_<avg>_company_as_of_<year>.csv            # estimates using the company's own pattern
        │   ├── <method>_<avg>_industry_as_of_<year>.csv           # estimates using the industry pattern
        │   └── <method>_<avg>_weighting_by_lag_as_of_<year>.csv   # fitted r per lag, MAE and RMSE
        ├── credibility_vs_share_{pooled,correlation,binned}_<tag>.csv
        ├── noise_and_accuracy_by_company_<tag>.csv
        ├── noise_vs_size_binned_<tag>.csv
        └── industry_paid_benchmark_by_accident_year.csv
reports/figures/   # PNGs and the combined PDF, written only with --download
                   # (the pipeline also saves the credibility curve here)
```

`<tag>` is `<method>_<avg>_as_of_<year>`, for example `paid_chain_ladder_volume_as_of_2007`.

## Project layout

```
scripts/
├── run_pipeline.py            # end-to-end orchestrator
└── visualize.py               # charts from saved results
src/reserve_lab/
├── initial_data_validation.py # checks the raw columns and time fields
├── clean_table.py             # renames columns, adds reserve and data-quality fields
├── build_loss_structures.py   # triangles and squares
├── manual_chain_ladder.py     # age-to-age and age-to-lag factors, reserves
├── backtesting.py             # error against the square's actual outcome
├── analysis.py                # walk-forward backtest, fits r, cross-company statistics
├── compare.py                 # all plotting (matplotlib and Streamlit)
└── utilities.py               # file paths and shared I/O helpers
reports/
└── technical_report.tex       # full write-up
```

Computation and plotting are kept apart: `analysis.py` and the modules before it only compute and save CSVs, and `compare.py` is the only module that draws charts.

## Building the report

```bash
cd reports
pdflatex technical_report.tex
pdflatex technical_report.tex   # second pass fills in the table of contents
```

## Limitations

- **Small samples.** Many per-lag fits rest on only a few observations. Lags with fewer than 2 are excluded from the cross-company analysis. Companies with fewer than 5 accident years get `thin_history_flag = True` and a warning on their charts.
- **No shrinkage.** The fitted ratios are exact optima with no shrinkage toward a prior, so they often land at 0 or 1.
- **Narrow scope.** The study covers one line of business (U.S. private passenger auto) over one ten-year window. Longer-tailed lines may behave differently.
