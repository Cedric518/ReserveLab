# Project Name

First Research Objective: Using only the information available at December 31, 2007, estimate each insurer's future private-passenger-auto payments through development lag 10.
## Repository structure

- `src/`: reusable Python code
- `scripts/`: operational and data-ingestion scripts
- `notebooks/`: exploratory analysis
- `data/raw/`: original source files, not committed
- `data/interim/`: intermediate files, not committed
- `data/processed/`: analysis-ready files, not committed
- `data/sample/`: small safe samples
- `tests/`: automated tests
- `reports/figures/`: generated figures

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"