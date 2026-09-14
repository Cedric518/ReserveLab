"""Visualize an already-computed company vs industry comparison.

This does NOT recompute anything - it just reads whatever run_pipeline.py
already saved to data/processed/results/<company_code>/ and charts it. Use
this to look at one company at a time without re-running the full
(possibly 143-company) pipeline.

Examples:
    python scripts/visualize.py --list           # see which companies are ready to view
    python scripts/visualize.py 43                # chart company 43 with the project defaults
    python scripts/visualize.py 43 --style interactive
    python scripts/visualize.py --credibility     # cross-company credibility-vs-size charts
"""
import argparse
from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from reserve_lab import compare
from reserve_lab import utilities as ut

#same defaults run_pipeline.py uses - override at the command line if a
#particular company was processed with different settings
DEFAULT_VALUATION_YEAR = 2007
DEFAULT_METHOD_NAME = 'paid_chain_ladder'
DEFAULT_FACTOR_AVERAGE = 'volume'


def list_available_companies() -> list[str]:
    results_dir = ut.PROJECT_ROOT / 'data' / 'processed' / 'results'
    if not results_dir.exists():
        return []
    return sorted((p.name for p in results_dir.iterdir() if p.is_dir()), key=lambda code: int(code))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'company_code', type=int, nargs='?',
        help='company code to visualize - must already have been processed by scripts/run_pipeline.py',
    )
    parser.add_argument('--list', action='store_true', help='list company codes that already have results to visualize, then exit')
    parser.add_argument(
        '--credibility', action='store_true',
        help='chart the cross-company credibility-vs-size scatter and curve instead of one company '
             '(needs analysis.analyze_credibility_vs_size to have been run - e.g. scripts/run_pipeline.py)',
    )
    parser.add_argument('--valuation-year', type=int, default=DEFAULT_VALUATION_YEAR)
    parser.add_argument('--method-name', default=DEFAULT_METHOD_NAME)
    parser.add_argument('--factor-average', default=DEFAULT_FACTOR_AVERAGE)
    parser.add_argument('--style', choices=['classical', 'interactive'], default='classical')
    args = parser.parse_args()

    if args.credibility:
        compare.plot_credibility_scatter(args.method_name, args.factor_average, args.valuation_year)
        compare.plot_credibility_curve(args.method_name, args.factor_average, args.valuation_year)
        return

    available = list_available_companies()

    if args.list or args.company_code is None:
        if available:
            print(f'{len(available)} companies have results ready to visualize:')
            print(', '.join(available))
        else:
            print('No companies have results yet - run scripts/run_pipeline.py first.')
        if args.company_code is None:
            return

    if str(args.company_code) not in available:
        print(
            f'company_code={args.company_code} has no saved results '
            f'(as_of valuation_year={args.valuation_year}). '
            'Run scripts/run_pipeline.py first, or pass --list to see what is available.'
        )
        return

    compare.start_visualization(
        args.style, args.company_code, args.method_name, args.factor_average, args.valuation_year
    )


if __name__ == '__main__':
    main()
