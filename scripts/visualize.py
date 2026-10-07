"""Visualize an already-computed company vs industry comparison.

This does NOT recompute anything - it just reads whatever run_pipeline.py
already saved to data/processed/results/<company_code>/ and charts it. Use
this to look at one company at a time without re-running the full
(possibly 143-company) pipeline.

Nothing is written to reports/figures/ unless --download is passed - by
default every chart here just displays (or, for --pdf, refuses to run at
all, since a 100+ page PDF has no non-file way to "just display").

Examples:
    python scripts/visualize.py --list                # see which companies are ready to view (code + name)
    python scripts/visualize.py 43                     # show company 43's charts, don't save anything
    python scripts/visualize.py 43 --download          # same, and also save the 2 PNGs to reports/figures/
    python scripts/visualize.py 43 --style interactive
    python scripts/visualize.py --credibility          # cross-company credibility-vs-size curve
    python scripts/visualize.py --pdf --download       # save one combined PDF (2 charts/company) for every company
    python scripts/visualize.py --noise                # noise-vs-error, share-vs-r, share-vs-error dot charts, and the noise curve
"""
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt

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


def list_available_companies() -> list[int]:
    results_dir = ut.PROJECT_ROOT / 'data' / 'processed' / 'results'
    if not results_dir.exists():
        return []
    return sorted(int(p.name) for p in results_dir.iterdir() if p.is_dir())


#one (code, name) tuple per line rather than a single comma-separated
#line - with 100+ companies a one-liner just wraps illegibly in a
#terminal, and this way it still reads as the [(...), (...), ...] shape
#of a plain list of tuples, just spread one entry per row.
def print_available_companies(available):
    company_names = ut.get_company_names()
    print(f'{len(available)} companies have results ready to visualize:')
    print('[')
    for code in available:
        print(f'    ({code}, {company_names.get(code, "Unknown company name")!r}),')
    print(']')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'company_code', type=int, nargs='?',
        help='company code to visualize - must already have been processed by scripts/run_pipeline.py',
    )
    parser.add_argument('--list', action='store_true', help='list company codes (and names) that already have results to visualize, then exit')
    parser.add_argument(
        '--credibility', action='store_true',
        help='chart the cross-company credibility-vs-size curve instead of one company '
             '(needs analysis.analyze_credibility_vs_size to have been run - e.g. scripts/run_pipeline.py)',
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='target every company that has saved results instead of one company - combine with --download '
             'to save one combined PDF (cumulative-paid-loss + median-trend charts, headed by company code '
             'and name), one page per company '
             '(needs analysis.compute_industry_paid_benchmark to have been run - e.g. scripts/run_pipeline.py)',
    )
    parser.add_argument(
        '--noise', action='store_true',
        help='chart the noise-vs-error, share-vs-r, and share-vs-error dot charts, plus the noise curve, '
             'instead of one company - combine with --download to also save them '
             '(needs analysis.compute_noise_and_accuracy and analyze_noise_vs_size to have been run - '
             'e.g. scripts/run_pipeline.py)',
    )
    parser.add_argument(
        '--download', action='store_true',
        help='also save chart(s) to reports/figures/ instead of only displaying them - a single company '
             'saves its 2 PNGs, --pdf saves the one combined PDF for every company',
    )
    parser.add_argument('--valuation-year', type=int, default=DEFAULT_VALUATION_YEAR)
    parser.add_argument('--method-name', default=DEFAULT_METHOD_NAME)
    parser.add_argument('--factor-average', default=DEFAULT_FACTOR_AVERAGE)
    parser.add_argument('--style', choices=['classical', 'interactive'], default='classical')
    args = parser.parse_args()

    if args.credibility:
        compare.plot_credibility_curve(args.method_name, args.factor_average, args.valuation_year, download=args.download)
        plt.show()
        return

    if args.pdf:
        if not args.download:
            print(
                'Nothing saved: a 100+ page combined PDF has no way to "just display" - '
                'pass --download along with --pdf to actually generate and save it.'
            )
            return
        compare.generate_all_companies_pdf(args.method_name, args.factor_average, args.valuation_year)
        return

    if args.noise:
        #build all 4 figures first, then show them together in one
        #plt.show() call - matplotlib's plt.show() blocks until every
        #currently-open figure's window is closed, so calling it once
        #here (rather than once per plot_*() call) pops all 4 windows up
        #at the same time instead of showing them one at a time.
        compare.plot_noise_vs_error(args.method_name, args.factor_average, args.valuation_year, download=args.download)
        compare.plot_share_vs_r(args.method_name, args.factor_average, args.valuation_year, download=args.download)
        compare.plot_share_vs_error(args.method_name, args.factor_average, args.valuation_year, download=args.download)
        compare.plot_noise_curve(args.method_name, args.factor_average, args.valuation_year, download=args.download)
        plt.show()
        return

    available = list_available_companies()

    if args.list or args.company_code is None:
        if available:
            print_available_companies(available)
        else:
            print('No companies have results yet - run scripts/run_pipeline.py first.')
        if args.company_code is None:
            return

    if args.company_code not in available:
        print(
            f'company_code={args.company_code} has no saved results '
            f'(as_of valuation_year={args.valuation_year}). '
            'Run scripts/run_pipeline.py first, or pass --list to see what is available.'
        )
        return

    compare.start_visualization(
        args.style, args.company_code, args.method_name, args.factor_average, args.valuation_year,
        download=args.download,
    )


if __name__ == '__main__':
    main()
