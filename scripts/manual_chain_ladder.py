from pathlib import Path

import pandas as pd

VALUATION_YEAR = 2007
COMPANY_CODE = 43
FINAL_DEVELOPMENT_LAG = 10

METHOD_NAME = "paid_chain_ladder"
FACTOR_AVERAGE = "volume"
PATTERN_SOURCE = "company_specific"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOSS_TRIANGLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "triangles"
    / f"{COMPANY_CODE}_{VALUATION_YEAR}.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reserve_estimates"
)

FACTOR_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / (
        f"company_{COMPANY_CODE}_"
        f"paid_chain_ladder_volume_factors_"
        f"as_of_{VALUATION_YEAR}.csv"
    )
)

ESTIMATE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / (
        f"company_{COMPANY_CODE}_"
        f"paid_chain_ladder_volume_estimates_"
        f"as_of_{VALUATION_YEAR}.csv"
    )
)


loss_triangle = pd.read_csv(
    LOSS_TRIANGLE_PATH,
    index_col=0,
)

chain_ladder = loss_triangle.copy()

chain_ladder.columns = (
    chain_ladder.columns.astype(int)
)


# ---------------------------------------------------------
# 1. Calculate volume-weighted age-to-age factors
# ---------------------------------------------------------

factor_records = []

for current_lag in range(
    1,
    FINAL_DEVELOPMENT_LAG,
):
    next_lag = current_lag + 1

    matched_pairs = (
        chain_ladder
        .loc[:, [current_lag, next_lag]]
        .dropna()
    )

    denominator_sum = (
        matched_pairs[current_lag].sum()
    )

    numerator_sum = (
        matched_pairs[next_lag].sum()
    )

    if denominator_sum == 0:
        raise ValueError(
            "Cannot calculate the "
            f"{current_lag}-to-{next_lag} factor "
            "because its denominator sum is zero."
        )

    age_to_age_factor = (
        numerator_sum / denominator_sum
    )

    factor_records.append(
        {
            "current_lag": current_lag,
            "next_lag": next_lag,
            "matched_accident_years": len(
                matched_pairs
            ),
            "denominator_sum": denominator_sum,
            "numerator_sum": numerator_sum,
            "age_to_age_factor": (
                age_to_age_factor
            ),
        }
    )


age_to_age_table = (
    pd.DataFrame(factor_records)
    .set_index("current_lag")
)


# ---------------------------------------------------------
# 2. Convert age-to-age factors to age-to-lag-10 factors
# ---------------------------------------------------------

cumulative_development_factor = (
    age_to_age_table[
        "age_to_age_factor"
    ]
    .sort_index(ascending=False)
    .cumprod()
    .sort_index()
)

cumulative_development_factor.loc[
    FINAL_DEVELOPMENT_LAG
] = 1.0

cumulative_development_factor = (
    cumulative_development_factor
    .sort_index()
)

cumulative_development_factor.name = (
    "age_to_lag_10_factor"
)


# ---------------------------------------------------------
# 3. Find the latest observed position for each accident year
# ---------------------------------------------------------

latest_observed_lag = (
    chain_ladder
    .notna()
    .sum(axis=1)
    .astype(int)
)

cumulative_paid_at_valuation = (
    chain_ladder
    .ffill(axis=1)
    .iloc[:, -1]
)

selected_cumulative_factor = (
    latest_observed_lag.map(
        cumulative_development_factor
    )
)


# ---------------------------------------------------------
# 4. Estimate cumulative paid and reserve through lag 10
# ---------------------------------------------------------

estimates = pd.DataFrame(
    {
        "latest_observed_lag": (
            latest_observed_lag
        ),
        "cumulative_paid_at_valuation": (
            cumulative_paid_at_valuation
        ),
        "age_to_lag_10_factor": (
            selected_cumulative_factor
        ),
    }
)

estimates[
    "estimated_cumulative_paid_lag_10"
] = (
    estimates[
        "cumulative_paid_at_valuation"
    ]
    * estimates[
        "age_to_lag_10_factor"
    ]
)

estimates[
    "estimated_reserve_to_lag_10"
] = (
    estimates[
        "estimated_cumulative_paid_lag_10"
    ]
    - estimates[
        "cumulative_paid_at_valuation"
    ]
)

estimates.index.name = "accident_year"

estimates.insert(
    0,
    "company_code",
    COMPANY_CODE,
)

estimates.insert(
    1,
    "valuation_year",
    VALUATION_YEAR,
)

estimates.insert(
    2,
    "method",
    METHOD_NAME,
)

estimates.insert(
    3,
    "factor_average",
    FACTOR_AVERAGE,
)

estimates.insert(
    4,
    "pattern_source",
    PATTERN_SOURCE,
)


# ---------------------------------------------------------
# 5. Prepare the factor output table
# ---------------------------------------------------------

factor_output = age_to_age_table.reindex(
    range(
        1,
        FINAL_DEVELOPMENT_LAG + 1,
    )
)

factor_output[
    "age_to_lag_10_factor"
] = cumulative_development_factor

factor_output.insert(
    0,
    "company_code",
    COMPANY_CODE,
)

factor_output.insert(
    1,
    "valuation_year",
    VALUATION_YEAR,
)

factor_output.insert(
    2,
    "method",
    METHOD_NAME,
)

factor_output.insert(
    3,
    "factor_average",
    FACTOR_AVERAGE,
)

factor_output.insert(
    4,
    "pattern_source",
    PATTERN_SOURCE,
)


# ---------------------------------------------------------
# 6. Save and display the results
# ---------------------------------------------------------

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

factor_output.to_csv(
    FACTOR_OUTPUT_PATH,
    index=True,
    index_label="current_lag",
)

estimates.to_csv(
    ESTIMATE_OUTPUT_PATH,
    index=True,
    index_label="accident_year",
)

total_estimated_reserve = (
    estimates[
        "estimated_reserve_to_lag_10"
    ].sum()
)

print(
    "\nVolume-weighted factors:"
)

print(
    factor_output
    .round(6)
    .to_string()
)

print(
    "\nPaid chain-ladder estimates:"
)

print(
    estimates
    .round(2)
    .to_string()
)

print(
    "\nTotal estimated reserve "
    "through lag 10: "
    f"{total_estimated_reserve:,.2f} "
    "USD thousands"
)

print(
    "\nFactor table saved to:"
)

print(
    FACTOR_OUTPUT_PATH
)

print(
    "\nEstimate table saved to:"
)

print(
    ESTIMATE_OUTPUT_PATH
)