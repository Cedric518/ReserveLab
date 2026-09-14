import pandas as pd
import pytest

from reserve_lab import analysis


@pytest.fixture
def two_observation_lag() -> tuple[pd.DataFrame, pd.DataFrame]:
    # real numbers for company 43, development lag 9, from accident years
    # 1998 (as observed at valuation year 2006) and 1999 (as observed at
    # valuation year 2007) - the exact worked example this project's
    # weighting-by-lag calculation was derived and hand-verified against.
    company = pd.DataFrame(
        {
            "accident_year": [1998, 1999],
            "valuation_year": [2006, 2007],
            "latest_observed_lag": [9, 9],
            "last_observed_paid": [39876.0, 45090.0],
            "age_to_lag_factor": [1.0, 1.000501554819942],
            "actual_paid": [39896, 45092],
            # made-up industry totals (this company is a made-up 10% of the
            # industry at this lag) - only used for the
            # company_share_of_industry diagnostic, not the r/MAE/RMSE fit
            "industry_last_observed_paid": [398760.0, 450900.0],
        }
    )
    industry = pd.DataFrame(
        {
            "accident_year": [1998, 1999],
            "valuation_year": [2006, 2007],
            "latest_observed_lag": [9, 9],
            "last_observed_paid": [39876.0, 45090.0],
            "age_to_lag_factor": [1.0, 1.0016448080624736],
            "actual_paid": [39896, 45092],
        }
    )
    return company, industry


def test_calculate_best_ratio_matches_hand_derivation(
    two_observation_lag: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    company, industry = two_observation_lag

    result = analysis.calculate_best_ratio(company, industry)

    assert result.loc[9, "num_observations"] == 2
    # both accident years favor the company factor once blended, per the
    # by-hand MAE/RMSE walkthrough for this exact pair
    assert result.loc[9, "best_r_w_mae"] == pytest.approx(1.0)
    assert result.loc[9, "best_mae"] == pytest.approx(20.307553, rel=1e-4)
    assert result.loc[9, "best_r_w_rmse"] == pytest.approx(1.0)
    assert result.loc[9, "best_rmse"] == pytest.approx(20.309882, rel=1e-4)
    # last_observed_paid / industry_last_observed_paid = 39876/398760 =
    # 45090/450900 = exactly 0.1 for both observations, so the mean is 0.1
    assert result.loc[9, "company_share_of_industry"] == pytest.approx(0.1)


def test_calculate_best_ratio_flags_low_sample_size(
    two_observation_lag: tuple[pd.DataFrame, pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    company, industry = two_observation_lag
    # drop down to a single observation - a ratio fit to one point exactly
    # matches that point by construction and isn't a meaningful pattern
    single_company = company.iloc[[0]]
    single_industry = industry.iloc[[0]]

    analysis.calculate_best_ratio(single_company, single_industry)

    assert "fewer than 2 observations" in capsys.readouterr().out
