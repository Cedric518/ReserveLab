import pandas as pd
import pytest

from reserve_lab import manual_chain_ladder as mcl


@pytest.fixture
def toy_triangle() -> pd.DataFrame:
    # 3 accident years, development lags 1-3, with the usual triangular
    # censoring (each accident year has one fewer observed lag than the one
    # before it) - just enough to exercise both age-to-age transitions.
    return pd.DataFrame(
        {
            1: [100.0, 110.0, 120.0],
            2: [150.0, 160.0, None],
            3: [165.0, None, None],
        },
        index=pd.Index([2001, 2002, 2003], name="accident_year"),
    )


def test_calculate_selected_factors(toy_triangle: pd.DataFrame) -> None:
    selected = mcl.calculate_selected_factors(toy_triangle)

    # lag 1->2: both accident years 2001 and 2002 have observed this
    # transition, so it's volume-weighted across both
    assert selected.loc[1, "age_to_age_factor"] == pytest.approx((150 + 160) / (100 + 110))
    # lag 2->3: only accident year 2001 has observed this transition
    assert selected.loc[2, "age_to_age_factor"] == pytest.approx(165 / 150)


def test_calculate_age_to_lag_factors(toy_triangle: pd.DataFrame) -> None:
    selected = mcl.calculate_selected_factors(toy_triangle)
    age_to_lag = mcl.calculate_age_to_lag_factors(selected)

    lag_2_to_3 = 165 / 150
    lag_1_to_2 = (150 + 160) / (100 + 110)

    # age-to-ultimate factor at the final observed lag is always 1.0 -
    # there's nothing left to develop
    assert age_to_lag.loc[3, "age_to_lag_factor"] == pytest.approx(1.0)
    # age-to-ultimate at an earlier lag is the product of every age-to-age
    # factor from there to the end
    assert age_to_lag.loc[2, "age_to_lag_factor"] == pytest.approx(lag_2_to_3)
    assert age_to_lag.loc[1, "age_to_lag_factor"] == pytest.approx(lag_2_to_3 * lag_1_to_2)
