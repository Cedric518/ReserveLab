"""Import smoke tests.

Every reserve_lab submodule must be importable on its own, with no data
files, network access, or other side effects required at import time -
only inside the functions that actually need them. (initial_data_validation
and compare used to violate this by reading data/raw/ppauto_pos.csv at
module level; fixed alongside these tests.)
"""

import importlib

import pytest

SUBMODULES = [
    "reserve_lab",
    "reserve_lab.utilities",
    "reserve_lab.initial_data_validation",
    "reserve_lab.clean_table",
    "reserve_lab.build_loss_structures",
    "reserve_lab.manual_chain_ladder",
    "reserve_lab.backtesting",
    "reserve_lab.compare",
    "reserve_lab.analysis",
]


@pytest.mark.parametrize("module_name", SUBMODULES)
def test_module_imports_without_side_effects(module_name: str) -> None:
    importlib.import_module(module_name)
