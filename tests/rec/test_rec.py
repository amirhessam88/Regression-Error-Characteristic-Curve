from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that
from matplotlib.figure import Figure

from rec import RegressionErrorCharacteristic


def ids(kwargs: Dict[str, Any]) -> str:
    """Returns a user-friendly test case ID from the parametrized key-value pairs."""
    return ", ".join(f"{k} : {v}" for (k, v) in kwargs.items())


class TestRegressionErrorCharacteristic:
    """Validates RegressionErrorCharacteristic instantiation."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [3, -0.5, 2, 7],
                "y_pred": [2.5, 0.0, 2, 8],
            },
            {
                "y_true": np.array([3, -0.5, 2, 7]),
                "y_pred": np.array([2.5, 0.0, 2, 8]),
            },
            {
                "y_true": pd.Series([3, -0.5, 2, 7]),
                "y_pred": pd.Series([2.5, 0.0, 2, 8]),
            },
        ],
        ids=ids,
    )
    def test_rec_instantiation__passes__with_default_inputs(self, kwargs: Dict[str, Any]) -> None:
        r = RegressionErrorCharacteristic(**kwargs)
        f = r.plot(
            display_plot=False,
            return_fig=True,
        )

        assert_that(r.y_true).is_instance_of(np.ndarray)
        assert_that(r.y_pred).is_instance_of(np.ndarray)
        assert_that(r.auc_rec).is_instance_of(float)
        assert_that(r.deviation).is_instance_of(np.ndarray)
        assert_that(r.accuracy).is_instance_of(np.ndarray)
        assert_that(f).is_instance_of(Figure)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "figsize": [8, 8],
                "display_plot": False,
            },
            {
                "color": 123,
                "display_plot": False,
            },
            {
                "linestyle": 123,
                "display_plot": False,
            },
            {
                "fontsize": "123",
                "display_plot": False,
            },
            {
                "save_path": 123,
                "display_plot": False,
            },
        ],
        ids=ids,
    )
    def test_rec_plot__fails__with_invalid_inputs(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        r = RegressionErrorCharacteristic(
            y_true=[3, -0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
        )
        with pytest.raises(TypeError):
            _ = r.plot(**kwargs)
