from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that
from matplotlib.figure import Figure

from rec import RegressionErrorCharacteristic


def ids(kwargs: dict[str, Any]) -> str:
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
    def test_rec_instantiation__passes__with_default_inputs(self, kwargs: dict[str, Any]) -> None:
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

    def test_rec_plot__passes__with_save_path_and_no_return_fig(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Covers savefig path and return_fig=False branch."""
        show_mock = MagicMock()
        monkeypatch.setattr("rec._rec.plt.show", show_mock)

        r = RegressionErrorCharacteristic(
            y_true=[3, -0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
        )
        save_path = tmp_path / "rec_curve.png"
        result = r.plot(
            save_path=str(save_path),
            display_plot=False,
            return_fig=False,
        )

        assert_that(result).is_none()
        assert_that(save_path.exists()).is_true()
        show_mock.assert_not_called()

    def test_rec_plot__passes__with_display_plot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Covers display_plot=True branch without opening a GUI window."""
        show_mock = MagicMock()
        monkeypatch.setattr("rec._rec.plt.show", show_mock)

        r = RegressionErrorCharacteristic(
            y_true=[3, -0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
        )
        result = r.plot(
            display_plot=True,
            return_fig=False,
        )

        assert_that(result).is_none()
        show_mock.assert_called_once()

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
        kwargs: dict[str, Any],
    ) -> None:
        r = RegressionErrorCharacteristic(
            y_true=[3, -0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
        )
        with pytest.raises(TypeError):
            _ = r.plot(**kwargs)
