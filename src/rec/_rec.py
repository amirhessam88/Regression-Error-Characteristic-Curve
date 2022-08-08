# -------------------------------
# rec.py
# Author: Amirhessam Tahmassebi
# Email: admin@amirhessam.com
# Website: www.amirhessam.com
# Date: November-07-2017
# -------------------------------

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
from matplotlib.figure import Figure


@dataclass
class RegressionErrorCharacteristic:
    """Regression Error Characteristics (REC).

    Notes
    -----
    This is wrapper to implement the REC algorithm. The REC is implemented based on the _[1] paper
    and the initial idea originally presented in _[2] paper. The Simpson method is used as the
    integral method to calculate the area under REC.

    References
    ---------
    .. [1] Tahmassebi, A., Gandomi, A. H., & Meyer-Baese, A. (2018, July). A Pareto front based
           evolutionary model for airfoil self-noise prediction. In 2018 IEEE Congress on
           Evolutionary Computation (CEC) (pp. 1-8). IEEE.
           https://www.amirhessam.com/assets/pdf/projects/cec-airfoil2018.pdf

    .. [2] Bi, J., & Bennett, K. P. (2003). Regression error characteristic curves. In Proceedings
           of the 20th international conference on machine learning (ICML-03) (pp. 43-50).
           https://www.aaai.org/Papers/ICML/2003/ICML03-009.pdf

    Parameters
    ----------
    y_true : Union[List[float], pd.Series, np.ndarray]
        Ground truth target (response) values

    y_pred : Union[List[float], pd.Series, np.ndarray]
        Predicted target values

    Attributes
    ----------
    auc_rec : float
        Area under REC curve with a possible value between 0.0 and 1.0

    deviation :  np.ndarray
        Array of deviations to plot REC curve

    accuracy :  np.ndarray
        Array of calculated accuracy at each deviation to plot REC curve

    plotting_dict : Dict[str, Any]
        Plotting properties

    Methods
    -------
    plot(figsize=(8, 5), color="navy", linestyle="--", fontsize=15, save_path=None, display_plot=False, return_fig=False)
        Plots the REC curve
    """

    y_true: Union[List[float], pd.Series, np.ndarray]
    y_pred: Union[List[float], pd.Series, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.y_true, np.ndarray):
            self.y_true = np.array(self.y_true)
        if not isinstance(self.y_pred, np.ndarray):
            self.y_pred = np.array(self.y_pred)
        (
            self.deviation,
            self.accuracy,
            self.auc_rec,
        ) = self._rec_curve()

    def _rec_curve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculates the rec curve elements: deviation, accuracy, auc.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
        """
        begin = 0.0
        end = 1.0
        interval = 0.01
        accuracy = []
        deviation = np.arange(begin, end, interval)

        # TODO(amir): come back to this and remove the `np.array()` casting
        # somehow, mypy does not like List[float], however, the list is already cast in post-init
        norms = np.abs(np.array(self.y_true) - np.array(self.y_pred)) / np.sqrt(
            np.array(self.y_true) ** 2 + np.array(self.y_pred) ** 2,
        )

        # main loop to count the number of times that the calculated norm is less than deviation
        for _, dev in enumerate(deviation):
            count = 0.0
            for _, norm in enumerate(norms):
                if norm < dev:
                    count += 1
            accuracy.append(count / len(self.y_true))

        auc_rec = scp.integrate.simps(accuracy, deviation) / end

        return (deviation, np.array(accuracy), auc_rec)

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = (8, 5),
        color: Optional[str] = "navy",
        linestyle: Optional[str] = "--",
        fontsize: Optional[float] = 15.0,
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = True,
        return_fig: Optional[bool] = False,
    ) -> Optional[Figure]:
        """Plots the REC curve.

        Parameters
        ----------
        figsize : tuple, optional, (default=(8, 5))
            Figure size, by default (8, 5)

        color : str, optional
            Color of the curve, by default "navy"

        linestyle : str, optional
            Line style, by default "--"

        fontsize : float, optional
            Fontsize for xlabel and ylabel, and ticks parameters, by default 15

        save_path : str, optional
            Relative or absolute save path, by default None

        display_plot : bool, optional
            Whether to display plot, by default True

        return_fig : bool, optional
            Whether to return figure object, by default False

        Returns
        -------
        Figure, optional
        """
        sns.set_style("ticks")
        mpl.rcParams["axes.linewidth"] = 3
        mpl.rcParams["lines.linewidth"] = 3

        if not isinstance(figsize, tuple):
            raise TypeError("Only tuple type is allowed for figsize.")
        if not isinstance(color, str):
            raise TypeError("Only str type is allowed for color.")
        if not isinstance(linestyle, str):
            raise TypeError("Only str type is allowed for linestyle.")
        if not isinstance(fontsize, float):
            raise TypeError("Only float type is allowed for fontsize.")
        if save_path is not None:
            if not isinstance(save_path, str):
                raise TypeError("Only str type is allowed for save_path.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            self.deviation,
            self.accuracy,
            color=color,
            linestyle=linestyle,
            label=f"AUC = {self.auc_rec:.3f}",
        )
        ax.set(
            xlim=[-0.01, 1.01],
            ylim=[-0.01, 1.01],
        )
        ax.set_xlabel(
            "Deviation",
            fontsize=fontsize,
        )
        ax.set_ylabel(
            "Accuracy",
            fontsize=fontsize,
        )
        ax.set_title(
            "REC Curve",
            fontsize=fontsize,
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=14,
        )
        ax.legend(
            prop={"size": fontsize},
            loc=4,
            framealpha=0.0,
        )
        if save_path:
            plt.savefig(
                save_path,
                bbox_inches="tight",
                dpi=200,
            )
        if display_plot:
            plt.show()

        if return_fig:
            return fig

        return None
