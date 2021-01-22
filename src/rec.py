# -------------------------------
# rec.py
# Author: Amirhessam Tahmassebi
# Email: admin@amirhessam.com
# Website: www.amirhessam.com
# Date: November-07-2017
# -------------------------------

import numpy as np
import scipy as scp
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


class RegressionErrorCharacteristic:
    """Regression Error Characteristics (REC).
    This is wrapper to implement the REC algorithm.
    REC is implemented based on the following paper:
    Bi, J., & Bennett, K. P. (2003). Regression error characteristic curves.
    In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 43-50).
    https://www.aaai.org/Papers/ICML/2003/ICML03-009.pdf
    Parameters
    ----------
    y_true: numpy.array[int] or list[float]
        List of ground truth target (response) values
    y_pred: numpy.array[float] or list[float]
        List of predicted target values list[float]
    Attributes
    ----------
    auc_rec: float value between 0. and 1
        Area under REC curve.
    deviation:  numpy.array[float] or list[float]
        List of deviations to plot REC curve.
    accuracy:  numpy.array[float] or list[float]
        Calculated accuracy at each deviation to plot REC curve.
    plotting_dict: dict()
        Plotting object as a dictionary consists of all
        calculated metrics which was used to plot curves
    plot_rec(): Func
        Function to plot the REC curve.
    """

    def __init__(self, y_true, y_pred):
        if not isinstance(y_true, np.ndarray):
            self.y_true = np.array(y_true)
        else:
            self.y_true = y_true
        if not isinstance(y_pred, np.ndarray):
            self.y_pred = np.array(y_pred)
        else:
            self.y_pred = y_pred
        self.deviation, self.accuracy, self.auc_rec = self._rec_curve()

    def _rec_curve(self):
        """
        Function to calculate the rec curve elements: deviation, accuracy, auc.
        Simpson method is used as the integral method to calculate the area under
        regression error characteristics (REC).
        REC is implemented based on the following paper:
        Bi, J., & Bennett, K. P. (2003). Regression error characteristic curves.
        In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 43-50).
        https://www.aaai.org/Papers/ICML/2003/ICML03-009.pdf
        """
        begin = 0.0
        end = 1.0
        interval = 0.01
        accuracy = []
        deviation = np.arange(begin, end, interval)

        # main loop to calculate norm and compare with each deviation
        for i in range(len(deviation)):
            count = 0.0
            for j in range(len(self.y_true)):
                calc_norm = np.linalg.norm(self.y_true[j] - self.y_pred[j]) / np.sqrt(
                    np.linalg.norm(self.y_true[j]) ** 2
                    + np.linalg.norm(self.y_pred[j]) ** 2
                )
                if calc_norm < deviation[i]:
                    count += 1
            accuracy.append(count / len(self.y_true))

        auc_rec = scp.integrate.simps(accuracy, deviation) / end

        return deviation, accuracy, auc_rec

    def plot_rec(self, figsize=None, color=None, linestyle=None, fontsize=None):
        """Function to plot REC curve.
        Parameters
        ----------
        figsize: tuple, optional, (default=(8, 5))
            Figure size
        color: str, optional, (default="navy")
            Color of the curve.
        linestyle: str, optional, (default="--")
        fontsize: int or float, optional, (default=15)
            Fontsize for xlabel and ylabel, and ticks parameters
        """
        sns.set_style("ticks")
        mpl.rcParams["axes.linewidth"] = 3
        mpl.rcParams["lines.linewidth"] = 3

        # initializing figsize
        if figsize is None:
            figsize = (8, 5)
        elif isinstance(figsize, list) or isinstance(figsize, tuple):
            figsize = figsize
        else:
            raise TypeError("Only tuple and list types are allowed for figsize.")

        # initializing color
        if color is None:
            color = "navy"
        elif isinstance(color, str):
            color = color
        else:
            raise TypeError("Only str type is allowed for color.")

        # initializing linestyle
        if linestyle is None:
            linestyle = "--"
        elif isinstance(linestyle, str):
            linestyle = linestyle
        else:
            raise TypeError("Only str type is allowed for linestyle.")

        # initializing fontsize
        if fontsize is None:
            fontsize = 15
        elif isinstance(fontsize, float) or isinstance(fontsize, int):
            fontsize = fontsize
        else:
            raise TypeError("Only int and float types are allowed for fontsize.")

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
        ax.set_xlabel("Deviation", fontsize=fontsize)
        ax.set_ylabel("Accuracy", fontsize=fontsize)
        ax.set_title("REC Curve", fontsize=fontsize)

        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(prop={"size": fontsize}, loc=4, framealpha=0.0)

        plt.show()
