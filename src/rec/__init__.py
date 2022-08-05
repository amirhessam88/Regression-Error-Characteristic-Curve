from importlib.metadata import version

from rec._rec import RegressionErrorCharacteristic

__version__ = version("rec")

__all__ = [
    "RegressionErrorCharacteristic",
]
