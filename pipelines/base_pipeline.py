from abc import ABC, abstractmethod

from pyspark.sql import SparkSession

from utils import get_dataframe


class Pipeline(ABC):
    """
    Base class for ML pipelines
    """
    def __init__(self,
                 session: SparkSession,
                 df_name: str,
                 cols: list[str],
                 target: str,
                 mapping: dict):
        self.df_name = df_name
        self.session = session
        self.cols = cols
        self.target = target
        self.mapping = mapping
        self.df = get_dataframe(self.df_name, self.session, self.cols, self.mapping)

    @abstractmethod
    def apply(self):
        ...
