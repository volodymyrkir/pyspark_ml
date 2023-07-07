from abc import ABC, abstractmethod

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame, functions as f

from utils import fill_numeric_manual
from consts import (TITANIC_NULL_COLUMNS, TITANIC_TARGET, TITANIC_MEAN_COLS, TITANIC_CAT_COLUMNS,
                    TITANIC_MEDIAN_COLS, TITANIC_FEATURE_COLS, columns as c)


class PreProcessing(ABC):

    def __init__(self, df: DataFrame):
        self.df = df

    @abstractmethod
    def apply_transformations(self) -> DataFrame:
        ...


class CovidPreProcessing(PreProcessing):

    def _encode_date(self):
        self.df = self.df.withColumn(
            c.date_died,
            f.when(f.col(c.date_died).isNotNull(), 1).otherwise(0)
        )

    def _convert_target(self):
        self.df = self.df.withColumn(
            c.classification_final,
            f.when(
                (f.col(c.classification_final) >= 1) &
                (f.col(c.classification_final) <= 3),
                1
            ).otherwise(0)
        )

    def apply_transformations(self) -> DataFrame:
        self._encode_date()
        self._convert_target()
        return self.df


class TitanicPreProcessing(PreProcessing):

    def index_strings(self):

        indexers = [StringIndexer(inputCol=col, outputCol=col + 'index').fit(self.df)
                    for col in TITANIC_CAT_COLUMNS]
        for indexer, col in zip(indexers, TITANIC_CAT_COLUMNS):
            self.df = indexer.transform(self.df)
            self.df = self.df.drop(col).withColumnRenamed(col + 'index', col)

    def _fill_categorical_nas(self):
        for column in TITANIC_NULL_COLUMNS:
            best_df = (self.df
                       .filter(f.col(column).isNotNull())
                       .groupBy(TITANIC_TARGET, column)
                       .count()
                       .orderBy(f.col("count").desc())
                       .groupBy(TITANIC_TARGET)
                       .agg(f.first(column).alias(f'{column}_best'))
                       )
            joined_df = self.df.join(best_df, on=TITANIC_TARGET, how='left')

            self.df = (joined_df
                       .withColumn(column,
                                   f.when(joined_df[column].isNull(),
                                          joined_df[f'{column}_best'])
                                   .otherwise(joined_df[column]))
                       .drop(f'{column}_best')
                       )

    def _extract_features(self):
        self.df = (self.df
                   .withColumn(c.passenger_id,
                               f.split(c.passenger_id, '_').getItem(1))
                   .withColumn(c.name,
                               f.split(c.name, ' ').getItem(1))
                   .withColumn(c.cabin,
                               f.split(c.cabin, '/').getItem(0))
                   )

    def apply_transformations(self) -> DataFrame:
        self._extract_features()
        self.df = fill_numeric_manual(self.df, TITANIC_TARGET, TITANIC_MEDIAN_COLS, TITANIC_MEAN_COLS)
        self._fill_categorical_nas()
        self.index_strings()
        return self.df
