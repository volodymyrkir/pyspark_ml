import os

from pyspark.sql import DataFrame
from pyspark.sql import types as t, functions as f
from pyspark.sql import SparkSession

from consts import COVID_DATE_FORMAT


def get_dataframe(name: str, session: SparkSession,
                  cols: list[str], type_mapping: dict,
                  date_format: str = COVID_DATE_FORMAT) -> DataFrame:
    return session.read.csv(
        os.path.join(os.getcwd(), 'input_data', name),
        schema=t.StructType([
            t.StructField(col, type_mapping[col])
            for col in cols
        ]),
        header=True,
        dateFormat=date_format,

    )


def fill_numeric_manual(df: DataFrame,
                        target_col: str,
                        median_cols: list[str],
                        mean_cols: list[str]) -> DataFrame:
    for median_column in median_cols:
        best_median_df = (df
                          .filter(f.col(median_column).isNotNull())
                          .groupBy(target_col)
                          .agg(f.percentile_approx(median_column, 0.5)
                               .alias(f"{median_column}_median")))

        joined_df = df.join(best_median_df, on=target_col, how='left')

        df = (joined_df
              .withColumn(median_column,
                          f.when(joined_df[median_column].isNull(),
                                 joined_df[f'{median_column}_median'])
                          .otherwise(joined_df[median_column]))
              .drop(f'{median_column}_median')
              )

    for mean_column in mean_cols:
        best_mean_df = (df
                        .filter(f.col(mean_column).isNotNull())
                        .groupBy(target_col)
                        .agg(f.percentile_approx(mean_column, 0.5)
                             .alias(f"{mean_column}_mean")))

        joined_df = df.join(best_mean_df, on=target_col, how='left')

        df = (joined_df
              .withColumn(mean_column,
                          f.when(joined_df[mean_column].isNull(),
                                 joined_df[f'{mean_column}_mean'])
                          .otherwise(joined_df[mean_column]))
              .drop(f'{mean_column}_mean')
              )

    return df


def split_dataframe(dataframe: DataFrame, thresholds: list) -> list[DataFrame]:
    dataframe = dataframe.orderBy(f.rand())
    return dataframe.randomSplit(thresholds, seed=42)
