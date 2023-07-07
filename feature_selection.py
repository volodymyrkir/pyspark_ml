from pyspark.sql import DataFrame
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

import seaborn as sns
import matplotlib.pyplot as plt

from consts import FEATURES_NAME


def assemble_features(df: DataFrame, columns: list[str]) -> DataFrame:
    assembler = VectorAssembler(inputCols=columns, outputCol=FEATURES_NAME)
    return assembler.transform(df)


def perform_correlation_analysis(df: DataFrame,
                                 cols: list[str],
                                 target: str,
                                 target_index: int,
                                 visualize_corr_matrix: bool,
                                 correlation_threshold: float) -> list[str]:
    assembler = assemble_features(df, cols)
    correlation_matrix = Correlation.corr(assembler, FEATURES_NAME).head()[0].toArray()

    if visualize_corr_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            xticklabels=cols,
            yticklabels=cols
        )
        plt.title("Correlation Matrix")
        plt.show()

    relevant_cols = []
    for col, col_name in zip(correlation_matrix[target_index], cols):
        if abs(col) < correlation_threshold:
            print(f'Discarding {col_name} attribute because of low correlation')
        else:
            relevant_cols.append(col_name)
            print(f'{col_name} attribute is left for training')

    relevant_cols.remove(target)
    return relevant_cols


def get_features_df(dataframe: DataFrame,
                    cols: list[str],
                    target: str,
                    target_index: int,
                    select_relevant_cols: bool = True,
                    visualize_corr_matrix: bool = True,
                    correlation_threshold: float = 0.01) -> DataFrame:

    relevant_columns = [col for col in cols if col != target]
    if select_relevant_cols:
        relevant_columns = perform_correlation_analysis(
            dataframe,
            cols,
            target,
            target_index,
            visualize_corr_matrix,
            correlation_threshold
        )

    assembler_df = assemble_features(dataframe, relevant_columns)
    return assembler_df
