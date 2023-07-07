from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.naive_bayes_model import NaiveBayesModel
from analytics.analytic_utils import get_titanic_model, get_data_titanic

smoothing_values = [0.0, 0.1, 0.5, 1.0]
f1s = [0.7464687633644654, 0.7464687633644654, 0.7464687633644654, 0.7464687633644654]


def evaluate_smoothing(df: DataFrame,
                      model_getter: Callable,
                      name: str,
                      ax: plt.Axes) -> int:
    for max_iter in smoothing_values:
        res = model_getter(df, NaiveBayesModel, maxIter=max_iter)
        print(res)
        f1s.append(res)

    sns.lineplot(x=smoothing_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of smoothing on F1 Score for {name} in SVC')

    return smoothing_values[f1s.index(max(f1s))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # titanic = get_data_titanic(session)
    #
    # titanic_smoothing = get_titanic_model(titanic, NaiveBayesModel,
    #                                  smoothing=smoothing_values[2],
    #                                  modelType="gaussian",)
    # print(titanic_smoothing)
    # titanic_smoothing1 = get_titanic_model(titanic, NaiveBayesModel,
    #                                   smoothing=smoothing_values[3],
    #                                   modelType="gaussian",)
    # print(titanic_smoothing1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=smoothing_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('smoothing value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of smoothing on F1 Score for titanic')
    plt.show()
