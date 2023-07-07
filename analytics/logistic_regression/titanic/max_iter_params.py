from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame

from models.lr_model import LogisticRegressionModel
from analytics.analytic_utils import get_titanic_model, get_data_titanic

max_iter_values = [1, 5, 20, 50]
f1s = [0.7045749296739043, 0.7316735789405941, 0.7497514163487109, 0.7497514163487109]


def evaluate_max_iter(df: DataFrame,
                      model_getter: Callable,
                      name: str,
                      ax: plt.Axes) -> int:
    results = []
    for max_iter in max_iter_values:
        res = model_getter(df, LogisticRegressionModel, family='binomial', maxIter=max_iter)
        print(res)
        results.append(res)

    sns.lineplot(x=max_iter_values, y=results, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxIter on F1 Score for {name}')

    return max_iter_values[results.index(max(results))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    fig, ax = plt.subplots(1, 1)
    # titanic = get_data_titanic(session)
    #
    # titanic_iter = get_titanic_model(titanic, LogisticRegressionModel,
    #                                        family='binomial', maxIter=max_iter_values[0])
    # print(titanic_iter)
    sns.lineplot(x=max_iter_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxIter on F1 Score for titanic')
    plt.show()
