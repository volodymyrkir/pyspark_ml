from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame

from models.lr_model import LogisticRegressionModel
from analytics.analytic_utils import get_titanic_model, get_data_titanic

reg_param_values = [0.1, 0.3, 0.5, 0.7]
f1s = [0.7633653029418795, 0.755743276891687, 0.7510451412287849, 0.7506962184362196]


def evaluate_reg_param(df: DataFrame,
                       model_getter: Callable,
                       name: str,
                       max_iter: int,
                       ax: plt.Axes):
    results = []
    for reg_param in reg_param_values:
        res = model_getter(df, LogisticRegressionModel, family='binomial', maxIter=max_iter, regParam=reg_param)
        print(res)
        results.append(res)

    sns.lineplot(x=reg_param_values, y=results, marker='o', ax=ax)
    ax.set_xlabel('regParam value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of regParam on F1 Score for {name}')

    return reg_param_values[results.index(max(results))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    fig, ax = plt.subplots(1, 1)
    # titanic = get_data_titanic(session)
    #
    # titanic_reg = get_titanic_model(titanic, LogisticRegressionModel,
    #                                 family='binomial', maxIter=30,
    #                                 regParam=reg_param_values[2])
    # print(titanic_reg)

    sns.lineplot(x=reg_param_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('regParam value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of regParam on F1 Score for titanic')
    plt.show()

