from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame

from models.lr_model import LogisticRegressionModel
from analytics.analytic_utils import get_covid_model, get_data_covid

reg_param_values = [0.1, 0.3, 0.5, 0.7]
f1s = [0.6253506830017009, 0.6233509447698835, 0.6117693362032285, 0.5843909376350913
]


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

    # titanic = get_data_covid(session)
    #
    # covid_reg = get_covid_model(titanic, LogisticRegressionModel,
    #                                 family='binomial', maxIter=30,
    #                                 regParam=reg_param_values[2])
    # print(covid_reg)
    #
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=reg_param_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('regParam value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of regParam on F1 Score for covid')
    plt.show()

