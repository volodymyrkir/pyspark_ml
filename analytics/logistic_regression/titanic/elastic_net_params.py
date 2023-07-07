from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame

from models.lr_model import LogisticRegressionModel
from analytics.analytic_utils import get_titanic_model, get_data_titanic

elastic_net_values = [0, 0.3, 0.8, 1]
f1s = [0.7633653029418795, 0.775284400097207, 0.34869186634468574, 0.34869186634468574]

def evaluate_elastic_net(df: DataFrame,
                         model_getter: Callable,
                         name: str,
                         max_iter: int,
                         reg_param: float,
                         ax: plt.Axes):
    results = []
    for elastic_net in elastic_net_values:
        res = model_getter(df, LogisticRegressionModel, family='binomial', maxIter=max_iter, regParam=reg_param,
                           elasticNetParam=elastic_net)
        results.append(res)

    sns.lineplot(x=elastic_net_values, y=results, marker='o', ax=ax)
    ax.set_xlabel('elasticNetParam value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of elasticNetParam on F1 Score for {name}')

    return results


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # titanic = get_data_titanic(session)
    #
    # titanic_net = get_titanic_model(titanic, LogisticRegressionModel,
    #                                 family='binomial', maxIter=30,
    #                                 regParam=0.1, elasticNetParam=elastic_net_values[2])
    # print(titanic_net)

    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=elastic_net_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('elasticNetParam value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of elasticNetParam on F1 Score for titanic')
    plt.show()
