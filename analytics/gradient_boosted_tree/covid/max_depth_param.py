from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.gbt_model import GBTModel
from analytics.analytic_utils import get_covid_model, get_data_covid

max_depth_values = [2, 5, 20, 25]
f1s = [0.6045937994509475, 0.6276887024973792, 0.6132887024973792, 0.5944887024973792]


def evaluate_max_iter(df: DataFrame,
                      model_getter: Callable,
                      name: str,
                      ax: plt.Axes) -> int:
    for max_iter in max_depth_values:
        res = model_getter(df, GBTModel, maxIter=max_iter)
        print(res)
        f1s.append(res)

    sns.lineplot(x=max_depth_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxDepth value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxDepth on F1 Score for {name} in SVC')

    return max_depth_values[f1s.index(max(f1s))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # covid = get_data_covid(session)
    #
    # covid_depth = get_covid_model(covid, GBTModel,
    #                               maxDepth=max_depth_values[2])
    # print(covid_depth)
    # covid_depth1 = get_covid_model(covid, GBTModel,
    #                                 maxDepth=max_depth_values[3])
    # print(covid_depth1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=max_depth_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxDepth value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxDepth on F1 Score for covid')
    plt.show()
