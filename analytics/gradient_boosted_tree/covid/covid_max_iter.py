from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.gbt_model import GBTModel
from analytics.analytic_utils import get_covid_model, get_data_covid

max_iters = [5, 20, 50, 60]
f1s = [0.6266212943281915,0.628186116823973, 0.631858130639988, 0.6297877339200826]


def evaluate_step_param(df: DataFrame,
                        model_getter: Callable,
                        name: str,
                        ax: plt.Axes) -> int:
    for max_iter in max_iters:
        res = model_getter(df, GBTModel, maxIter=max_iter)
        print(res)
        f1s.append(res)

    sns.lineplot(x=max_iters, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('numTrees value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of stepSize on F1 Score for {name} in GBT')

    return max_iters[f1s.index(max(f1s))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # covid = get_data_covid(session)
    #
    # covid_trees = get_covid_model(covid, GBTModel,
    #                               maxDepth=5, stepSize=0.5,
    #                               maxIter=max_iters[2]
    #                               )
    # print(covid_trees)
    # covid_trees1 = get_covid_model(covid, GBTModel,
    #                                maxDepth=5, stepSize=0.5,
    #                                maxIter=max_iters[3])
    # print(covid_trees1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=max_iters, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxIter on F1 Score for covid')
    plt.show()
