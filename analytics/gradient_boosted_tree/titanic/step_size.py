from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.gbt_model import GBTModel
from analytics.analytic_utils import get_titanic_model, get_data_titanic

step_sizes = [0.1, 0.5, 1]
f1s = [0.7933675106866225, 0.793492205389374, 0.7785991445371461]


def evaluate_step_param(df: DataFrame,
                        model_getter: Callable,
                        name: str,
                        ax: plt.Axes) -> int:
    for max_iter in step_sizes:
        res = model_getter(df, GBTModel, maxIter=max_iter)
        print(res)
        f1s.append(res)

    sns.lineplot(x=step_sizes, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('step Size value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of stepSize on F1 Score for {name} in GBT')

    return step_sizes[f1s.index(max(f1s))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # titanic = get_data_titanic(session, exclude_name=True)
    #
    # titanic_step = get_titanic_model(titanic, GBTModel,
    #                                  maxDepth=5, stepSize=step_sizes[2])
    # print(titanic_step)
    # titanic_step1 = get_titanic_model(titanic, GBTModel,
    #                                   maxDepth=5, stepSize=step_sizes[3])
    # print(titanic_step1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=step_sizes, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('Step Size')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of stepSize on F1 Score for titanic')
    plt.show()
