from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.svc_model import SVCModel
from analytics.analytic_utils import get_titanic_model, get_data_titanic

max_iter_values = [1, 5, 20, 50]
f1s = [0.7507806163972288, 0.7802847012476397, 0.7895938044638222, 0.7913757793407123]


def evaluate_max_iter(df: DataFrame,
                      model_getter: Callable,
                      name: str,
                      ax: plt.Axes) -> int:
    for max_iter in max_iter_values:
        res = model_getter(df, SVCModel, maxIter=max_iter)
        print(res)
        f1s.append(res)

    sns.lineplot(x=max_iter_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxIter on F1 Score for {name} in SVC')

    return max_iter_values[f1s.index(max(f1s))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # titanic = get_data_titanic(session)
    #
    # titanic_iter = get_titanic_model(titanic, SVCModel,
    #                                  maxIter=max_iter_values[2])
    # print(titanic_iter)
    # titanic_iter1 = get_titanic_model(titanic, SVCModel,
    #                                   maxIter=max_iter_values[3])
    # print(titanic_iter1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=max_iter_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxIter on F1 Score for titanic')
    plt.show()
