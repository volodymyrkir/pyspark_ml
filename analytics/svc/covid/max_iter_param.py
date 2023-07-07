from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.svc_model import SVCModel
from analytics.analytic_utils import get_covid_model, get_data_covid

max_iter_values = [1, 5, 60, 70]
f1s = [0.6239760184286545, 0.6160536884964728, 0.5649497242980944, 0.6025648168641282]


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
    # covid = get_data_covid(session)
    #
    # covid_iter = get_covid_model(covid, SVCModel,
    #                                maxIter=max_iter_values[2])
    # print(covid_iter)
    # covid_iter1 = get_covid_model(covid, SVCModel,
    #                                 maxIter=max_iter_values[3])
    # print(covid_iter1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=max_iter_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxIter on F1 Score for covid')
    plt.show()
