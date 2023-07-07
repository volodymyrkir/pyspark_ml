from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.svc_model import SVCModel
from analytics.analytic_utils import get_covid_model, get_data_covid

tol_values = [1e-02, 1e-08, 1e-12, 1e-20]
f1s = [0.6217455554628022, 0.6239378670801959, 0.6202689633302418, 0.6217902475490249]


def evaluate_tol_param(df: DataFrame,
                      model_getter: Callable,
                      name: str,
                      ax: plt.Axes) -> int:
    for max_iter in tol_values:
        res = model_getter(df, SVCModel, maxIter=max_iter)
        print(res)
        f1s.append(res)

    sns.lineplot(x=tol_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of maxIter on F1 Score for {name} in SVC')

    return tol_values[f1s.index(max(f1s))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # covid = get_data_covid(session)
    #
    # covid_tol = get_covid_model(covid, SVCModel,
    #                                  maxIter=1, tol=tol_values[2])
    # print(covid_tol)
    # titanic_tol1 = get_covid_model(covid, SVCModel,
    #                                   maxIter=1, tol=tol_values[3])
    # print(titanic_tol1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=tol_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('tol value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of tol on F1 Score for covid')
    plt.show()
