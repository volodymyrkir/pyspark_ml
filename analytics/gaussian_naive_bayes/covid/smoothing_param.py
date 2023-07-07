from typing import Callable

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession

from models.naive_bayes_model import NaiveBayesModel
from analytics.analytic_utils import get_covid_model, get_data_covid

smoothing_values = [0.0, 0.1, 0.5, 1.0]
f1s = [0.5554025511184411, 0.5575882317905294, 0.5565689297440075, 0.5724592390944014]


def evaluate_smoothing(df: DataFrame,
                       model_getter: Callable,
                       name: str,
                       ax: plt.Axes) -> int:
    for max_iter in smoothing_values:
        res = model_getter(df, NaiveBayesModel, maxIter=max_iter)
        print(res)
        f1s.append(res)

    sns.lineplot(x=smoothing_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('maxIter value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of smoothing on F1 Score for {name} in SVC')

    return smoothing_values[f1s.index(max(f1s))]


if __name__ == '__main__':
    # session = SparkSession.builder.getOrCreate()
    #
    # covid = get_data_covid(session)
    #
    # covid_smoothing = get_covid_model(covid, NaiveBayesModel,
    #                                   smoothing=smoothing_values[2],
    #                                   modelType="gaussian", )
    # print(covid_smoothing)
    # covid_smoothing1 = get_covid_model(covid, NaiveBayesModel,
    #                                    smoothing=smoothing_values[3],
    #                                    modelType="gaussian", )
    # print(covid_smoothing1)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x=smoothing_values, y=f1s, marker='o', ax=ax)
    ax.set_xlabel('smoothing value')
    ax.set_ylabel('F1 score')
    ax.set_title(f'Effect of smoothing on F1 Score for covid')
    plt.show()
