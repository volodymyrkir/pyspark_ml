from pyspark.sql import SparkSession

from pipelines.logistic_regression_pipeline import LogisticRegressionCovid, LogisticRegressionTitanic
from consts import (TITANIC_ALL_COLS, TITANIC_MAPPING, TITANIC_TARGET,
                    COVID_ALL_COLS, COVID_MAPPING, COVID_TARGET)

if __name__ == '__main__':
    session = SparkSession.builder.getOrCreate()

    # lr_covid_pipeline = LogisticRegressionCovid(
    #     session,
    #     'covid.csv',
    #     COVID_ALL_COLS,
    #     COVID_TARGET,
    #     COVID_MAPPING,
    # )
    # lr_covid_pipeline.apply()

    lr_titanic_pipeline = LogisticRegressionTitanic(
        session,
        'titanic.csv',
        TITANIC_ALL_COLS,
        TITANIC_TARGET,
        TITANIC_MAPPING,
    )
    lr_titanic_pipeline.apply()
