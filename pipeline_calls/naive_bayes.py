from pyspark.sql import SparkSession

from pipelines.naive_bayes_pipeline import NaiveBayesTitanic, NaiveBayesCovid
from consts import (TITANIC_ALL_COLS, TITANIC_MAPPING, TITANIC_TARGET,
                    COVID_ALL_COLS, COVID_MAPPING, COVID_TARGET)

if __name__ == '__main__':
    session = SparkSession.builder.getOrCreate()

    bayes_covid_pipeline = NaiveBayesCovid(
        session,
        'covid.csv',
        COVID_ALL_COLS,
        COVID_TARGET,
        COVID_MAPPING,
    )
    bayes_covid_pipeline.apply()

    bayes_titanic_pipeline = NaiveBayesTitanic(
        session,
        'titanic.csv',
        TITANIC_ALL_COLS,
        TITANIC_TARGET,
        TITANIC_MAPPING,
    )
    bayes_titanic_pipeline.apply()
