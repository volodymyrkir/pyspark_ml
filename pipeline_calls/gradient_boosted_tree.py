from pyspark.sql import SparkSession

from pipelines.gbt_pipeline import GBTCovid, GBTTitanic
from consts import (TITANIC_ALL_COLS, TITANIC_MAPPING, TITANIC_TARGET,
                    COVID_ALL_COLS, COVID_MAPPING, COVID_TARGET)

if __name__ == '__main__':
    session = SparkSession.builder.getOrCreate()

    gbt_covid_pipeline = GBTCovid(
        session,
        'covid.csv',
        COVID_ALL_COLS,
        COVID_TARGET,
        COVID_MAPPING,
    )
    gbt_covid_pipeline.apply()

    gbt_titanic_pipeline = GBTTitanic(
        session,
        'titanic.csv',
        TITANIC_ALL_COLS,
        TITANIC_TARGET,
        TITANIC_MAPPING,
    )
    gbt_titanic_pipeline.apply()
