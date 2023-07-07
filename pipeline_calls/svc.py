from pyspark.sql import SparkSession

from pipelines.svc_pipeline import SVCCovid, SVCTitanic
from consts import (TITANIC_ALL_COLS, TITANIC_MAPPING, TITANIC_TARGET,
                    COVID_ALL_COLS, COVID_MAPPING, COVID_TARGET)

if __name__ == '__main__':
    session = SparkSession.builder.getOrCreate()

    svc_covid_pipeline = SVCCovid(
        session,
        'covid.csv',
        COVID_ALL_COLS,
        COVID_TARGET,
        COVID_MAPPING,
    )
    svc_covid_pipeline.apply()

    svc_titanic_pipeline = SVCTitanic(
        session,
        'titanic.csv',
        TITANIC_ALL_COLS,
        TITANIC_TARGET,
        TITANIC_MAPPING,
    )
    svc_titanic_pipeline.apply()
