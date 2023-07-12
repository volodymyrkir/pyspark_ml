"""
This module provides visual charts for covid dataset.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, functions as f

from utils import get_dataframe
from analytics.utils import get_spark_session

from consts import COVID_MAPPING, COVID_ALL_COLS


def display_count_chart():
    """
    Displays count chart for classification column.
    """
    classification_counts = df.groupBy('CLASIFFICATION_FINAL').count().toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=classification_counts,
                x=classification_counts['CLASIFFICATION_FINAL'],
                y=classification_counts['count'])
    plt.xlabel('Patient Classification')
    plt.ylabel('Count')
    plt.title('Distribution of Patient Classifications')
    plt.show()


def display_distribution_chart(base_df: DataFrame):
    """
    Displays classification columns distribution.

    Args:
        base_df (DataFrame): The base df for this chart.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=base_df,
                x=base_df['CLASIFFICATION_FINAL'],
                y=base_df['count'])
    plt.xlabel('Patient Classification')
    plt.ylabel('Count')
    plt.title('Distribution of Patient Classifications')
    plt.show()


def display_patient_types_distribution(patient_types: DataFrame):
    """
    Displays classification columns distribution.

    Args:
        patient_types (DataFrame): The patient types df for this chart.
    """
    plt.figure(figsize=(6, 6))
    plt.pie(patient_types['count'], labels=patient_types['PATIENT_TYPE'], autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Distribution of Patient Types')
    plt.legend(['Returned Home', 'Hospitalization'])
    plt.show()


def display_deaths_line_chart(deaths_df: DataFrame):
    """
    Displays classification columns distribution.

    Args:
        deaths_df (DataFrame): The deaths df for this chart.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=deaths_df, x='DATE_DIED', y='count')
    plt.xlabel('Date')
    plt.ylabel('Daily Deaths')
    plt.title('Daily Deaths due to COVID-19')
    plt.show()


def display_medical_counts_chart(med_counts_df: DataFrame):
    """
    Displays classification columns distribution.

    Args:
        med_counts_df (DataFrame): The medical counts df for this chart.
    """
    plt.figure(figsize=(10, 6))
    plt.pie(med_counts_df['count'], labels=med_counts_df['MEDICAL_UNIT'], autopct='%1.1f%%', )
    plt.title('Distribution of Patients across Medical Units')
    plt.show()


if __name__ == '__main__':
    session = get_spark_session()

    df = get_dataframe('covid.csv', session, COVID_ALL_COLS, COVID_MAPPING)
    display_count_chart()

    cls_converted = df.withColumn(
        'CLASIFFICATION_FINAL',
        f.when(
            (f.col('CLASIFFICATION_FINAL') >= 1) &
            (f.col('CLASIFFICATION_FINAL') <= 3),
            1
        ).otherwise(0)
    ).groupBy('CLASIFFICATION_FINAL').count().toPandas()
    display_distribution_chart(cls_converted)

    patient_type_counts = df.groupBy('PATIENT_TYPE').count().toPandas()
    display_patient_types_distribution(patient_type_counts)

    daily_deaths = df.groupBy('DATE_DIED').count().sort('DATE_DIED').toPandas()
    display_deaths_line_chart(daily_deaths)

    medical_unit_counts = df.groupBy('MEDICAL_UNIT').count().orderBy('count').limit(5).toPandas()
    display_medical_counts_chart(medical_unit_counts)
