import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as f

from utils import get_dataframe

from consts import COVID_MAPPING, COVID_TARGET, COVID_ALL_COLS


if __name__ == '__main__':
    session = SparkSession.builder.getOrCreate()
    df = get_dataframe('covid.csv', session, COVID_ALL_COLS, COVID_MAPPING)
    classification_counts = df.groupBy('CLASIFFICATION_FINAL').count().toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=classification_counts,
                x=classification_counts['CLASIFFICATION_FINAL'],
                y=classification_counts['count'])
    plt.xlabel('Patient Classification')
    plt.ylabel('Count')
    plt.title('Distribution of Patient Classifications')
    plt.show()

    сls_converted = df.withColumn(
        'CLASIFFICATION_FINAL',
        f.when(
            (f.col('CLASIFFICATION_FINAL') >= 1) &
            (f.col('CLASIFFICATION_FINAL') <= 3),
            1
        ).otherwise(0)
    ).groupBy('CLASIFFICATION_FINAL').count().toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=classification_counts,
                x=сls_converted['CLASIFFICATION_FINAL'],
                y=сls_converted['count'])
    plt.xlabel('Patient Classification')
    plt.ylabel('Count')
    plt.title('Distribution of Patient Classifications')
    plt.show()

    patient_type_counts = df.groupBy('PATIENT_TYPE').count().toPandas()
    plt.figure(figsize=(6, 6))
    plt.pie(patient_type_counts['count'], labels=patient_type_counts['PATIENT_TYPE'], autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Distribution of Patient Types')
    plt.legend(['Returned Home', 'Hospitalization'])
    plt.show()

    daily_deaths = df.groupBy('DATE_DIED').count().sort('DATE_DIED').toPandas()
    print(daily_deaths)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=daily_deaths, x='DATE_DIED', y='count')
    plt.xlabel('Date')
    plt.ylabel('Daily Deaths')
    plt.title('Daily Deaths due to COVID-19')
    plt.show()

    medical_unit_counts = df.groupBy('MEDICAL_UNIT').count().orderBy('count').limit(5).toPandas()
    plt.figure(figsize=(10, 6))
    plt.pie(medical_unit_counts['count'], labels=medical_unit_counts['MEDICAL_UNIT'], autopct='%1.1f%%',)
    plt.title('Distribution of Patients across Medical Units')
    plt.show()


