"""
This module encapsulates visual charts for titanic df.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, functions as f

from utils import get_dataframe
from analytics.utils import get_spark_session

from consts import TITANIC_ALL_COLS, TITANIC_MAPPING


def display_target_pie(transported_df: DataFrame):
    """
    Displays pie chart for target variable of titanic df.

    Args:
        transported_df (DataFrame): transported dataframe.

    Returns:
        None

    """
    plt.figure(figsize=(6, 6))
    plt.pie(transported_df['count'], labels=transported_df['Transported'], autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Proportion of Passengers Transported')
    plt.show()


def display_billing_amounts(billing_df: DataFrame):
    """
    Displays bar chart for billing column of titanic df.

    Args:
        billing_df (DataFrame): billing dataframe.

    Returns:
        None

    """
    plt.figure(figsize=(20, 20))
    billing_df.plot(kind='bar')
    plt.xlabel('Luxury Amenities')
    plt.ylabel('Average Billing Amount')
    plt.title('Average Billing Amount by Luxury Amenities')
    plt.show()


def display_age_distribution(base_df: DataFrame):
    """
    Displays histogram chart for counts of ages for titanic df.

    Args:
        base_df (DataFrame): base titanic df dataframe.

    Returns:
        None

    """
    plt.figure(figsize=(15, 6))
    sns.histplot(data=base_df.toPandas(), x='Age', bins=20)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Distribution of Passenger Ages')
    plt.show()


if __name__ == '__main__':
    session = get_spark_session()
    df = get_dataframe('titanic.csv', session, TITANIC_ALL_COLS, TITANIC_MAPPING)
    transported_counts = df.groupBy('Transported').count().toPandas()
    display_target_pie(transported_counts)

    luxury_amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    avg_billing_amounts = (df.agg(*(f.avg(f.col(c)).alias(c)
                                    for c in luxury_amenities)
                                  ).toPandas().transpose())
    avg_billing_amounts.columns = ['Average Billing Amount']
    display_billing_amounts(avg_billing_amounts)

    display_age_distribution(df)
