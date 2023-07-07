import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as f

from utils import get_dataframe

from consts import TITANIC_ALL_COLS, TITANIC_MAPPING


if __name__ == '__main__':
    session = SparkSession.builder.getOrCreate()
    df = get_dataframe('titanic.csv', session, TITANIC_ALL_COLS, TITANIC_MAPPING)
    transported_counts = df.groupBy('Transported').count().toPandas()
    plt.figure(figsize=(6, 6))
    plt.pie(transported_counts['count'], labels=transported_counts['Transported'], autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Proportion of Passengers Transported')
    plt.show()

    luxury_amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    avg_billing_amounts = df.agg(*(f.avg(f.col(c)).alias(c) for c in luxury_amenities)).toPandas().transpose()
    avg_billing_amounts.columns = ['Average Billing Amount']
    plt.figure(figsize=(20, 20))
    avg_billing_amounts.plot(kind='bar')
    plt.xlabel('Luxury Amenities')
    plt.ylabel('Average Billing Amount')
    plt.title('Average Billing Amount by Luxury Amenities')
    plt.show()

    plt.figure(figsize=(15, 6))
    sns.histplot(data=df.toPandas(), x='Age', bins=20)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Distribution of Passenger Ages')
    plt.show()
