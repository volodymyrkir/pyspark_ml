from pyspark.sql import DataFrame, SparkSession

from consts import COVID_ALL_COLS, COVID_MAPPING, COVID_TARGET, TITANIC_ALL_COLS, TITANIC_MAPPING, TITANIC_TARGET
from pre_processing import PreProcessing, CovidPreProcessing, TitanicPreProcessing
from models.base_model import Model
from feature_selection import get_features_df
from utils import split_dataframe, get_dataframe


def get_data_covid(session: SparkSession):
    data = get_dataframe('covid.csv', session, COVID_ALL_COLS, COVID_MAPPING)
    processed_data = CovidPreProcessing(data).apply_transformations()
    features_df = get_features_df(processed_data, COVID_ALL_COLS, COVID_TARGET, 0, False, False)
    return features_df


def get_data_titanic(session: SparkSession, exclude_name = False):
    data = get_dataframe('titanic.csv', session, TITANIC_ALL_COLS, TITANIC_MAPPING)
    processed_data = TitanicPreProcessing(data).apply_transformations()
    if exclude_name:
        processed_data = processed_data.drop('Name')
        TITANIC_ALL_COLS.remove('Name')
    features_df = get_features_df(processed_data, TITANIC_ALL_COLS, TITANIC_TARGET, 0, False, False)
    return features_df


def get_covid_model(df, model: Model, **kwargs) -> float:
    return get_analytics(df, model, 'covid', COVID_TARGET, **kwargs)


def get_titanic_model(df, model: type(Model), **kwargs) -> float:
    return get_analytics(df, model,  'titanic', TITANIC_TARGET, **kwargs)


def get_analytics(df: DataFrame,
                  model: type(Model),
                  name: str,
                  label: str,
                  **kwargs) -> float:
    train, test = split_dataframe(df, [.8, .2])
    model_handler = model(train, test, name, label, **kwargs)
    model_obj = model_handler.assemble_model()
    preds = model_handler.get_predictions(model_obj)
    f1_score = model_handler.evaluate_results(preds)
    return f1_score
