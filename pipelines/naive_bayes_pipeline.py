from feature_selection import get_features_df
from pre_processing import CovidPreProcessing, TitanicPreProcessing

from utils import split_dataframe

from models.naive_bayes_model import NaiveBayesModel
from consts import COVID_TARGET_INDEX, TITANIC_TARGET_INDEX, columns as c
from pipelines.base_pipeline import Pipeline


class NaiveBayesCovid(Pipeline):

    def apply(self):
        df = CovidPreProcessing(self.df).apply_transformations()

        features_dataframe = get_features_df(
            df,
            self.cols,
            self.target,
            COVID_TARGET_INDEX,
            False,
            False,
        )

        train_df, test_df = split_dataframe(features_dataframe, [.7, .3])

        model_handler = NaiveBayesModel(
            train_df,
            test_df,
            'covid',
            self.target,
            smoothing=1.0,
            modelType="gaussian",
            )
        lr_model = model_handler.assemble_model()
        predictions = model_handler.get_predictions(lr_model)
        model_handler.evaluate_results(predictions)


class NaiveBayesTitanic(Pipeline):

    def apply(self):
        df = TitanicPreProcessing(self.df).apply_transformations()

        features_dataframe = get_features_df(
            df,
            self.cols,
            self.target,
            TITANIC_TARGET_INDEX,
            False,
            False,
            0.1
        )

        train_df, test_df = split_dataframe(features_dataframe, [.7, .3])
        model_handler = NaiveBayesModel(
            train_df,
            test_df,
            'space_titanic',
            self.target,
            smoothing=1.0,
            modelType="gaussian",
        )
        lr_model = model_handler.assemble_model()
        predictions = model_handler.get_predictions(lr_model)

        model_handler.evaluate_results(predictions)
