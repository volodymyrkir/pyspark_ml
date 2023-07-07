from feature_selection import get_features_df
from pre_processing import CovidPreProcessing, TitanicPreProcessing

from utils import split_dataframe

from models.lr_model import LogisticRegressionModel
from consts import COVID_TARGET_INDEX, TITANIC_TARGET_INDEX
from pipelines.base_pipeline import Pipeline


class LogisticRegressionCovid(Pipeline):

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

        model_handler = LogisticRegressionModel(train_df, test_df, 'covid', self.target,
                                                maxIter=20,
                                                regParam=0.1,
                                                elasticNetParam=0.0,
                                                family="binomial"
                                                )
        lr_model = model_handler.assemble_model()
        predictions = model_handler.get_predictions(lr_model)
        model_handler.evaluate_results(predictions)


class LogisticRegressionTitanic(Pipeline):

    def apply(self):
        df = TitanicPreProcessing(self.df).apply_transformations()

        features_dataframe = get_features_df(
            df,
            self.cols,
            self.target,
            TITANIC_TARGET_INDEX,
            True,
            True
        )

        train_df, test_df = split_dataframe(features_dataframe, [.7, .3])
        model_handler = LogisticRegressionModel(train_df, test_df, 'space_titanic', self.target,
                                                maxIter=20,
                                                regParam=0.1,
                                                elasticNetParam=0.3,
                                                family="binomial"
                                                )
        lr_model = model_handler.assemble_model()
        predictions = model_handler.get_predictions(lr_model)

        model_handler.evaluate_results(predictions)
