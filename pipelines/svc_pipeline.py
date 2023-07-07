from feature_selection import get_features_df
from pre_processing import CovidPreProcessing, TitanicPreProcessing

from utils import split_dataframe

from models.svc_model import SVCModel
from consts import COVID_TARGET_INDEX, TITANIC_TARGET_INDEX, columns as c
from pipelines.base_pipeline import Pipeline


class SVCCovid(Pipeline):

    def apply(self):
        df = CovidPreProcessing(self.df).apply_transformations()

        features_dataframe = get_features_df(
            df,
            self.cols,
            self.target,
            COVID_TARGET_INDEX,
            True,
            False,
        )

        train_df, test_df = split_dataframe(features_dataframe, [.7, .3])

        model_handler = SVCModel(
            train_df,
            test_df,
            'covid',
            self.target,
            maxIter=1,
            tol=1e-02,

        )
        lr_model = model_handler.assemble_model()
        predictions = model_handler.get_predictions(lr_model)
        model_handler.evaluate_results(predictions)


class SVCTitanic(Pipeline):

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
        model_handler = SVCModel(
            train_df,
            test_df,
            'space_titanic',
            self.target,
            maxIter=50,
            tol=1e-02,
        )
        lr_model = model_handler.assemble_model()
        predictions = model_handler.get_predictions(lr_model)

        model_handler.evaluate_results(predictions)
