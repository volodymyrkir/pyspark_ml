from feature_selection import get_features_df
from pre_processing import CovidPreProcessing, TitanicPreProcessing

from utils import split_dataframe

from models.gbt_model import GBTModel
from consts import COVID_TARGET_INDEX, TITANIC_TARGET_INDEX, columns as c
from pipelines.base_pipeline import Pipeline


class GBTCovid(Pipeline):

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

        model_handler = GBTModel(
            train_df,
            test_df,
            'covid',
            self.target,
            maxDepth=5,
            stepSize=0.5,
            maxIter=50
            )
        lr_model = model_handler.assemble_model()
        print(f'Model trained using {lr_model.getNumTrees} trees')
        predictions = model_handler.get_predictions(lr_model)
        model_handler.evaluate_results(predictions)


class GBTTitanic(Pipeline):

    def apply(self):
        df = TitanicPreProcessing(self.df).apply_transformations()
        df = df.drop(c.name)
        self.cols.remove(c.name)
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
        model_handler = GBTModel(
            train_df,
            test_df,
            'space_titanic',
            self.target,
            maxDepth=5,
            stepSize=0.5,
            maxIter=50,
        )
        lr_model = model_handler.assemble_model()
        predictions = model_handler.get_predictions(lr_model)

        model_handler.evaluate_results(predictions)
