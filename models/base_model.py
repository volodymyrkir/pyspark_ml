from abc import ABC, abstractmethod

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.sql import DataFrame

from consts import FEATURES_NAME


class Model(ABC):

    def __init__(self,
                 train_df: DataFrame,
                 test_df: DataFrame,
                 name: str,
                 label: str,
                 **kwargs):
        self._train_df = train_df
        self._test_df = test_df
        self._name = name
        self._label = label
        self.kwargs = kwargs

    @abstractmethod
    def assemble_model(self):
        pass

    def get_predictions(self, model) -> DataFrame:
        return model.transform(self._test_df)

    def standardize_features(self):
        scaler = StandardScaler(inputCol=FEATURES_NAME,
                                outputCol="scaledFeatures",
                                withStd=False,
                                withMean=True)
        scaler = scaler.fit(self._train_df)
        self._train_df = (scaler
                          .transform(self._train_df)
                          .drop(FEATURES_NAME)
                          .withColumnRenamed('scaledFeatures', FEATURES_NAME))

    def evaluate_results(self, predictions):
        evaluator = MulticlassClassificationEvaluator(labelCol=self._label, predictionCol='prediction')

        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

        print(f'{type(self).__name__} is applied on {self._name} dataset')
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")

        return f1_score
