import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


from consts import FEATURES_NAME
from models.base_model import Model


class LogisticRegressionModel(Model):

    def assemble_model(self):
        self.standardize_features()
        lr_model = LogisticRegression(featuresCol=FEATURES_NAME, labelCol=self._label,
                                      **self.kwargs)
        # best_model = self.tune_threshold(lr_model, self._train_df)
        lr_model = lr_model.fit(self._train_df)
        return lr_model

    def tune_threshold(self, model, data) -> float:
        evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol=self._label)

        paramGrid = (ParamGridBuilder()
                     .addGrid(model.threshold, list(np.arange(0.5, 0.6, 0.01)))
                     .build())

        crossval = CrossValidator(estimator=model,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5)

        cvModel = crossval.fit(data)
        bestModel = cvModel.bestModel

        return bestModel.getThreshold()
