from pyspark.ml.classification import NaiveBayes

from consts import FEATURES_NAME
from models.base_model import Model


class NaiveBayesModel(Model):

    def assemble_model(self):
        self.standardize_features()
        lr_model = NaiveBayes(featuresCol=FEATURES_NAME, labelCol=self._label,
                              **self.kwargs)

        lr_model = lr_model.fit(self._train_df)
        return lr_model
