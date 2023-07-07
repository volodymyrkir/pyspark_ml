from pyspark.ml.classification import GBTClassifier

from consts import FEATURES_NAME
from models.base_model import Model


class GBTModel(Model):

    def assemble_model(self):
        # self.standardize_features()
        lr_model = GBTClassifier(featuresCol=FEATURES_NAME, labelCol=self._label,
                                 **self.kwargs)

        lr_model = lr_model.fit(self._train_df)
        return lr_model
