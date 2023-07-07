from pyspark.ml.classification import LinearSVC

from consts import FEATURES_NAME
from models.base_model import Model


class SVCModel(Model):

    def assemble_model(self):
        self.standardize_features()
        lr_model = LinearSVC(featuresCol=FEATURES_NAME, labelCol=self._label,
                                      **self.kwargs)

        lr_model = lr_model.fit(self._train_df)
        return lr_model

