
import pandas as pd
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier


@dataclass
class myRandomForestClassifier:
  x_train: pd.DataFrame = None
  y_train: pd.DataFrame = None

  x_test: pd.DataFrame = None
  y_test: pd.DataFrame = None

  def train(self):
    forest_model = RandomForestClassifier(random_state=1)
    forest_model.fit(self.x_train, self.y_train)
    y_pred = forest_model.predict(self.x_test)

    visualization.model_evaluation(self.y_test, y_pred)
  