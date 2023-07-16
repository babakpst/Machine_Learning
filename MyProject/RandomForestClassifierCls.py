
import pandas as pd
from dataclasses import dataclass, field
from visualization import *
from sklearn.ensemble import RandomForestClassifier


@dataclass
class myRandomForestClassifier:
  x_train: pd.DataFrame = None
  y_train: pd.DataFrame = None

  x_valid: pd.DataFrame = None
  y_valid: pd.DataFrame = None

  x_test: pd.DataFrame = None
  y_test: pd.DataFrame = None

  n_estimators: int = 50

  def train(self):
    forest_model = RandomForestClassifier(n_estimators = self.n_estimators, random_state=1)
    forest_model.fit(self.x_train, self.y_train)
    y_pred = forest_model.predict(self.x_valid)

    visualization.model_evaluation(self.y_valid, y_pred)
  