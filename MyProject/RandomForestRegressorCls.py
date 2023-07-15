
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from helpers import *

@dataclass
class myRandomForestRegressor:
  """_summary_
  """
  x_train: pd.DataFrame = None
  y_train: pd.DataFrame = None

  x_valid: pd.DataFrame = None
  y_valid: pd.DataFrame = None

  x_test: pd.DataFrame = None
  y_test: pd.DataFrame = None  

  n_estimators: int = 50

  def __post_init__(self):
    pass
  
  def trainRandomForestRegressor(self):
    rf_model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=1)

    rf_model.fit(self.x_train, self.y_train)
    rf_val_mae = mean_absolute_error(rf_model.predict(self.x_valid),self.y_valid)

    print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))





