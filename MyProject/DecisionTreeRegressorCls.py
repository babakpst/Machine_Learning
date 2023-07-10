
import pandas as pd
from dataclasses import dataclass
from helpers import *
from sklearn.tree import DecisionTreeRegressor

@dataclass
class myDecisionTreeRegressor:
  """_summary_
  """
  x_train: pd.DataFrame = None
  y_train: pd.DataFrame = None

  x_test: pd.DataFrame = None
  y_test: pd.DataFrame = None  

  def __post_init__(self):
    pass


  def trainDecisionTreeRegressor(self, max_leaf_nodes):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(self.x_train, self.y_train)
    y_preds = model.predict(self.x_test)
    mae = mean_absolute_error(self.y_test, y_preds)
    return (mae)