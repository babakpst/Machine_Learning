
from sklearn.metrics import mean_absolute_error

class helpers:
  
  @staticmethod
  def MAE(val_y, val_predictions):
    return mean_absolute_error(val_y, val_predictions)