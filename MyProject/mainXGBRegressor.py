
import readData as rd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from visualization import *
from helpers import *

from xgboost import XGBRegressor

#************************************************
def main():
  
  #TODO: pipeline +  cross validation

  #  input params -----------
  # ARGUMENTS:  
  MLType = True
  
  # various options of xgboost
  # specifies how many times to go through the modeling cycle. It is equal to the number of models that 
  # we include in the ensemble. Too low: underfitting, leads to inaccurate predictions on both training 
  # data and test data. Too high: overfitting, causes accurate predictions on training data, but inaccurate 
  # predictions on test data (which is what we care about). Typical values range from 100-1000, though 
  # this depends a lot on the learning_rate parameter discussed below.
  n_estimators = 500

  # early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. Early stopping causes 
  # the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for 
  # n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the 
  # optimal time to stop iterating. Since random chance sometimes causes a single round where validation scores 
  # don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. 
  # Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of 
  # deteriorating validation scores. When using early_stopping_rounds, you also need to set aside some data for 
  # calculating the validation scores - this is done by setting the eval_set parameter.
  early_stopping_rounds = 5 


  # Instead of getting predictions by simply adding up the predictions from each component model, we can multiply 
  # the predictions from each model by a small number (known as the learning rate) before adding them in. This means 
  # each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting. 
  # If we use early stopping, the appropriate number of trees will be determined automatically. In general, a small 
  # learning rate and large number of estimators will yield more accurate XGBoost models, though it will also 
  # take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets 
  # learning_rate=0.1.
  learning_rate = 0.05

  # On larger datasets where runtime is a consideration, you can use parallelism to build your models faster.
  n_jobs = 4

  # reading data ------------
  data = rd.DataPreprocessing(train_filename="train2.csv", test_filename="test2.csv", dataPath="./data/home-data-kaggle", 
                              train_size = 0.8, categoricalFeatures = 3, imputeStrategy="mean", Index_col = "Id",
                              target='SalePrice', addImputeCol=True, debugMode = True)
  
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.pickFeatures(feaetures)

  data.handleMissingValues()
  data.categoricalFeatures_processing()
  # data.normalizeNumericalFeatures()
  data.splitData()
  data.alignDataframes()
  
  print("done with preprocessing")

  # xg boosting -----------------------------------------------------
  if MLType:

    my_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs = n_jobs)

    my_model.fit(data.x_train, data.y_train,
             early_stopping_rounds=early_stopping_rounds, 
             eval_set=[(data.x_test, data.y_valid)],
             verbose=False)

    predictions = my_model.predict(data.x_test)
    print("Mean Absolute Error: " + str(helpers.MAE(predictions, data.y_valid)))


    helpers.MAE(self.y_valid, y_preds)



  print("\n ==================")
  print(" end of the code")

if __name__== "__main__":
  main()



