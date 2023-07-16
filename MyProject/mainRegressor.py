
import readData as rd
import DecisionTreeRegressorCls as dtr
import RandomForestRegressorCls as rfr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from visualization import *


#************************************************
def main():
  
  #  input params -----------
  # ARGUMENTS:  
  # MLType = "DTR" # Decision Tree Regressor
  # MLType = "RFR" # Random Forest Regressor
  MLType = "CV" # cross validation

  # reading data ------------
  data = rd.DataPreprocessing(train_filename="train.csv", test_filename="test.csv", dataPath="./data/home-data-kaggle", 
                              train_size = 0.8, categoricalFeatures = 3, imputeStrategy="mean", target='SalePrice', 
                              # addImputeCol=True, debugMode = True)
                              addImputeCol=True, debugMode = False)
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.pickFeatures(feaetures)

  data.handleMissingValues()
  data.categoricalFeatures_processing()
  data.normalizeNumericalFeatures()
  print("done with preprocessing")

  def RandomForestWithCrossValidation(n_estimators, numberofCrossValidation: int):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    #pass
    print(f"random foerst with cross validation {n_estimators}: {numberofCrossValidation}")
    APipeline = Pipeline(steps=[
                                #('preprocessor', SimpleImputer()),
                                # ('separate', data.SeparateTarget()),
                                ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
                                ])
    return (-1*cross_val_score(APipeline, data.X, data.y.values.ravel(), cv = numberofCrossValidation, scoring='neg_mean_absolute_error')).mean()
  
  # decision tree classifier ---------------------------
  if MLType == "DTR":  
    data.splitData()
    myDecisionTree = dtr.myDecisionTreeRegressor(x_train = data.x_train, y_train = data.y_train, x_valid = data.x_valid, y_valid = data.y_valid)
    lowest_mae = 1e10
    best_tree_size = 0

    for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
      mae = myDecisionTree.trainDecisionTreeRegressor(max_leaf_nodes)
      if mae<lowest_mae: 
        best_tree_size= max_leaf_nodes
        lowest_mae = mae

      print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))

    print(f"lowest error {lowest_mae} at max leaf {best_tree_size}")
  
  
  # random forest classifier ---------------------------
  elif MLType == "RFR":
    data.splitData()
  
    myRF = rfr.myRandomForestRegressor(x_train = data.x_train, y_train = data.y_train, x_valid = data.x_valid, y_valid = data.y_valid)
    myRF.trainRandomForestRegressor()

  # elif MLType == "CV":
  else:
    
    print("cross validation in main Regressor")

    data.SeparateTarget()
    NumberOfCrossValidation = 5
    results = {i:RandomForestWithCrossValidation(i, NumberOfCrossValidation) for i in [j for j in range(50,450,50)]} 
    # results = {i:RandomForestWithCrossValidation(i, NumberOfCrossValidation) for i in [j for j in range(50,250,100)]} 
    visualization.dictionary(results)

  print("\n ==================")
  print(" end of the code")






if __name__== "__main__":
  main()



