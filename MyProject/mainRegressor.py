
import readData as rd
import DecisionTreeRegressorCls as dtr
import RandomForestRegressorCls as rfr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from visualization import *


#************************************************
def main():
  
  # ARGUMENTS:  
  # "DTC": DecisionTree Classifier
  # "DTR": DecisionTree Regressor
  # "RFC": RandomForest Classifier
  # "RFR": RandomForest Regressor 
  
  # MLType = "DTR"
  # MLType = "RFR"
  MLType = "CrossValidation"
  
  data = rd.DataPreprocessing(trainfilename="train.csv", testfilename="test.csv", dataPath="./data/home-data-kaggle", split = 0.8, categoricalFeatures = 3, imputeStrategy="mean", target='SalePrice', addImputeCol=True, debugMode = False)
  
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.pickFeatures(feaetures)

  data.handleMissingValues()
  data.categoricalFeatures_processing()
  data.normalizeNumericalFeatures()


  # decition tree classifier ---------------------------
  if MLType == "DTR":  
    data.splitData()
    myDecisionTree = dtr.myDecisionTreeRegressor(x_train = data.x_train, y_train = data.y_train, x_test = data.x_test, y_test = data.y_test)
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
  
    myRF = rfr.myRandomForestRegressor(x_train = data.x_train, y_train = data.y_train, x_test = data.x_test, y_test = data.y_test)
    myRF.trainRandomForestRegressor()

  elif MLType == "CrossValidation":
    
    def RandomForestWithCrossValidation(n_estimators, numberofCrossValidation: int):
      """Return the average MAE over 3 CV folds of random forest model.
      
      Keyword argument:
      n_estimators -- the number of trees in the forest
      """
      # Replace this body with your own code
      #pass
      APipeline = Pipeline(steps=[
                                  #('preprocessor', SimpleImputer()),
                                  # ('separate', data.SeparateTarget()),
                                  ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
                                  ])
      
      return (-1*cross_val_score(APipeline, data.X, data.y.values.ravel(), cv = numberofCrossValidation, scoring='neg_mean_absolute_error')).mean()

    data.SeparateTarget()
    
    NumberOfCrossValidation = 5
    
    results = {i:RandomForestWithCrossValidation(i, NumberOfCrossValidation) for i in [j for j in range(50,450,50)]} 

    visualization.dictionary(results)



  print("\n ==================")
  print(" end of the code")

if __name__== "__main__":
  main()



