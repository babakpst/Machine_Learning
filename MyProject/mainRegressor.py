
import readData as rd
import DecisionTreeRegressorCls as dtr
import RandomForestRegressorCls as rfr
import visualization

#************************************************
def main():
  
  # ARGUMENTS:  
  # "DTC": DecisionTree Classifier
  # "DTR": DecisionTree Regressor
  # "RFC": RandomForest Classifier
  # "RFR": RandomForest Regressor 
  # MLType = "DTR"
  MLType = "RFR"
  
  data = rd.DataPreprocessing(trainfilename="train.csv", testfilename="test.csv", dataPath="./data/home-data-kaggle", split = 0.8, categoricalFeatures = 3, imputeStrategy="mean", target='SalePrice', addImputeCol=True, debugMode = False)
  
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.pickFeatures(feaetures)

  data.handleMissingValues()
  data.categoricalFeatures_processing()
  data.normalizeNumericalFeatures()

  data.splitData()

  # decition tree classifier ---------------------------
  if MLType == "DTR":  
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
    myRF = rfr.myRandomForestRegressor(x_train = data.x_train, y_train = data.y_train, x_test = data.x_test, y_test = data.y_test)
    myRF.trainRandomForestRegressor()



  print("\n ==================")
  print(" end of the code")

if __name__== "__main__":
  main()