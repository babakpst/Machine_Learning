
import readData as rd
import DecisionTreeClassifierCls as dtc
import RandomForestClassifierCls as rfc
from visualization import *

#************************************************
def main():
  
  # ARGUMENTS:  
  # "DTC": DecisionTree Classifier
  # "DTR": DecisionTree Regressor
  # "RFC": RandomForest Classifier
  # "RFR": RandomForest Regressor 
  # MLType = "DTC"
  MLType = "RFC"
  
  data = rd.DataPreprocessing(train_filename="BankMarketingData.csv", test_filename="", dataPath="./data", 
                              train_size = 0.8, categoricalFeatures = 3, imputeStrategy="mean", target='y',
                              addImputeCol=True, debugMode = True)
  # data = rd.DataPreprocessing(train_filename="PhishingWebsitesData.csv", test_filename="", dataPath="./data", 
  #                             train_size = 0.8, categoricalFeatures = 1, imputeStrategy="mean", target='Result', 
  #                             addImputeCol=True, debugMode = False)
  
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.pickFeatures(feaetures)

  data.handleMissingValues()
  
  
  data.categoricalFeatures_processing()

  data.splitData()
  # data.convert_target_to_ordinal()
  # data.make_mi_scores()


  # decision tree classifier ---------------------------
  if MLType == "DTC":
    print("\n Decision Tree Classifier: ")
    myDecisionTree = dtc.myDecisionTreeClassifier(x_train = data.x_train, y_train = data.y_train, 
                                                  x_valid = data.x_valid, y_valid = data.y_valid)
    
    # training a decision tree classifier with max depth optimizer (simpler decision tree). 
    myDecisionTree.FindBestDTwithDepth()

    print(round(0.001*len(data.x_train)), round(0.005*len(data.x_train)),round(0.05*len(data.x_train)), round(0.1*len(data.x_train)))

    print("working on the comprehensive grid search ")
    max_depth, min_samples_leaf, max_leaf_nodes = myDecisionTree.DecisionTreeOptimizer_GridSearchCV( 
                                                  round(0.001*len(data.x_train)), round(0.003*len(data.x_train)),
                                                  10,15,
                                                  round(0.01*len(data.x_train)), round(0.1*len(data.x_train))
                                                  )

    print(" best tree")
    estimator_data = dtc.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                            max_leaf_nodes=max_leaf_nodes, random_state=100, criterion='entropy')

    # train_samp_data, DT_train_score_data, DT_fit_time_data, DT_pred_time_data = visualization.plot_learning_curve(clf = estimator_data, X=data.x_train, y=data.y_train, title="Decision Tree")
    visualization.final_classifier_evaluation(clf=estimator_data, X_train = data.x_train, x_valid = data.x_valid, 
                                              y_train = data.y_train, y_valid = data.y_valid)

  # random forest classifier ---------------------------
  elif MLType == "RFC":
    print("\n Random Forest Classifier: ")
    myRF = rfc.myRandomForestClassifier(x_train = data.x_train, y_train = data.y_train, x_valid = data.x_valid, y_valid = data.y_valid)
    myRF.train()

  print("\n ==================")
  print(" end of the code")

if __name__== "__main__":
  main()