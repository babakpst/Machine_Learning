
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import os
#import argparser
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import cross_validate

from visualization import *

@dataclass
class myDecisionTreeClassifier:
  """_summary_
  """

  maxdepth: int = 32
  x_train: pd.DataFrame = None
  y_train: pd.DataFrame = None

  x_valid: pd.DataFrame = None
  y_valid: pd.DataFrame = None

  x_test: pd.DataFrame = None
  y_test: pd.DataFrame = None

  def __post_init__(self):
    pass


  # notes
  # f1_score: a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
  # F1 = 2 * (precision * recall) / (precision + recall)

  # train decision tree, with max depth searcher
  def FindBestDTwithDepth(self):

    f1_train = [] # to store the score of training dataset
    f1_test = [] # to store the score of test dataset
    max_depth = list(range(1, self.maxdepth)) # for plotting

    maxScore = 0
    for i in max_depth:         
        DTclassifier = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=1, criterion='entropy')
        DTclassifier.fit(self.x_train, self.y_train)

        y_pred_train = DTclassifier.predict(self.x_train)
        f1_train.append(f1_score(self.y_train, y_pred_train, pos_label='yes' if self.y_train.dtypes[0] == object else 1) )

        y_pred_valid = DTclassifier.predict(self.x_valid)
        newScore = f1_score(self.y_valid, y_pred_valid, pos_label='yes' if self.y_valid.dtypes[0] == object else 1)

        if newScore>maxScore: bestDT = DTclassifier
        f1_test.append(newScore)

    visualization.plotScoreVSDepth(max_depth, f1_test, f1_train, "Decision Tree score - Hyperparameter : Tree Max Depth")
    
    print(f"\n best decision tree params: {bestDT.get_params()}") # to get the decision tree parameters
    
    featuresDF = pd.DataFrame(bestDT.feature_importances_, index=self.x_train.columns).sort_values(0,ascending=False)
    
    print(f"\n feature importance: {featuresDF}") # tells you which feature has the highest weight. 

    featuresDF.head(10).plot(kind='bar')
    
    
    print(f"\n probabilities: {DTclassifier.predict_proba(self.x_valid)}")
    
    bestScore = max(f1_test) # should be equal to maxScore
    bestIndex = f1_test.index(bestScore)
    print(f"best tree is {bestIndex} level deep with a score of { bestScore }.")

    y_pred = bestDT.predict(self.x_valid)
    
    visualization.model_evaluation(self.y_valid, y_pred)
    
    cm = confusion_matrix(self.y_valid, y_pred)    
    visualization.plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')

#     plotDecisionTree(bestDT, feature_names)
    
    return bestIndex, bestScore

  # comprehensive Decision Tree optimizer
  def DecisionTreeOptimizer_GridSearchCV(self, min_min_sample_leaf, max_min_sample_leaf_n, 
                                       min_depth,max_depth, min_max_leaf_nodes, max_max_leaf_nodes):

    #parameters to search:
    # 1- max_leaf_nodes (provides a very sensible way to control overfitting vs underfitting. 
    #    The more leaves we allow the model to make, the more we move from the underfitting area in the above 
    #    graph to the overfitting area.)
    # 2- max_depth: from 1 to max_depth-the higher the depth, the more the chance of overfitting
    # 3- min_samples_leaf: the smaller the number, the higher the chance we have overftting
    param_grid = {'min_samples_leaf':np.linspace(min_min_sample_leaf, max_min_sample_leaf_n,10).round().astype('int'), 
                  'max_depth':np.arange(min_depth,max_depth), 
                  'max_leaf_nodes':np.linspace(min_max_leaf_nodes, max_max_leaf_nodes,10).round().astype('int'), 
                  'ccp_alpha': [0.0]} # , 0.01, 0.05

    #Exhaustive search over specified parameter values for an estimator.
    tree = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid=param_grid, cv=5, verbose=3) #cv stritifiedKFold?!
    tree.fit(self.x_train, self.y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(tree.best_params_)
    
    return tree.best_params_['max_depth'], tree.best_params_['min_samples_leaf'], tree.best_params_['max_leaf_nodes']
