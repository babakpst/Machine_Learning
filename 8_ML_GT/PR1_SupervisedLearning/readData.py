
# reads data and clean it. 

import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import os
#import argparser
from sklearn.impute import SimpleImputer
from IPython.display import display # to display dataframe
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import tree
import itertools
import timeit
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

#================================================
#================================================
# reading the trainging data
@dataclass
class DataPreprocessing:
  """_summary_
  """
  
  trainfilename: str = ""
  testfilename: str = ""
  dataPath: str = ""
  split: float = 0.8
  debugMode: bool = False
  categoricalFeatures: int = 3 # 1: drop, 2: ordinal, 3: one-hot ##TODO USE ENUMERATE
  imputeStrategy: str = "drop"  # drop: drop columns (works with numeric or object features) 
                                # mean: mean of data (only works with numeric features) - applies most_freq for categorical data
                                # median: uses the median (only works with numeric features) - applies most_freq for categorical data
                                # most_frequent: replaces with the most frequent value (works with numeric or object features).
                                # constant: uses fill_value to replace all nans with a constant fill_value.
  
  addImputeCol: bool = False    # Ture/False
  
  #missing_values: ? = np.nan # for impute
  fill_value: any = "" # for impute, when using the constant strategy, otherwise it would be ignored.
  target: str = ""
  cardinalityThreshold: int = 15

  def __post_init__(self):
    self.fullpath = os.path.join(self.dataPath, self.trainfilename)

  # read data -----------------------------------
  def readData(self):
    self.df_data = pd.read_csv(self.fullpath)
    
    print("\n data frame info: ")
    print(self.df_data.info())

    if self.target not in self.df_data.columns:
      #raise ValueError(f"The target columns '{self.target}' does not exist in the data.")
      print(f"Error: The target columns '{self.target}' does not exist in the data. Check the data.")
      quit()
      
    print("\n info about the data: ")
    print(f' {"number of samples":<22} | {"number of features":<22} | {"Any missing data":<22} | {"Missing target data":<22}\n {len(self.df_data):<22} | {len(self.df_data.columns):<22} | {True if self.df_data.isnull().values.any() else False:<22} | {self.df_data[self.target].isnull().sum()}') 
    
    self.objectFeatures = self.df_data.select_dtypes(include=['object']).columns.to_list()
    if self.target in self.objectFeatures:  # drop the target columns
      self.objectFeatures.remove(self.target)
    if self.debugMode:
      print("\n Here are the object features: \n", self.objectFeatures)
    
    self.numericFeatures = self.df_data.select_dtypes(exclude=['object']).columns.to_list()
    if self.target in self.numericFeatures:  # drop the target columns
      self.numericFeatures.remove(self.target)
    if self.debugMode:
      print("\n Here are the numerical features: \n", self.numericFeatures)

    missing_val_count_by_column = (self.df_data.isnull().sum())
    print(type(missing_val_count_by_column))
    print("\n missing values: ")
    print(missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False))

    self.objectFeatures_with_missing_data = [col for col in self.df_data[self.objectFeatures].columns if self.df_data[col].isnull().any()] # categorrical features with missing data
    if self.target in self.objectFeatures_with_missing_data:  # target should not be here, but let's check it out. 
      self.objectFeatures_with_missing_data.remove(self.target)

    self.numericFeatures_with_missing_data = [col for col in self.df_data[self.numericFeatures].columns if self.df_data[col].isnull().any()] # numerical features with missing data
    if self.target in self.numericFeatures_with_missing_data:  # target should not be here, but let's check it out. 
      self.numericFeatures_with_missing_data.remove(self.target)

    self.Features_with_missing_data = self.objectFeatures_with_missing_data + self.numericFeatures_with_missing_data
    # self.Features_with_missing_data = self.objectFeatures_with_missing_data

        
    if self.debugMode:
      print("\n object features with missing data: \n", self.objectFeatures_with_missing_data)
      print("\n numerical features with missing data: \n", self.numericFeatures_with_missing_data)
      print("\n Features with missing data (total {}): \n".format(len(self.Features_with_missing_data)), self.Features_with_missing_data)

    print("\n data head")
    print(self.df_data.head())

  # remove rows/instances with missing target ---
  def missingTarget(self):
    if self.df_data[self.target].isnull().values.any():
      print(f"\n {self.df_data[self.target].isnull().sum()} instances are missing the target value. Dropping the instances." )
      self.df_data.dropna(axis=0, subset=[self.target], how='any', inplace=True)
    else:
      print("\n There is no missing target.")


  #  a subset of features in the data
  def pickFeatures(self, features):
    self.data = self.data[features]

  def handleMissingValues(self):
    if not self.objectFeatures_with_missing_data and not self.numericFeatures_with_missing_data:
      print("There is no missing data")
    else:
      if self.addImputeCol and not self.imputeStrategy == 'drop':
        for col in self.Features_with_missing_data:
          self.df_data[col + '_was_missing'] = self.df_data[col].isnull()

      if self.imputeStrategy == "drop": # for categorical and numerical features
        print(" impute strategy: drop features with missing data (categorical and numerical)")

        if self.debugMode:
          print(" features to be dropped: \n", self.Features_with_missing_data)
          print(f"\n before impute: \n", self.df_data.to_string())

          print(type(self.Features_with_missing_data))

        # self.df_data = self.df_data.drop(self.Features_with_missing_data, axis=1)
        self.df_data.drop(self.Features_with_missing_data, axis=1, inplace=True)

        # removing deleted features from the list
        self.objectFeatures = [item for item in self.objectFeatures if item not in self.objectFeatures_with_missing_data]
        self.numericFeatures= [item for item in self.numericFeatures if item not in self.numericFeatures_with_missing_data]

        if self.debugMode:
          print("\n object features after dropping: \n", self.objectFeatures)
          print("\n numerical features after dropping: \n", self.numericFeatures)
          print(f"\n After impute: \n", self.df_data.to_string())

      elif self.imputeStrategy in ["mean", "median", "most_frequent"]:  
        self.ImputeTheData()

      elif self.imputeStrategy == "constant": #  
        print(" The constant  impute strategy affects the numerical and categorical features.")
        self.ImputeConstant()

    if self.debugMode:
      print("\n data head after Impute")
      print(self.df_data.head())


  # TODO: add an option for imute categorical features to drop it if more than half is NaN.
  def ImputeTheData(self): # for impute strategy mean, median, most_frequent
    if self.debugMode:
      print(f"\n Before {self.imputeStrategy} impute for num features: \n", self.df_data[self.numericFeatures_with_missing_data].to_string())

    imputer = SimpleImputer(strategy=self.imputeStrategy, copy=False)
    
    self.df_data[self.numericFeatures_with_missing_data] = pd.DataFrame(imputer.fit_transform(self.df_data[self.numericFeatures_with_missing_data]), columns = self.numericFeatures_with_missing_data)
    # imputed_num_features = pd.DataFrame(imputer.fit_transform(self.df_data[self.numericFeatures_with_missing_data]), columns = self.numericFeatures_with_missing_data)

    # self.df_data = self.df_data.drop(self.numericFeatures_with_missing_data,axis=1)
    # self.df_data = self.df_data.join(imputed_num_features)

    if self.debugMode:
      print(f"\n After {self.imputeStrategy} impute for num features: \n", self.df_data[self.numericFeatures_with_missing_data].to_string())

    if self.debugMode:
      print(f"\n Before most_frequent impute for categorical features: \n", self.df_data[self.objectFeatures_with_missing_data].to_string())

    cat_imputer = SimpleImputer(strategy='most_frequent', copy=False)
    self.df_data[self.objectFeatures_with_missing_data] = pd.DataFrame(cat_imputer.fit_transform(self.df_data[self.objectFeatures_with_missing_data]), columns = self.objectFeatures_with_missing_data)
    # imputed_cat_features = pd.DataFrame(cat_imputer.fit_transform(self.df_data[self.objectFeatures_with_missing_data]), columns = self.objectFeatures_with_missing_data)

    # self.df_data = self.df_data.drop(self.objectFeatures_with_missing_data,axis=1)
    # self.df_data = self.df_data.join(imputed_cat_features)

    if self.debugMode:
      print("\n After most frequent impute for categorical features: \n", self.df_data[self.objectFeatures_with_missing_data].to_string())


  def ImputeConstant(self):
    if self.debugMode:
      print("\n Before most frequent impute for num features: \n", self.df_data[self.numericFeatures_with_missing_data])

    imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value, copy=False)
    imputed_num_features = pd.DataFrame(imputer.fit_transform(self.df_data[self.numericFeatures_with_missing_data]), columns = self.numericFeatures_with_missing_data)

    self.df_data = self.df_data.drop(self.numericFeatures_with_missing_data,axis=1)
    self.df_data = self.df_data.join(imputed_num_features)

    if self.debugMode:
      print("\n After most frequent impute for num features: \n", self.df_data[self.numericFeatures_with_missing_data])

    if self.debugMode:
      print("\n Before most frequent impute for categorical features: \n", self.df_data[self.objectFeatures_with_missing_data])

    imputed_cat_features = pd.DataFrame(imputer.fit_transform(self.df_data[self.objectFeatures_with_missing_data]), columns = self.objectFeatures_with_missing_data)

    self.df_data = self.df_data.drop(self.objectFeatures_with_missing_data,axis=1)
    self.df_data = self.df_data.join(imputed_cat_features)

    if self.debugMode:
      print("\n After most frequent impute for categorical features: \n", self.df_data[self.objectFeatures_with_missing_data])

  # normalize numerical data
  # TODO: it should not normalize everything (yearbuild, Id). Fix it.
  def normalizeNumericalFeatures(self):
    if self.debugMode:
      print("\n Before normalization: \n", self.df_data[self.numericFeatures].to_string())
    self.df_data[self.numericFeatures] = (self.df_data[self.numericFeatures]-self.df_data[self.numericFeatures].min()) / (self.df_data[self.numericFeatures].max()-self.df_data[self.numericFeatures].min())

    if self.debugMode:
      print("\n After normalization: \n", self.df_data[self.numericFeatures].to_string())

  def categoricalFeatures_processing(self):

    # first find how many categories exist for each col
    unique_cats_of_objectFeatures = list(map(lambda col: self.df_data[col].nunique(), self.objectFeatures))
    d = dict(zip(self.objectFeatures, unique_cats_of_objectFeatures))
    
    print("unique categories of object features: ")
    print(sorted(d.items(), key = lambda x:x[1]))

    # features that will be one-hot encoded
    low_cardinality_cols = [col for col in self.objectFeatures if self.df_data[col].nunique() < self.cardinalityThreshold]

    # features that will be dropped from the dataset
    high_cardinality_cols = list(set(self.objectFeatures)-set(low_cardinality_cols))

    if self.debugMode:
      print(f'\n object features (total {len(self.objectFeatures)}):\n', self.objectFeatures)
      print(f'\n Low cardinality categorical columns (candidates for one-hot encoding if selected (less than {self.cardinalityThreshold} unique categories)): \n', low_cardinality_cols)
      print(f'\n Categorical columns that will be dropped from the dataset (more than {self.cardinalityThreshold} unique categories):\n', high_cardinality_cols)

    if self.categoricalFeatures == 1: # drop
      if self.debugMode:
        print("\n Before drop categorical features: \n", self.df_data[self.objectFeatures].to_string())

        count = 0
        for col in self.objectFeatures:
          count = count + 1
          print(f'{count} {col} exits in data: {True if col in self.df_data.columns else False}')
          # self.df_data.drop(col, axis=1, inplace=True)
      
      print(type(self.objectFeatures), len(self.objectFeatures))
      print("objects\n", self.objectFeatures)
      # self.df_data=self.df_data.drop(self.objectFeatures, axis=1)
      self.df_data.drop(self.objectFeatures, axis=1, inplace=True)
      # self.df_data.drop(['MSZoning'], axis=1, inplace=True)

      if self.debugMode:
        print("\n After drop categorical features: \n", self.df_data.to_string())

    elif self.categoricalFeatures == 2: # ordinal numbering of the categorical features
      ordinal_encoder = OrdinalEncoder()
      if self.debugMode:
        print("\n Before ordinal encoding: \n", self.df_data[self.objectFeatures].to_string())
  
      self.df_data[self.objectFeatures] = ordinal_encoder.fit_transform(self.df_data[self.objectFeatures])

      if self.debugMode:
        print("\n After ordinal encoding: \n", self.df_data[self.objectFeatures].to_string())

    elif self.categoricalFeatures == 3: # One-hot encoding of the categorical features

      if self.debugMode:
        
        print("\n Before one-hot encoding: \n", self.df_data[low_cardinality_cols].to_string())
        print("\n dropping high cardinality features: \n", high_cardinality_cols)

      # 'ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
      OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # 
      #OH_encoder.get_feature_names(['MSZoning','Street'])
      OH_df_data = pd.DataFrame(OH_encoder.fit_transform(self.df_data[low_cardinality_cols]))
      # OH_df_data = pd.DataFrame(OH_encoder.fit_transform(self.df_data[['MSZoning','Street']]))
      self.df_data.drop(self.objectFeatures, axis=1,inplace=True)
      self.df_data = self.df_data.join(OH_df_data)    

      if self.debugMode:
        print("\n After one-hot encoding: \n", self.df_data.to_string())

    return self


  def custom_combiner(feature, category):
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)

  # split train to train and validate
  def splitData(self):
    train, test = train_test_split(self.df_data, train_size=self.split , random_state=50, shuffle=True)
    print(type(train))
       
    self.x_train = train.loc[:, train.columns != self.target]
    self.y_train = train.loc[:, train.columns == self.target]
    
    self.x_test = test.loc[:, test.columns != self.target]
    self.y_test = test.loc[:, test.columns == self.target]

    if self.debugMode:
      print("\nx_train: \n", self.x_train)
      print("\ny_train: \n", self.y_train)

      print("\nx_test: \n", self.x_test)
      print("\ny_test: \n", self.y_test)

    return self

  # separate target from the data.

#================================================
#================================================
@dataclass
class myDecisionTreeClassifier:
  """_summary_
  """

  maxdepth: int = 32
  x_train: pd.DataFrame = None
  y_train: pd.DataFrame = None

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
        f1_train.append(f1_score(self.y_train, y_pred_train,pos_label=1))

        y_pred_test = DTclassifier.predict(self.x_test)
        newScore = f1_score(self.y_test, y_pred_test,pos_label=1)
        if newScore>maxScore: bestDT = DTclassifier
        f1_test.append(newScore)

    visualization.plotScoreVSDepth(max_depth, f1_test, f1_train, "Decision Tree score - Hyperparameter : Tree Max Depth")
    
    print(f"\n best decision tree params: {bestDT.get_params()}") # to get the decision tree parameters
    
    featuresDF = pd.DataFrame(bestDT.feature_importances_, index=self.x_train.columns).sort_values(0,ascending=False)
    
    print(f"\n feature importance: {featuresDF}") # tells you which feature has the highest weight. 

    featuresDF.head(10).plot(kind='bar')
    
    
    print(f"\n probabilities: {DTclassifier.predict_proba(self.x_test)}")
    
    bestScore = max(f1_test) # should be equal to maxScore
    bestIndex = f1_test.index(bestScore)
    print(f"best tree is {bestIndex} level deep with a score of { bestScore }.")

    y_pred = bestDT.predict(self.x_test)
    
    visualization.model_evaluation(self.y_test, y_pred)
    
    cm = confusion_matrix(self.y_test, y_pred)    
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



#================================================
#================================================
class DecisionTreeRegressor:
  """_summary_
  """
  pass

#================================================
#================================================
@dataclass
class myRandomForestClassifier:
  x_train: pd.DataFrame = None
  y_train: pd.DataFrame = None

  x_test: pd.DataFrame = None
  y_test: pd.DataFrame = None

  def train(self):
    forest_model = RandomForestClassifier(random_state=1)
    forest_model.fit(self.x_train, self.y_train)
    y_pred = forest_model.predict(self.x_test)

    visualization.model_evaluation(self.y_test, y_pred)
  

#================================================
#================================================
class RandomForestRegressor:
  pass
  
  # train with n_estimator as the paramter, plot the results, and select the best estimator.  


#================================================
#================================================
class helpers:
  
  @staticmethod
  def MAE(val_y, val_predictions):
    return mean_absolute_error(val_y, val_predictions)

#================================================
#================================================
class visualization:
  
  @staticmethod
  def plotScoreVSDepth(max_depth, f1_score_test, f1_score_train, title):
    print(" f1 score vs max depth of the tree ")
    plt.plot(max_depth, f1_score_test,  'o-', color = 'r', label='Test F1 Score')
    plt.plot(max_depth, f1_score_train, 'o-', color = 'b', label='Train F1 Score')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Model F1 Score')
   
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

  @staticmethod
  def plotDecisionTree(clf, feature_names):
    fig = plt.figure(figsize=(50,40))
    _ = tree.plot_tree(clf, 
                   feature_names=feature_names,  
                   #class_names={0:'Malignant', 1:'Benign'},
                   filled=True,
                  fontsize=15)
    
  @staticmethod
  def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')        
    plt.show()    

  @staticmethod
  def model_evaluation(y_test, y_pred):
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred,pos_label=1)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,pos_label=1)
    recall = recall_score(y_test,y_pred,pos_label=1)
    
    print("mean abs error:  "+"{:.2f}".format(mean_absolute_error(y_test, y_pred)))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(auc)+"     AUC:          "+"{:.2f}".format(auc))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"  Accuracy:     "+"{:.2f}".format(accuracy))
    print("Precision: "+"{:.2f}".format(precision)+"  Precision: "+"{:.2f}".format(precision))
    print("Precision: "+"{:.2f}".format(recall)+"     Recall:    "+"{:.2f}".format(recall))

  @staticmethod
  def plot_learning_curve(clf, X, y, title="Insert Title"):
    
    nn = len(y)
    train_mean = []; train_std = [] #model performance score (f1)
    cv_mean = []; cv_std = [] #model performance score (f1) cross validation
    fit_mean = []; fit_std = [] #model fit/training time
    pred_mean = []; pred_std = [] #model test/prediction times
    train_sizes=(np.linspace(.05, 1.0, 10)*nn).astype('int')  
    print(train_sizes)
    
    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        print(len(X))
        print(idx)
        X_subset = X[idx,:]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)
        
        train_mean.append(np.mean(scores['train_score'])); train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score'])); cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time'])); fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time'])); pred_std.append(np.std(scores['score_time']))
    
    train_mean = np.array(train_mean); train_std = np.array(train_std)
    cv_mean = np.array(cv_mean); cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean); fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean); pred_std = np.array(pred_std)
    
    visualization.plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    visualization.plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)
    
    return train_sizes, train_mean, fit_mean, pred_mean
  
  
  @staticmethod
  def final_classifier_evaluation(clf,X_train, X_test, y_train, y_test):
    
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    start_time = timeit.default_timer()    
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time
    
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred,pos_label=1)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,pos_label=1)
    recall = recall_score(y_test,y_pred,pos_label=1)
    cm = confusion_matrix(y_test,y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    visualization.plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.show()


  @staticmethod
  def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    
    plt.figure()
    plt.title("Learning Curve: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()
    
  @staticmethod  
  def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    plt.figure()
    plt.title("Modeling Time: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()    
    plt.figure()
    plt.title("Learning Curve: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()
    
  @staticmethod
  def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    plt.figure()
    plt.title("Modeling Time: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()    



#================================================
#================================================
class parser:
  pass

#================================================
#================================================
class crossValidation:
  pass


#================================================
#================================================
# reading the test dataobject fea
class testData:
  pass


#************************************************
def main():
  
  # ARGUMENTS:  
  # "DTC": DecisionTree Classifier
  # "DTR": DecisionTree Regressor
  # "RFC": RandomForest Classifier
  # "RFR": RandomForest Regressor 
  MLType = "RFR"
  
  #data = DataPreprocessing(trainfilename="BankMarketingData.csv", dataPath="../data", split = 0.8, categoricalFeatures = 3, imputeStrategy="fix", target='y')
  #data = DataPreprocessing(trainfilename="PhishingWebsitesData.csv", testfilename="", dataPath="../data", split = 0.2, categoricalFeatures = 3, imputeStrategy="mean", target='Result', addImputeCol=True, debugMode = False)
  data = DataPreprocessing(trainfilename="train.csv", testfilename="test.csv", dataPath="../data/home-data-kaggle", split = 0.8, categoricalFeatures = 3, imputeStrategy="drop", target='SalePrice', addImputeCol=True, debugMode = False)
  
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.pickFeatures(feaetures)

  data.handleMissingValues()
  data.categoricalFeatures_processing()
  data.normalizeNumericalFeatures()

  data.splitData()

  # decition tree classifier ---------------------------
  if MLType == "DTC":
    myDecisionTree = myDecisionTreeClassifier(x_train = data.x_train, y_train = data.y_train, x_test = data.x_test, y_test = data.y_test)
    
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
    estimator_data = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                            max_leaf_nodes=max_leaf_nodes, random_state=100, criterion='entropy')

    # train_samp_data, DT_train_score_data, DT_fit_time_data, DT_pred_time_data = visualization.plot_learning_curve(clf = estimator_data, X=data.x_train, y=data.y_train, title="Decision Tree")
    visualization.final_classifier_evaluation(clf=estimator_data, X_train = data.x_train, X_test = data.x_test, y_train = data.y_train, y_test = data.y_test)

  # random forest classifier ---------------------------
  elif MLType == "RFC":
    myRF = myRandomForestClassifier(x_train = data.x_train, y_train = data.y_train, x_test = data.x_test, y_test = data.y_test)
    myRF.train()

  elif MLType == "RFR":
    pass

  print("\n ==================")
  print(" end of the code")

if __name__== "__main__":
  main()



