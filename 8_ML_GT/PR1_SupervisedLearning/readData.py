
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

      # 'ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
      OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False ) # 
      OH_df_data = OH_encoder.fit_transform(self.df_data[low_cardinality_cols])
      self.df_data = self.df_data.drop(self.objectFeatures, axis=1)
      self.df_data = self.df_data.join(OH_df_data)
    

      if self.debugMode:
        print("\n After one-hot encoding: \n", self.df_data[low_cardinality_cols].to_string())

    return self


  # separate target from the data.

  # split train to train and validate
 


#================================================
#================================================
class DecisionTree:
  pass

  # regressor
  # classifier


#================================================
#================================================
class RandomForest:
  pass
  
  # train with n_estimator as the paramter, plot the results, and select the best estimator.  
  # regressor
  # classifier  

#================================================
#================================================
class helpers:
  pass

#================================================
#================================================
class visualization:
  pass


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
  #data = DataPreprocessing(trainfilename="BankMarketingData.csv", dataPath="../data", split = 0.8, categoricalFeatures = 3, imputeStrategy="fix", target='y')
  #data = DataPreprocessing(trainfilename="PhishingWebsitesData.csv", dataPath="../data", split = 0.8, categoricalFeatures = 3, imputeStrategy="fix", target='Result')
  data = DataPreprocessing(trainfilename="train.csv", testfilename="test.csv", dataPath="../data/home-data-kaggle", 
                           split = 0.8, categoricalFeatures = 1, imputeStrategy="drop", target='SalePrice', 
                           addImputeCol=True, debugMode = True)
  
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.pickFeatures(feaetures)

  data.handleMissingValues()
  data.normalizeNumericalFeatures()
  
  data.categoricalFeatures_processing()
  

  print("\n ==================")
  print(" end of the code")

if __name__== "__main__":
  main()



