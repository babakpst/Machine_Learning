
# reads data and clean it. 

import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import os
#import argparser
from sklearn.preprocessing import OneHotEncoder



#================================================
#================================================
# reading the trainging data
@dataclass
class DataPreprocessing:
  trainfilename: str = ""
  testfilename: str = ""
  dataPath: str = ""
  split: float = 0.8
  debugMode: bool = False
  categoriecalFeatures: int = 3 # 1: drop, 2: ordinal, 3: one-hot ##TODO USE ENUMERATE
  imputeStrategy: str = "imputation"       # drop: drop columns,         2: mean, 3: median, 4: most_frequent ##TODO USE ENUMERATE
  target: str = ""

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
    
    print("\n missing values: ")
    missing_val_count_by_column = (self.df_data.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False))
    self.cols_with_missing_data = [col for col in df_data.columns if df_data[col].isnull().any()]

    print("\n data head")
    print(self.df_data.head())
  
  # remove rows/instances with missing target ---
  def missingTarget(self):
    if self.df_data[self.target].isnull().values.any():
      print(f"\n {self.df_data[self.target].isnull().sum()} instances are missing the target value. Dropping the instances." ) # here
      self.df_data.dropna(axis=0, subset=[self.target], how='any', inplace=True)
    else:
      print("\n There is no missing target.")


  # select a subset of features
  def selectFeatures(self, features):
    self.data = self.data[featurs]

  # select object features
  def selectObjectFeatures(self):
    self.objectFeatures = self.df_data.dtypes == 'object'



  def handleMissingValues(self):
    if not self.cols_with_missing_data:
      print("There is no missing data")
    else:
      if self.imputeStrategy == "drop":
        print(" impute strategy is: drop columns with misstin data")
        ImputeDrop()
      elif imputeStrategy == 2:



  # impute options: 
  # Impute approach one: drop column
  def ImputeDrop(self):
    pass


  # fix missing values in the column - approach two: Impute column
  def missingAppTwoImpute(self):
    pass

  # fix missing values in the column - approach three: impute and extend
  def missingAppThreeImpute(self):
    pass


  # separate target from the data.

  # split train to train and validate


  # deal with categorical data get columns, , 

  # get numerical data, impune
  
  
  
  # normalize the data



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
# reading the test data
class testData:
  pass


#************************************************
def main():
  #data = DataPreprocessing(trainfilename="BankMarketingData.csv", dataPath="../data", split = 0.8, categoriecalFeatures = 3, imputeStrategy="fix", target='y')
  #data = DataPreprocessing(trainfilename="PhishingWebsitesData.csv", dataPath="../data", split = 0.8, categoriecalFeatures = 3, imputeStrategy="fix", target='Result')
  data = DataPreprocessing(trainfilename="train.csv", testfilename="test.csv", dataPath="../data/home-data-kaggle", split = 0.8, categoriecalFeatures = 3, imputeStrategy="fix", target='SalePrice')
  data.readData()
  data.missingTarget()

  #features = [f1, f2, f3]
  #data.selectFeatures(feaetures)
  
  data.handleMissingValues()

  data.selectObjectFeatures()



  print(" ==================")
  print(" end of the code")

if __name__== "__main__":
  main()



