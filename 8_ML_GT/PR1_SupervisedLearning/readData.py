
# reads data and clean it. 

import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import os
#import argparser
from sklearn.preprocessing import OneHotEncoder

@dataclass
class DataPreparation:
  filename: str = ""
  dataPath: str = ""
  split: float = 0.8
  debugMode: bool = False
  categoriecalFeatures: int = 3 # 1: drop, 2: ordinal, 3: one-hot ##TODO USE ENUMERATE
  imputeStrategy: int = 1  # 1: Constant, 2: mean, 3: median, 4: most_frequent ##TODO USE ENUMERATE


  def __post_init__(self):
    self.fullpath = os.path.join(self.dataPath, self.filename)


  # read data
  def readData(self):
    self.df_data = pd.read_csv(self.fullpath)
    
    print("\n info about the data: ")
    print(f' {"number of samples":<22} | {"number of features":<22} | {"Any missing data":<22} \n {len(self.df_data):<22} | {len(self.df_data.columns):<22} | {True if self.df_data.isnull().values.any() else False:<22}') 
    
    print("\n missing values: ")

    missing_val_count_by_column = (self.df_data.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    print("\n data head")
    print(self.df_data.head())
  

# remove rows/instances with missing target

# separate target from the data.

# split train to train and validate


# get columns, categorical, 

# get numerical data, impune


class DecisionTree:
  pass

class RandomForest:
  pass


class helpers:
  pass

class visualization:
  pass

class parser:
  pass

#=============================================
def main():
  data = DataPreparation(filename="BankMarketingData.csv", dataPath="../data", split = 0.8, categoriecalFeatures = 3, imputeStrategy=2)
  data.readData()


if __name__== "__main__":
  main()


