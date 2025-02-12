

import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import os
#import argparser
from IPython.display import display # to display dataframe
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize
from sklearn import tree
from sklearn.feature_selection import mutual_info_regression

import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


# from mlxtend.preprocessing import minmax_scaling
import itertools
import timeit


@dataclass
class DataPreprocessing:
  """_summary_
  """
  
  train_filename: str = ""
  test_filename: str = ""
  dataPath: str = ""
  train_size: float = 0.8
  Index_col: str = None # Index col should start from 0 if it is integer! Check it out. 
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
    self.traindata_fullpath = os.path.join(self.dataPath, self.train_filename)
    self.testdata_fullpath = os.path.join(self.dataPath, self.test_filename) if self.test_filename !="" else ""

  # read data from csv file, get numerical/categorical features, and find missing data -------------
  def readData(self):
    self.df_train = pd.read_csv(self.traindata_fullpath, index_col=self.Index_col)
    self.df_test = pd.read_csv(self.testdata_fullpath, index_col=self.Index_col) if self.testdata_fullpath else None
    
    print("\n train data frame info: ")
    print(self.df_train.info())

    if self.testdata_fullpath:
      print("\n test data frame info: ")
      print(self.df_test.info())

    if self.target not in self.df_train.columns:
      print(f"Error: The target columns '{self.target}' does not exist in the train data. Check the train data.")
      quit()

    if (not self.df_test is None) and (self.target not in self.df_test.columns):
      print(f"Error: The target columns '{self.target}' does not exist in the test data. Check the test data.")
      #quit()

    print("\n info about the train data: ")
    # calculate the percentage of missing samples
    total = np.product(self.df_train.shape)
    total_missing = self.df_train.isnull().sum().sum()
    percent_missing_in_train = total_missing/total*100
    
    if self.testdata_fullpath:
      total = np.product(self.df_test.shape)
      total_missing = self.df_test.isnull().sum().sum()
      percent_missing_in_test = total_missing/total*100
    else:
      percent_missing_in_test = 0
    print(f' {"number of samples":<22} | {"number of features":<22} | {"Any missing data":<22} | {"Missing target data":<22} | {"% missing in train":<22} | {"% missing in test":<22}  \n {len(self.df_train):<22} | {len(self.df_train.columns):<22} | {True if self.df_train.isnull().values.any() else False:<22} | {self.df_train[self.target].isnull().sum():<22} | {percent_missing_in_train:<22} | {percent_missing_in_test:<22}')

    if self.testdata_fullpath:
      print("\n info about the test data: ")
      print(f"{self.df_test[self.target].isnull().sum() if (self.target in self.df_test.columns) else 0 }")
      print(f' {"number of samples":<22} | {"number of features":<22} | {"Any missing data":<22} | {"Missing target data":<22}\n {len(self.df_test):<22} | {len(self.df_test.columns):<22} | {True if self.df_test.isnull().values.any() else False:<22} | {self.df_test[self.target].isnull().sum() if (self.target in self.df_test.columns) else 0 }')

    self.objectFeatures = self.df_train.select_dtypes(include=['object']).columns.to_list()# train object features 
    if self.target in self.objectFeatures:  # drop the target columns
      self.objectFeatures.remove(self.target)
    if self.debugMode:
      print("\n Here are the object features: \n", self.objectFeatures)
    
    self.numericFeatures = self.df_train.select_dtypes(exclude=['object']).columns.to_list()
    if self.target in self.numericFeatures:  # drop the target columns
      self.numericFeatures.remove(self.target)
    if self.debugMode:
      print("\n Here are the numerical features: \n", self.numericFeatures)

    if self.testdata_fullpath and self.objectFeatures != self.df_test.select_dtypes(include=['object']).columns.to_list():
      print(f"Error: The train object data features are different from the test data object features.")
      quit()

    if self.testdata_fullpath and self.numericFeatures != self.df_test.select_dtypes(exclude=['object']).columns.to_list():
      print(f"Error: The train numerical data features are different from the test data numerical features.")
      quit()

    missing_val_count_by_column = (self.df_train.isnull().sum())
    print("\n train data missing values: ")
    print(missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False))

    if self.test_filename !="":
      missing_val_count_by_column = (self.df_test.isnull().sum())
      print("\n test data missing values: ")
      print(missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False))

    self.objectFeatures_with_missing_data_train = [col for col in self.df_train[self.objectFeatures].columns if self.df_train[col].isnull().any()] # categorrical features with missing data
    if self.target in self.objectFeatures_with_missing_data_train:  # target should not be here, but let's check it out. 
      self.objectFeatures_with_missing_data_train.remove(self.target)

    self.numericFeatures_with_missing_data_train = [col for col in self.df_train[self.numericFeatures].columns if self.df_train[col].isnull().any()] # numerical features with missing data
    if self.target in self.numericFeatures_with_missing_data_train:  # target should not be here, but let's check it out. 
      self.numericFeatures_with_missing_data_train.remove(self.target)

    self.Features_with_missing_data_train = self.objectFeatures_with_missing_data_train + self.numericFeatures_with_missing_data_train
    # self.Features_with_missing_data_train = self.objectFeatures_with_missing_data_train

    self.objectFeatures_with_missing_data_test = None
    self.numericFeatures_with_missing_data_test = None
    self.Features_with_missing_data_test = None
    if self.test_filename !="":
      self.objectFeatures_with_missing_data_test = [col for col in self.df_test[self.objectFeatures].columns if self.df_test[col].isnull().any()] # categorrical features with missing data in test
      if self.target in self.objectFeatures_with_missing_data_test:  # target should not be here, but let's check it out. 
        self.objectFeatures_with_missing_data_test.remove(self.target)

      self.numericFeatures_with_missing_data_test = [col for col in self.df_train[self.numericFeatures].columns if self.df_train[col].isnull().any()] # numerical features with missing data in test
      if self.target in self.numericFeatures_with_missing_data_test:  # target should not be here, but let's check it out. 
        self.numericFeatures_with_missing_data_test.remove(self.target)

      self.Features_with_missing_data_test = self.objectFeatures_with_missing_data_test + self.numericFeatures_with_missing_data_test
        
    if self.debugMode:
      print("\n object features with missing data in train: \n", self.objectFeatures_with_missing_data_train)
      print("\n numerical features with missing data in train: \n", self.numericFeatures_with_missing_data_train)
      print("\n Features with missing data (total {}) in train: \n".format(len(self.Features_with_missing_data_train)), self.Features_with_missing_data_train)
      print()
      print("\n object features with missing data in test: \n", self.objectFeatures_with_missing_data_test)
      print("\n numerical features with missing data in test: \n", self.numericFeatures_with_missing_data_test)
      print("\n Features with missing data (total {}) in test: \n".format(len(self.Features_with_missing_data_train)), self.Features_with_missing_data_test)

    print("\n data head")
    print(self.df_train.head())

  # remove rows/instances with missing target --------------------------------------------------------------------------
  def missingTarget(self):
    if self.df_train[self.target].isnull().values.any():
      print(f"\n {self.df_train[self.target].isnull().sum()} instances are missing the target value in train data. Dropping the instances." )
      self.df_train.dropna(axis=0, subset=[self.target], how='any', inplace=True)
    else:
      print("\n There is no missing target in the train data.")

    #TODO Target might not be in test
    if self.testdata_fullpath and  (self.target in self.df_test.columns) and self.df_test[self.target].isnull().values.any():
      print(f"\n {self.df_test[self.target].isnull().sum()} instances are missing the target value in test data. Dropping the instances." )
      self.df_test.dropna(axis=0, subset=[self.target], how='any', inplace=True)
    elif self.testdata_fullpath:
      print("\n There is no missing target in the test data.")

  # pick a subset of features in the dataset ---------------------------------------------------------------------------
  def pickFeatures(self, features):
    self.df_train = self.df_train[features]
    if self.testdata_fullpath:
      self.df_test = self.df_test[features]

  # delete a subset of features in the dataset -------------------------------------------------------------------------
  def deleteFeatures(self, features):
    self.df_train = self.df_train.drop(features, axis=1)
    if self.testdata_fullpath:
      self.df_test = self.df_test.drop(features, axis=1)


  # impute missing values in the dataset, there a couple of options ----------------------------------------------------
  def handleMissingValues(self):
    isThereAnyMissingDataInTrain = True if self.Features_with_missing_data_train else False
    isThereAnyMissingDataInTest = True if self.test_filename !="" and  self.Features_with_missing_data_test else False

    if self.debugMode:
      print("\n data head before Impute")
      print(self.df_train.head())
      if self.testdata_fullpath:
        print(self.df_test.head())
    
    if not isThereAnyMissingDataInTrain:
      print(" There is no missing data in train")
    if not isThereAnyMissingDataInTest:
      print(" There is no missing data in test")

    if isThereAnyMissingDataInTest or isThereAnyMissingDataInTrain:
      self.objectFeatures_with_missing_data = list(set(self.objectFeatures_with_missing_data_test) | set(self.objectFeatures_with_missing_data_train)) if self.objectFeatures_with_missing_data_test else self.objectFeatures_with_missing_data_train
      self.numericFeatures_with_missing_data = list(set(self.numericFeatures_with_missing_data_test) | set(self.numericFeatures_with_missing_data_train)) if self.numericFeatures_with_missing_data_test else self.numericFeatures_with_missing_data_train
      self.Features_with_missing_data = list(set(self.Features_with_missing_data_test) | set(self.Features_with_missing_data_train)) if self.Features_with_missing_data_test else self.Features_with_missing_data_train
      
      if self.addImputeCol and not self.imputeStrategy == 'drop':
        for col in self.Features_with_missing_data:
          self.df_train[col + '_was_missing'] = self.df_train[col].isnull().astype(int)
          if self.testdata_fullpath:
            self.df_test[col + '_was_missing'] = self.df_test[col].isnull().astype(int)

      if self.imputeStrategy == "drop": # for categorical and numerical features
        print(" impute strategy: drop features with missing data (categorical and numerical)")

        if self.debugMode:
          print(" features to be dropped: \n", self.Features_with_missing_data)
          print(f"\n train data before impute: \n", self.df_train.to_string())
          if self.testdata_fullpath:
            print(f"\n test data before impute: \n", self.df_test.to_string())

        self.df_train.drop(self.Features_with_missing_data, axis=1, inplace=True)
        if self.testdata_fullpath:
          self.df_test.drop(self.Features_with_missing_data, axis=1, inplace=True)

        # removing deleted features from the list
        self.objectFeatures = [item for item in self.objectFeatures if item not in self.objectFeatures_with_missing_data]
        self.numericFeatures= [item for item in self.numericFeatures if item not in self.numericFeatures_with_missing_data]

        if self.debugMode:
          print("\n object features after dropping: \n", self.objectFeatures)
          print("\n numerical features after dropping: \n", self.numericFeatures)
          print(f"\n After impute: \n", self.df_train.to_string())

      elif self.imputeStrategy in ["mean", "median", "most_frequent"]:  
        self.ImputeTheData()

      elif self.imputeStrategy == "constant": #  
        print(" The constant  impute strategy affects the numerical and categorical features.")
        self.ImputeConstant()

    if self.debugMode:
      print("\n data head after Impute-train")
      print(self.df_train.head())
      print(self.df_train.tail())
      if self.testdata_fullpath:
        print("\n data head after Impute-test")
        print(self.df_test.head())
        print(self.df_test.tail())

  # TODO: add an option for impute categorical features to drop it if more than half is NaN.
  # impute missing values for mean, median, most_frequent methods ------------------------------------------------------
  def ImputeTheData(self): 
    if self.debugMode:
      print(f"\n Before {self.imputeStrategy} impute for numerical features in train data: \n", self.df_train[self.numericFeatures_with_missing_data_train].to_string())
      if self.testdata_fullpath:
        print(f"\n Before {self.imputeStrategy} impute for numerical features in test data: \n", self.df_test[self.numericFeatures_with_missing_data_test].to_string())

    imputer_test = SimpleImputer(strategy=self.imputeStrategy, copy=False)
    imputer_train = SimpleImputer(strategy=self.imputeStrategy, copy=False)
    
    self.df_train[self.numericFeatures_with_missing_data_train] = pd.DataFrame(imputer_train.fit_transform(self.df_train[self.numericFeatures_with_missing_data_train]), columns = self.numericFeatures_with_missing_data_train)
    if self.testdata_fullpath:
      self.df_test[self.numericFeatures_with_missing_data_test] = pd.DataFrame(imputer_test.fit_transform(self.df_test[self.numericFeatures_with_missing_data_test]), columns = self.numericFeatures_with_missing_data_test)

    if self.debugMode:
      print(f"\n After {self.imputeStrategy} impute for numerical features in train data: \n", self.df_train[self.numericFeatures_with_missing_data_train].to_string())
      if self.testdata_fullpath:
        print(f"\n After {self.imputeStrategy} impute for numerical features in test data: \n", self.df_test[self.numericFeatures_with_missing_data_test].to_string())

    if self.debugMode:
      print(f"\n Before most_frequent impute for categorical features in train data: \n", self.df_train[self.objectFeatures_with_missing_data_train].to_string())
      if self.testdata_fullpath:
        print(f"\n Before most_frequent impute for categorical features in test data: \n", self.df_test[self.objectFeatures_with_missing_data_test].to_string())

    cat_imputer = SimpleImputer(strategy='most_frequent', copy=False)
    self.df_train[self.objectFeatures_with_missing_data_train] = pd.DataFrame(cat_imputer.fit_transform(self.df_train[self.objectFeatures_with_missing_data_train]), columns = self.objectFeatures_with_missing_data_train)
    if self.testdata_fullpath:
      self.df_test[self.objectFeatures_with_missing_data_test] = pd.DataFrame(cat_imputer.fit_transform(self.df_test[self.objectFeatures_with_missing_data_test]), columns = self.objectFeatures_with_missing_data_test)

    if self.debugMode:
      print("\n After most frequent impute for categorical features in train data: \n", self.df_train[self.objectFeatures_with_missing_data_train].to_string())
      if self.testdata_fullpath:
        print("\n After most frequent impute for categorical features in test data: \n", self.df_test[self.objectFeatures_with_missing_data_test].to_string())

  # TODO add test data + check the entire function
  # impute missing values for const methods ----------------------------------------------------------------------------
  def ImputeConstant(self):
    if self.debugMode:
      print("\n Before const impute for num features: \n", self.df_train[self.numericFeatures_with_missing_data_train])

    imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value, copy=False)
    imputed_num_features = pd.DataFrame(imputer.fit_transform(self.df_train[self.numericFeatures_with_missing_data_train]), columns = self.numericFeatures_with_missing_data_train)

    self.df_train = self.df_train.drop(self.numericFeatures_with_missing_data_train,axis=1)
    self.df_train = self.df_train.join(imputed_num_features)

    if self.debugMode:
      print("\n After const impute for num features: \n", self.df_train[self.numericFeatures_with_missing_data_train])

    if self.debugMode:
      print("\n Before const impute for categorical features: \n", self.df_train[self.objectFeatures_with_missing_data_train])

    imputed_cat_features = pd.DataFrame(imputer.fit_transform(self.df_train[self.objectFeatures_with_missing_data_train]), columns = self.objectFeatures_with_missing_data_train)

    self.df_train = self.df_train.drop(self.objectFeatures_with_missing_data_train,axis=1)
    self.df_train = self.df_train.join(imputed_cat_features)

    if self.debugMode:
      print("\n After const impute for categorical features: \n", self.df_train[self.objectFeatures_with_missing_data_train])
  

  # TODO: it should not normalize everything (yearbuild, Id). Fix it.
  # normalize numerical data (has not been tested) -------------------------------------------------------------------
  def normalizeNumericalFeatures(self):
    if self.debugMode:
      print("\n Before normalization-train data: \n", self.df_train[self.numericFeatures].to_string())
      if self.testdata_fullpath:
        print("\n Before normalization-test data: \n", self.df_test[self.numericFeatures].to_string())

    self.df_train[self.numericFeatures] = normalize(self.df_train[self.numericFeatures], norm='l2',axis = 0, copy=True)
    if self.testdata_fullpath:
      self.df_test[self.numericFeatures] = normalize(self.df_test[self.numericFeatures], norm='l2',axis = 0, copy=True)

    if self.debugMode:
      print("\n After normalization-train data: \n", self.df_train[self.numericFeatures].to_string())
      if self.testdata_fullpath:
        print("\n After normalization-test data: \n", self.df_test[self.numericFeatures].to_string())
    print("done with normalization")

  # scale numerical features (minmax method) ---------------------------------------------------------------------------
  def scaleNumericalFeatures(self):
    # if self.debugMode:
    if True:
      print("\n Before scaling-train data: \n", self.df_train[self.numericFeatures].to_string())
      if self.testdata_fullpath:
        print("\n Before scaling-test data: \n", self.df_test[self.numericFeatures].to_string())

    print(" min/max of training data before scaling: ")
    print('Minimum value:\n', self.df_train[self.numericFeatures].min(),
          '\n\nMaximum value:\n', self.df_train[self.numericFeatures].max())

    if self.testdata_fullpath:
      print(" min/max of testing data before scaling: ")
      print('Minimum value:\n', self.df_test[self.numericFeatures].min(),
            '\n\nMaximum value:\n', self.df_test[self.numericFeatures].max())

    self.df_train[self.numericFeatures] = minmax_scale(self.df_train[self.numericFeatures], feature_range=(0,1), axis=0, copy=False) # false means inplace, but for nparray
    if self.testdata_fullpath:
      self.df_test[self.numericFeatures] = minmax_scale(self.df_test[self.numericFeatures], feature_range=(0,1), axis=0, copy=False) # false means inplace

    print(" min/max of training data after scaling: ")
    print('Minimum value:\n', self.df_train[self.numericFeatures].min(),
          '\n\nMaximum value:\n', self.df_train[self.numericFeatures].max())
    
    if self.testdata_fullpath:
      print(" min/max of testing data after scaling: ")
      print('Minimum value:\n', self.df_test[self.numericFeatures].min(),
            '\n\nMaximum value:\n', self.df_test[self.numericFeatures].max())

    if self.debugMode:
      print("\n After scaling-train data: \n", self.df_train[self.numericFeatures].to_string())
      if self.testdata_fullpath:
        print("\n After scaling-test data: \n", self.df_test[self.numericFeatures].to_string())
    print("done with scaling")


  # convert categorical features to numerical features (cordinal, one-hot, etc.)
  def categoricalFeatures_processing(self):

    # first find how many categories exist for each col
    unique_cats_of_objectFeatures = list(map(lambda col: self.df_train[col].nunique(), self.objectFeatures))
    d = dict(zip(self.objectFeatures, unique_cats_of_objectFeatures))
    
    print("unique categories of object features: ")
    print(sorted(d.items(), key = lambda x:x[1]))

    # features that will be one-hot encoded
    low_cardinality_cols = [col for col in self.objectFeatures if self.df_train[col].nunique() < self.cardinalityThreshold]

    # features that will be dropped from the dataset
    high_cardinality_cols = list(set(self.objectFeatures)-set(low_cardinality_cols))

    if self.debugMode:
      print(f'\n object features (total {len(self.objectFeatures)}):\n', self.objectFeatures)

    if self.categoricalFeatures == 1: # drop
      if self.debugMode:
        #print("\n Before drop categorical features-train: \n", self.df_train[self.objectFeatures].to_string()) # print all data
        print("\n Before drop categorical features-train: \n", self.df_train[self.objectFeatures].head()) # print all data
        if self.testdata_fullpath:
          print("\n Before drop categorical features-test: \n", self.df_test[self.objectFeatures].to_string())

        count = 0
        for col in self.objectFeatures:
          count = count + 1
          print(f'{count} {col} exits in data: {True if col in self.df_train.columns else False}')
          # self.df_train.drop(col, axis=1, inplace=True)
      
      print(f"\n objects features (total {len(self.objectFeatures)}): \n", self.objectFeatures)
      
      self.df_train.drop(self.objectFeatures, axis=1, inplace=True)
      if self.testdata_fullpath:
        self.df_test.drop(self.objectFeatures, axis=1, inplace=True)

      if self.debugMode:
        # print("\n After drop categorical features-train: \n", self.df_train.to_string()) # to print all data
        print("\n After drop categorical features-train: \n", self.df_train.head())  
        if self.testdata_fullpath:
          print("\n After drop categorical features-test: \n", self.df_test.to_string())

      # set object features to null after dropping all features
      self.objectFeatures = []
      if self.debugMode:
        print(f"\n objects features after drop (total {len(self.objectFeatures)}): \n", self.objectFeatures)

    elif self.categoricalFeatures == 2: # ordinal numbering of the categorical features
      
      print(f"\n objects features (total {len(self.objectFeatures)}): \n", self.objectFeatures)

      ordinal_encoder = OrdinalEncoder()
      if self.debugMode:
        # print("\n Before ordinal encoding-train: \n", self.df_train[self.objectFeatures].to_string()) # print all data
        print("\n Before ordinal encoding-train: \n", self.df_train[self.objectFeatures].head())
        if self.testdata_fullpath:
          # print("\n Before ordinal encoding-test: \n", self.df_test[self.objectFeatures].to_string()) # print all dadta
          print("\n Before ordinal encoding-test: \n", self.df_test[self.objectFeatures].head())
  
      ordinal_encoder.fit(self.df_train[self.objectFeatures])
      self.df_train[self.objectFeatures] = ordinal_encoder.transform(self.df_train[self.objectFeatures])
      if self.testdata_fullpath:
        self.df_test[self.objectFeatures] = ordinal_encoder.transform(self.df_test[self.objectFeatures])

      if self.debugMode:
        # print("\n After ordinal encoding-train: \n", self.df_train[self.objectFeatures].to_string()) # to print all data
        print("\n After ordinal encoding-train: \n", self.df_train[self.objectFeatures].head())
        if self.testdata_fullpath:
          #print("\n After ordinal encoding-test: \n", self.df_test[self.objectFeatures].to_string()) # to print all data
          print("\n After ordinal encoding-test: \n", self.df_test[self.objectFeatures].head())
      
      # set object features to null, and add object features to numerics
      self.numericFeatures.extend(self.objectFeatures)
      self.objectFeatures = []
      
      if self.debugMode:
        print(f"\n objects features after ordinal numbering (total {len(self.objectFeatures)}): \n", self.objectFeatures)
        print(f"\n numerical features after ordinal numbering (total {len(self.numericFeatures)}): \n", self.numericFeatures)

    elif self.categoricalFeatures == 3: # One-hot encoding of the categorical features

      if self.debugMode:
        print(f'\n Low cardinality categorical columns (candidates for one-hot encoding if selected (less than {self.cardinalityThreshold} unique categories)): \n', low_cardinality_cols)
        print(f'\n Categorical columns that will be dropped from the dataset (more than {self.cardinalityThreshold} unique categories):\n', high_cardinality_cols)
        # print("\n Before one-hot encoding-train: \n", self.df_train[low_cardinality_cols].to_string()) # to print all data
        print("\n Before one-hot encoding-train: \n", self.df_train[low_cardinality_cols].head())
        if self.testdata_fullpath:
          #print("\n Before one-hot encoding-test: \n", self.df_test[low_cardinality_cols].to_string()) # to print all data
          print("\n Before one-hot encoding-test: \n", self.df_test[low_cardinality_cols].head())
        print("\n dropping high cardinality features: \n", high_cardinality_cols)

      # 'ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
      OH_encoder = OneHotEncoder(handle_unknown='ignore', dtype='int', sparse_output=False) #, feature_name_combiner='concat')
      # OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, feature_name_combiner=self.custom_combiner)

      OH_encoder.fit(self.df_train[low_cardinality_cols])
      new_feature_names= OH_encoder.get_feature_names_out(low_cardinality_cols)
      
      OH_df_train = pd.DataFrame(OH_encoder.transform(self.df_train[low_cardinality_cols]), columns=new_feature_names )
      # OH_df_train.columns = OH_df_train.columns.astype(str) # convert the name of one-hot features from int to string
      self.df_train.drop(self.objectFeatures, axis=1,inplace=True)
      self.df_train = self.df_train.join(OH_df_train)

      if self.testdata_fullpath:
        OH_df_test = pd.DataFrame(OH_encoder.transform(self.df_test[low_cardinality_cols]), columns=new_feature_names )
        # OH_df_test.columns = OH_df_test.columns.astype(str) # convert the name of one-hot features from int to string
        self.df_test.drop(self.objectFeatures, axis=1,inplace=True)
        self.df_test = self.df_test.join(OH_df_test)    

      # set object features to null, and add object features to numerics
      self.numericFeatures.extend(new_feature_names)
      self.objectFeatures = []
      
      if self.debugMode:
        print(f"\n objects features after one-hot (total {len(self.objectFeatures)}): \n", self.objectFeatures)
        print(f"\n numerical features after one-hot (total {len(self.numericFeatures)}): \n", self.numericFeatures)


      if self.debugMode:
        print("\nnew feature names after encoding: \n", new_feature_names)

        print(f"\n objects features after one-hot (total {len(self.objectFeatures)}): \n", self.objectFeatures)
        print(f"\n numerical features after one-hot (total {len(self.numericFeatures)}): \n", self.numericFeatures)

        # print("\n After one-hot encoding-train: \n", self.df_train.to_string()) # to print all data
        print("\n After one-hot encoding-train: \n", self.df_train.head()) 
        if self.testdata_fullpath:
          # print("\n After one-hot encoding-test: \n", self.df_test.to_string()) # to print all data
          print("\n After one-hot encoding-test: \n", self.df_test.head())

    return self

  def custom_combiner(feature, category):
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)

  # split train to train and validate ----------------------------------------------------------------------------------
  def splitData(self):
    train, valid = train_test_split(self.df_train, train_size=self.train_size , random_state=50, shuffle=True)
    #print(type(train))
       
    self.x_train = train.loc[:, train.columns != self.target]
    self.y_train = train.loc[:, train.columns == self.target]
    
    self.x_valid = valid.loc[:, valid.columns != self.target]
    self.y_valid = valid.loc[:, valid.columns == self.target]

    if self.testdata_fullpath:
      self.x_test = self.df_test.loc[:, self.df_test.columns != self.target]
      self.y_test = self.df_test.loc[:, self.df_test.columns == self.target] # target might not be in test


    if self.debugMode:
      print("\nx_train: \n", self.x_train)
      print("\ny_train: \n", self.y_train)

      print("\nx_valid: \n", self.x_valid)
      print("\ny_valid: \n", self.y_valid)

      if self.testdata_fullpath:
        print("\nx_test: \n", self.x_test)
        print("\ny_test: \n", self.y_test)

    return self

  # ?
  def alignDataframes(self):

    if self.debugMode:
      print(" before alignment: ")
      print("\nx_train: \n", self.x_train)
      print("\ny_train: \n", self.y_train)

      print("\nx_valid: \n", self.x_valid)
      print("\ny_valid: \n", self.y_valid)

      if self.testdata_fullpath:
        print("\nx_test: \n", self.x_test)
        print("\ny_test: \n", self.y_test)


    self.x_train, self.x_valid = self.x_train.align(self.x_valid, join='left', axis=1)
    self.x_train, self.x_test = self.x_train.align(self.x_test, join='left', axis=1)

    if self.debugMode:
      print(" after alignment: ")
      print("\nx_train: \n", self.x_train)
      print("\ny_train: \n", self.y_train)

      print("\nx_valid: \n", self.x_valid)
      print("\ny_valid: \n", self.y_valid)

      if self.testdata_fullpath:
        print("split test data")
        print("\nx_test: \n", self.x_test)
        print("\ny_test: \n", self.y_test)

  # separate target from the data before split -------------------------------------------------------------------------
  def SeparateTarget(self):       
    self.X = self.df_train.loc[:, self.df_train.columns != self.target]
    self.y = pd.DataFrame(self.df_train.loc[:, self.df_train.columns == self.target])

    if self.debugMode:
      print("\nX: \n", self.X)
      print("\ny: \n", self.y)

    return self

  # Convert the target variable from {no,yes} to {0,1} -----------------------------------------------------------------
  def convert_target_to_binary(self):
    self.y_train[self.target].replace("no",0,inplace=True)
    self.y_train[self.target].replace("yes",1,inplace=True)

    self.y_valid[self.target].replace("no",0,inplace=True)
    self.y_valid[self.target].replace("yes",1,inplace=True)

    # convert target col/feature to ordinal
  def convert_target_to_ordinal(self):
      print(" convert target to ordinal")
      ordinal_encoder = OrdinalEncoder()
      ordinal_encoder.fit(self.y_train)
      self.y_train = ordinal_encoder.transform(self.y_train)


  # calculate mutual information for discrete values -------------------------------------------------------------------
  def make_mi_scores(self):
      
      def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")


      if self.debugMode:
        print("MI_score")
        print(self.x_train.to_string())
        print("MI_score_here")
        print(self.y_train)
      

      discrete_features = [pd.api.types.is_integer_dtype(t) for t in self.x_train.dtypes] # select int data columns

      mi_scores_numeric = mutual_info_regression(self.x_train, self.y_train, discrete_features=discrete_features)
      # mi_scores_numeric = mutual_info_regression(self.x_train, self.y_train)
      mi_scores_numeric = pd.Series(mi_scores_numeric, name="MI Scores", index=self.x_train.columns)
      mi_scores_numeric = mi_scores_numeric.sort_values(ascending=False)

      print("Mutual Information for numeric features:")
      print(mi_scores_numeric)

      plt.figure(dpi=100, figsize=(8, 5))
      plot_mi_scores(mi_scores_numeric.head(20))

      #return mi_scores_numeric

  # Principal Component Analysis (PCA) --------------------------------------------------------------------------------
  def apply_pca(X, standardize=True):
      # Standardize
      if standardize:
          X = (X - X.mean(axis=0)) / X.std(axis=0)

      print("original dataframe")
      print(X)

      # Create principal components
      pca = PCA()
      
      # transform X dataframe to pca table of data (the output is not a dataframe)
      X_pca = pca.fit_transform(X)

      print("pca dataframe before feature names")
      print(X_pca)

      # Convert to PCA output to a dataframe
      component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
      X_pca = pd.DataFrame(X_pca, columns=component_names)

      print("pca dataframe after feature names")
      print(X_pca)

      # Create loadings
      loadings = pd.DataFrame(
          pca.components_.T,  # transpose the matrix of loadings
          columns=component_names,  # so the columns are the principal components
          index=X.columns,  # and the rows are the original features
      )
      return pca, X_pca, loadings


  # statistical analysis of the categorical features for target encodeing ----------------------------------------------
  def review_Categorical_Features(self):
    print("\n categorical features for target encoding: ")
    print(self.objectFeatures)

    for item in self.objectFeatures:
      print("\n", item)
      print(self.df_train[item].value_counts())
      print(self.df_train[item].value_counts(normalize=True))
      print(self.df_train[item].unique())

#============================
# data parser
# create a new column, date_parsed, with the parsed dates-this will convert the dtype from object to datetime64
# df['date_parsed'] = pd.to_datetime(df['date'], format="%m/%d/%y") # mm/dd/yy
# df['date_parsed'] = pd.to_datetime(df['date'], format="%m/%d/%Y") # mm/dd/yyyy

#if there are multiple dataformat in the dataframe, we can let pandas decide about the data format.
# Don't use this option, unless you have to, because it is slow, and inaccurate.
# df['date_parsed'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

# review the object files:
# convert to lower case
#professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces
#professors['Country'] = professors['Country'].str.strip()



