

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


from sklearn import tree
import itertools
import timeit


@dataclass
class DataPreprocessing:
  """_summary_
  """
  
  train_filename: str = ""
  test_filename: str = ""
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
    self.traindata_fullpath = os.path.join(self.dataPath, self.train_filename)
    self.testdata_fullpath = os.path.join(self.dataPath, self.test_filename) if self.test_filename !="" else ""

  # read data -----------------------------------
  def readData(self):
    self.df_train = pd.read_csv(self.traindata_fullpath)
    self.df_test = pd.read_csv(self.testdata_fullpath) if self.testdata_fullpath else None
    
    print("\n train data frame info: ")
    print(self.df_train.info())

    if self.testdata_fullpath:
      print("\n test data frame info: ")
      print(self.df_test.info())


    if self.target not in self.df_train.columns:
      #raise ValueError(f"The target columns '{self.target}' does not exist in the data.")
      print(f"Error: The target columns '{self.target}' does not exist in the train data. Check the train data.")
      quit()

    if self.df_test & self.target not in self.df_test.columns:
      #raise ValueError(f"The target columns '{self.target}' does not exist in the data.")
      print(f"Error: The target columns '{self.target}' does not exist in the test data. Check the test data.")
      quit()

    print("\n info about the train data: ")
    print(f' {"number of samples":<22} | {"number of features":<22} | {"Any missing data":<22} | {"Missing target data":<22}\n {len(self.df_train):<22} | {len(self.df_train.columns):<22} | {True if self.df_train.isnull().values.any() else False:<22} | {self.df_train[self.target].isnull().sum()}')

    if self.testdata_fullpath:
      print("\n info about the test data: ")
      print(f' {"number of samples":<22} | {"number of features":<22} | {"Any missing data":<22} | {"Missing target data":<22}\n {len(self.df_test):<22} | {len(self.df_test.columns):<22} | {True if self.df_test.isnull().values.any() else False:<22} | {self.df_test[self.target].isnull().sum()}')
    
    
    self.objectFeatures = self.df_train.select_dtypes(include=['object']).columns.to_list()# train object features 
    if self.testdata_fullpath and self.objectFeatures != self.df_test.select_dtypes(include=['object']).columns.to_list():
      print(f"Error: The train object data features are different from the test data object features.")
      quit()

    if self.target in self.objectFeatures:  # drop the target columns
      self.objectFeatures.remove(self.target)
    if self.debugMode:
      print("\n Here are the object features: \n", self.objectFeatures)
    
    self.numericFeatures = self.df_train.select_dtypes(exclude=['object']).columns.to_list()
    if self.testdata_fullpath and self.numericFeatures != self.df_test.select_dtypes(exclude=['object']).columns.to_list():
      print(f"Error: The train numerical data features are different from the test data numerical features.")
      quit()

    if self.target in self.numericFeatures:  # drop the target columns
      self.numericFeatures.remove(self.target)
    if self.debugMode:
      print("\n Here are the numerical features: \n", self.numericFeatures)

    missing_val_count_by_column = (self.df_train.isnull().sum())
    print("\n train data missing values: ")
    print(missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False))

    if self.test_filename !="":
      missing_val_count_by_column = (self.df_test.isnull().sum())
      print("\n test data missing values: ")
      print(missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False))

    self.objectFeatures_with_missing_data_trian = [col for col in self.df_train[self.objectFeatures].columns if self.df_train[col].isnull().any()] # categorrical features with missing data
    if self.target in self.objectFeatures_with_missing_data_trian:  # target should not be here, but let's check it out. 
      self.objectFeatures_with_missing_data_trian.remove(self.target)

    self.numericFeatures_with_missing_data_train = [col for col in self.df_train[self.numericFeatures].columns if self.df_train[col].isnull().any()] # numerical features with missing data
    if self.target in self.numericFeatures_with_missing_data_train:  # target should not be here, but let's check it out. 
      self.numericFeatures_with_missing_data_train.remove(self.target)

    self.Features_with_missing_data_train = self.objectFeatures_with_missing_data_trian + self.numericFeatures_with_missing_data_train
    # self.Features_with_missing_data_train = self.objectFeatures_with_missing_data_trian

    if self.test_filename !="":
      self.objectFeatures_with_missing_data_test = [col for col in self.df_test[self.objectFeatures].columns if self.df_test[col].isnull().any()] # categorrical features with missing data in test
      if self.target in self.objectFeatures_with_missing_data_test:  # target should not be here, but let's check it out. 
        self.objectFeatures_with_missing_data_test.remove(self.target)

      self.numericFeatures_with_missing_data_test = [col for col in self.df_train[self.numericFeatures].columns if self.df_train[col].isnull().any()] # numerical features with missing data in test
      if self.target in self.numericFeatures_with_missing_data_test:  # target should not be here, but let's check it out. 
        self.numericFeatures_with_missing_data_test.remove(self.target)

      self.Features_with_missing_data_test = self.objectFeatures_with_missing_data_test + self.numericFeatures_with_missing_data_test
        
    if self.debugMode:
      print("\n object features with missing data in train: \n", self.objectFeatures_with_missing_data_trian)
      print("\n numerical features with missing data in train: \n", self.numericFeatures_with_missing_data_train)
      print("\n Features with missing data (total {}) in train: \n".format(len(self.Features_with_missing_data_train)), self.Features_with_missing_data_train)
      print()
      print("\n object features with missing data in test: \n", self.objectFeatures_with_missing_data_test)
      print("\n numerical features with missing data in test: \n", self.numericFeatures_with_missing_data_test)
      print("\n Features with missing data (total {}) in test: \n".format(len(self.Features_with_missing_data_train)), self.Features_with_missing_data_test)

    print("\n data head")
    print(self.df_train.head())

  # remove rows/instances with missing target ---
  def missingTarget(self):
    if self.df_train[self.target].isnull().values.any():
      print(f"\n {self.df_train[self.target].isnull().sum()} instances are missing the target value in train data. Dropping the instances." )
      self.df_train.dropna(axis=0, subset=[self.target], how='any', inplace=True)
    else:
      print("\n There is no missing target in the train data.")

    if self.testdata_fullpath and self.df_test[self.target].isnull().values.any():
      print(f"\n {self.df_test[self.target].isnull().sum()} instances are missing the target value in test data. Dropping the instances." )
      self.df_test.dropna(axis=0, subset=[self.target], how='any', inplace=True)
    elif self.testdata_fullpath:
      print("\n There is no missing target in the test data.")

  #  a subset of features in the data
  def pickFeatures(self, features):
    self.df_train = self.df_train[features]
    if self.testdata_fullpath:
      self.df_test = self.df_test[features]







  def handleMissingValues(self):

    isThereAnyMissingDataInTrain = True
    isThereAnyMissingDataInTest = True
    # train data
    if not self.Features_with_missing_data_train:
      print("There is no missing data in train")
      isThereAnyMissingDataInTrain = False
    if not self.Features_with_missing_data_test:
      print("There is no missing data in test")
      isThereAnyMissingDataInTest = False
    
    if isThereAnyMissingDataInTest or isThereAnyMissingDataInTrain:
      self.objectFeatures_with_missing_data = list(set(self.objectFeatures_with_missing_data_test) | set(self.objectFeatures_with_missing_data_trian))
      self.numericFeatures_with_missing_data = list(set(self.numericFeatures_with_missing_data_test) | set(self.numericFeatures_with_missing_data_trian))
      self.Features_with_missing_data = list(set(self.Features_with_missing_data_test) | set(self.Features_with_missing_data_trian))
      
      if self.addImputeCol and not self.imputeStrategy == 'drop':
        for col in self.Features_with_missing_data:
          self.df_train[col + '_was_missing'] = self.df_train[col].isnull()
          if self.testdata_fullpath:
            self.df_test[col + '_was_missing'] = self.df_test[col].isnull()

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
      print("\n data head after Impute")
      print(self.df_train.head())
      if self.testdata_fullpath:
        print(self.df_test.head())

  # TODO: add an option for impute categorical features to drop it if more than half is NaN.
  def ImputeTheData(self): # for impute strategy mean, median, most_frequent
    if self.debugMode:
      print(f"\n Before {self.imputeStrategy} impute for num features in train data: \n", self.df_train[self.numericFeatures_with_missing_data_train].to_string())
      if self.testdata_fullpath:
        print(f"\n Before {self.imputeStrategy} impute for num features in test data: \n", self.df_test[self.numericFeatures_with_missing_data_test].to_string())

    imputer = SimpleImputer(strategy=self.imputeStrategy, copy=False)
    
    self.df_train[self.numericFeatures_with_missing_data_train] = pd.DataFrame(imputer.fit_transform(self.df_train[self.numericFeatures_with_missing_data_train]), columns = self.numericFeatures_with_missing_data_train)
    if self.testdata_fullpath:
      self.df_test[self.numericFeatures_with_missing_data_test] = pd.DataFrame(imputer.fit_transform(self.df_test[self.numericFeatures_with_missing_data_test]), columns = self.numericFeatures_with_missing_data_test)

    if self.debugMode:
      print(f"\n After {self.imputeStrategy} impute for num features in train data: \n", self.df_train[self.numericFeatures_with_missing_data_train].to_string())
      if self.testdata_fullpath:
        print(f"\n After {self.imputeStrategy} impute for num features in test data: \n", self.df_test[self.numericFeatures_with_missing_data_test].to_string())

    if self.debugMode:
      print(f"\n Before most_frequent impute for categorical features in train data: \n", self.df_train[self.objectFeatures_with_missing_data_trian].to_string())
      if self.testdata_fullpath:
        print(f"\n Before most_frequent impute for categorical features in test data: \n", self.df_test[self.objectFeatures_with_missing_data_test].to_string())

    cat_imputer = SimpleImputer(strategy='most_frequent', copy=False)
    self.df_train[self.objectFeatures_with_missing_data_trian] = pd.DataFrame(cat_imputer.fit_transform(self.df_train[self.objectFeatures_with_missing_data_trian]), columns = self.objectFeatures_with_missing_data_trian)
    if self.testdata_fullpath:
      self.df_test[self.objectFeatures_with_missing_data_test] = pd.DataFrame(cat_imputer.fit_transform(self.df_test[self.objectFeatures_with_missing_data_test]), columns = self.objectFeatures_with_missing_data_test)

    if self.debugMode:
      print("\n After most frequent impute for categorical features in train data: \n", self.df_train[self.objectFeatures_with_missing_data_trian].to_string())
      if self.testdata_fullpath:
        print("\n After most frequent impute for categorical features in test data: \n", self.df_test[self.objectFeatures_with_missing_data_test].to_string())

  # TODO add test data + check the entire function
  def ImputeConstant(self):
    if self.debugMode:
      print("\n Before most frequent impute for num features: \n", self.df_train[self.numericFeatures_with_missing_data_train])

    imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value, copy=False)
    imputed_num_features = pd.DataFrame(imputer.fit_transform(self.df_train[self.numericFeatures_with_missing_data_train]), columns = self.numericFeatures_with_missing_data_train)

    self.df_train = self.df_train.drop(self.numericFeatures_with_missing_data_train,axis=1)
    self.df_train = self.df_train.join(imputed_num_features)

    if self.debugMode:
      print("\n After most frequent impute for num features: \n", self.df_train[self.numericFeatures_with_missing_data_train])

    if self.debugMode:
      print("\n Before most frequent impute for categorical features: \n", self.df_train[self.objectFeatures_with_missing_data_trian])

    imputed_cat_features = pd.DataFrame(imputer.fit_transform(self.df_train[self.objectFeatures_with_missing_data_trian]), columns = self.objectFeatures_with_missing_data_trian)

    self.df_train = self.df_train.drop(self.objectFeatures_with_missing_data_trian,axis=1)
    self.df_train = self.df_train.join(imputed_cat_features)

    if self.debugMode:
      print("\n After most frequent impute for categorical features: \n", self.df_train[self.objectFeatures_with_missing_data_trian])

  # normalize numerical data
  # TODO: it should not normalize everything (yearbuild, Id). Fix it.
  def normalizeNumericalFeatures(self):
    if self.debugMode:
      print("\n Before normalization-train data: \n", self.df_train[self.numericFeatures].to_string())
      if self.testdata_fullpath:
        print("\n Before normalization-test data: \n", self.df_test[self.numericFeatures].to_string())
    self.df_train[self.numericFeatures] = (self.df_train[self.numericFeatures]-self.df_train[self.numericFeatures].min()) / (self.df_train[self.numericFeatures].max()-self.df_train[self.numericFeatures].min())
    if self.testdata_fullpath:
      self.df_test[self.numericFeatures] = (self.df_test[self.numericFeatures]-self.df_test[self.numericFeatures].min()) / (self.df_test[self.numericFeatures].max()-self.df_test[self.numericFeatures].min())

    if self.debugMode:
      print("\n After normalization-train data: \n", self.df_train[self.numericFeatures].to_string())
      if self.testdata_fullpath:
        print("\n After normalization-test data: \n", self.df_test[self.numericFeatures].to_string())

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
      print(f'\n Low cardinality categorical columns (candidates for one-hot encoding if selected (less than {self.cardinalityThreshold} unique categories)): \n', low_cardinality_cols)
      print(f'\n Categorical columns that will be dropped from the dataset (more than {self.cardinalityThreshold} unique categories):\n', high_cardinality_cols)

    if self.categoricalFeatures == 1: # drop
      if self.debugMode:
        print("\n Before drop categorical features-train: \n", self.df_train[self.objectFeatures].to_string())
        if self.testdata_fullpath:
          print("\n Before drop categorical features-test: \n", self.df_test[self.objectFeatures].to_string())

        count = 0
        for col in self.objectFeatures:
          count = count + 1
          print(f'{count} {col} exits in data: {True if col in self.df_train.columns else False}')
          # self.df_train.drop(col, axis=1, inplace=True)
      
      print(type(self.objectFeatures), len(self.objectFeatures))
      print("objects\n", self.objectFeatures)
      
      self.df_train.drop(self.objectFeatures, axis=1, inplace=True)
      if self.testdata_fullpath:
        self.df_test.drop(self.objectFeatures, axis=1, inplace=True)

      if self.debugMode:
        print("\n After drop categorical features-train: \n", self.df_train.to_string())
        if self.testdata_fullpath:
          print("\n After drop categorical features-test: \n", self.df_test.to_string())

    elif self.categoricalFeatures == 2: # ordinal numbering of the categorical features
      ordinal_encoder = OrdinalEncoder()
      if self.debugMode:
        print("\n Before ordinal encoding-train: \n", self.df_train[self.objectFeatures].to_string())
        if self.testdata_fullpath:
          print("\n Before ordinal encoding-test: \n", self.df_test[self.objectFeatures].to_string())
  
      ordinal_encoder.fit(self.df_train[self.objectFeatures])
      self.df_train[self.objectFeatures] = ordinal_encoder.transform(self.df_train[self.objectFeatures])
      if self.testdata_fullpath:
        self.df_test[self.objectFeatures] = ordinal_encoder.transform(self.df_test[self.objectFeatures])

      if self.debugMode:
        print("\n After ordinal encoding-train: \n", self.df_train[self.objectFeatures].to_string())
        if self.testdata_fullpath:
          print("\n After ordinal encoding-test: \n", self.df_test[self.objectFeatures].to_string())

    elif self.categoricalFeatures == 3: # One-hot encoding of the categorical features

      if self.debugMode:
        print("\n Before one-hot encoding-train: \n", self.df_train[low_cardinality_cols].to_string())
        if self.testdata_fullpath:
          print("\n Before one-hot encoding-test: \n", self.df_test[low_cardinality_cols].to_string())
        print("\n dropping high cardinality features: \n", high_cardinality_cols)

      # 'ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
      OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # 
      
      OH_encoder.fit(self.df_train[low_cardinality_cols])
      OH_df_train = pd.DataFrame(OH_encoder.transform(self.df_train[low_cardinality_cols]))
      OH_df_train.columns = OH_df_train.columns.astype(str) # convert the name of one-hot features from int to string
      self.df_train.drop(self.objectFeatures, axis=1,inplace=True)
      self.df_train = self.df_train.join(OH_df_train)

      if self.testdata_fullpath:
        OH_df_test = pd.DataFrame(OH_encoder.transform(self.df_test[low_cardinality_cols]))
        OH_df_test.columns = OH_df_test.columns.astype(str) # convert the name of one-hot features from int to string
        self.df_test.drop(self.objectFeatures, axis=1,inplace=True)
        self.df_test = self.df_test.join(OH_df_test)    

      if self.debugMode:
        print("\n After one-hot encoding-train: \n", self.df_train.to_string())
        if self.testdata_fullpath:
          print("\n After one-hot encoding-test: \n", self.df_test.to_string())

    return self

  def custom_combiner(feature, category):
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)

  # split train to train and validate
  def splitData(self):
    train, valid = train_test_split(self.df_train, train_size=self.split , random_state=50, shuffle=True)
    #print(type(train))
       
    self.x_train = train.loc[:, train.columns != self.target]
    self.y_train = train.loc[:, train.columns == self.target]
    
    self.x_valid = valid.loc[:, valid.columns != self.target]
    self.y_valid = valid.loc[:, valid.columns == self.target]

    if self.testdata_fullpath:
      self.x_test = self.df_test.loc[:, self.df_test.columns != self.target]
      self.y_test = self.df_test.loc[:, self.df_test.columns == self.target]


    if self.debugMode:
      print("\nx_train: \n", self.x_train)
      print("\ny_train: \n", self.y_train)

      print("\nx_valid: \n", self.x_valid)
      print("\ny_valid: \n", self.y_valid)

      if self.testdata_fullpath:
        print("\nx_test: \n", self.x_test)
        print("\ny_test: \n", self.y_test)

    return self

  # separate target from the data without split.
  def SeparateTarget(self):       
    self.X = self.df_train.loc[:, self.df_train.columns != self.target]
    self.y = pd.DataFrame(self.df_train.loc[:, self.df_train.columns == self.target])

    if self.debugMode:
      print("\nX: \n", self.X)
      print("\ny: \n", self.y)

    return self




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






