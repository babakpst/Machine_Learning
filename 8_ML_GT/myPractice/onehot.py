
import pandas as pd
from IPython.display import display # to display dataframe


diff = pd.DataFrame({'R': ['a', 'c', 'd'],
                     'T': ['d', 'a', 'c'],
                     'S_': [1, 2, 3]})


display(diff)
#display(pd.get_dummies(diff, prefix=['column1', 'column2']))
display(pd.get_dummies(diff, prefix=['R', 'T']))


#=====================================================
df = pd.DataFrame({'R': [-1, 0, 1],
                     'T': [-1, 1, 1],
                     'S_': [1, 2, 3]})



display(df.head())

# 1- Some features are {-1,0,1}. Convert these features to one-hot encoding. 
ThreeCatFeatures = ['R','T']

#display(pd.get_dummies(df_phish, prefix=ThreeCatFeatures))

Onehot = df[ThreeCatFeatures]
print("original onehot ")
display(Onehot.head())


trimmed_df = df.drop(ThreeCatFeatures,axis=1)
print("dataframe dropped")
display(trimmed_df.head())


Onehot = pd.get_dummies(Onehot.astype(str)) # one-hot encoding the feature using pandas 
print("encoded onehot ")
display(Onehot.head())


df = pd.concat([Onehot,trimmed_df],axis=1)

print("dataframe joined")
display(df.head())



print("\n\n\n\n")

dataPath = "../data/"
print(" Loading Phishing Websites Data")
df_phish = pd.read_csv(dataPath+"PhishingWebsitesData.csv")

print(" \n\nsample bank data:")
display(df_phish.head(20))

print(" Cleansing/Preprocessing the data ...")

# 1- Some features are {-1,0,1}. Convert these features to one-hot encoding. 
ThreeCatFeatures = ['URL_Length','having_Sub_Domain','SSLfinal_State','URL_of_Anchor','Links_in_tags','SFH','web_traffic','Links_pointing_to_page']

#df_phish= pd.get_dummies(df_phish.astype(str), prefix=ThreeCatFeatures)
df_phish= pd.get_dummies(df_phish.astype(str), columns=ThreeCatFeatures)
display(df_phish.head(20))
print(df_phish.head(20))
df_phish.head(20)

#df_phish = df_phish.replace('-1','0') #.astype(str) #.astype('category')
df_phish = df_phish.replace(-1,0) #.astype(str) #.astype('category')
print("after replace")
display(df_phish.head())


print("===========================")

print(" Loading Bank Marketing Data... \n")
df_bank = pd.read_csv(dataPath+"BankMarketingData.csv")

print(" \n\nsample bank data:")
display(df_bank.head())


print(" Preprocessing the data ...")

# 1- Some features have many categories, as listed below. Convert these features to one-hot encoding. 
ToOneHot = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
df_bank= pd.get_dummies(df_bank.astype(str), columns=ToOneHot)
print("after dummies")
display(df_bank.head())

# 2- reordering
column_order = list(df_bank)
column_order.insert(0, column_order.pop(column_order.index('y')))
df_bank = df_bank.loc[:, column_order]    
print("after ordering")
display(df_bank.head())    

# 3- Convert the target variable from {no,yes} to {0,1} 
df_bank['y'].replace("no",0,inplace=True)
df_bank['y'].replace("yes",1,inplace=True) 

df_bank['y'] = df_bank['y'].astype('category')    
print("after replace ")
display(df_bank.head())    

# 4- Normalize numerical values
numCols = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']


print("checkpoint 1")
df_num = df_bank[numCols].astype(float)
print("checkpoint 2")
df_stand =(df_num-df_num.min())/(df_num.max()-df_num.min())

df_bank_categorical = df_bank.drop(numCols,axis=1)
print("checkpoint 3")
df_bank = pd.concat([df_bank_categorical,df_stand],axis=1)
print("checkpoint 4")
df_bank.describe(include='all')    
print("after final ")
display(df_bank.head())

#=============
#print("checkpoint 1")
#df_num = df_bank[numCols].astype(float)
#print("checkpoint 2")
#df_bank = df_bank.drop(numCols,axis=1)
#print("after drop ")
#display(df_bank.head())    

#print("checkpoint 3")
#df_bank=df_bank.join( (df_num-df_num.min())/(df_num.max()-df_num.min()) )
#print("checkpoint 4")
#df_bank.describe(include='all')    
#print("after final ")
#display(df_bank.head())    

    


