# Part 2. Q2.2
# Implementing the model using descision tree
# Feature Extraction



import numpy as np
import pandas as pd

datasetPath = r'C:\adam\Coding\Bykea\Data\Sample.xlsx' #Locating the dataset
datasetModel = pd.read_excel(datasetPath) #Reading the excel using pandas
datasetModel['acquired_date'] = pd.to_datetime(datasetModel['acquired_date'], format='mixed') #Using format = mixed since the format is mixed
datasetModel['transaction_time'] = pd.to_datetime(datasetModel.transaction_time).dt.tz_localize('UTC') #Setting time to UTC
datasetModel['product_type'] = datasetModel['product_type'].astype('str')

# Removing outliers
from scipy.stats import zscore
datasetModel['z_score'] = zscore(datasetModel['value'])
threshold = 3 #This threshold to detect outliers using std can be changed
datasetModel = datasetModel[datasetModel['z_score'] < threshold] #Updating the dataset by removing outliers
datasetModel.info()
datasetModel = datasetModel.drop(['z_score'], axis=1)
datasetModel.info()

# Analyzing the data to find missing values

dataset.isna().sum() #Looking for missing values of each feature
dataset.isna().sum().sum() #Total missing values
df.dropna()

# Analyzing the data to find infinity
print(dataset[dataset == np.inf].count() + dataset[dataset == -np.inf].count()) #Total infinity values per each feature
print((dataset[dataset == np.inf].count() + dataset[dataset == -np.inf].count()).sum()) #Total infinity values present







