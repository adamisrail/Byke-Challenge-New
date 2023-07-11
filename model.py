# Part 2. Q2.2
# Implementing the model using descision tree

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

datasetModel.isna().sum() #Looking for missing values of each feature
datasetModel.isna().sum().sum() #Total missing values
datasetModel.dropna()

# Analyzing the data to find infinity
print(datasetModel[datasetModel == np.inf].count() + datasetModel[datasetModel == -np.inf].count()) #Total infinity values per each feature
print((datasetModel[datasetModel == np.inf].count() + datasetModel[datasetModel == -np.inf].count()).sum()) #Total infinity values present

#Feature extraction

#Dropping irrelevant features
datasetModel.info()
datasetModelFeatures = pd.DataFrame(datasetModel)
datasetModelFeatures = datasetModelFeatures.drop(['customer_id', 'child_region_id', 'product_type'], axis=1)
datasetModelFeatures.info()

datasetModelFeatures['acquired_date'] = pd.to_datetime(datasetModelFeatures['acquired_date'].dt.strftime('%Y-%m'))
datasetModelFeatures['acquired_date'] = (pd.to_datetime("today") - datasetModelFeatures['acquired_date']).dt.days
datasetModelFeatures.head()

datasetModelFeatures = datasetModelFeatures.rename(columns={'acquired_date': 'DaysPassedAfterAquired'})
datasetModelFeatures.info()



datasetModelFeatures['transaction_time'] = pd.to_datetime(datasetModelFeatures['transaction_time'].dt.strftime('%Y-%m'))
datasetModelFeatures['transaction_time'] = (pd.to_datetime("today") - datasetModelFeatures['transaction_time']).dt.days
datasetModelFeatures.head()

datasetModelFeatures = datasetModelFeatures.rename(columns={'transaction_time': 'TrasactionDifference'})
datasetModelFeatures.info()

datasetModelFeatures['acquired_by'] = datasetModelFeatures['acquired_by'].map(dict(OFFLINE=1, ONLINE=0))
datasetModelFeatures.info()

dataDummyVar = pd.get_dummies(datasetModelFeatures['parent_region_id'], dtype=int, drop_first=True)
dataDummyVar.info()

datasetModelFeaturesWithDummy = pd.concat([datasetModelFeatures, dataDummyVar], axis=1)
datasetModelFeaturesWithDummy.info()

datasetModelFeaturesWithDummy = datasetModelFeaturesWithDummy.drop(['parent_region_id'], axis=1)
datasetModelFeaturesWithDummy.info()


datasetModelFeaturesWithDummy['value'] = datasetModelFeaturesWithDummy['value'].gt(datasetModelFeaturesWithDummy['value'].mean())
datasetModelFeaturesWithDummy.head()



y = datasetModelFeaturesWithDummy['value']
y.info()
X = pd.DataFrame(datasetModelFeaturesWithDummy)
X = X.drop(['value'], axis=1)
X.info()


#Splitting data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt


clf = DecisionTreeClassifier() # Create the Decision Tree classifier
# clf = DecisionTreeClassifier(criterion="entropy")

clf.fit(x_train, y_train) # Train the classifier

y_pred = clf.predict(x_test) # Make predictions on the test set

accuracy_score(y_test, y_pred)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("TP = ", tp, "\nFP = ", fp, "\nTN = ", tn, "\nFN = ", fn)


from sklearn.metrics import classification_report
print('\nClasification report:\n', classification_report(y_test, y_pred))



