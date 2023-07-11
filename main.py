import numpy as np
import pandas as pd

# HEADING: Importing the dataset
datasetPath = r'C:\adam\Coding\Bykea\Data\Sample.xlsx' #Locating the dataset
dataset = pd.read_excel(datasetPath) #Reading the excel using pandas
type(dataset) #Checking the type of dataset

#HEADING: Pre-Processing
#SUB-HEADING: Analyzing the data to check if all datatypes have been read accurate

dataset.info() #Checking if the datatypes of columns are okay
pd.set_option('display.max_columns', None) #So that when use pandas head function, columns are not hidden
dataset.head() #Checking if the data is either cutout, for example the columns having value starting with 0 might loose the 0

#Aquired data was read as object hence changing it to date time
dataset['acquired_date'] = pd.to_datetime(dataset['acquired_date'], format='mixed') #Using format = mixed since the format is mixed
dataset.info() ##Transaction time is not set to UTC


dataset['transaction_time'] = pd.to_datetime(dataset.transaction_time).dt.tz_localize('UTC') #Setting time to UTC
dataset.info() #Looks Good

#Now changing the product type to object as well since this is supposed to be an replacement for name of product, not an int
dataset['product_type'] = dataset['product_type'].astype('str')
dataset.info() #Checking if the datatypes of columns are okay
#Datatypes look good for now! Might change them in future if required


#SUB-HEADING: Analyzing the data to find missing values

dataset.isna().sum() #Looking for missing values of each feature
dataset.isna().sum().sum() #Total missing values
dataset.dropna()
#No missing values found

print(dataset[dataset == np.inf].count() + dataset[dataset == -np.inf].count()) #Total infinity values per each feature
print((dataset[dataset == np.inf].count() + dataset[dataset == -np.inf].count()).sum()) #Total infinity values present
#No infinity values found

#SUB-HEADING: Analyzing the data to find outliers
#Visualizing the feature
import matplotlib.pyplot as plt

#creating a box plot of feature value
plt.boxplot(dataset['value'])
plt.show()
#creating a scatter plot with x axis being constant 1 for viewing
plt.xlabel("Constant 1 just for viewing")
plt.ylabel("Value of dataset['value']")
plt.scatter(np.ones((len(dataset['value']))), dataset['value'], s=1) #setting the size of dot small to have a better visual


#creating a scatter plot with z-scores and value just for different views
import scipy.stats as stats
statsData = stats.zscore(dataset['value']) #Looking at the Z-Scores of feature value

plt.xlabel("z-score of dataset['value']")
plt.ylabel("Value of dataset['value']")
plt.scatter(statsData, dataset['value'], s=1) #setting the size of dot small to have a better visual

#creating a histogram with 50 bins to view outliers
plt.hist(dataset['value'], bins=50)
plt.show()

from scipy.stats import zscore
dataset['z_score'] = zscore(dataset['value'])
dataset.info() #Checking if the datatypes of columns are okay
threshold = 3 #This threshold to detect outliers using std can be changed
df_exceeding_threshold = dataset[dataset['z_score'] > threshold] #Storing and viewing customers that are outliers
df_exceeding_threshold.info()
pd.set_option('display.max_columns', None) #So that when use pandas head function, columns are not hidden
df_exceeding_threshold.head(5)

#It can be seen that outliers exist in the data. Hence removing the outliers.
#NOTE: All these outliers are based on a positive z score which means high valued clients
#Storing them to as they are a good fit to target!


threshold = 3 #This threshold to detect outliers using std can be changed
dataset = dataset[dataset['z_score'] < threshold] #Updating the dataset by removing outliers
dataset.info()

#Pre-processing is complete and the data is now ready for analysis.

# Q1. As a Data Analyst, the company requires a way to,
# 1. Identify User based on Purchase Habits, Spending Ability.

# Purchase habits can be
# a. Number of trasactions of a user
# b. Total spendings
# c. Buying high value products


# 2. Region Based analysis of Transactions and Users.

# a. Spending of a user of specific region
# b. Spending of a user who is offline and online
# c. Transaction of a user of a specific region
# d. Number of high spending users relative to the number of users


# 3. Analysis which will help navigate the direction of this new business.
# All of the above


# putting releveant code altogether for self ease
import numpy as np
import pandas as pd
datasetPath = r'C:\adam\Coding\Bykea\Data\Sample.xlsx' #Locating the dataset
dataset = pd.read_excel(datasetPath) #Reading the excel using pandas
dataset['acquired_date'] = pd.to_datetime(dataset['acquired_date'], format='mixed') #Using format = mixed since the format is mixed
dataset['transaction_time'] = pd.to_datetime(dataset.transaction_time).dt.tz_localize('UTC') #Setting time to UTC
dataset['product_type'] = dataset['product_type'].astype('str')
from scipy.stats import zscore
dataset['z_score'] = zscore(dataset['value'])
threshold = 3 #This threshold to detect outliers using std can be changed
dataset = dataset[dataset['z_score'] < threshold] #Updating the dataset by removing outliers
dataset.info()





# 1. Identify User based on Purchase Habits, Spending Ability.

# Purchase habits can be
# a. Number of trasactions of a user
# b. Total spendings
# c. Buying high value products


# a. Number of transactions of a user
dataAvgChildSpending = dataset.groupby(['customer_id']).agg({'transaction_time': 'count'}).reset_index() #Groupby to count trasactions. Reseting index to not make customerid the index
dataAvgChildSpending = dataAvgChildSpending.rename(columns={'transaction_time': 'Total_Trasactions'})
dataAvgChildSpending.info()
dataAvgChildSpending.head(5)
dataAvgChildSpending.max()

plt.xlabel("User Transaction count")
plt.ylabel("Transactions")
plt.bar_label(bars)
counts, edges, bars = plt.hist(dataAvgChildSpending['Total_Trasactions'], bins=3)
plt.show()



dataAvgChildSpending['Total_Trasactions'].mean() #Looking at the mean to know the average trasaction of a user
trasanctionThreshold = dataAvgChildSpending['Total_Trasactions'].mean() #Setting the threshold to the mean, can be changed accordingly
dataAvgChildSpending = dataAvgChildSpending[dataAvgChildSpending['Total_Trasactions'] > trasanctionThreshold]
dataAvgChildSpending.info()
dataAvgChildSpending.head(5)

# b. Total spendings
dataHighvalue = dataset.groupby(['customer_id']).agg({'value': 'sum'}).reset_index() #Groupby to sum value. Reseting index to not make customerid the index
dataHighvalue = dataHighvalue.rename(columns={'value': 'Total_value'})
dataHighvalue.info()
dataHighvalue.head(5)

dataHighvalue['Total_value'].mean() #Looking at the mean to know the average trasaction of a user
valueThreshold = dataHighvalue['Total_value'].mean() #Setting the threshold to the mean, can be changed accordingly
dataHighvalue = dataHighvalue[dataHighvalue['Total_value'] > valueThreshold] #Only storing values that exceed the threshold
dataHighvalue.info()
dataHighvalue.head(5)

# c. Buying high value products

#Finding the high value product by checking the mean value of each product and diving them by total transactions of that product
dataHighValueProduct = dataset.groupby(['product_type']).agg({'transaction_time': 'count', 'value': 'sum'}).reset_index()
dataHighValueProduct = dataHighValueProduct.rename(columns={'value': 'TotalProductValue'}) #changing column names
dataHighValueProduct = dataHighValueProduct.rename(columns={'transaction_time': 'TotalTrasactions'}) #changing column names
dataHighValueProduct.info()
dataHighValueProduct.head(10)

dataHighValueProduct['AvgValuePerTransaction'] = dataHighValueProduct['TotalProductValue'] / dataHighValueProduct['TotalTrasactions'] #Avg product value per each trasaction
dataHighValueProduct.head(10)
dataHighValueProduct['AvgValuePerTransaction'].mean() #Looking at the mean to know the average trasaction of a user
meanProductValueThreshold = dataHighValueProduct['AvgValuePerTransaction'].mean() #Setting the threshold to the mean, can be changed accordingly
dataHighValueProduct = dataHighValueProduct[dataHighValueProduct['AvgValuePerTransaction'] > meanProductValueThreshold] #Only storing values that exceed the threshold
dataHighValueProduct.info()
dataHighValueProduct.head(10)


#Finding customers that purchased the high valued products
print(dataset['product_type'].value_counts())  # The total count of each product purchased

#Looping through customers that have purchased high valued products
dataCustHighValueProd = pd.DataFrame()
for idProd, prod in enumerate(dataHighValueProduct['product_type']):
    if idProd == 0:
        dataCustHighValueProd = pd.DataFrame()
    TempdataCustHighValueProd = dataset[dataset['product_type'] == prod]
    dataCustHighValueProd = pd.concat([dataCustHighValueProd, TempdataCustHighValueProd], ignore_index=True)

dataCustHighValueProd.info()
print(dataHighValueProduct['TotalTrasactions'].sum())  #Checking to see if all is working fine






# 2. Region Based analysis of Transactions and Users.

# a. Average value per transaction of specific region
# b. Spending of a user who is offline and online
# c. Number of high spending users relative to the number of users


# Part 1: 2a. Average value per transaction of specific region

#Parent region
dataAvgParentSpending = dataset.groupby(['parent_region_id']).agg({'transaction_time': 'count', 'value': 'sum'}).reset_index() #Groupby to count trasactions. Reseting index to not make customerid the index
dataAvgParentSpending = dataAvgParentSpending.rename(columns={'transaction_time': 'TotalTrasactions'})
dataAvgParentSpending = dataAvgParentSpending.rename(columns={'value': 'TotalValue'})
dataAvgParentSpending.info()
dataset['parent_region_id'].value_counts().value_counts().sum() #Checking for errors
dataAvgParentSpending.head(5)


dataAvgParentSpending['AvgValuePerTransaction'] = dataAvgParentSpending['TotalValue'] / dataAvgParentSpending['TotalTrasactions'] #Avg product value per each trasaction
dataAvgParentSpending['TotalTrasactions'].mean() #Looking at the mean to know the average trasaction of a user
avgParentSpendingThreshold = dataAvgParentSpending['TotalTrasactions'].mean() #Setting the threshold to the mean, can be changed accordingly
dataAvgParentSpending = dataAvgParentSpending[dataAvgParentSpending['TotalTrasactions'] > avgParentSpendingThreshold]
dataAvgParentSpending.info()
dataAvgParentSpending.head(5)


#Child region
dataAvgChildSpending = dataset.groupby(['child_region_id']).agg({'transaction_time': 'count', 'value': 'sum'}).reset_index() #Groupby to count trasactions. Reseting index to not make customerid the index
dataAvgChildSpending = dataAvgChildSpending.rename(columns={'transaction_time': 'TotalTrasactions'})
dataAvgChildSpending = dataAvgChildSpending.rename(columns={'value': 'TotalValue'})
dataAvgChildSpending.info()
dataset['child_region_id'].value_counts().value_counts().sum() #Checking for errors
dataAvgChildSpending.head(5)


dataAvgChildSpending['AvgValuePerTransaction'] = dataAvgChildSpending['TotalValue'] / dataAvgChildSpending['TotalTrasactions'] #Avg product value per each trasaction
dataAvgChildSpending['TotalTrasactions'].mean() #Looking at the mean to know the average trasaction of a user
avgChildSpendingThreshold = dataAvgChildSpending['TotalTrasactions'].mean() #Setting the threshold to the mean, can be changed accordingly
dataAvgChildSpending = dataAvgChildSpending[dataAvgChildSpending['TotalTrasactions'] > avgChildSpendingThreshold]
dataAvgChildSpending.info()
dataAvgChildSpending.head(5)

# Part 1: 2b. Spending of a user who is offline and online

dataAquiredBy = dataset.groupby(['acquired_by']).agg({'transaction_time': 'count', 'value': 'sum'}).reset_index() #Groupby to sum value. Reseting index to not make customerid the index
dataAquiredBy = dataAquiredBy.rename(columns={'transaction_time': 'TotalTrasactions'})
dataAquiredBy = dataAquiredBy.rename(columns={'value': 'TotalValue'})

dataAquiredBy.info()
dataAquiredBy.head(5)


dataAquiredBy['AvgValuePerTransaction'] = dataAquiredBy['TotalValue'] / dataAquiredBy['TotalTrasactions'] #Avg product value per each trasaction
dataAquiredBy['TotalTrasactions'].mean() #Looking at the mean to know the average trasaction of a user
dataAquiredByThreshold = dataAquiredBy['TotalTrasactions'].mean() #Setting the threshold to the mean, can be changed accordingly
dataAquiredBy = dataAquiredBy[dataAquiredBy['TotalTrasactions'] > dataAquiredByThreshold]
dataAquiredBy.info()
dataAquiredBy.head(5)

# Part1 2c. Region where number of high spending users relative to the number of users
dataset.info()
dataHigherSpendingUsersRegion = pd.DataFrame(dataset) #Creating a dataframe that replicates dataset
dataHigherSpendingUsersRegion = dataHigherSpendingUsersRegion.groupby(['customer_id']).agg({'value': 'sum'
                                                                 , 'acquired_by': 'first'
                                                                 ,'acquired_date': 'first'
                                                                 ,'transaction_time': 'count'
                                                                 ,'product_type': 'first'
                                                                 ,'parent_region_id': 'first'
                                                                 ,'child_region_id': 'first'
                                                                 }).reset_index() #Groupby to sum value. Reseting index to not make customerid the index
dataHigherSpendingUsersRegion = dataHigherSpendingUsersRegion.rename(columns={'value': 'TotalValue'})
dataHigherSpendingUsersRegion = dataHigherSpendingUsersRegion.rename(columns={'transaction_time': 'TotalTransactions'})

dataHigherSpendingUsersRegion['AvgUserValue'] = dataHigherSpendingUsersRegion['TotalValue'] / dataHigherSpendingUsersRegion['TotalTransactions']
dataHigherSpendingUsersRegion['HighSpendUser'] = dataHigherSpendingUsersRegion['AvgUserValue'].gt(dataHigherSpendingUsersRegion['AvgUserValue'].mean()).map({True: 1, False: 0}) #Where the value of user is greater than mean than adding a column that tell the user is high valued
dataHigherSpendingUsersRegion.info()
dataHigherSpendingUsersRegion.head(20)

dataHigherSpendingUsersRegion = dataHigherSpendingUsersRegion.groupby('child_region_id').agg({'HighSpendUser': 'sum'
                                                                 , 'acquired_by': 'first'
                                                                 ,'acquired_date': 'first'
                                                                 ,'TotalTransactions': 'count'
                                                                 ,'product_type': 'first'

                                                                 ,'TotalValue': 'sum'

                                                                 }).reset_index() #Groupby to sum and count high vallued user and total users. Reseting index to not make customerid the index
dataHigherSpendingUsersRegion = dataHigherSpendingUsersRegion.rename(columns={'TotalTransactions': 'TotalUsers'})
dataHigherSpendingUsersRegion.info()
dataHigherSpendingUsersRegion.head(5)


dataHigherSpendingUsersRegion['averageRegionHighValuedUser'] = dataHigherSpendingUsersRegion['HighSpendUser'] / dataHigherSpendingUsersRegion['TotalUsers'] #high spend user to total user ratio acc to region
dataHigherSpendingUsersRegion['averageRegionHighValuedUser'].mean() #Looking at the mean
HigherSpendingUsersRegionThreshold = dataHigherSpendingUsersRegion['averageRegionHighValuedUser'].mean() #Setting the threshold to the mean, can be changed accordingly
dataHigherSpendingUsersRegion = dataHigherSpendingUsersRegion[dataHigherSpendingUsersRegion['averageRegionHighValuedUser'] > HigherSpendingUsersRegionThreshold]
dataHigherSpendingUsersRegion.info()
dataHigherSpendingUsersRegion.head(5)


# 3. Analysis which will help navigate the direction of this new business.
dataset['value'].mean() #Average value/spending of user
dataHighSpendCustomer = pd.DataFrame(dataset) #Creating a dataframe that replicates dataset
dataHighSpendCustomer['HighSpendUser'] = dataset['value'].gt(dataset['value'].mean()).map({True: 1, False: 0}) #Where the value of user is greater than mean than adding a column that tell the user is high valued
dataHighSpendCustomer.info()
dataHighSpendCustomer.head()

print('This tells us about the percentage of good high paying customers in the total data', dataHighSpendCustomer['HighSpendUser'].value_counts()[1] / (dataHighSpendCustomer['HighSpendUser'].value_counts()[0] + dataHighSpendCustomer['HighSpendUser'].value_counts()[1]))

import matplotlib.pyplot as plt
y = np.array([dataHighSpendCustomer['HighSpendUser'].value_counts()[1], dataHighSpendCustomer['HighSpendUser'].value_counts()[0]])
plt.pie(y, labels=["High Spend Customers", "Low Spend Customers"])
plt.show()

#Plotting using answer 2c
#Regions

dataHigherSpendingUsersRegion.info()
dataHigherSpendingUsersRegion.head()

# Bar Plot
plt.xlabel("Region")
plt.ylabel("average Region High Valued User")
plt.bar(dataHigherSpendingUsersRegion.child_region_id, dataHigherSpendingUsersRegion.averageRegionHighValuedUser)

#Scatter Plot
import scipy.stats as stats
plt.xlabel("Region")
plt.ylabel("average Region High Valued User")
plt.scatter(dataHigherSpendingUsersRegion.child_region_id, dataHigherSpendingUsersRegion.averageRegionHighValuedUser, s=1) #setting the size of dot small to have a better visual

# Pie Chart
GoodRegion = dataHigherSpendingUsersRegion['averageRegionHighValuedUser'][dataHigherSpendingUsersRegion['averageRegionHighValuedUser'] >= 0.5]
BadRegion = dataHigherSpendingUsersRegion['averageRegionHighValuedUser'][dataHigherSpendingUsersRegion['averageRegionHighValuedUser'] < 0.5]
GoodRegion.count()
plt.pie([GoodRegion.count(), BadRegion.count()], labels = ['High Spending Region', "Low Spending Region"])

# As a Data Scientist, the company want to utilize Machine Learning to help,
# 1. Identify the types of Users who will be a perfect target for the new company.

# 2. Identify the susceptible conditions to target and acquire the user.

# 3. A Predictor which helps tag newly acquired users according to the
# a. Location
# b. Time
# c. Purchase Amount
# d. Acquired By
# e. Acquisition Time
# The purpose of this is to identify and segregate potential sticky customers.


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#PART 2

# Checkpoint 2
# putting releveant code altogether for self ease
import numpy as np
import pandas as pd
datasetPath = r'C:\adam\Coding\Bykea\Data\Sample.xlsx' #Locating the dataset
dataset = pd.read_excel(datasetPath) #Reading the excel using pandas
dataset['acquired_date'] = pd.to_datetime(dataset['acquired_date'], format='mixed') #Using format = mixed since the format is mixed
dataset['transaction_time'] = pd.to_datetime(dataset.transaction_time).dt.tz_localize('UTC') #Setting time to UTC
dataset['product_type'] = dataset['product_type'].astype('str')
from scipy.stats import zscore
dataset['z_score'] = zscore(dataset['value'])
threshold = 3 #This threshold to detect outliers using std can be changed
dataset = dataset[dataset['z_score'] < threshold] #Updating the dataset by removing outliers
dataset.info()

# Part 2
# Answer 1
# 1. Identify the types of Users who will be a perfect target for the new company.

#Labeling the dataset
# User who have an above average spending
newDataHighSpendCustomer = dataset.groupby(['customer_id']).agg({'value': 'sum'
                                                                 , 'acquired_by': 'first'
                                                                 ,'acquired_date': 'first'
                                                                 ,'transaction_time': 'count'
                                                                 ,'product_type': 'first'
                                                                 ,'parent_region_id': 'first'
                                                                 ,'child_region_id': 'first'
                                                                 }).reset_index() #Groupby to sum value. Reseting index to not make customerid the index
newDataHighSpendCustomer = newDataHighSpendCustomer.rename(columns={'value': 'TotalValue'})
newDataHighSpendCustomer = newDataHighSpendCustomer.rename(columns={'transaction_time': 'TotalTransactions'})

newDataHighSpendCustomer['AvgUserValue'] = newDataHighSpendCustomer['TotalValue'] / newDataHighSpendCustomer['TotalTransactions']
newDataHighSpendCustomer['HighSpendUser'] = newDataHighSpendCustomer['AvgUserValue'].gt(newDataHighSpendCustomer['AvgUserValue'].mean()).map({True: 1, False: 0}) #Where the value of user is greater than mean than adding a column that tell the user is high valued
newDataHighSpendCustomer.info()
newDataHighSpendCustomer.head(20)


# These are the user customers to target
newDataHighSpendCustomer = newDataHighSpendCustomer[newDataHighSpendCustomer['HighSpendUser'] == 1]
newDataHighSpendCustomer.info()
newDataHighSpendCustomer.head(20)



# Applying K-Means to form clusters
dataset.info()
datasetKMeans = dataset.groupby(['customer_id']).agg({'value': 'sum'
                                                                 , 'acquired_by': 'first'
                                                                 ,'acquired_date': 'first'
                                                                 ,'transaction_time': 'count'
                                                                 ,'product_type': 'first'
                                                                 ,'parent_region_id': 'first'
                                                                 ,'child_region_id': 'first'
                                                                 }).reset_index() #Groupby to sum value. Reseting index to not make customerid the index
datasetKMeans = datasetKMeans.rename(columns={'value': 'TotalValue'})
datasetKMeans = datasetKMeans.rename(columns={'transaction_time': 'TotalTransactions'})
datasetKMeans.info()
datasetKMeans = datasetKMeans.drop(['customer_id', 'acquired_date', 'product_type', 'parent_region_id', 'child_region_id'], axis=1)
datasetKMeans['acquired_by'] = datasetKMeans['acquired_by'].map(dict(OFFLINE=1, ONLINE=0))
datasetKMeans.info()
datasetKMeans.head()


import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


#Viewing the elblow using inertia (elbow method)
scaled_df = StandardScaler().fit_transform(datasetKMeans) #Scaling the data to fit kmeans requirement
print(scaled_df[:5])
inertias = []
kmeans_kwargs = {"init": "random", "n_init": 10, "random_state": 1}

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, **kmeans_kwargs)
    kmeans.fit(scaled_df)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Using 2 clusters
kmeans_kwargs = {"init": "random", "n_init": 10, "random_state": 1}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(scaled_df)
datasetKMeans['cluster'] = kmeans.labels_
datasetKMeans.info()
datasetKMeans['cluster'].value_counts()

plt.pie([GoodRegion.count(), BadRegion.count()], labels = ['High Spending Region', "Low Spending Region"])
plt.pie([datasetKMeans['cluster'].value_counts()[1], datasetKMeans['cluster'].value_counts()[0]], labels = ['Cluster 1', "Cluster 2"])



# 2. Identify the susceptible conditions to target and acquire the user.

from scipy.stats import pearsonr
def pvalues(df):
    cols = pd.DataFrame(columns=df.columns)
    p = cols.transpose().join(cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            p[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return p

dataset.info()
datasetPvalues = pd.DataFrame(dataset)
datasetPvalues = datasetPvalues.drop(['customer_id', 'transaction_time', 'acquired_date', 'product_type', 'parent_region_id', 'child_region_id'], axis=1)
datasetPvalues = datasetPvalues.drop(['z_score'], axis=1)
datasetPvalues['acquired_by'] = datasetPvalues['acquired_by'].map(dict(OFFLINE=1, ONLINE=0))
datasetPvalues.info()
pvalues(datasetPvalues)


# 3. A Predictor which helps tag newly acquired users according to the
# a. Location
# b. Time
# c. Purchase Amount
# d. Acquired By
# e. Acquisition Time
# The purpose of this is to identify and segregate potential sticky customers.



#Labeling the dataset
# User who have an above average spending
newDataHighSpendCustomer = dataset.groupby(['customer_id']).agg({'value': 'sum'
                                                                 , 'acquired_by': 'first'
                                                                 ,'acquired_date': 'first'
                                                                 ,'transaction_time': 'count'
                                                                 ,'product_type': 'first'
                                                                 ,'parent_region_id': 'first'
                                                                 ,'child_region_id': 'first'
                                                                 }).reset_index() #Groupby to sum value. Reseting index to not make customerid the index
newDataHighSpendCustomer = newDataHighSpendCustomer.rename(columns={'value': 'TotalValue'})
newDataHighSpendCustomer = newDataHighSpendCustomer.rename(columns={'transaction_time': 'TotalTransactions'})

newDataHighSpendCustomer['AvgUserValue'] = newDataHighSpendCustomer['TotalValue'] / newDataHighSpendCustomer['TotalTransactions']
newDataHighSpendCustomer['HighSpendUser'] = newDataHighSpendCustomer['AvgUserValue'].gt(newDataHighSpendCustomer['AvgUserValue'].mean()).map({True: 1, False: 0}) #Where the value of user is greater than mean than adding a column that tell the user is high valued
newDataHighSpendCustomer.info()
newDataHighSpendCustomer.head(20)


# These are the user customers to target
newDataHighSpendCustomer = newDataHighSpendCustomer[newDataHighSpendCustomer['HighSpendUser'] == 1]
newDataHighSpendCustomer.info()
newDataHighSpendCustomer.head(20)



