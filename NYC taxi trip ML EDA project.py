#!/usr/bin/env python
# coding: utf-8

# # Nyc taxi trip duration prediction

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as datetime
import seaborn as sns


# In[2]:


#importing data
data=pd.read_csv(r"C:\Users\YOGENDRA SINGH\Desktop\EDA-ML-Final-Project\EDA+ML-Final Project\nyc_taxi_trip_duration.csv")


# In[3]:


#checking initial 5 rows
data.head()


# In[4]:


#finding number of rows and columns in data
data.shape


# In[5]:


#printing all the columns name
data.columns


# In[6]:


#checking for blank or null value
data.isna().sum()


# In[7]:


#checking datatypes of all the columns
data.dtypes


# In[8]:


#converting yes and no into 1 and 0 respectively
data['store_and_fwd_flag']=1 * (data.store_and_fwd_flag.values == 'Y')


# In[9]:


data['store_and_fwd_flag'] = data['store_and_fwd_flag'].astype('int64')


# In[10]:


#managing datetime and extracting month, day and hour for both pickup and dropoff
for i in['pickup','dropoff']:
    data['{}_datetime'.format(i)]=pd.to_datetime(data['{}_datetime'.format(i)])
    data['{}_month'.format(i)]=data['{}_datetime'.format(i)].apply(lambda x:x.month)
    data['{}_day_name'.format(i)]=data['{}_datetime'.format(i)].apply(lambda x:x.day_name())
    data['{}_hour'.format(i)]=data['{}_datetime'.format(i)].apply(lambda x:x.hour)


# In[11]:


data.dtypes


# In[12]:


#converting pickup_day_name and dropoff_day-name into category
#converting passenger_count,dropoff_month and pickup_month  into category
data['pickup_day_name'] = data['pickup_day_name'].astype('category')
data['dropoff_day_name'] = data['dropoff_day_name'].astype('category')
data['passenger_count'] = data['passenger_count'].astype('category')
data['pickup_month'] = data['pickup_month'].astype('category')
data['dropoff_month'] = data['dropoff_month'].astype('category')


# In[13]:


data.dtypes


# In[14]:


#new data
data.head()


# In[15]:


#trip duration in hour
data['trip_duration'].describe()/3600


# In[16]:


data.boxplot( column =['trip_duration'], grid = False)


# As you can see there is a huge outlier, so we have to deal with it otherwise it might create problems at prediction stage. To deal with it we need to log transform the trip duration before prediction to visualise it better.

# In[17]:


data['log_trip_duration'] = np.log(data['trip_duration'].values + 1)
sns.distplot(data['log_trip_duration'], kde = False, bins = 200)
plt.show()


# In[18]:


# trip duration in hour
data['trip_duration']=data['trip_duration'].apply(lambda x:x/3600)


# # Benchmark model

# In[19]:


#creating and shuffling test and train data

from sklearn.utils import shuffle

# Shuffling the Dataset
data = shuffle(data, random_state = 42)

#creating 4 divisions
div = int(data.shape[0]/4)

# 3 parts to train set and 1 part to test set
train = data.loc[:3*div+1,:]
test = data.loc[3*div+1:]


# In[20]:


train.head()


# In[21]:


test.head()


# In[22]:


# storing simple mean in a new column in the test set as "simple_mean"
test['simple_mean'] = train['trip_duration'].mean()


# In[23]:


#calculating root mean squared error
from sklearn.metrics import mean_squared_error as MSE

simple_mean_error = np.sqrt(MSE(test['trip_duration'] , test['simple_mean']))
simple_mean_error


# # Mean trip_duration with respect to vendor_id

# In[24]:


vend = pd.pivot_table(train, values='trip_duration', index = ['vendor_id'], aggfunc=np.mean)
vend


# In[25]:


# initializing new column to zero
test['vend_mean'] = 0

for i in train['vendor_id'].unique():
  # Assign the mean value corresponding to unique entry
  test['vend_mean'][test['vendor_id'] == str(i)] = train['trip_duration'][train['vendor_id'] == str(i)].mean()


# In[26]:


#calculating root mean squared error
vend_error = np.sqrt(MSE(test['trip_duration'] , test['vend_mean'] ))
vend_error


# # mean trip_duration with respect to passenger_count

# In[27]:


passenger_data = pd.pivot_table(train, values='trip_duration', index = ['passenger_count'], aggfunc=np.mean)
passenger_data


# In[28]:


# initializing new column to zero
test['passenger_data_mean'] = 0

for i in train['passenger_count'].unique():
  # Assign the mean value corresponding to unique entry
  test['passenger_data_mean'][test['passenger_count'] == i] = train['trip_duration'][train['passenger_count'] == i].mean()


# In[29]:


#calculating root mean squared error
passenger_data_error = np.sqrt(MSE(test['trip_duration'] , test['passenger_data_mean'] ))
passenger_data_error


# # mean trip_duration with respect to store_and_fwd_flag

# In[30]:


store_and_fwd = pd.pivot_table(train, values='trip_duration', index = ['store_and_fwd_flag'], aggfunc=np.mean)
store_and_fwd


# In[31]:


# initializing new column to zero
test['store_and_fwd_mean'] = 0

for i in train['store_and_fwd_flag'].unique():
  # Assign the mean value corresponding to unique entry
  test['store_and_fwd_mean'][test['store_and_fwd_flag'] == i] = train['trip_duration'][train['store_and_fwd_flag'] == i].mean()


# In[32]:


#calculating root mean squared error
store_and_fwd_error = np.sqrt(MSE(test['trip_duration'] , test['store_and_fwd_mean'] ))
store_and_fwd_error


# # KNN Model

# In[33]:


data.head()


# In[34]:


#seperating independent and dependent variables
x = data.drop(['trip_duration','id','pickup_datetime','dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','pickup_day_name','dropoff_day_name'], axis=1)
y = data['trip_duration']
x.shape, y.shape


# In[35]:


# Importing the MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[36]:


x = pd.DataFrame(x_scaled, columns = x.columns)


# In[37]:


x.head()


# In[38]:


# Importing the train test split function
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)


# In[39]:


#importing KNN classifier and metric F1score
from sklearn.neighbors import KNeighborsClassifier as KNN


# In[40]:


from sklearn import preprocessing
#converting y values tocategorical values

label_encoder=preprocessing.LabelEncoder()
train_y=label_encoder.fit_transform(train_y)


# In[45]:


# Creating instance of KNN
regg = KNN(n_neighbors = 20)

# Fitting the model
regg.fit(train_x, train_y)

# Predicting over the Train Set and calculating RMSE
test_predict = regg.predict(test_x)
k = np.sqrt(MSE(test_predict, test_y))
print('Test RMSE', k )


# In[46]:


def Elbow(K):
    #initiating empty list
    test_rmse = []
   
    #training model for evey value of K
    for i in K:
        #Instance on KNN
        regg = KNN(n_neighbors = i)
        regg.fit(train_x, train_y)
        # Appending rmse to empty list claculated using the predictions
        tmp = regg.predict(test_x)
        tmp = np.sqrt(MSE(tmp,test_y))
        test_rmse.append(tmp)
    
    return test_rmse


# In[47]:


#Defining K range
k = range(1,50,5)


# In[48]:


# calling above defined function
test = Elbow(k)


# In[49]:


# plotting the Curves
plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test RMSE')
plt.title('Elbow Curve for test')
plt.show()


# In[50]:


# Creating instance of KNN
regg = KNN(n_neighbors = 500)

# Fitting the model
regg.fit(train_x, train_y)

# Predicting over the Train Set and calculating RMSE
test_predict = regg.predict(test_x)
k = np.sqrt(MSE(test_predict, test_y))
print('Test RMSE ', k )


# The value of k is decreases as we increase the value of n_neighbours

# # Linear Regression Model

# In[51]:


#importing Linear Regression
from sklearn.linear_model import LinearRegression as LR


# In[52]:


# Creating instance of Linear Regresssion
lr = LR()

# Fitting the model
lr.fit(train_x, train_y)


# In[53]:


# Predicting over the Train Set and calculating error
train_predict = lr.predict(train_x)
k = np.sqrt(MSE(train_predict, train_y))
print('Training Root Mean Squared Error ', k )


# In[54]:


# Predicting over the Test Set and calculating error
test_predict = lr.predict(test_x)
k = np.sqrt(MSE(test_predict, test_y))
print('Test Root Mean squared Error    ', k )


# In[55]:


lr.coef_


# In[56]:


#plotting the coefficients

plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')
x = range(len(train_x.columns))
y = lr.coef_
plt.bar( x, y )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Coefficient plot')
plt.show()


# In[57]:


# Arranging and calculating the Residuals
residuals = pd.DataFrame({
    'fitted values' : test_y,
    'predicted values' : test_predict,
})

residuals['residuals'] = residuals['fitted values'] - residuals['predicted values']
residuals.head()


# In[58]:


# Histogram for distribution
plt.figure(figsize=(10, 6), dpi=120, facecolor='w', edgecolor='b')
plt.hist(residuals.residuals, bins = 150)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Distribution of Error Terms')
plt.show()


# In[59]:


# importing the QQ-plot from the from the statsmodels
from statsmodels.graphics.gofplots import qqplot

## Plotting the QQ plot
fig, ax = plt.subplots(figsize=(5,5) , dpi = 120)
qqplot(residuals.residuals, line = 's' , ax = ax)
plt.ylabel('Residual Quantiles')
plt.xlabel('Ideal Scaled Quantiles')
plt.title('distribution of Residual Errors')
plt.show()


# The QQ-plot clearly verifies our findings from the the histogram of the residuals, the data is mostly normal in nature, but there sre some outliers on the higher and lower end of the Residues.

# # Conclusion
# * From the above all three models I would prefer Linear Regression model also I prefer RMSE over all the evaluation metrics. 
# * Test RMSE is way more than Train RMSE, so we can say that the data is either overfitted or because of complexity of data.
