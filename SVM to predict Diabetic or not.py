#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') 


# In[3]:


# printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[4]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# In[5]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[6]:


diabetes_dataset['Outcome'].value_counts()


# In[7]:


diabetes_dataset.groupby('Outcome').mean()


# In[8]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[11]:


scaler = StandardScaler()
scaler.fit(X)


# In[12]:


standardized_data = scaler.transform(X)


# In[13]:


print(standardized_data)


# In[14]:


X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[16]:


classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[17]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)


# In[18]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


# In[19]:


input_data = (6,85,79,49,75,22.8,2.587,31)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[20]:


import pickle


# In[21]:


filename ='trained_model.sav'
pickle.dump(classifier,open(filename,'wb'))


# In[22]:


loaded_model=pickle.load(open('trained_model.sav','rb'))


# In[23]:


input_data = (6,85,79,49,75,22.8,2.587,31)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




