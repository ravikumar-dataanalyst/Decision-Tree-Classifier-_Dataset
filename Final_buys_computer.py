#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import openpyxl
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import UndefinedMetricWarning


# In[5]:


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Function importing Dataset
def importdata():
    file_path = r'E:\buys_computer.xlsx'
    # read the file:
    dataset = pd.read_excel(file_path)

    # Encode categorical data + labelencoding
 
    labelencoder = LabelEncoder()

    # Assigning numerical values + Store
    dataset['age'] = labelencoder.fit_transform(dataset['age'])
    dataset['income'] = labelencoder.fit_transform(dataset['income'])
    dataset['student'] = labelencoder.fit_transform(dataset['student'])
    dataset['credit_rating'] = labelencoder.fit_transform(dataset['credit_rating'])
    dataset['Class: buys_computer'] = labelencoder.fit_transform(dataset['Class: buys_computer'])

    # Print shape:
    print("Dataset Length: ", len(dataset))
    print("Dataset Shape: ", dataset.shape)

    # Print obseravtions:
    print("Dataset: ", dataset.head())
    return dataset


# In[6]:


# split the dataset
def splitdataset(dataset):
    # Drop 'RID' column as it is not a feature
    dataset = dataset.drop(columns=['RID'], axis=1)
    # Separating the target variable
    X = dataset.values[:, 0:3]
    Y = dataset.values[:, 4]

    # Splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


# In[7]:


# Perform training with giniIndex.
def train_using_gini(X_train, y_train):
    # Create the classifier object
    try:
        clf_gini = DecisionTreeClassifier(criterion="gini",
                                          random_state=100, max_depth=3, min_samples_leaf=5)

        # Performing training
        clf_gini.fit(X_train, y_train)
        return clf_gini
    except Exception as e:
        print(f"An error occurred while training with Gini index: {e}")
        return None  


# In[8]:


# Perform training with entropy.
def train_using_entropy(X_train, y_train):
    try:
        # Decision tree with entropy
        clf_entropy = DecisionTreeClassifier(
            criterion="entropy", random_state=100,
            max_depth=3, min_samples_leaf=5)

        # Performing training
        clf_entropy.fit(X_train, y_train)
        return clf_entropy
    except Exception as e:
        print(f"An error occurred while training with Entropy: {e}")
        return None  


# In[10]:


# To calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred, zero_division=1))


# To make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred




# In[11]:


# Driver code
def main():
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, y_train)  # Corrected here
    if clf_gini is not None:
        print("Gini Index-Result:")
        y_pred_gini = prediction(X_test, clf_gini)
        cal_accuracy(y_test, y_pred_gini)
    else:
        print("Gini index reult fail.")

    clf_entropy = train_using_entropy(X_train, y_train)  # Corrected here
    if clf_entropy is not None:
        print("\nEntropy-Result:")
        y_pred_entropy = prediction(X_test, clf_entropy)
        cal_accuracy(y_test, y_pred_entropy)
    else:
        print("Entropy result fail.")

    
# Calling main function
if __name__ == "__main__":
    main()


# In[ ]:




