#!/usr/bin/env python
# coding: utf-8

# ## The below code does the following actions:

# We import the RandomForestClassifier from the sklearn.ensemble module.

# We replace the LogisticRegression model with RandomForestClassifier.
# 

# We initialize the RandomForestClassifier with 100 trees (n_estimators=100) and a random_state of 42 for reproducibility.
# 

# We fit the model to the training data and make predictions on the testing data.

# Finally, we evaluate the model using accuracy_score and classification_report.

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv(r'C:\Users\Shivani\OneDrive\Desktop\wdbc.data', header=None)
print(data)


# In[3]:


# Drop the first column as it represents the ID, and the second column as it represents the diagnosis
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build and train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




