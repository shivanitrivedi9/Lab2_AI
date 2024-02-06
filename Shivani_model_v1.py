#!/usr/bin/env python
# coding: utf-8

# ## The below code does the following actions:

# Loads the dataset from the file "wdbc.data".
# 

# Preprocesses the data by separating features (X) and labels (y).

# Splits the dataset into training and testing sets using a 80-20 split.
# 

# Performs feature scaling on the training and testing sets.
# 

# Builds a logistic regression model and trains it using the training data.

# Makes predictions on the testing set.
# 

# Evaluates the model using accuracy score and classification report.

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv(r'C:\Users\Shivani\OneDrive\Desktop\wdbc.data', header=None)

X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




