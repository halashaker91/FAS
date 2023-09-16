# -*- coding: utf-8 -*-
"""
Created on Sun May 28 15:56:31 2023
@author: AL-NABAA
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# Read the CSV file
df = pd.read_csv('fusiondata.csv')
#######################################################################
# Extract the input features and output
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column
#######################################################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#######################################################################
# Continue with the classifier code
# Neural Network classifier
nn_clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
nn_clf.fit(X_train, y_train)
nn_predictions = nn_clf.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
print("Neural Network Accuracy:", nn_accuracy)
f = open ('nn.pkl', 'wb')
pickle.dump(nn_clf, f)
print('NNCLF has been saved.')
#######################################################################
# K-Nearest Neighbors classifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_predictions = knn_clf.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
f = open ('knn.pkl', 'wb')
pickle.dump(knn_clf, f)
print('KNNCLF has been saved.')
#######################################################################
# Logistic Regression classifier
logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
logreg_predictions = logreg_clf.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print("Logistic Regression Accuracy:", logreg_accuracy)
f = open ('logreg.pkl', 'wb')
pickle.dump(logreg_clf, f)
print('logregCLF has been saved.')

