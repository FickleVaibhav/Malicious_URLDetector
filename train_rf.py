# Importing modules and libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
import joblib
import os
from dotenv import load_dotenv

# Assigning label and data
label = []  # List to store labels from the dataset
data = []  # List to store features from the dataset

# Creating the path of the model
load_dotenv()  # Load environment variables from .env file
MODEL_PATH_RF = r"C:\Users\fickl\Desktop\final_model\mal_link\models\random_forest.pkl"  # Path to save the trained model

# Reading the dataset from a file
with open('C:/Users/fickl/Desktop/final_model/mal_link/train/data/web_data.arff') as fh:
    for line in fh:
        line = line.strip()  # Remove leading/trailing whitespaces
        temp = line.split(',')  # Split line by commas
        label.append(temp[-1])  # Append the last element as the label
        data.append(temp[0:-1])  # Append all elements except the last as features

# Converting data and labels to NumPy arrays
X = np.array(data)  # Features
y = np.array(label)  # Labels

# Selecting specific features (column indices)
X = X[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12,
          13, 14, 15, 16, 17, 22, 23, 24, 25, 27, 29]]
X = np.array(X).astype(np.float64)  # Ensure all data is of type float64

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # 75% train, 25% test

# Initializing and training the Random Forest Classifier
clf = RandomForestClassifier(random_state=42, verbose=1)  # Verbose mode to show training progress
clf.fit(X_train, y_train)  # Train the model using training data

# Getting feature importance from the trained model
importance = clf.feature_importances_  # Importance scores for each feature
print(importance)  # Print feature importance

# Evaluating the model
print(clf.score(X_test, y_test))  # Print the accuracy on the test set

# Saving the trained model to a file
joblib.dump(clf, MODEL_PATH_RF, compress=9)  # Save model with high compression
