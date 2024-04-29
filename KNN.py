import matplotlib
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import time

import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  # Add this import
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv("/content/diabetes_dataset__2019.csv")

# df.describe()
# df.info()

# Renaming columns with incorrect spelling

df.rename(columns={"Pregancies": "Pregnancies", "UriationFreq": "UrinationFrequency", "Pdiabetes": "GestationDiabetes"}, inplace=True)

# getting total NA values for each column

df.isna().sum()

# for removing rows with NA values (BMI 4, GestationDiabetes 1, Diabetic 1)

na_indices = df[df['Diabetic'].isna() | df['GestationDiabetes'].isna() | df['BMI'].isna()].index.to_list()

# print(na_indices)
df.drop(index=na_indices, inplace=True)

# checking if all rows with NA values were dropped

na_indices = df[df['Diabetic'].isna() | df['GestationDiabetes'].isna() | df['BMI'].isna()].index.to_list()
df['Pregnancies'] = df['Pregnancies'].fillna(value=0.0)

# Changing type to int instead of float
df['Pregnancies'] = df['Pregnancies'].astype(int)

# original value counts before changes
changeCols = ["RegularMedicine", "BPLevel", "GestationDiabetes", "Diabetic"]
# for col in changeCols:
#     print(df[col].value_counts())

df["RegularMedicine"] = df["RegularMedicine"].replace('o', 'no')

df["BPLevel"] = df["BPLevel"].replace('High', 'high')
df["BPLevel"] = df["BPLevel"].replace('Low', 'low')
df["BPLevel"] = df["BPLevel"].replace('normal ', 'normal')

df["GestationDiabetes"] = df["GestationDiabetes"].replace('0', 'no')
df["Diabetic"] = df["Diabetic"].replace(' no', 'no')

# Changing Gender values
df['Gender'] = df['Gender'].replace(to_replace=['Male', 'Female'], value=[0, 1]).infer_objects(copy=False)
# Changing Family_Diabetes
df['Family_Diabetes'] = df['Family_Diabetes'].replace(to_replace=['no', 'yes'], value=[0, 1]).infer_objects(copy=False)

# Changing highBP
df['highBP'] = df['highBP'].replace(to_replace=['no', 'yes'], value=[0, 1]).infer_objects(copy=False)

# Changing Smoking
df['Smoking'] = df['Smoking'].replace(to_replace=['no', 'yes'], value=[0, 1]).infer_objects(copy=False)

# Changing Smoking
df['Alcohol'] = df['Alcohol'].replace(to_replace=['no', 'yes'], value=[0, 1]).infer_objects(copy=False)

# Changing regular medicine
df['RegularMedicine'] = df['RegularMedicine'].replace(to_replace=['no', 'yes'], value=[0, 1]).infer_objects(copy=False)

# Changing GestationDiabetes
df['GestationDiabetes'] = df['GestationDiabetes'].replace(to_replace=['no', 'yes'], value=[0, 1]).infer_objects(copy=False)

# Changing UrinationFrequency
df['UrinationFrequency'] = df['UrinationFrequency'].replace(to_replace=['not much', 'quite often'], value=[0, 1]).infer_objects(copy=False)

# Changing diabetic
df['Diabetic'] = df['Diabetic'].replace(to_replace=['no', 'yes'], value=[0, 1]).infer_objects(copy=False)

# df.info()

# Getting dummy indicator variables from categorical variables

# Age
Age_dummies = pd.get_dummies(df['Age'], dtype=int, prefix='Age')
df.drop(['Age'], axis=1, inplace=True)
df = pd.concat([df, Age_dummies], axis=1)

# Physically Active
PhysicallyActive_dummies = pd.get_dummies(df['PhysicallyActive'], dtype=int, prefix='PhysicallyActive')
df.drop(['PhysicallyActive'], axis=1, inplace=True)
df = pd.concat([df, PhysicallyActive_dummies], axis=1)

# JunkFood
JunkFood_dummies = pd.get_dummies(df['JunkFood'], dtype=int, prefix='JunkFood')
df.drop(['JunkFood'], axis=1, inplace=True)
df = pd.concat([df, JunkFood_dummies], axis=1)

# Stress
Stress_dummies = pd.get_dummies(df['Stress'], dtype=int, prefix='Stress')
df.drop(['Stress'], axis=1, inplace=True)
df = pd.concat([df, Stress_dummies], axis=1)

# BPLevel
BPLevel_dummies = pd.get_dummies(df['BPLevel'], dtype=int, prefix='BPLevel')
df.drop(['BPLevel'], axis=1, inplace=True)
df = pd.concat([df, BPLevel_dummies], axis=1)

# dropping highBP
df.drop(['highBP'], axis=1, inplace=True)

# Checking if there are any males that have values other than 0 for Pregnancies and GestationDiabetes

male_and_gestation = df[(df['Gender'] == 0) & (df['GestationDiabetes'] == 1)]  # found one at index 115, need to remove
df.drop((male_and_gestation).index, inplace=True)

### 12 OBSERVATIONS WITH MALE AND PREGNANT --------------------------------------------------------------------------------------------------------DECIDE TO DELETE OR NOT
df[(df['Gender'] == 0) & (df['Pregnancies'] != 0)]

# drop the SoundSleep column to help reduce multicollinearity
df.drop(['SoundSleep'], axis=1, inplace=True)

# Train - Test Split

X = df.drop('Diabetic', axis=1)
y = df['Diabetic']

# random_state=3 allows for data to be split the same way each time (reproducible)
# stratify=y_train ---> training and testing data both have approx. same portion of diabetic and non-diabetic patients.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# split for training and validation data
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=0,
                                                                stratify=y_train)

sc_X = MinMaxScaler(feature_range=(0, 1))
X_train = sc_X.fit_transform(X_train)

X_validation = sc_X.transform(X_validation)  # Use transform instead of fit_transform for validation data

X_test = sc_X.transform(X_test)  # Use transform instead of fit_transform for test data

# Predictions on training set with KNN classifier
knn_predictions_training = best_knn.predict(X_train)

# Classification report for training set with KNN classifier
print("\nClassification Report for Training Set with Best KNN Model:")
print(classification_report(y_train, knn_predictions_training))

# Confusion matrix for training set with KNN classifier
cm_training = confusion_matrix(y_train, knn_predictions_training)
sns.heatmap(cm_training,
            annot=True,
            fmt='g',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Training Set with Best KNN Model', fontsize=15)
plt.show()

# Define the parameter grid to search
param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}

# Initialize the KNN classifier for validation set before tuning
knn_validation_before_tuning = KNeighborsClassifier()

# Train the KNN classifier on the training data (excluding validation set)
knn_validation_before_tuning.fit(X_train, y_train)

# Predictions on validation set before tuning with KNN
knn_predictions_validation_before_tuning = knn_validation_before_tuning.predict(X_validation)

# Classification report for validation set before tuning with KNN
print("\nClassification Report for Validation Set Before Tuning with KNN:")
print(classification_report(y_validation, knn_predictions_validation_before_tuning))

# Confusion matrix for validation set before tuning with KNN
cm_validation_before_tuning = confusion_matrix(y_validation, knn_predictions_validation_before_tuning)
sns.heatmap(cm_validation_before_tuning,
            annot=True,
            fmt='g',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Validation Set Before Tuning with KNN', fontsize=15)
plt.show()

# Re-initialize KNN classifier with best parameters from grid search
best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                                weights=grid_search.best_params_['weights'],
                                metric=grid_search.best_params_['metric'])

# Train the model on the entire training set (including validation data)
best_knn.fit(X_train, y_train)

# Predictions on validation set after tuning with best model
best_knn_predictions_validation_after_tuning = best_knn.predict(X_validation)

# Classification report for validation set after tuning with best model
print("\nClassification Report for Validation Set After Tuning with Best KNN Model:")
print(classification_report(y_validation, best_knn_predictions_validation_after_tuning))

# Confusion matrix for validation set after tuning with best model
cm_validation_after_tuning = confusion_matrix(y_validation, best_knn_predictions_validation_after_tuning)
sns.heatmap(cm_validation_after_tuning,
            annot=True,
            fmt='g',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Validation Set After Tuning with Best KNN Model', fontsize=15)
plt.show()

# Re-initialize KNN classifier with best parameters from grid search
best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                                weights=grid_search.best_params_['weights'],
                                metric=grid_search.best_params_['metric'])

# Train the model on the entire training set (including validation data)
best_knn.fit(X_train, y_train)

# Predictions on validation set after tuning with best model
best_knn_predictions_validation_after_tuning = best_knn.predict(X_validation)

# Classification report for validation set after tuning with best model
print("\nClassification Report for Validation Set After Tuning with Best Model:")
print(classification_report(y_validation, best_knn_predictions_validation_after_tuning))

# Confusion matrix for validation set after tuning with best model
cm_validation_after_tuning = confusion_matrix(y_validation, best_knn_predictions_validation_after_tuning)
sns.heatmap(cm_validation_after_tuning,
            annot=True,
            fmt='g',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Validation Set After Tuning with Best Model', fontsize=15)
plt.show()

# Predictions on test set
knn_test_predictions = best_knn.predict(X_test)

# Classification report for test set with best model
print("\nClassification Report for Test Set with Best Model:")
print(classification_report(y_test, knn_test_predictions))

# Confusion matrix for test set with best model
cm_test = confusion_matrix(y_test, knn_test_predictions)
sns.heatmap(cm_test,
            annot=True,
            fmt='g',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Test Set with Best Model', fontsize=15)
plt.show()

# Predict probabilities for the positive class
knn_probs_training = best_knn.predict_proba(X_train)[:, 1]

# AUC score for training set with KNN classifier
auc_test = roc_auc_score(y_test, y_test_probs)
print("AUC for Test Set with Best KNN Model:", auc_test)

# Compute probabilities for the test set
y_test_probs = best_knn.predict_proba(X_test)[:, 1]

# Compute ROC-AUC value for the test set
roc_auc_test = roc_auc_score(y_test, y_test_probs)

# Compute ROC curve for the test set
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)

# Plot ROC curve for the test set
plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve\nReceiver Operating Characteristic (ROC) Curve for Test Set')
plt.legend(loc="lower right")
plt.show()

# Print ROC-AUC value for the test set
print("ROC-AUC for Test Set:", roc_auc_test)
