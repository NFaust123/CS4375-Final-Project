import matplotlib
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import time

import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



df = pd.read_csv("diabetes_dataset__2019.csv")

#df.describe()
#df.info()

# Renaming columns with incorrect spelling

df.rename(columns={"Pregancies" : "Pregnancies", "UriationFreq" : "UrinationFrequency", "Pdiabetes" : "GestationDiabetes"}, inplace=True)

# getting total NA values for each column

df.isna().sum()

# for removing rows with NA values (BMI 4, GestationDiabetes 1, Diabetic 1)

na_indices = df[df['Diabetic'].isna() | df['GestationDiabetes'].isna() | df['BMI'].isna()].index.to_list()

#print(na_indices)
df.drop(index= na_indices,inplace = True)

# checking if all rows with NA values were dropped

na_indices = df[df['Diabetic'].isna() | df['GestationDiabetes'].isna() | df['BMI'].isna()].index.to_list()

#print(na_indices)

#df.info()

# Checking values of "Pregnancies"
#print(df.Pregnancies.unique())
#print(df['Pregnancies'].value_counts())

# Changing NA values in "Pregnancies" to 0

df['Pregnancies'] = df['Pregnancies'].fillna(value = 0.0)

#print(df['Pregnancies'].value_counts())

# Changing type to int instead of float
df['Pregnancies'] = df['Pregnancies'].astype(int)

# original value counts before changes
changeCols = ["RegularMedicine", "BPLevel", "GestationDiabetes", "Diabetic"]
#for col in changeCols:
 # print(df[col].value_counts())

df["RegularMedicine"] = df["RegularMedicine"].replace('o', 'no')

df["BPLevel"] = df["BPLevel"].replace('High', 'high')
df["BPLevel"] = df["BPLevel"].replace('Low', 'low')
df["BPLevel"] = df["BPLevel"].replace('normal ', 'normal')

df["GestationDiabetes"] = df["GestationDiabetes"].replace('0', 'no')
df["Diabetic"] = df["Diabetic"].replace(' no', 'no')

# checking if changes applied
#changeCols = ["RegularMedicine", "BPLevel", "GestationDiabetes", "Diabetic"]
#for col in changeCols:
 # print(df[col].value_counts())

# Changing Gender values
df['Gender'] = df['Gender'].replace(to_replace = ['Male', 'Female'], value = [0, 1]).infer_objects(copy=False)
# Changing Family_Diabetes
df['Family_Diabetes'] = df['Family_Diabetes'].replace(to_replace = ['no', 'yes'], value = [0, 1]).infer_objects(copy=False)

# Changing highBP
df['highBP'] = df['highBP'].replace(to_replace = ['no', 'yes'], value = [0, 1]).infer_objects(copy=False)

# Changing Smoking
df['Smoking'] = df['Smoking'].replace(to_replace = ['no', 'yes'], value = [0, 1]).infer_objects(copy=False)

# Changing Smoking
df['Alcohol'] = df['Alcohol'].replace(to_replace = ['no', 'yes'], value = [0, 1]).infer_objects(copy=False)

# Changing regular medicine
df['RegularMedicine'] = df['RegularMedicine'].replace(to_replace = ['no', 'yes'], value = [0, 1]).infer_objects(copy=False)

# Changing GestationDiabetes
df['GestationDiabetes'] = df['GestationDiabetes'].replace(to_replace = ['no', 'yes'], value = [0, 1]).infer_objects(copy=False)

# Changing UrinationFrequency
df['UrinationFrequency'] = df['UrinationFrequency'].replace(to_replace = ['not much', 'quite often'], value = [0, 1]).infer_objects(copy=False)

# Changing diabetic
df['Diabetic'] = df['Diabetic'].replace(to_replace = ['no', 'yes'], value = [0, 1]).infer_objects(copy=False)

#df.info()

# Getting dummy indicator variables from categorical variables

# Age
Age_dummies = pd.get_dummies(df['Age'], dtype = int, prefix='Age')
df.drop(['Age'], axis = 1, inplace=True)
df = pd.concat([df, Age_dummies], axis = 1)

# Physically Active
PhysicallyActive_dummies = pd.get_dummies(df['PhysicallyActive'], dtype = int, prefix='PhysicallyActive')
df.drop(['PhysicallyActive'], axis = 1, inplace=True)
df = pd.concat([df, PhysicallyActive_dummies], axis = 1)

# JunkFood
JunkFood_dummies = pd.get_dummies(df['JunkFood'], dtype = int, prefix='JunkFood')
df.drop(['JunkFood'], axis = 1, inplace=True)
df = pd.concat([df, JunkFood_dummies], axis = 1)

# Stress
Stress_dummies = pd.get_dummies(df['Stress'], dtype = int, prefix='Stress')
df.drop(['Stress'], axis = 1, inplace=True)
df = pd.concat([df, Stress_dummies], axis = 1)

# BPLevel
BPLevel_dummies = pd.get_dummies(df['BPLevel'], dtype = int, prefix='BPLevel')
df.drop(['BPLevel'], axis = 1, inplace=True)
df = pd.concat([df, BPLevel_dummies], axis = 1)

# dropping highBP
df.drop(['highBP'], axis = 1, inplace=True)

# Checking if there are any males that have values other than 0 for Pregnancies and GestationDiabetes

male_and_gestation = df[(df['Gender']==0) & (df['GestationDiabetes']==1)]  # found one at index 115, need to remove
df.drop((male_and_gestation).index, inplace = True)

### 12 OBSERVATIONS WITH MALE AND PREGNANT --------------------------------------------------------------------------------------------------------DECIDE TO DELETE OR NOT
df[(df['Gender']==0) & (df['Pregnancies']!=0)]

# Renaming/Rearranging columns
#df = df.iloc[ : , [15,12,13,14,0,1,18,16,17,19,2,3,4,5,6,7,21,22,23,20,25,26,27,24,29,30,28,8,9,10,11]]

#df.columns = ['Age_UNDER40', 'Age_40-49', 'Age_50-59', 'Age_OVER60', 'Gender', 'FamilyHistory',
#              'Exercise_NONE', 'Exercise_UNDER30MIN', 'Exercise_30MIN-1HR', 'Exercise_OVER1HR',
#              'BMI', 'Smoker', 'AlcoholConsumption', 'SleepHrs', 'SoundSleepHrs', 'RegMedicineIntake',
#              'JunkFood_OCCASIONAL', 'JunkFood_OFTEN', 'JunkFood_VERYOFTEN', 'JunkFood_ALWAYS',
#              'Stress_NONE', 'Stress_OCCASIONAL', 'Stress_OFTEN', 'Stress_ALWAYS', 'BPLevel_LOW', 'BPLevel_NORMAL', 'BPLevel_HIGH',
#              'NumPregnancies', 'GestationalDiabetes', 'UrinationFrequency', 'Diabetic']
#df.info()

# Train - Test Split

X = df.drop('Diabetic', axis=1)
y = df['Diabetic']

# random_state=3 allows for data to be split the same way each time (reproducible)
# stratify=y_train ---> training and testing data both have approx. same portion of diabetic and non-diabetic patients.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# split for training and validation data
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#print("With Standard Scaler X_Train")
#print(X_train)

X_validation = sc_X.fit_transform(X_validation)

X_test = sc_X.fit_transform(X_test)
#print("\nWith Standard Scaler X_Test")
#print(X_test)

print(y_train.value_counts())
print(y_validation.value_counts())
print(y_test.value_counts())


# determine the best solver to use based on time to complete and accuracy
# solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

'''
for sol in solvers:
    start = time.time()
    logisticRegr = LogisticRegression(solver=sol)
    logisticRegr.fit(X_train, y_train)
    end = time.time()
    print(sol + " Fit time: ", end-start)
    predictions = logisticRegr.predict(X_validation)
    print(classification_report(y_validation, predictions))
'''


# train the model on the training data
logisticRegr = LogisticRegression(solver='liblinear')
logisticRegr.fit(X_train, y_train)

# results for validation data
predictions = logisticRegr.predict(X_validation)
print("Classification report for validation set: ")
print(classification_report(y_validation, predictions))

# confusion matrix before parameter tuning
cm = confusion_matrix(y_validation, predictions)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Diabetic', 'Not Diabetic'],
            yticklabels=['Diabetic', 'Not Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Validation Set', fontsize=15)
plt.show()

# using gridsearch to tune parameters
logisticRegr = LogisticRegression(solver='liblinear')
parameters = {'C': [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 1], 'penalty': ('l1', 'l2')}

gridsearch = GridSearchCV(estimator=logisticRegr, param_grid=parameters, scoring='f1', refit=True)
gridsearch.fit(X_train, y_train)

# print best parameters
print("Best Parameters: ", gridsearch.best_params_)
print("Best Estimator: ", gridsearch.best_estimator_)

# obtain predictions for gridsearch validation data
gridsearch_pred = gridsearch.predict(X_validation)
print("Classification report after tuning parameters: ")
print(classification_report(y_validation, gridsearch_pred))

# obtain confusion matrix for gridsearch_pred
cm = confusion_matrix(y_validation, gridsearch_pred)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Diabetic', 'Not Diabetic'],
            yticklabels=['Diabetic', 'Not Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Validation Set with Tuned Parameters', fontsize=15)
plt.show()


# optimized results for testing data
test_predictions = gridsearch.predict(X_test)
print("Classification report for Test Set: ")
print(classification_report(y_test, test_predictions))

# obtain the ROC-AUC value of the model
fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions)
print(metrics.auc(fpr, tpr))

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Diabetic', 'Not Diabetic'],
            yticklabels=['Diabetic', 'Not Diabetic'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix for Test Set', fontsize=15)
plt.show()

# accuracy of test data
#score = logisticRegr.score(X_test, y_test)
#print(score)