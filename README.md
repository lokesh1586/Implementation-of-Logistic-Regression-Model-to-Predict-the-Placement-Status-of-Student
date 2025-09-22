# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Import the required packages and print the present data
Print the placement data and salary data.
Find the null and duplicate values.
Using logistic regression find the predicted values of accuracy , confusion matrices.
``` 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LOKESH M
RegisterNumber:212224040173
import pandas as pd
import numpy as np
print("NAME:LOKESH M")
print("REG.NO:212224040173")
# Load dataset
datapd = pd.read_csv("Placement_Data.csv")
print(datapd.head())

# Copy data
data1 = datapd.copy()

# Drop column 'sl_no' and 'salary'
data1 = data1.drop(["sl_no", "salary"], axis=1)
print(data1.head())

# Check for missing values
print(data1.isnull().sum())

# Check for duplicates
print(data1.duplicated().sum())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Split features and target
x = data1.iloc[:, :-1]
y = data1["status"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)
print("11+24=35 -> correct predictions, 5+3=8 -> incorrect predictions")
from sklearn.metrics import classification_report

classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
# Add missing feature and wrap in DataFrame with column names
# ------------------------------
# Correct Prediction Example
# ------------------------------
# Build a DataFrame with correct column names and 12 feature values
sample_input = pd.DataFrame([[
    1,   # gender
    80,  # ssc_p
    1,   # ssc_b
    90,  # hsc_p
    1,   # hsc_b
    1,   # hsc_s
    90,  # degree_p
    1,   # degree_t
    0,   # workex
    85,  # etest_p
    1,   # specialisation
    60   # mba_p
]], columns=x.columns)

print("Prediction:", lr.predict(sample_input))


*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

<img width="925" height="602" alt="Screenshot 2025-09-22 095012" src="https://github.com/user-attachments/assets/0dbbc21e-6ff2-4925-865f-986a33d7e8ef" />
<img width="1154" height="675" alt="Screenshot 2025-09-22 095024" src="https://github.com/user-attachments/assets/0507e4c9-93ba-4bd6-bdd9-b4c247d35a74" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
