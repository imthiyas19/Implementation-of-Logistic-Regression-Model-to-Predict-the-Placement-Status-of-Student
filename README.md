# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MOHAMMED IMTHIYAS.M
RegisterNumber: 212222230083
*/
```

```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:



![312763127-4c291307-624d-4758-89b0-3f62de521914](https://github.com/imthiyas19/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120353416/9893e508-392e-4d78-8e05-6ed040518412)





![312763176-c4d47cc3-b2c0-4a3d-9c90-de5779421b81](https://github.com/imthiyas19/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120353416/a5a6aa05-2cd7-4a16-ab77-9937d95a123d)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
