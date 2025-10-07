                                             CODE1

import numpy as np 
from sklearn import datasets,metrics 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 

iris=datasets.load_iris() 
X=iris.data 
y=iris.target 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) 

nb_classifier=GaussianNB() 

nb_classifier.fit(X_train,y_train) 

y_pred=nb_classifier.predict(X_test) 

accuracy=accuracy_score(y_test,y_pred) 
print(accuracy) 
print(f"Accuracy: {accuracy*100:.2f}%")

OUTPUT

1.0
Accuracy: 100.00%


                            CODE2

import numpy as np 
from sklearn import datasets,metrics 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report  
iris=datasets.load_iris() 
X=iris.data 
y=iris.target 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) 
nb_classifier=GaussianNB() 
nb_classifier.fit(X_train,y_train) 
y_pred=nb_classifier.predict(X_test) 
res=confusion_matrix(y_test,y_pred) 
print("confusion_matrix \n",res) 
res1=classification_report(y_test,y_pred) 
print("Classification_report \n",res1) 
accuracy=accuracy_score(y_test,y_pred) 
print("Accuracy:\n",accuracy) 

OUTPUT
confusion_matrix 
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
Classification_report 
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Accuracy:
 1.0
