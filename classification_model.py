import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,confusion_matrix ,classification_report
from sklearn.datasets import load_iris
import joblib

 iris = load_iris()
x = iris.data[: ,:2]  #just take the first two features (colunms)
y = (iris.target == 0).astype(int) # take the 0 types (versicolar) and transform it into integer 0 or 1 

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state=42)
model = LogisticRegression()
model.fit(x_train ,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test , y_pred)
conf = confusion_matrix(y_test , y_pred)
classfication = classification_report(y_test , y_pred)
joblib.dump(model , "logestic_regression.pk1")

print("accuracy score : ",acc*100)
print(f"confuson_matrix : {conf}")
print(f"classification : {classfication}")

#just to save the model and make new predictions if u don't want to do so simply commit it it doesn't effect above code 
load_model = joblib.load("logestic_regression.pk1")
smaple = [[2.5,3.5]]
pred = load_model.predict(smaple)
name = iris.target_names[pred][0]
print(name)
