import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error , r2_score, classification_report 
from sklearn.datasets import load_iris
import joblib 

iris = load_iris()
df = pd.DataFrame(iris.data ,columns = iris.feature_names)
df['species'] = iris.target

x =df.iloc[: , :-1].values
y = df['species'].values
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42 )
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
joblib.dump(knn , "Knnclassifier.pk1")
# new prediction
print(f"prediction result :" , y_pred[:10]) # the first 10 columns and rows 
print("actual results : ", y_test[:10])

# let's test it like a pro 
mse = mean_squared_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)
report = classification_report(y_test , y_pred)
print(f"mse score : {mse}")
print(f"r2 score : {r2}")
print(f'classificatoin report : {report}')
 
#save the state of the model and test it again 
load_model = joblib.load("Knnclassifier.pk1")
smaple = [[2.5,3.5,3.2,3.1]]
predict = load_model.predict (smaple)
name = iris.target_names[predict][0]
print(name)


