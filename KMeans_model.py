import numpy as np 
import pandas as pd 
from sklearn.clusters import KMeans
from sklearn.datasets import load_iris 
import joblib

iris = load_iris() 
x = iris.data[:,:2] # use first two clumns for visualisation 
Kmeans = KMeans(n_clusters = 3 , random_state = 42 ) 
kmeans.fit(x) 
y_pred = kmeans.predict(x)

print("kmeans clusters center " , Kmeans.clusters_center_)
joblib.dump(Kmeans , "kmeans_model.pk1")


#it's time to test 
model = joblib.load("kmeans.pk1")
sample = np.array([[3.0 ,3.0]])
pred = model.predict(sample)
print(pred)
