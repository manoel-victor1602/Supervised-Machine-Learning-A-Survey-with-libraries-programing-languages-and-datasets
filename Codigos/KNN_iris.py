from pandas import DataFrame, Series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn import neighbors
from datetime import datetime

t1 = datetime.now()

names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']

df = pd.read_csv('iris.data.txt', names=names)

features = df.columns.difference(['Class'])

x = df[features].values
y = df['Class'].values

knn = neighbors.KNeighborsClassifier(15, weights='distance')
scores = cross_val_score(knn, x, y, scoring='accuracy', cv=10)

print("%.2f" %(scores.mean()*100) + '%')

t2 = datetime.now()
print("%dms" %(float(t2.microsecond - t1.microsecond)/1000))