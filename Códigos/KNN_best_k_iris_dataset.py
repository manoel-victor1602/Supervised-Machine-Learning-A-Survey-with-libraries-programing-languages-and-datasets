from pandas import DataFrame, Series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn import neighbors

names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']

df = pd.read_csv('iris.data.txt', names=names)

features = df.columns.difference(['Class'])

x = df[features].values
y = df['Class'].values

scores = []

k = [i for i in range(1,50,2)]

best_k = 0
best_score = 0

for i in k:
	knn = neighbors.KNeighborsClassifier(i, weights='distance')
	score = cross_val_score(knn, x, y, scoring='accuracy', cv=10)
	scores.append(score.mean()*100)
	
	if(best_score < score.mean()):
		best_score = score.mean()
		best_k = i


plt.plot(k,scores)

plt.grid(True)

plt.show()

print(best_k)
print("%.2f" %(best_score*100) + '%')