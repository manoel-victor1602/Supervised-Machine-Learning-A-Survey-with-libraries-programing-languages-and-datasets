from sklearn import tree
from pandas import DataFrame, Series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

tempo = []
scores = []

for i in range(1000):
	t1 = datetime.now()

	names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']

	df = pd.read_csv('iris.data.txt', names=names)

	features = df.columns.difference(['Class'])

	x = df[features].values
	y = df['Class'].values

	# classifier_rf = RandomForestClassifier(random_state=1986, criterion='entropy', max_depth=None, n_estimators=50, n_jobs=-1, bootstrap=True)
	classifier_dt = tree.DecisionTreeClassifier()

	# scores = cross_val_score(classifier_rf, x, y, scoring='accuracy', cv=10)
	score = cross_val_score(classifier_dt, x, y, scoring='accuracy', cv=10)

	t2 = datetime.now()

	if(t2.second-t1.second > 0):
		tempo.append(float((t2.second-t1.second)*(10**6) + t2.microsecond-t1.microsecond)/1000)
	else:
		tempo.append(float(t2.microsecond-t1.microsecond)/1000)
		
	scores.append(score.mean()*100)

	# print("%.2f" %(scores.mean()*100) + '%')

	# print("%dms" %(float(t2.microsecond - t1.microsecond)/1000))

print('%dms' %np.mean(tempo))
print('%.2f' %np.mean(scores) + "%")