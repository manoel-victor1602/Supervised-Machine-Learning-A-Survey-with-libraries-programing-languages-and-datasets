from pandas import DataFrame, Series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from datetime import datetime

tempo = []
scores = []

for i in range(1000):

	t1 = datetime.now()

	names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']

	df = pd.read_csv('iris.data.txt', names=names)

	features = df.columns.difference(['Class'])

	x = df[features].values
	y = df['Class'].values

	gnb = GaussianNB()

	score = cross_val_score(gnb, x, y, scoring='accuracy', cv=10)

	t2 = datetime.now()

	if(t2.second-t1.second > 0):
		tempo.append(float((t2.second-t1.second)*(10**6) + t2.microsecond-t1.microsecond)/1000)
	else:
		tempo.append(float(t2.microsecond-t1.microsecond)/1000)

	scores.append(score.mean()*100)

print('%dms' %np.mean(tempo))
print('%.2f' %np.mean(scores) + "%")