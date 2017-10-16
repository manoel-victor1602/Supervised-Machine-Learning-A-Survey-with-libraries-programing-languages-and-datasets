from pandas import DataFrame, Series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from datetime import datetime

t1 = datetime.now()

names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']

df = pd.read_csv('iris.data.txt', names=names)

features = df.columns.difference(['Class'])

x = df[features].values
y = df['Class'].values

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

scores_gnb = cross_val_score(gnb, x, y, scoring='accuracy', cv=10)
scores_mnb = cross_val_score(mnb, x, y, scoring='accuracy', cv=10)
scores_bnb = cross_val_score(bnb, x, y, scoring='accuracy', cv=10)

t2 = datetime.now()

print("%.2f" %(scores_gnb.mean()*100) + '%')
print("%.2f" %(scores_mnb.mean()*100) + '%')
print("%.2f" %(scores_bnb.mean()*100) + '%')

print(t2-t1)
print("%dms" %(float(t2.microsecond - t1.microsecond)/1000))