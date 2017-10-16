from sklearn import tree
from pandas import DataFrame, Series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

iteracoes = [1,100,200,300,400,500,600,700,800,900,1000]
# iteracoes = [1,2,3]

tempo_it = []
scores_it = []

names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
	
df = pd.read_csv('iris.data.txt', names=names)
	
features = df.columns.difference(['Class'])
	
x = df[features].values
y = df['Class'].values	

for it in iteracoes:

	tempo = []
	scores = []

	for i in range(it):
		t1 = datetime.now()
		
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
	
	scores_it.append(np.mean(scores))
	tempo_it.append(np.mean(tempo))

plt.plot(np.float64(iteracoes),scores_it)

plt.xlabel('Iteracoes')
plt.ylabel('Pontuacoes')

plt.savefig('pontuacao_por_N_iteracoes.png')

plt.grid(True)

plt.show()

#----------------------------------------------

plt.plot(np.float64(iteracoes), tempo_it)

plt.xlabel('Iteracoes')
plt.ylabel('Tempo')

plt.savefig('tempo_por_N_iteracoes.png')

plt.grid(True)

plt.show()

#-----------------------------------------------

# print scores_it
# print tempo_it
# print iteracoes