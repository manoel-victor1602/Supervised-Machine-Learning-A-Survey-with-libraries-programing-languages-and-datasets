from pandas import DataFrame, Series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from datetime import datetime

def trunc(score):
    aux = score * 100

    aux2 = int(aux)

    return aux2/100

iteracoes = [1,100,200,300,400,500,600,700,800,900,1000]

tempo_it = []
scores_it = []

names = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Class']

df = pd.read_csv('iris.data.csv', names=names)

features = df.columns.difference(['Class'])

x = df[features].values
y = df['Class'].values

for it in iteracoes:
    
    tempo = []
    scores = []
     
    for i in range(it):
        
        t1 = datetime.now()

        gnb = GaussianNB()

        score = cross_val_score(gnb, x, y, scoring='accuracy', cv=10)

        t2 = datetime.now()

        if(t2.second-t1.second > 0):
            tempo.append(float((t2.second-t1.second)*(10**6) + t2.microsecond-t1.microsecond)/1000)
        else:
            tempo.append(float(t2.microsecond-t1.microsecond)/1000)

        scores.append(trunc(score.mean()*100))
    
    scores_it.append(np.mean(scores))
    tempo_it.append(np.mean(tempo))

plt.plot(np.float64(iteracoes), scores_it)
print(scores_it)

plt.xlabel('Iteracoes')
plt.ylabel('Pontuacoes (%)')

plt.grid(True)

plt.savefig('pontuacao_por_N_iteracoes.png')

plt.show()

#----------------------------------------------

plt.plot(np.float64(iteracoes), tempo_it)

plt.xlabel('Iteracoes')
plt.ylabel('Tempo (ms)')

plt.grid(True)

plt.savefig('tempo_por_N_iteracoes.png')

plt.show()