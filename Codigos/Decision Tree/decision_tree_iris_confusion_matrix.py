from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


iris = datasets.load_iris()
class_names = iris.target_names

classifier_dt = tree.DecisionTreeClassifier()

cv = cross_validation.KFold(150, n_folds=10,shuffle=True,random_state=20)

test = []
predicted = []

for train_index, test_index in cv:

	X_tr, X_tes = iris.data[train_index], iris.data[test_index]
	y_tr, y_tes = iris.target[train_index],iris.target[test_index]
	classifier_dt.fit(X_tr,y_tr)

	y_pred=classifier_dt.predict(X_tes)

	test.extend(y_tes)
	predicted.extend(y_pred)

cnf_matrix = confusion_matrix(test, predicted)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                  title='Confusion matrix, without normalization')

plt.savefig("without_normalization.png")

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                  title='Normalized confusion matrix')

plt.savefig("normalized.png")

plt.show()