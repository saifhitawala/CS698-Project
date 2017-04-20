import pandas as pd 
import numpy as np
import sys
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, neural_network as nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap = plt.cm.Blues):
	plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation = 45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
		print('Normalized confusion matrix')
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment = 'center', color = 'white' if cm[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def predictor(classifier, trainData, trainLebelE, testData, testLabelE, encoder, classifierName):
	classifier.fit(trainData, trainLebelE)
	p_te = classifier.predict_proba(testData)
	y_te_pred = classifier.predict(testData)
	acc = accuracy_score(testLabelE, y_te_pred)
	prec = precision_score(testLabelE, y_te_pred, average = 'macro')
	rec = recall_score(testLabelE, y_te_pred, average = 'macro')
	cfs = confusion_matrix(testLabelE, y_te_pred)
	print("Accuracy :", acc, "Precision :", prec, "Recall :", rec)
	plt.figure()
	class_names = encoder.classes_
	plot_confusion_matrix(cfs, classes = class_names, title = classifierName+' Confusion matrix, without normalization')

def main():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	train = shuffle(train)
	test = shuffle(test)

	trainData = train.drop('Activity', axis=1).values
	trainLabel = train.Activity.values
	testData = test.drop('Activity', axis=1).values
	testLabel = test.Activity.values

	encoder = preprocessing.LabelEncoder()
	encoder.fit(trainLabel)
	trainLabelE = encoder.transform(trainLabel)
	encoder.fit(testLabel)
	testLabelE = encoder.transform(testLabel)

	pca = PCA(n_components = 3)
	print('Initial Train Data Shape :', trainData.shape)
	trainData = pca.fit_transform(trainData)
	print('Train Data Shape :', trainData.shape)
	testData = pca.transform(testData)
	
	mlpSGD = nn.MLPClassifier(hidden_layer_sizes = (90, ), max_iter = 1000, alpha = 1e-4, solver = 'sgd', 
		verbose = 10, tol = 1e-19, random_state = 1, learning_rate_init = 0.001)
	predictor(mlpSGD, trainData, trainLabelE, testData, testLabelE, encoder, 'MLP SGD')

	mlpADAM = nn.MLPClassifier(hidden_layer_sizes = (90, ), max_iter = 1000, alpha = 1e-4, solver = 'adam', 
		verbose = 10, tol = 1e-19, random_state = 1, learning_rate_init = 0.001)
	predictor(mlpADAM, trainData, trainLabelE, testData, testLabelE, encoder, 'MLP ADAM')
	
	rfclf = RandomForestClassifier(n_estimators = 200, n_jobs = 4, min_samples_leaf = 10)
	predictor(rfclf, trainData, trainLabelE, testData, testLabelE, encoder, 'Random Forest')
	
	svmclf = OneVsRestClassifier(svm.SVC(kernel = 'linear', probability = True, random_state = 0))
	predictor(svmclf, trainData, trainLabelE, testData, testLabelE, encoder, 'SVM')
	
	adaclf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), n_estimators = 200, learning_rate = 1.5, algorithm = 'SAMME')
	predictor(adaclf, trainData, trainLabelE, testData, testLabelE, encoder, 'AdaBoost')

	knnclf = knn(n_neighbors = 20, n_jobs = 2, weights = 'distance')
	predictor(knnclf, trainData, trainLabelE, testData, testLabelE, encoder, 'KNN')	

if __name__ == '__main__':
	main()