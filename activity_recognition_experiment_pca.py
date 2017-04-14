import pandas as pd 
import numpy as np
from sklearn import preprocessing, neural_network as nn
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn


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

	pca = PCA(n_components = 260)
	print('Initial Train Data Shape :', trainData.shape)
	trainData = pca.fit_transform(trainData)
	print('Train Data Shape :', trainData.shape)
	testData = pca.transform(testData)
	'''
	knnclf = knn(n_neighbors = 20, n_jobs = 2, weights = 'distance')
	knnModel = knnclf.fit(trainData, trainLabelE)

	print("Training set score for KNN :", knnModel.score(trainData, trainLabelE))
	print("Test set score for KNN :", knnModel.score(testData, testLabelE))
	'''
	mlpSGD = nn.MLPClassifier(hidden_layer_sizes = (90, ), max_iter = 1000, alpha = 1e-4, solver = 'sgd', 
		verbose = 10, tol = 1e-19, random_state = 1, learning_rate_init = 0.001)
	mlpADAM = nn.MLPClassifier(hidden_layer_sizes = (90, ), max_iter = 1000, alpha = 1e-4, solver = 'adam', 
		verbose = 10, tol = 1e-19, random_state = 1, learning_rate_init = 0.001)
	nnModelSGD = mlpSGD.fit(trainData, trainLabelE)
	nnModelADAM = mlpADAM.fit(trainData, trainLabelE)

	X1 = np.linspace(1, nnModelSGD.n_iter_, nnModelSGD.n_iter_)
	X2 = np.linspace(1, nnModelADAM.n_iter_, nnModelADAM.n_iter_)

	print("Training set score for SDG :", mlpSGD.score(trainData, trainLabelE))
	print("Test Set score for SDG :", mlpSGD.score(testData, testLabelE))
	print("Training set score for ADAM :", mlpADAM.score(trainData, trainLabelE))
	print("Test Set score for ADAM :", mlpADAM.score(testData, testLabelE))	

if __name__ == '__main__':
	main()