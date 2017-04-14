import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import preprocessing, neural_network as nn
from sklearn.utils import shuffle

def main():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	train = shuffle(train)
	test = shuffle(test)

	trainData = train.drop('Activity', axis = 1).values
	trainLabel = train.Activity.values
	testData = test.drop('Activity', axis=1).values
	testLabel = test.Activity.values

	encoder = preprocessing.LabelEncoder()
	encoder.fit(testLabel)
	testLabelE = encoder.transform(testLabel)
	encoder.fit(trainLabel)
	trainLabelE = encoder.transform(trainLabel)

	mlpSGD = nn.MLPClassifier(hidden_layer_sizes = (90, ), max_iter = 1000, alpha = 1e-4, solver = 'sgd', 
		verbose = 10, tol = 1e-19, random_state = 1, learning_rate_init = 0.001)
	mlpADAM = nn.MLPClassifier(hidden_layer_sizes = (90, ), max_iter = 1000, alpha = 1e-4, solver = 'adam', 
		verbose = 10, tol = 1e-19, random_state = 1, learning_rate_init = 0.001)
	nnModelSGD = mlpSGD.fit(trainData, trainLabelE)
	nnModelADAM = mlpADAM.fit(trainData, trainLabelE)

	X1 = np.linspace(1, nnModelSGD.n_iter_, nnModelSGD.n_iter_)
	X2 = np.linspace(1, nnModelADAM.n_iter_, nnModelADAM.n_iter_)

	plt.plot(X1, nnModelSGD.loss_curve_, label='SGD Convergence')
	plt.plot(X2, nnModelADAM.loss_curve_, label='ADAM Convergence')
	plt.title('Error Convergence')
	plt.ylabel('Cost Function')
	plt.xlabel('Iterations')
	plt.legend()
	plt.show()

	print("Training set score for SDG :", mlpSGD.score(trainData, trainLabelE))
	print("Test Set score for SDG :", mlpSGD.score(testData, testLabelE))
	print("Training set score for ADAM :", mlpADAM.score(trainData, trainLabelE))
	print("Test Set score for ADAM :", mlpADAM.score(testData, testLabelE))	

if __name__ == '__main__':
	main()