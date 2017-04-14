import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as knn

def main():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	train = shuffle(train)
	test = shuffle(test)

	trainData = train.drop('Activity', axis = 1).values
	trainLabel = train.Activity.values

	testData = test.drop('Activity', axis = 1).values
	testLabel = test.Activity.values

	encoder = preprocessing.LabelEncoder()
	encoder.fit(trainLabel)
	trainLabelE = encoder.transform(trainLabel)
	encoder.fit(testLabel)
	testLabelE = encoder.transform(testLabel)

	knnclf = knn(n_neighbors = 20, n_jobs = 2, weights = 'distance')
	knnModel = knnclf.fit(trainData, trainLabelE)

	print("Training set score for KNN :", knnModel.score(trainData, trainLabelE))
	print("Test set score for KNN :", knnModel.score(testData, testLabelE))

if __name__ == '__main__':
	main()