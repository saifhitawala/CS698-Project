import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import timeit
import matplotlib.pyplot as plt
import numpy as np

def main():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	features = train.drop('Activity', axis=1).values
	label = train.Activity.values
	clf = ExtraTreesClassifier()
	clf = clf.fit(features, label)
	model = SelectFromModel(clf, prefit = True)
	new_features = model.transform(features)

	lsvc = LinearSVC(C = 0.01, penalty = 'l1', dual = False).fit(features, label)
	model_2 = SelectFromModel(lsvc, prefit = True)
	new_features_2 = model_2.transform(features)

	classifiers = [DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 200), GradientBoostingClassifier(n_estimators = 200)]

	test_features = test.drop('Activity', axis=1).values
	time_1, model_1, out_accuracy_1 = [], [], []
	for clf in classifiers:
		start_time = timeit.default_timer()
		fit = clf.fit(features, label)
		pred = fit.predict(test_features)
		elapsed = timeit.default_timer() - start_time
		time_1.append(elapsed)
		model_1.append(clf.__class__.__name__)
		out_accuracy_1.append(accuracy_score(test.Activity.values, pred))
	'''
	test_features = model.transform(test.drop('Activity', axis=1).values)
	time_2, model_2, out_accuracy_2 = [], [], []
	for clf in classifiers:
		start_time = timeit.default_timer()
		fit = clf.fit(new_features, label)
		pred = fit.predict(test_features)
		elapsed = timeit.default_timer() - start_time
		time_2.append(elapsed)
		model_2.append(clf.__class__.__name__)
		out_accuracy_2.append(accuracy_score(test.Activity.values, pred))

	test_features = model_2.transform(test.drop('Activity', axis=1).values)
	time_3, model_3, out_accuracy_3 = [], [], []
	for clf in classifiers:
		start_time = timeit.default_timer()
		fit = clf.fit(new_features_2, label)
		pred = fit.predict(test_features)
		elapsed = timeit.default_timer() - start_time
		time_3.append(elapsed)
		model_3.append(clf.__class__.__name__)
		out_accuracy_3.append(accuracy_score(test.Activity.values, pred))
	'''
	ind = np.arange(3)
	width = 0.1
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, out_accuracy_1, width, color = 'r')
	# rects2 = ax.bar(ind + width, out_accuracy_2, width, color = 'g')
	#rects3 = ax.bar(ind + width * 2, out_accuracy_3, width, color = 'y')
	ax.set_ylabel('Accuracy')
	ax.set_title('Accuracy by Models and Selection Process')
	ax.set_xticks(ind + width)
	ax.set_xticklabels(model_3, rotation = 45)
	plt.show()

if __name__ == '__main__':
	main()