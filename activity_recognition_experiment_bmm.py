import pandas as pd
import numpy as np
import sys
import timeit
from random import uniform
from sklearn.decomposition import PCA
from numpy.linalg import inv, pinv, eigvals

activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

def learnSourceHMM(subject_ids, train_subject_data, train_subject_label, n_class, n_components, n_features):
	m, n, d = n_components, n_class, n_features
	all_theta, all_weight, all_mu, all_sigma= [], [], [], []
	for subject_id in range(len(subject_ids)):
		subject_data = train_subject_data[subject_id]
		subject_label = train_subject_label[subject_id]
		#print('Subject Data Shape: ', np.matrix(subject_data).shape)
		#print('Subject Label Shape: ', np.matrix(subject_label).shape)
		alpha, beta, delta, kappa, W, nu = initialHyperparameters(n, m, d)
		# print('Alpha Shape:', np.matrix(alpha).shape, 'Beta Shape:', np.matrix(beta).shape, 'Delta: ', delta, 'Kappa Shape:', np.matrix(kappa).shape, 'Nu Shape:', np.matrix(nu).shape, )
		# print('Delta Size :', len(delta), 'Delta[i] Size :', len(delta[0]))
		# sys.exit()
		exp_theta, exp_theta_2, exp_weight, exp_weight_2, exp_mu, exp_mu_2, exp_sigma, exp_sigma_2 = [[] * n] * n, [[] * n] * n, [[] * m] * n, [[] * m] * n, [[] * m] * n, [[] * m] * n, [[] * m] * n, [[] * m] * n
		prev_activity = -1
		for activity_index in range(len(subject_data)):
			data_point = subject_data[activity_index]
			#print("Data point: ", data_point)
			activity = activities.index(subject_label[activity_index])
			if prev_activity != -1:
				alpha[prev_activity] = calcAlphaCap(alpha, prev_activity)
			beta[activity] = calcAlphaCap(beta, activity)
			W[activity] = calcWCap(W, kappa, delta, data_point, activity)
			delta[activity] = calcDeltaCap(delta, kappa, data_point, activity)
			kappa[activity] = calcAlphaCap(kappa, activity)
			nu[activity] = calcAlphaCap(nu, activity)
			#print('Alpha Shape :', np.matrix(alpha).shape)
			#print('Beta Shape :', np.matrix(beta).shape)
			# print('Alpha :', alpha)
			# print('Delta :', delta)
			# print('Delta Shape :', np.matrix(delta).shape)
			#print('Kappa Shape :', np.matrix(kappa).shape)
			#print('Nu Shape :', np.matrix(nu).shape)
			# print('W Shape :', np.matrix(W).shape)
			# sys.exit()
			exp_theta = calcExpTheta(exp_theta, alpha)
			exp_theta_2 = calcExpTheta2(exp_theta_2, alpha)
			exp_weight = calcExpTheta(exp_weight, beta)
			exp_weight_2 = calcExpTheta2(exp_weight_2, beta)
			exp_mu = delta
			print('Activity Index:', activity_index)
			print("W", W)
			exp_mu_2 = calcExpMu2(exp_mu_2, kappa, nu, d, W)
			exp_sigma = calcExpSigma(exp_sigma, nu, W)
			exp_sigma_2 = calcExpSigma2(exp_sigma_2, nu, W)
			#print('E[theta] shape :', np.matrix(exp_theta).shape)
			#print('E[theta2] shape :', np.matrix(exp_theta_2).shape)
			#print('E[weight] shape :', np.matrix(exp_weight).shape)
			#print('E[weight2] shape :', np.matrix(exp_weight_2).shape)
			#print('E[mu] shape :', np.matrix(exp_mu).shape)
			#print('E[mu2] shape :', np.matrix(exp_mu_2).shape)
			#print('E[sigma] shape :', np.matrix(exp_sigma).shape)
			#print('E[sigma2] shape :', np.matrix(exp_sigma_2).shape)
			alpha_new = updateAlpha(alpha, exp_theta, exp_theta_2)
			beta_new = updateAlpha(beta, exp_weight, exp_weight_2)
			delta_new = exp_mu
			W_new = updateW(W, exp_sigma_2, nu)
			kappa_new = updateKappa(kappa, nu, d, exp_mu_2, W)
			alpha, beta, delta, W, kappa, prev_activity = alpha_new, beta_new, delta_new, W_new, kappa_new, activity
		theta, weight, mu, sigma = exp_theta, exp_weight, exp_mu, exp_sigma 
		all_theta.append(theta)
		all_weight.append(weight)
		all_mu.append(mu)
		all_sigma.append(sigma)
		# print('Subject ID :', subject_ids[subject_id])
		# print('Subject Labels :', subject_label)
		# print('Alpha :', alpha, '\nBeta :', beta, '\nDelta :', delta, '\nKappa :', kappa, '\nW :', W, '\nNu :', nu, '\n')
	return all_theta, all_weight, all_mu, all_sigma

def initialHyperparameters(n, m, d):
	alpha = [[1] * n] * n
	beta = [[1] * m] * n
	delta = np.random.uniform(1, 10, d)
	kappa = uniform(1, 10)
	W = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	nu = uniform(d - 1, d + 100)
	delta = [[delta.tolist()] * m] * n
	# print('Delta :', delta)
	kappa = [[kappa] * m] * n
	W = [[W] * m] * n
	nu = [[nu] * m] * n
	return alpha, beta, delta, kappa, W, nu

def calcAlphaCap(alpha, activity):
	alpha_i = alpha[activity]
	return incrementAlpha(alpha_i)

def incrementAlpha(alpha):
	new_alpha = [x+1 for x in alpha]
	return new_alpha

def calcDeltaCap(delta, kappa, data_point, activity):
	for index in range(len(delta[activity])):
		# print('Delta[activity][index] :', delta[activity][index])
		numerator = [x + y for (x, y) in zip([i * kappa[activity][index] for i in delta[activity][index]], data_point)]
		denominator = kappa[activity][index] + 1
		delta[activity][index] = [num/denominator for num in numerator]
	return delta[activity]

def calcWCap(W, kappa, delta, data_point, activity):
	for index in range(len(W[activity])):
		diff = np.matrix(delta[activity][index]) - np.matrix(data_point)
		prod = np.dot(diff, diff.transpose())
		kappa_val = kappa[activity][index]
		prod = np.dot(kappa_val/(kappa_val+1), prod)
		W[activity][index] = W[activity][index] + prod
	return W[activity] 

def calcExpTheta(exp_theta, alpha):
	for index in range(len(exp_theta)):
		alpha_sum = sum(alpha[index])
		for val_i in range(len(alpha[index])):
			exp_theta[index].append(alpha[index][val_i]/alpha_sum)
	return exp_theta

def calcExpTheta2(exp_theta_2, alpha):
	for index in range(len(exp_theta_2)):
		alpha_sum = sum(alpha[index])
		for val_i in range(len(alpha[index])):
			exp_theta_2[index].append((alpha[index][val_i] * (1 + alpha[index][val_i]))/(alpha_sum * (1 + alpha_sum)))
	return exp_theta_2

def calcExpMu2(exp_mu_2, kappa, nu, d, W):
	for index in range(len(exp_mu_2)):
		for val_i in range(len(kappa[index])):
			kappa_val = kappa[index][val_i]
			nu_val = nu[index][val_i]
			W_val = W[index][val_i]
			numerator = kappa_val + 1
			denominator = kappa_val * (nu_val - d - 1)
			const = numerator / denominator
			W_inv = pinv(W_val)
			exp_mu_2[index].append(np.dot(const, W_inv))
	return exp_mu_2

def calcExpSigma(exp_sigma, nu, W):
	for index in range(len(exp_sigma)):
		for val_i in range(len(nu[index])):
			exp_sigma[index].append(np.dot(nu[index][val_i], W[index][val_i]))
	return exp_sigma

def calcExpSigma2(exp_sigma_2, nu, W):
	for index in range(len(exp_sigma_2)):
		for val_i in range(len(nu[index])):
			nu_val = nu[index][val_i]
			W_val = W[index][val_i]
			exp_sigma_2_val = np.zeros((len(W_val), len(W_val)))
			# print('E[sigma2] shape', exp_sigma_2_val.shape)
			# print('W', W_val.shape)
			for i in range(len(W_val)):
				for j in range(len(W_val)):
					exp_sigma_2_val[i, j] = nu_val * (W_val[i, j] * W_val[i, j] + W_val[i, i] * W_val[j, j])
			exp_sigma_2[index].append(exp_sigma_2_val)
	return exp_sigma_2

def updateAlpha(alpha, exp_theta, exp_theta_2):
	for index in range(len(alpha)):
		# print('Alpha length :', len(alpha))
		# print('E[theta] length :', len(exp_theta))
		for val_i in range(len(alpha[index])):
			# print('Alpha[index] length :', len(alpha[index]))
			# print('E[theta][index] length :', len(exp_theta[index]))
			exp_theta_val = exp_theta[index][val_i]
			exp_theta_2_val = exp_theta_2[index][val_i]
			alpha[index][val_i] = (exp_theta_val * (exp_theta_val - exp_theta_2_val)) / (exp_theta_2_val - exp_theta_val ** 2)
	return alpha

def updateW(W, exp_sigma_2, nu):
	for index in range(len(W)):
		for val_i in range(len(W[index])):
			W_val = W[index][val_i]
			exp_sigma_2_val = exp_sigma_2[index][val_i]
			nu_val = nu[index][val_i]
			for i in range(len(W_val)):
				for j in range(len(W_val[i])):
					W_val[i][j] = exp_sigma_2_val[i][j]/(nu_val*W_val[i][j])
			W[index][val_i] = W_val
	return W

def updateKappa(kappa, nu, d, exp_mu_2, W):
	for index in range(len(kappa)):
		for val_i in range(len(kappa[index])):
			kappa_val = kappa[index][val_i]
			nu_val = nu[index][val_i]
			exp_mu_2_val = exp_mu_2[index][val_i]
			W_val = W[index][val_i]
			prod = np.dot(W_val, exp_mu_2_val)
			prod = np.dot((kappa_val - d - 1), prod)
			prod = np.eye(d) - inv(prod)
			#print('Kappa[index][val_i] :', kappa_val)
			#print('Prod Shape :', prod)
			kappa[index][val_i] = getMaxEigenValue(prod)
			print('Kappa', kappa[index][val_i])
	return kappa

def getMaxEigenValue(mat):
	maxeig = -1
	maxeig = max(eigvals(mat))
	return maxeig

def predictTargetDomain(test_subject_ids, test_subject_data, test_subject_label, thetas, weights, mus, sigmas):
	K = len(thetas)
	for subject_id in range(len(test_subject_ids)):
		subject_data = test_subject_data[subject_id].tolist()
		subject_label = test_subject_label[subject_id].tolist()
		lambdaa, pi, gamma, nu = initDistributionWeights(K)
		prior_params = calcPriorParams(lambdaa, pi, gamma, nu)
		for activity_index in range(len(subject_data)):
			if activity_index == 0:
				continue
			data_point = subject_data[activity_index]
			

def initDistributionWeights(K):
	lambdaa = [1] * K
	pi = [1] * K
	gamma = [1] * K
	nu = [1] * K
	return lambdaa, pi, gamma, nu

def calcPriorParams(lambdaa, pi, gamma, nu):
	dir_lambda = calcDirichlet(lambdaa, gamma)
	dir_pi = calcDirichlet(pi, nu)
	return dir_lambda*dir_pi

def calcDirichlet(x, a):
	numerator = calcGammaFn(sum(a))
	denominator = 1
	for i in range(len(a)):
		denominator *= calcGammaFn(a[i])
		numerator *= (x[i] ** a[i])
	return numerator/denominator

def calcGammaFn(a):
	prod = 1
	for i in range(1, a):
		prod *= i
	return prod

def main():
	start_time = timeit.default_timer()
	train = pd.read_csv('train.csv')
	train.set_index(keys = ['subject'], drop = False, inplace = True)
	train_subject_ids = train['subject'].unique().tolist()
	# print('Train subject ids: ', train_subject_ids)
	train_data = train.drop('subject', axis = 1).drop('Activity', axis = 1).values
	train_label = train.Activity.values
	# print('Initial Train Data Shape :', train_data.shape)
	pca = PCA(n_components = 3)
	pca.fit(train_data)
	train_data = pca.transform(train_data)
	# print('New Train Data Shape :', train_data.shape)
	train_subject_data, train_subject_label = [], []
	for subject_id in train_subject_ids:
		subject_rows = train[train.subject == subject_id].index.tolist()
		subject_data = [train_data[i] for i in subject_rows]
		subject_label = [train_label[i] for i in subject_rows]
		# print('Train Subject Data Shape: ', np.matrix(subject_data).shape)
		train_subject_data.append(subject_data)
		train_subject_label.append(subject_label)
	n_class, n_components, n_features = 6, 4, 3
	thetas, weights, mus, sigmas = learnSourceHMM(train_subject_ids, train_subject_data, train_subject_label, n_class, n_components, n_features)
	print('All thetas: ', thetas, '\nAll weights: ', weights, '\nAll mus: ', mus, '\nAll sigmas: ', sigmas)
	print('Elapsed Time: ', timeit.default_timer() - start_time)
	sys.exit()
	test = pd.read_csv('test.csv')
	test.set_index(keys = ['subject'], drop = False, inplace = True)
	test_subject_ids = test['subject'].unique().tolist()
	test_subject, test_subject_data, test_subject_label, n_features = [], [], [], 3
	for subject_id in test_subject_ids:
		subject = test.loc[test.subject == subject_id]
		subject_data = subject.drop('subject', axis = 1).drop('Activity', axis = 1).values
		# print('Initial Subject Data Shape :', subject_data.shape)
		pca = PCA(n_components = n_features)
		# pca.fit(subject_data)
		subject_data_new = pca.fit_transform(subject_data)
		subject_label = subject.Activity.values
		test_subject.append(subject)
		test_subject_data.append(subject_data_new)
		test_subject_label.append(subject_label)
		# print('Subject ID :', subject_id, '\nSubject Data :', subject_data_new, '\nSubject Data Shape :', subject_data_new.shape, '\nSubject Label :', subject_label, '\n')
	predictions = predictTargetDomain(test_subject_ids, test_subject_data, test_subject_label, thetas, weights, mus, sigmas, c_start)

if __name__ == '__main__':
	main()