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
		alpha, beta, delta, kappa, W, nu = initialHyperparameters(n, m, d)
		exp_theta, exp_theta_2, exp_weight, exp_weight_2, exp_mu, exp_mu_2, exp_sigma, exp_sigma_2 = [[0] * n] * n, [[0] * n] * n, [[0] * m] * n, [[0] * m] * n, [[0] * m] * n, [[0] * m] * n, [[0] * m] * n, [[0] * m] * n
		print('Subject ID: ', subject_id)
		prev_activity = -1
		for activity_index in range(len(subject_data)):
			data_point = subject_data[activity_index]
			activity = activities.index(subject_label[activity_index])
			if prev_activity != -1:
				alpha[prev_activity] = calcAlphaCap(alpha, prev_activity)
			beta[activity] = calcAlphaCap(beta, activity)
			W[activity] = calcWCap(W, kappa, delta, data_point, activity)
			delta[activity] = calcDeltaCap(delta, kappa, data_point, activity)
			kappa[activity] = calcAlphaCap(kappa, activity)
			nu[activity] = calcAlphaCap(nu, activity)
			exp_theta = calcExpTheta(exp_theta, alpha)
			exp_theta_2 = calcExpTheta2(exp_theta_2, alpha)
			exp_weight = calcExpTheta(exp_weight, beta)
			exp_weight_2 = calcExpTheta2(exp_weight_2, beta)
			exp_mu = delta
			exp_mu_2 = calcExpMu2(exp_mu_2, kappa, nu, d, W)
			exp_sigma = calcExpSigma(exp_sigma, nu, W)
			exp_sigma_2 = calcExpSigma2(exp_sigma_2, nu, W)
			alpha_new = updateAlpha(alpha, exp_theta, exp_theta_2)
			beta_new = updateAlpha(beta, exp_weight, exp_weight_2)
			delta_new = exp_mu
			W_new = updateW(W, exp_sigma_2, exp_sigma, nu)
			kappa_new = updateKappa(kappa, nu, d, exp_mu_2, W)
			nu_new = updateNu(nu, W, exp_sigma)
			exp_sigma = convertSigmas(exp_sigma)
			alpha, beta, delta, W, kappa, prev_activity = alpha_new, beta_new, delta_new, W_new, kappa_new, activity
		theta, weight, mu, sigma = exp_theta, exp_weight, exp_mu, exp_sigma 
		all_theta.append(theta)
		all_weight.append(weight)
		all_mu.append(mu)
		all_sigma.append(sigma)
	return all_theta, all_weight, all_mu, all_sigma

def initialHyperparameters(n, m, d):
	alpha = [[1] * n] * n
	beta = [[1] * m] * n
	delta = np.random.uniform(1, 10, d)
	kappa = uniform(1, 10)
	W = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	nu = uniform(d - 1, d + 100)
	delta = [[delta.tolist()] * m] * n
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
	for index in range(len(alpha)):
		alpha_sum = sum(alpha[index])
		for val_i in range(len(alpha[0])):
			exp_theta[index][val_i] = (alpha[index][val_i]/alpha_sum)
	return exp_theta

def calcExpTheta2(exp_theta_2, alpha):
	for index in range(len(exp_theta_2)):
		alpha_sum = sum(alpha[index])
		for val_i in range(len(alpha[index])):
			exp_theta_2[index][val_i] = ((alpha[index][val_i] * (1 + alpha[index][val_i]))/(alpha_sum * (1 + alpha_sum)))
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
			exp_mu_2[index][val_i] = (np.dot(const, W_inv))
	return exp_mu_2

def calcExpSigma(exp_sigma, nu, W):
	for index in range(len(exp_sigma)):
		for val_i in range(len(nu[index])):
			exp_sigma[index][val_i] = (np.dot(nu[index][val_i], W[index][val_i]))
	return exp_sigma

def calcExpSigma2(exp_sigma_2, nu, W):
	for index in range(len(exp_sigma_2)):
		for val_i in range(len(nu[index])):
			nu_val = nu[index][val_i]
			W_val = W[index][val_i]
			exp_sigma_2_val = np.zeros((len(W_val), len(W_val)))
			for i in range(len(W_val)):
				for j in range(len(W_val)):
					exp_sigma_2_val[i, j] = nu_val * (W_val[i, j] * W_val[i, j] + W_val[i, i] * W_val[j, j])
			exp_sigma_2[index][val_i] = exp_sigma_2_val
	return exp_sigma_2

def convertSigmas(exp_sigma):
	for index in range(len(exp_sigma)):
		for val_i in range(len(exp_sigma[index])):
			if type(exp_sigma[index][val_i]) == list:
				continue
			exp_sigma[index][val_i] = exp_sigma[index][val_i].tolist()
	return exp_sigma

def updateAlpha(alpha, exp_theta, exp_theta_2):
	for index in range(len(alpha)):
		for val_i in range(len(alpha[index])):
			exp_theta_val = exp_theta[index][val_i]
			exp_theta_2_val = exp_theta_2[index][val_i]
			alpha[index][val_i] = (exp_theta_val * (exp_theta_val - exp_theta_2_val)) / (exp_theta_2_val - exp_theta_val ** 2)
	return alpha

def updateW(W, exp_sigma_2, exp_sigma, nu):
	for index in range(len(W)):
		for val_i in range(len(W[index])):
			W_val = W[index][val_i]
			exp_sigma_val = exp_sigma[index][val_i]
			exp_sigma_2_val = exp_sigma_2[index][val_i]
			nu_val = nu[index][val_i]
			for i in range(len(W_val)):
				for j in range(len(W_val[i])):
					W_val[i][j] = exp_sigma_2_val[i][j]/(exp_sigma_val[i][j])
			W[index][val_i] = W_val
	return W

def updateNu(nu, W, exp_sigma):
	for index in range(len(nu)):
		for val_i in range(len(nu[index])):
			nu_val = nu[index][val_i]
			W_val = W[index][val_i]
			exp_sigma_val = exp_sigma[index][val_i]
			prod = np.dot(exp_sigma_val, inv(W_val))
			nu[index][val_i] = getMaxNu(prod)
	return nu

def getMaxNu(mat):
	maxnu = -1
	for i in range(len(mat)):
		for j in range(len(mat[0])):
			if maxnu < mat[i, j]:
				maxnu = mat[i, j]
	return maxnu
 
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
			kappa[index][val_i] = getMaxEigenValue(prod)
	return kappa

def getMaxEigenValue(mat):
	maxeig = -1
	maxeig = max(eigvals(mat))
	return maxeig

def predictTargetDomain(test_subject_ids, test_subject_data, test_subject_label, thetas, weights, mus, sigmas):
	K, N, M = len(thetas), len(thetas[0]), len(weights[0][0])
	accuracies = []
	for subject_id in range(len(test_subject_ids)):
		print('Test Subject ID: ', subject_id)
		subject_data = test_subject_data[subject_id]
		subject_label = test_subject_label[subject_id]
		lambdaa, pi, gamma, nu = initDistributionWeights(K)
		exp_lambda, exp_lambda2, exp_pi, exp_pi2, prev_activity, correct_pred = [1]*K, [1]*K, [1]*K, [1]*K, -1, 1
		for activity_index in range(len(subject_data)):
			curr_activity = activities.index(subject_label[activity_index])
			if activity_index == 0:
				prev_activity = curr_activity
				continue
			data_point = subject_data[activity_index]
			prob, prob_index, probs = -1, -1, []
			for j in range(N):
				c_ijkm_sum = 0
				for k in range(K):
					nu_k = calcNuK(nu, k)
					phi_k = calcEmissionDist(weights, mus, sigmas, data_point, j, k, M)
					nu = updateGammaCap(nu, k)
					gamma_m = calcGammaK(gamma, k)
					gamma = updateGammaCap(gamma, k)
					for i in range(N):
						theta_mij = thetas[k][i][j]
						c_ijkm = nu_k * gamma_m * phi_k * theta_mij
						c_ijkm_sum += c_ijkm
				exp_lambda = calcExpLambda(lambdaa, c_ijkm_sum, exp_lambda)
				exp_lambda2 = calcExpLambda2(lambdaa, c_ijkm_sum, exp_lambda2)
				lambdaa = exp_lambda
				exp_pi = calcExpLambda(pi, c_ijkm_sum, exp_pi)
				exp_pi2 = calcExpLambda2(pi, c_ijkm_sum, exp_pi2)
				pi = exp_pi
				prob_j = calcProb(lambdaa, pi, thetas, weights, mus, sigmas, data_point, K, M, N, j)
				probs.append(prob_j)
			prev_activity = curr_activity
			probs = normalize(probs)
			prob = max(probs)
			prob_index = probs.index(prob)
			print('prob_index: ', prob_index, 'prob: ', prob )
			if prob_index == curr_activity:
				correct_pred += 1
			print('Correct Prediction Count: ', correct_pred)
		acc_k = correct_pred/len(subject_data)
		print('Accuracy for ', subject_id, 'th Test Subject: ', acc_k)
		accuracies.append(acc_k)
	return accuracies
						
def initDistributionWeights(K):
	lambdaa = [1] * K
	pi = [1] * K
	gamma = [1] * K
	nu = [1] * K
	return lambdaa, pi, gamma, nu

def normalize(arr):
	arr_sum = sum(arr)
	for i in range(len(arr)):
		arr[i] = arr[i]/arr_sum
	return arr

def calcNuK(nu, k):
	nu_sum = sum(nu)
	nu_k = nu[k]/nu_sum
	return nu_k

def calcGammaK(gamma, m):
	gamma_sum = sum(gamma)
	gamma_m = gamma[m]/gamma_sum
	return gamma_m

def calcEmissionDist(weights, mus, sigmas, data_point, j, k, M):
	final_model = 0
	for u in range(M):
		w_kju = weights[k][j][u]
		mu_kju = mus[k][j][u]
		sigma_kju = sigmas[k][j][u]
		model_kju = calcGaussianDist(mu_kju, sigma_kju, data_point)
		model_kju = w_kju * model_kju
		final_model += model_kju
	return final_model

def calcGaussianDist(mu_kju, sigma_kju, data_point):
	diff = mu_kju - data_point
	first_term = np.dot(diff.transpose(), sigma_kju)
	second_term = np.dot(first_term, diff)
	final_term = np.exp(-second_term/2)
	return final_term

def updateGammaCap(gamma, m):
	gamma[m] += 1
	return gamma

def calcExpLambda(lambdaa, c_sum, exp_lambda):
	lambdaa_sum = sum(lambdaa) + len(lambdaa)
	for i in range(len(lambdaa)):
		exp_lambda[i]  = ((lambdaa[i]+1)*c_sum)/lambdaa_sum
	return exp_lambda

def calcExpLambda2(lambdaa, c_sum, exp_lambda2):
	lambdaa_sum = sum(lambdaa) + len(lambdaa)
	for i in range(len(lambdaa)):
		exp_lambda2[i]  = (lambdaa[i]*(lambdaa[i]+1)*c_sum)/(lambdaa_sum*(1 + lambdaa_sum))
	return exp_lambda2

def updateGamma(gamma, exp_lambda, exp_lambda2):
	for i in range(len(exp_lambda)):
		gamma[i] = (exp_lambda[i]*(exp_lambda[i] - exp_lambda2[i]))/(exp_lambda2[i] - exp_lambda[i] * exp_lambda[i])
	return gamma

def calcProb(lambdaa, pi, thetas, weights, mus, sigmas, data_point, K, M, N, j):
	prob = -1
	for k in range(K):
		lambda_k = lambdaa[k]
		pi_k = pi[k]
		phi_k = calcEmissionDist(weights, mus, sigmas, data_point, j, k, M)
		for i in range(N):
			theta_ki = thetas[k][i][j]
			prob_val = lambda_k * phi_k * pi_k * theta_ki
			prob += prob_val
	return prob

def main():
	start_time = timeit.default_timer()
	train = pd.read_csv('train.csv')
	train.set_index(keys = ['subject'], drop = False, inplace = True)
	train_subject_ids = train['subject'].unique().tolist()
	train_data = train.drop('subject', axis = 1).drop('Activity', axis = 1).values
	train_label = train.Activity.values
	pca = PCA(n_components = 3)
	pca.fit(train_data)
	train_data = pca.transform(train_data)
	train_subject_data, train_subject_label = [], []
	for subject_id in train_subject_ids:
		subject_rows = train[train.subject == subject_id].index.tolist()
		subject_data = [train_data[i] for i in subject_rows]
		subject_label = [train_label[i] for i in subject_rows]
		train_subject_data.append(subject_data)
		train_subject_label.append(subject_label)
	n_class, n_components, n_features = 6, 4, 3
	thetas, weights, mus, sigmas = learnSourceHMM(train_subject_ids, train_subject_data, train_subject_label, n_class, n_components, n_features)
	print('Elapsed Time: ', timeit.default_timer() - start_time)
	test = pd.read_csv('test.csv')
	test.set_index(keys = ['subject'], drop = False, inplace = True)
	test_subject_ids = test['subject'].unique().tolist()
	test_data = test.drop('subject', axis = 1).drop('Activity', axis = 1).values
	test_label = test.Activity.values
	pca = PCA(n_components = 3)
	pca.fit(test_data)
	test_data = pca.transform(test_data)
	test_subject_data, test_subject_label = [], []
	for subject_id in test_subject_ids:
		subject_rows = test[test.subject == subject_id].index.tolist()
		subject_data = [test_data[i] for i in subject_rows]
		subject_label = [test_label[i] for i in subject_rows]
		test_subject_data.append(subject_data)
		test_subject_label.append(subject_label)
	accuracies = predictTargetDomain(test_subject_ids, test_subject_data, test_subject_label, thetas, weights, mus, sigmas)
	print('Elapsed Time: ', timeit.default_timer() - start_time) 

if __name__ == '__main__':
	main()