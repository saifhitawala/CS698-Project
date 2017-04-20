'''
Author: Saifuddin Hitawala
Date: 6th April, 2017
'''

import csv
import os
import sys
import random
import math
import operator
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numpy.linalg import inv

# Activity Classes to be predicted
activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

# Calculates the number of data points corresponding to a particular class
def calculateCstart(train_subject_label, n_class):
    c_start = [0]*n_class
    for subject_id in range(len(train_subject_label)):
        subject_labels = train_subject_label[subject_id]
        c_start[activities.index(subject_labels[0])] += 1
    return c_start

# Calculates the prior probabilities
def calculatePriorProbs(c_start, K):
    pi = []
    for i in c_start:
        pi.append(i/K)
    return pi

# Calculates the sequence counts of classes in the training set
def calculateSeq(train_subject_label, n_class):
    cicj = np.zeros((n_class, n_class))
    for subject_id in range(len(train_subject_label)):
        subject_label = train_subject_label[subject_id]
        for iterator in range(len(subject_label) - 1):
            #print('Label i: ', subject_label[iterator], 'Label i+1: ', subject_label[iterator + 1])
            current_label = activities.index(subject_label[iterator])
            next_label = activities.index(subject_label[iterator + 1])
            cicj[next_label, current_label] += 1
    return cicj

def calculateTheta(cicj, n_class):
    thetas = np.zeros((n_class, n_class))
    column_sums = [0]*n_class
    for i in range(n_class):
        for j in range(n_class):
            column_sums[i] += cicj[j, i]
    for i in range(n_class):
        for j in range(n_class):
            thetas[i, j] = cicj[i, j]/column_sums[i]
    return thetas

# Calculates mu
def calculateMus(train_subject_label, train_subject_data, n_class, n_features):
    mus = [0]*n_class
    freq = [0]*n_class
    for i in range(n_class):
        mus[i] = np.zeros((n_features, 1))
    for subject_id in range(len(train_subject_data)):
        subject_data = train_subject_data[subject_id]
        subject_label = train_subject_label[subject_id]
        for data_i in range(len(subject_data)):
            data_point = subject_data[data_i]
            data_label = activities.index(subject_label[data_i])
            data_point_trans = np.matrix(data_point).transpose()
            mus[data_label] += data_point_trans
            freq[data_label] += 1
    for i in range(n_class):
        mus[i] = mus[i]/freq[i]
    return mus

# Calculate all sigmas
def calculateSigmas(train_subject_label, train_subject_data, mus, n_class, n_features):
    sigmas = [0]*n_class
    for i in range(n_class):
        sigmas[i] = np.zeros((n_features, n_features))
    for subject_id in range(len(train_subject_data)):
        subject_data = train_subject_data[subject_id]
        subject_label = train_subject_label[subject_id]
        for data_i in range(len(subject_data)):
            data_point = subject_data[data_i]
            data_label = activities.index(subject_label[data_i])
            data_point_trans = np.matrix(data_point).transpose()
            temp = data_point_trans - mus[data_label]
            temp = np.dot(temp, temp.transpose())
            #print('Temp Shape: ', temp.shape, '\nData Point Shape: ', data_point_trans.shape)
            sigmas[data_label] += temp
    return sigmas

# Calculate Sigma
def calculateSigma(sigmas, train_subject_data, n_features):
    sigma = np.zeros((n_features, n_features))
    for i in range(len(sigmas)):
        sigma += sigmas[i]
    train_len = 0
    for subject_data in train_subject_data:
        train_len += len(subject_data)
    sigma = sigma/train_len
    return sigma

# Viterbi algorithm for calculating the probabilities
def viterbi(sigma_inv, mus, thetas, pi, test_data, n_class):
    probCi = [0]*n_class
    predicted_class = []
    classCi = [0]*n_class
    for i in range(n_class):
        probCi[i] = []
        classCi[i] = []
    initial_data = test_data[0]
    prob_x1_y1 = calculateXGivenY(initial_data, mus, sigma_inv)
    prob_y1_x1 = []
    prob_c = [0]*n_class
    for i in range(n_class):
        max_prob, max_prob_index = -1, -1
        for j in range(n_class):
            temp = thetas[i, j]*prob_x1_y1[j]*pi[j]
            if max_prob < temp:
                max_prob = temp
                max_prob_index = j
        prob_c[i] = max_prob
        classCi[i].append(max_prob_index)
    sum_of_probs_y1 = sum(prob_c)
    for i in range(n_class):
        probCi[i].append(prob_c[i]/sum_of_probs_y1)
        prob_y1_x1.append(probCi[i][-1])
    prob_yi_xi_1 = prob_y1_x1
    for iterator in range(len(test_data) - 1):
        if iterator == 0:
            continue
        data_point = test_data[iterator]
        prob_xi_yi = calculateXGivenY(data_point, mus, sigma_inv)
        prob_yi_ci = [0]*n_class
        for i in range(n_class):
            prob_yi_ci[i] = prob_xi_yi[i]*prob_yi_xi_1[i]
        prob_c = [0]*n_class
        for i in range(n_class):
            max_prob, max_prob_index = -1, -1
            for j in range(n_class):
                temp = thetas[i, j]*prob_yi_ci[j]
                if max_prob < temp:
                    max_prob = temp
                    max_prob_index = j
            prob_c[i] = max_prob
            classCi[i].append(max_prob_index)
        sum_of_probs = sum(prob_c)
        prob_yi_xi_1 = []
        for i in range(n_class):
            probCi[i].append(prob_c[i]/sum_of_probs)
            prob_yi_xi_1.append(probCi[i][-1])
    data_point = test_data[-1]
    prob_xi_yi = calculateXGivenY(data_point, mus, sigma_inv)
    prob_yn_ci = [0]*n_class
    for i in range(n_class):
        prob_yn_ci[i] = prob_xi_yi[i]*prob_yi_xi_1[i]
    prob_yn = max(prob_yn_ci)
    for i in range(n_class):
        if prob_yn == prob_yn_ci[i]:
            predicted_class.append(i)
    for iterator in range(len(test_data)):
        next_class = predicted_class[-1]
        for i in range(n_class):
            if next_class == i:
                predicted_class.append(classCi[i][len(test_data) - iterator - 2])
    return predicted_class

# Returns the probability of x given class for all classes
def calculateXGivenY(test_data, mus, sigma_inv):
    prob_x1_y1, sum_of_probs_x1 = [], 0
    for mu in mus:
        prob_x1_y1_ci = calculateXGivenC(test_data, mu, sigma_inv)
        sum_of_probs_x1 += prob_x1_y1_ci
        prob_x1_y1.append(prob_x1_y1_ci)
    for i in range(len(mus)):
        prob_x1_y1[i] = prob_x1_y1[i]/sum_of_probs_x1
    return prob_x1_y1

# Calculates the probability of data given class
def calculateXGivenC(test_data, mu, sigma_inv):
    first_term = np.matrix(test_data).transpose() - mu
    product = np.dot(first_term.transpose(), sigma_inv)
    product = np.dot(product, first_term)
    return math.exp(-0.5*product)

# Returns the count of correctly identified labels
def getCorrectCount(predicted_class, test_label):
    correct = 0
    for instance in range(len(test_label)):
        if activities.index(test_label[instance]) == predicted_class[len(test_label) - instance - 1]:
            correct += 1
    return correct

# Main function performing actual linear regression calling all other helper functions
def main():
    start_time = timeit.default_timer()

    # Train and Test data obtained from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    train = pd.read_csv('train.csv')
    train.set_index(keys = ['subject'], drop = False, inplace = True)
    train_subject_ids = train['subject'].unique().tolist()
    train_data = train.drop('subject', axis = 1).drop('Activity', axis = 1).values
    train_label = train.Activity.values

    # PCA applied to train data reducing the number of features (dimensionality redcution) from 561 to 3
    pca = PCA(n_components = 3)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    train_subject_data, train_subject_label = [], []
    for subject_id in train_subject_ids:
        train_list = train.as_matrix().tolist()
        train_data_list = train_data.tolist()
        subject_data, subject_label = [], []
        for data_i in range(len(train_list)):
            if train_list[data_i][-2] == subject_id:
                subject_data.append(train_data_list[data_i])
                subject_label.append(train_list[data_i][-1])
        train_subject_data.append(subject_data)
        train_subject_label.append(subject_label)

    n_class, n_features, K = 6, 3, len(train_subject_ids)

    test = pd.read_csv('test.csv')
    test.set_index(keys = ['subject'], drop = False, inplace = True)
    test_subject_ids = test['subject'].unique().tolist()
    test_data = test.drop('subject', axis = 1).drop('Activity', axis = 1).values
    test_label = test.Activity.values

    # PCA applied to test data reducing the number of features (dimensionality reduction) from 561 to 3
    pca = PCA(n_components = 3)
    pca.fit(test_data)
    test_data = pca.transform(test_data)
    test_subject_data, test_subject_label = [], []

    for subject_id in test_subject_ids:
        test_list = test.as_matrix().tolist()
        test_data_list = test_data.tolist()
        subject_data, subject_label = [], []
        for data_i in range(len(test_list)):
            if test_list[data_i][-2] == subject_id:
                subject_data.append(test_data_list[data_i])
                subject_label.append(test_list[data_i][-1])
        test_subject_data.append(subject_data)
        test_subject_label.append(subject_label)

    # Initial Distributions
    c_start = calculateCstart(train_subject_label, n_class)
    print("\nc_start: ", c_start)
    pis = calculatePriorProbs(c_start, K)

    # Transition Diostributions
    cicj = calculateSeq(train_subject_label, n_class)
    thetas = calculateTheta(cicj, n_class)

    # Emission Distributions
    mus = calculateMus(train_subject_label, train_subject_data, n_class, n_features)
    sigmas = calculateSigmas(train_subject_label, train_subject_data, mus, n_class, n_features)
    sigma = calculateSigma(sigmas, train_subject_data, n_features)
    print('Pi: ', pis, '\ncicj: ', cicj, '\nTheta: ', thetas, '\nMus: ', mus, '\nSigma: ', sigma)
    print('Elapsed Time: ', timeit.default_timer() - start_time)

    # From here onwards class is being predicted and parameter estimation has been completed

    # Calculating the probability using Viterbi Algorithm
    sigma_inv = inv(sigma)
    accuracies = []

    for subject_id in range(len(test_subject_ids)):
        subject_data = test_subject_data[subject_id]
        subject_label = test_subject_label[subject_id]
        predicted_class = viterbi(sigma_inv, mus, thetas, pis, subject_data, n_class)
        correct_count = getCorrectCount(predicted_class, subject_label)
        accuracy = correct_count/len(subject_data)
        print('Test Subject ID: ', test_subject_ids[subject_id], 'has accuracy: ', accuracy)
        accuracies.append(accuracy)
    print("\nAccuracies for all subjects: "+str(accuracies))
    accuracy = sum(accuracies)/len(test_subject_ids)
    print("\nAccuracy using Viterbi Algorithm for Hidden Markov Models is: "+str(accuracy))

if __name__ == '__main__':
    main()