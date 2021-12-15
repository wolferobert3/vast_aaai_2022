import numpy as np
from helper_functions import cosine_similarity, std_deviation, create_permutation
from scipy.stats import norm
import random

def permutation_test(A, B, X, Y, test_stat, permutations):

    distribution = []

    for _ in range(permutations):
        j, k = create_permutation(X, Y)
        m = differential_association(A, B, j, k)
        distribution.append(m)
    
    dist_mean = np.mean(distribution)
    dist_dev = std_deviation(distribution)

    p_value = (1 - norm.cdf(test_stat, dist_mean, dist_dev))

    return p_value

def differential_association(A, B, X, Y):
    sigma_x, sigma_y = 0.0, 0.0

    for x in X:
        x_norm = np.sqrt(np.dot(x, x))
        mean_A_x = np.mean([np.dot(x, a) / (x_norm * np.sqrt(np.dot(a, a))) for a in A])
        mean_B_x = np.mean([np.dot(x, b) / (x_norm * np.sqrt(np.dot(b, b))) for b in B])
        sigma_x += mean_A_x - mean_B_x

    for y in Y:
        y_norm = np.sqrt(np.dot(y, y))
        mean_A_y = np.mean([np.dot(y, a) / (y_norm * np.sqrt(np.dot(a, a))) for a in A])
        mean_B_y = np.mean([np.dot(y, b) / (y_norm * np.sqrt(np.dot(b, b))) for b in B])
        sigma_y += mean_A_y - mean_B_y

    return sigma_x - sigma_y

def weat_effect_size(A, B, X, Y):
    distribution_X = []
    distribution_Y = []
    
    for x in X:
        mean_A = np.mean([cosine_similarity(a, x) for a in A])
        mean_B = np.mean([cosine_similarity(b, x) for b in B])
        distribution_X.append(mean_A - mean_B)

    for y in Y:
        mean_A = np.mean([cosine_similarity(a, y) for a in A])
        mean_B = np.mean([cosine_similarity(b, y) for b in B])
        distribution_Y.append(mean_A - mean_B)

    return (np.mean(distribution_X) - np.mean(distribution_Y)) / std_deviation(distribution_X + distribution_Y), std_deviation(distribution_X + distribution_Y)

#def WEAT(Attr_A, Attr_B, Target_X, Target_Y, permutations):
def WEAT(Attr_A, Attr_B, Target_X, Target_Y):

    #test_statistic = differential_association(Attr_A, Attr_B, Target_X, Target_Y)
    #p_value = permutation_test(Attr_A, Attr_B, Target_X, Target_Y, test_statistic, permutations)
    effect_size, variance = weat_effect_size(Attr_A, Attr_B, Target_X, Target_Y)

    return effect_size, variance#, p_value

def SV_WEAT_association(w, A, B):

    w_norm = np.sqrt(np.dot(w, w))
    mean_A_w = np.mean([np.dot(w, a) / (w_norm * np.sqrt(np.dot(a, a))) for a in A])
    mean_B_w = np.mean([np.dot(w, b) / (w_norm * np.sqrt(np.dot(b, b))) for b in B])
    
    return mean_A_w - mean_B_w

def SV_WEAT_effect_size(w, A, B):
    
    w_norm = np.sqrt(np.dot(w, w))

    distribution_a = [(np.dot(a, w) / (w_norm * np.sqrt(np.dot(a, a)))) for a in A]
    distribution_b = [(np.dot(b, w) / (w_norm * np.sqrt(np.dot(b, b)))) for b in B]    
    joint_distribution = distribution_a + distribution_b
    
    return ((np.mean(distribution_a) - np.mean(distribution_b)) / std_deviation(joint_distribution)), std_deviation(joint_distribution)

def SV_WEAT_permutation_test(w, A, B, test_stat, permutations):

    distribution = []

    for _ in range(permutations):
        j, k = create_permutation(A, B)
        m = SV_WEAT_association(w, j, k)
        distribution.append(m)
    
    dist_mean = np.mean(distribution)
    dist_dev = std_deviation(distribution)

    p_value = (1 - norm.cdf(test_stat, dist_mean, dist_dev))

    return p_value

#def SV_WEAT(target_w, Attr_A, Attr_B, permutations):
def SV_WEAT(target_w, Attr_A, Attr_B):

    effect_size, variance = SV_WEAT_effect_size(target_w, Attr_A, Attr_B)
    #test_statistic = SV_WEAT_association(target_w, Attr_A, Attr_B)
    #p_value = SV_WEAT_permutation_test(target_w, Attr_A, Attr_B, test_statistic, permutations)

    return effect_size, variance#, p_value