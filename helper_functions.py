import numpy as np
from os import path
from sklearn.decomposition import PCA
import pickle
import random
import torch
from matplotlib import pyplot as plt

#Helper Functions

def load_term_object(target, directory):

    with open(path.join(directory, target + '-object.pkl'), 'rb') as object_reader:
        return pickle.load(object_reader)

#Math Functions
def cosine_similarity(a, b):
    return ((np.dot(a, b)) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))))

def std_deviation(J):
    mean_J = np.mean(J)
    var_J = sum([(j - mean_J)**2 for j in J])
    return (np.sqrt(var_J / (len(J)-1)))

def create_permutation(a, b):
    permutation = random.sample(a+b, len(a+b))
    return permutation[:int(len(permutation)*.5)], permutation[int(len(permutation)*.5):]

#Returns a list of vectors from layers of a language model
def get_embeddings(term, context, model, tokenizer, layers = tuple(range(13)), tensor_type = 'tf', prev_token = False):
    
    context += ' '
    encoding = tokenizer.encode(term, add_special_tokens = False, add_prefix_space=True)
    encoded_context = tokenizer.encode(context, add_special_tokens=True)

    if prev_token:
        positions = [encoded_context.index(encoding[0])]
        if positions[0] == 0:
            context = ' ' + context
        else:
            positions[0] -= 1

    else:
        positions = []

        if len(encoding) == 1:
            positions = [encoded_context.index(encoding[0])]

        else:
            for i in range(len(encoded_context)):
                if encoding[0] == encoded_context[i] and encoding[1:] == encoded_context[i+1:i+len(encoding)]:
                    positions = [j for j in range(i, i + len(encoding))]

        if not positions:
            context = context.replace(term, ' ' + term)
            encoded_context = tokenizer.encode(context, add_special_tokens = True)

            if len(encoding) == 1:
                positions = [encoded_context.index(encoding[0])]

            else:
                for i in range(len(encoded_context)):
                    if encoding[0] == encoded_context[i] and encoding[1:] == encoded_context[i+1:i+len(encoding)]:
                        positions = [j for j in range(i, i + len(encoding))]

    inputs = tokenizer(context, return_tensors = tensor_type)

    if tensor_type == 'tf':
        output_ = model(inputs)
    if tensor_type == 'pt':
        with torch.no_grad():
            output_ = model(**inputs)

    np.squeeze(output_)

    embeddings = []

    for layer in layers:

        target_embedding = []
        for position in positions:
            sub_embedding = np.array(output_[-1][layer][0][position])
            target_embedding.append(sub_embedding)

        embeddings.append(target_embedding)

    return embeddings

def get_last_token_embeddings(context, model, tokenizer, layers = tuple(range(13)), tensor_type = 'tf'):
    inputs = tokenizer(context, return_tensors = tensor_type)

    if tensor_type == 'tf':
        output_ = model(inputs)
    if tensor_type == 'pt':
        with torch.no_grad():
            output_ = model(**inputs)

    np.squeeze(output_)

    embeddings = []

    for layer in layers:

        embeddings.append(np.array(output_[-1][layer][0][-1]))

    return embeddings

def form_representations(cwe_list, rep_type = 'Last'):

    representations = []

    if rep_type == 'Last':
        for vector in cwe_list:
            representations.append(vector[-1])
        return representations

    if rep_type == 'First':
        for vector in cwe_list:
            representations.append(vector[0])
        return representations

    if rep_type == 'Mean':
        for vector in cwe_list:
            cwe_arr = np.array([i for i in vector])
            representations.append(np.mean(cwe_arr, axis = 0))
        return representations

    if rep_type == 'Max':
        for vector in cwe_list:
            cwe_arr = np.array([i for i in vector])
            representations.append(np.nanmax(cwe_arr, axis = 0))
        return representations

    if rep_type == 'Min':
        for vector in cwe_list:
            cwe_arr = np.array([i for i in vector])
            representations.append(np.nanmin(cwe_arr, axis = 0))
        return representations

    if rep_type == 'Abs_Max':
        for vector in cwe_list:
            cwe_arr = np.array([i for i in vector])
            max_arr = np.nanmax(cwe_arr, axis = 0)
            min_arr = np.nanmin(cwe_arr, axis = 0)
            max_ind = abs(max_arr) >= abs(min_arr)
            min_ind = abs(min_arr) < abs(max_arr)
            abs_max = np.zeros(max_arr.shape)
            abs_max[max_ind] = max_arr[max_ind]
            abs_max[min_ind] = min_arr[min_ind]
            representations.append(abs_max)
        return representations

    if rep_type == 'Concat':
        for vector in cwe_list:
            concat_cwe = np.array(vector[0])
            if len(vector) > 1:
                for subword in vector[1:]:
                    concat_cwe = np.append(concat_cwe, subword)
            representations.append(concat_cwe)
        return representations
    
    return representations

def pca_transform(embedding_array, pcs, subtract_mean = True):

    pca = PCA(n_components = pcs)

    if subtract_mean:
        common_mean = np.mean(embedding_array, axis=0)
        transformed_array = embedding_array - common_mean
    else:
        transformed_array = np.array(embedding_array, copy = True)

    pcs = pca.fit_transform(transformed_array)
    pcas = pca.components_
    pc_remove = np.matmul(np.matmul(transformed_array, pcas.T), pcas)
    transformed_embeddings = transformed_array - pc_remove

    return transformed_embeddings, pcs