from typing import Counter
import numpy as np
import pandas as pd
import pickle
import copy
from helper_functions import pca_transform
from nltk import word_tokenize, pos_tag
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import sklearn
from string import punctuation
import seaborn as sns
import nltk

nltk.download('universal_tagset')

BIN = True
TEST = 'POS'

with open(f'D:\\cola_test\\prev_word_vectors.pkl', 'rb') as pkl_reader:
    cola = pickle.load(pkl_reader)

with open(f'D:\\cola_test\\prev_word_vectors_test.pkl', 'rb') as pkl_reader:
    cola_test = pickle.load(pkl_reader)

k = pd.read_csv(f'\\CoLA\\in_domain_train.tsv',sep='\t',header=None)
test_df = pd.read_csv(f'\\CoLA\\out_of_domain_dev.tsv',sep='\t',header=None)

sentences = k[3].to_list()
sentences = [i.strip() for i in sentences]
new_sentences = [i.rstrip(punctuation) for i in sentences]

last_pos = [pos_tag(word_tokenize(sentence),tagset='universal')[-1][-1] for sentence in new_sentences]
pos_types = list(set(last_pos))
pos_dict = {pos_types[idx]: idx for idx in range(len(pos_types))}

test_sentences = test_df[3].to_list()
test_sentences = [i.strip() for i in test_sentences]
test_new_sentences = [i.rstrip(punctuation) for i in test_sentences]

test_last_pos = [pos_tag(word_tokenize(sentence),tagset='universal')[-1][-1] for sentence in test_new_sentences]
test_pos_types = list(set(test_last_pos))
test_pos_dict = {test_pos_types[idx]: idx for idx in range(len(test_pos_types))}

ids = k.index.to_list()
labels = k[1].to_list()
cola_labels = k[1].to_list()

test_ids = test_df.index.to_list()
test_labels = test_df[1].to_list()
test_cola_labels = test_df[1].to_list()

if TEST == 'POS':
    labels = [pos_dict[pos] for pos in last_pos]
    label_dict = {ids[idx]:labels[idx] for idx in range(len(ids))}
    labels = np.array(labels)

    test_labels = [test_pos_dict[pos] for pos in test_last_pos]
    test_label_dict = {test_ids[idx]:test_labels[idx] for idx in range(len(test_ids))}
    test_labels = np.array(test_labels)

    only_acceptable = [idx for idx in range(len(cola_labels)) if cola_labels[idx] == 1]
    labels_ = [labels[idx] for idx in only_acceptable]
    labels = [last_pos[idx] for idx in only_acceptable]

    test_only_acceptable = [idx for idx in range(len(test_cola_labels)) if test_cola_labels[idx] == 1]
    test_labels_ = [test_labels[idx] for idx in test_only_acceptable]
    test_labels = [test_last_pos[idx] for idx in test_only_acceptable]

    if BIN == True:
        labels = [1 if labels_[idx] == 'NOUN' else 0 for idx in range(len(labels_))]
        test_labels = [1 if test_labels_[idx] == 'NOUN' else 0 for idx in range(len(test_labels_))]

    ids = [ids[idx] for idx in only_acceptable]
    test_ids = [test_ids[idx] for idx in test_only_acceptable]

top_scores = []
bottom_scores = []

cola_arr = np.array([value for key, value in cola.items() if key in ids])
keys = [key for key, value in cola.items() if key in ids]

cola_arr_test = np.array([value for key, value in cola_test.items() if key in test_ids])
keys_test = [key for key, value in cola_test.items() if key in test_ids]

PC_RANGE = range(13)
SUBTRACT_MEAN = True
F1_TYPE = 'weighted avg'
F1_WRITE = 'Weighted'

j = Counter(labels)
class_weights = {i:1-(1/j[i]) for i in j.keys()}

for i in PC_RANGE:

    PC_EXTRACT = i

    #Process training array
    bottom, n = pca_transform(cola_arr,PC_EXTRACT,subtract_mean = SUBTRACT_MEAN)
    n_, top = pca_transform(cola_arr,PC_EXTRACT,subtract_mean = False)

    if i == 0:
        top = copy.deepcopy(bottom)
    
    bottom_dict = {keys[i]: bottom[i] for i in range(len(bottom))}
    top_dict = {keys[i]: top[i] for i in range(len(top))}

    bottom = np.array([value for key, value in bottom_dict.items()])
    top = np.array([value for key, value in top_dict.items()])

    train_ = cola_arr

    #Process test array
    bottom_test, n = pca_transform(cola_arr_test,PC_EXTRACT,subtract_mean = SUBTRACT_MEAN)
    n_, top_test = pca_transform(cola_arr_test,PC_EXTRACT,subtract_mean = False)

    bottom_dict_test = {keys_test[i]: bottom_test[i] for i in range(len(bottom_test))}
    top_dict_test = {keys_test[i]: top_test[i] for i in range(len(top_test))}

    bottom_test = np.array([value for key, value in bottom_dict_test.items()])
    top_test = np.array([value for key, value in top_dict_test.items()])

    test_ = cola_arr_test

    if not BIN and TEST == 'POS':
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',verbose=0,class_weight=class_weights).fit(top,labels)
    else:
        clf = LogisticRegression(verbose=0,class_weight=class_weights).fit(top,labels)

    top_scores.append(sklearn.metrics.classification_report(test_labels,clf.predict(top_test),output_dict=True)[F1_TYPE]['f1-score'])

    if not BIN and TEST == 'POS':
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',verbose=0,class_weight=class_weights).fit(bottom,labels)
    else:
        clf = LogisticRegression(verbose=0,class_weight=class_weights).fit(bottom,labels)

    bottom_scores.append(sklearn.metrics.classification_report(test_labels,clf.predict(bottom_test),output_dict=True)[F1_TYPE]['f1-score'])

sns.lineplot(PC_RANGE,top_scores,label='Top PCs',marker='o')
sns.lineplot(PC_RANGE,bottom_scores,label='Vector After PCs Nullified',marker='o')
plt.xlabel('PCs Nullified')
plt.ylabel('F1 Score')
plt.xticks(list(PC_RANGE))
plt.legend()
plt.title(f'Binary POS Tagging {F1_WRITE} F1 Score - Current Word')
plt.show()

print(top_scores)
print(bottom_scores)