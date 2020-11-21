##-----------------------------------------------------------##
# Try to use: python fastText.py <dataset> to show performance
# Results till now:
#
# The normal fasttext
# R8: (2189, 97.12%, 97.12%)
# R52: (2568, 88.71%, 88.71%)
# ohsumed: (4043, 50.45%, 50.45%)
# THUTC: (600, 92.17%, 92.17%)
# To be continued...
#
# The bigram fasttext
# R8: (2189, 96.21%, 96.21%)
# R52: (2568, 85.63%, 85.63%)
# ohsumed: (4043, 48.29%, 48.29%)
# THUTC: (600, 89.50%, 89.50%)
##-----------------------------------------------------------##

import numpy as np
import pandas as pd
import fasttext

import time
import sys

if len(sys.argv)!=2:
    sys.exit("Use: python fastText.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'THUTC', 'Chinese_L']
dataset = sys.argv[1]
if dataset not in datasets:
    sys.exit("Error: wrong dataset!")

def generate_content_label(dataset_name):
    doc_content_list = []
    with open('../cleaned_data/' + dataset + '/corpus/' + dataset + '.txt', 'r') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip())
    f.close()

    doc_label_list = []
    with open('../cleaned_data/' + dataset + '/' + dataset + '.txt', 'r') as f:
        for line in f.readlines():
            doc_label_list.append(line.strip().split())
    f.close()

    return doc_content_list, doc_label_list

def generate_df(doc_content_list, doc_label_list):
    doc_label_list = np.array(doc_label_list)
    df = pd.DataFrame()

    df['content'] = doc_content_list
    df['set'] = doc_label_list[:,1]
    df['label'] = doc_label_list[:,2]

    return df

doc_content_list, doc_label_list = generate_content_label(dataset)
print("Content and labels generated!")
print(len(doc_content_list), len(doc_label_list))
df = generate_df(doc_content_list, doc_label_list)

# In case training/testing or more complex strings
train_df = df.loc[df['set'].str.find('train')!=-1]
test_df = df.loc[df['set'].str.find('test')!=-1]

def generate_fasttext_data(dataframe, extend, dataset_name):
    np_data = dataframe[['content', 'label']].to_numpy()
    with open('../cleaned_data/' + dataset_name + '/fastText/' + dataset_name + extend + '.txt', 'w') as f:
        for d in np_data:
            line = '__label__'+d[1]+' '+d[0]+'\n'
            f.write(line)
    f.close()

generate_fasttext_data(train_df, '_train', dataset)
generate_fasttext_data(test_df, '_test', dataset)
print("fastText data generated!")

model = fasttext.train_supervised(input='../cleaned_data/' + dataset + '/fastText/' + dataset + '_train.txt', lr=0.5, epoch=10, wordNgrams=2)
print("fasttext model trained!")
print("The Bigram fasttext result:")
print(model.test('../cleaned_data/' + dataset + '/fastText/' + dataset + '_test.txt'))