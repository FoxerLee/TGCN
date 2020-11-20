##-----------------------------------------------------------##
# Try to use: python tfidf_lr.py <dataset> to test the accuracy
# Results till now:
# R8: 94.88%
# R52: 87.42%
# ohsumed: 54.86%
# THUTC: 93.67%
# To be continued...

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sys

# Usage of code
if len(sys.argv)!=2:
    sys.exit("Use: python tfidf_lr.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'Chinese_L', 'THUTC']
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

def generate_content_label_20ng(dataset_name):
    doc_content_list = []
    with open('../cleaned_data/' + dataset + '/corpus/' + dataset + '.txt', 'r', encoding='latin1') as f:
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

if dataset!='20ng':
    doc_content_list, doc_label_list = generate_content_label(dataset)
else:
    doc_content_list, doc_label_list = generate_content_label_20ng(dataset)
    doc_content_list = doc_content_list[:len(doc_label_list)]
print("Content and labels generated!")
print(len(doc_content_list), len(doc_label_list))
df = generate_df(doc_content_list, doc_label_list)

# In case training/testing or more complex strings
train_df = df.loc[df['set'].str.find('train')!=-1]
test_df = df.loc[df['set'].str.find('test')!=-1]
#train_df = df.loc[df['set'] == 'train']
#test_df = df.loc[df['set'] == 'test']
X_train, y_train = train_df['content'].to_numpy(), train_df['label'].to_numpy()
X_test, y_test = test_df['content'].to_numpy(), test_df['label'].to_numpy()

vectorizer = TfidfVectorizer()
X_train_tv = vectorizer.fit_transform(X_train)
X_test_tv = vectorizer.transform(X_test)
print("Vectorization finished!")

clf = LogisticRegression(random_state=0)
clf.fit(X_train_tv, y_train)
print("Training finished!")
y_pre = clf.predict(X_test_tv)

print("Tfidf-LR on %s: %.4f." % (dataset, accuracy_score(y_test, y_pre)))