from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

import sys
sys.path.append('../')
from utils.utils import clean_str, loadWord2Vec  


if len(sys.argv) != 2:
    sys.exit("Use: python remove_words.py <dataset>")


dataset = sys.argv[1]


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


doc_content_list = []
if dataset == 'CHINESE' or dataset == 'THUCTC':
    with open('../cleaned_data/' + dataset + '/corpus/' + dataset + '.txt', 'r') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip())
else:
    with open('../cleaned_data/' + dataset + '/corpus/' + dataset + '.txt', 'rb') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip().decode('latin1'))


word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    if dataset == 'CHINESE' or dataset == 'THUCTC':
        temp = doc_content
    else:
        temp = clean_str(doc_content)
        
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    if dataset == 'CHINESE':
        temp = doc_content
    else:
        temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5
        if dataset == 'mr':
            doc_words.append(word)
        elif word not in stop_words and word_freq[word] >= 5:
            doc_words.append(word)
        

    doc_str = ' '.join(doc_words).strip()
    #if doc_str == '':
        #doc_str = temp
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)


with open('../cleaned_data/' + dataset + '/' + dataset + '_clean.txt', 'w') as f:
    f.write(clean_corpus_str)

print("finish")