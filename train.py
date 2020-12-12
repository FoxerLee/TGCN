import numpy as np


import time
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn import metrics

from utils.utils import *
from models.gcn import GCN

from config import CONFIG
import os


cfg = CONFIG()

# set random seed
seed = 6606
np.random.seed(seed)
tf.random.set_seed(seed)

if len(sys.argv) != 2:
    sys.exit("Use: python train.py <dataset>")
    
dataset = sys.argv[1]
cfg.dataset = dataset    
    


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(cfg.dataset)

features = sp.identity(features.shape[0])  # featureless
features = preprocess_features(features)


support = [preprocess_adj(adj)]


t_features = tf.SparseTensor(*features)
t_y_train = tf.convert_to_tensor(y_train)
t_y_val = tf.convert_to_tensor(y_val)
t_y_test = tf.convert_to_tensor(y_test)
tm_train_mask = tf.convert_to_tensor(train_mask)

tm_val_mask = tf.convert_to_tensor(val_mask)
tm_test_mask = tf.convert_to_tensor(test_mask)

t_support = []
for i in range(len(support)):
    t_support.append(tf.cast(tf.SparseTensor(*support[i]), dtype=tf.float64))


# Create model
model = GCN(input_dim=features[2][1], output_dim=y_train.shape[1], num_features_nonzero=features[1].shape)



# Loss and optimizer
optimizer = optimizers.Adam(lr=cfg.learning_rate)

cost_val = []

for epoch in range(cfg.epochs):
    
    t = time.time()
    with tf.GradientTape() as tape:
        _, loss, acc = model((t_features, t_y_train, tm_train_mask, t_support))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    _, val_loss, val_acc = model((t_features, t_y_val, tm_val_mask, t_support), training=False)
    cost_val.append(val_loss)
    
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss),
          "train_acc=", "{:.5f}".format(acc), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))
    
    if epoch > cfg.early_stopping and cost_val[-1] > np.mean(cost_val[-(cfg.early_stopping+1):-1]):
        print("Early stopping...")
        break

def evaluate(features, y, mask, support):
    t = time.time()
    
    pred, test_loss, test_acc = model((features, y, mask, support), training=False)
    
    
    return test_loss, test_acc, pred, np.argmax(y, axis=1), time.time() - t


test_cost, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, tm_test_mask, t_support)
print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


test_pred = []
test_labels = []

for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print("Average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))




embeddings = model.layers_[0].embedding

word_embeddings = embeddings[train_size: adj.shape[0] - test_size]
train_doc_embeddings = embeddings[:train_size]  # include val docs
test_doc_embeddings = embeddings[adj.shape[0] - test_size:]

print('storing word embeddings...')
f = open('./cleaned_data/' + cfg.dataset + '/corpus/' + cfg.dataset +  '_vocab.txt', 'r')
words = f.readlines()
f.close()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(tf.keras.backend.get_value(x)) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
f = open('./cleaned_data/' + cfg.dataset + '/' + cfg.dataset +  '_word_vectors.txt', 'w')
f.write(word_embeddings_str)
f.close()
print("finish...")



print('storing doc embeddings...')
doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(tf.keras.backend.get_value(x)) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(tf.keras.backend.get_value(x)) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
f = open('./cleaned_data/' + cfg.dataset + '/' + cfg.dataset +  '_doc_vectors.txt', 'w')
f.write(doc_embeddings_str)
f.close()

print("finish...")