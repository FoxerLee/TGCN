# from __future__ import division
# from __future__ import print_function
# from sklearn import metrics
# import random
# import time
# import sys
# import os

# import torch
# import torch.nn as nn

import numpy as np

# from utils.utils import *
# from models.gcn import GCN
# from models.mlp import MLP

# from config import CONFIG
# cfg = CONFIG()

import  time
import  tensorflow as tf
from    tensorflow.keras import optimizers

from    utils.utils import *
from    models import GCN, MLP
from    config import args

import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('tf version:', tf.__version__)
assert tf.__version__.startswith('2.')


# if len(sys.argv) != 2:
	# sys.exit("Use: python train.py <dataset>")

# dataset = sys.argv[1]

# if dataset not in datasets:
	# sys.exit("wrong dataset name")
# cfg.dataset = dataset

# set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(args.dataset)


features = sp.identity(features.shape[0])  # featureless
features = preprocess_features(features)



if args.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif args.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, args.max_degree)
    num_supports = 1 + args.max_degree
    model_func = GCN
elif args.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))



t_features = tf.convert_to_tensor(features)
t_y_train = tf.convert_to_tensor(y_train)
t_y_val = tf.convert_to_tensor(y_val)
t_y_test = tf.convert_to_tensor(y_test)
t_train_mask = tf.convert_to_tensor(train_mask.astype(np.float32))
tm_train_mask = tf.tile(tf.transpose(tf.expand_dims(t_train_mask, 0), perm=[1, 0]), [1, y_train.shape[1]])



# # Define placeholders
# t_features = torch.from_numpy(features)
# t_y_train = torch.from_numpy(y_train)
# t_y_val = torch.from_numpy(y_val)
# t_y_test = torch.from_numpy(y_test)
# t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
# tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

t_support = []
for i in range(len(support)):
    t_support.append(tf.convert_to_tensor(support[i]))

# if torch.cuda.is_available():
#     # model_func = model_func.to(device)
#     t_features = t_features.to(device)
#     t_y_train = t_y_train.to(device)
#     t_y_val = t_y_val.to(device)
#     t_y_test = t_y_test.to(device)
#     t_train_mask = t_train_mask.to(device)
#     tm_train_mask = tm_train_mask.to(device)
#     for i in range(len(support)):
#         t_support = [t.to(device) for t in t_support if True]
        
# model = model_func(input_dim=features.shape[0], support=t_support, num_classes=y_train.shape[1])
# model.to(device)

# Create model
model = GCN(input_dim=features.shape[0], output_dim=y_train.shape[1], num_features_nonzero=features[1].shape) # [1433]





# Loss and optimizer
optimizer = optimizers.Adam(lr=args.learning_rate)


for epoch in range(args.epochs):

    with tf.GradientTape() as tape:
        loss, acc = model((t_features, t_y_train, tm_train_mask, t_support))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    _, val_acc = model((t_features, t_y_val, val_mask, t_support), training=False)


    if epoch % 20 == 0:

        print(epoch, float(loss), float(acc), '\tval:', float(val_acc))



test_loss, test_acc = model((features, test_label, test_mask, support), training=False)


print('\ttest:', float(test_loss), float(test_acc))

# Define model evaluation function
# def evaluate(features, labels, mask):
#     t_test = time.time()
#     # feed_dict_val = construct_feed_dict(
#     #     features, support, labels, mask, placeholders)
#     # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
#     model.eval()
#     with torch.no_grad():
#         logits = model(features)
#         t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32)).to(device)
#         tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1]).to(device)
#         loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
#         pred = torch.max(logits, 1)[1]
#         acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
        
#     return loss.cpu().numpy(), acc, pred.cpu().numpy(), labels.cpu().numpy(), (time.time() - t_test)



# val_losses = []

# # Train model
# for epoch in range(cfg.epochs):

#     t = time.time()
    
#     # Forward pass
#     logits = model(t_features)
#     loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])    
#     acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
        
#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Validation
#     val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
#     val_losses.append(val_loss)

#     print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
#                 .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

#     if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping+1):-1]):
#         print_log("Early stopping...")
#         break


# print_log("Optimization Finished!")


# # Testing
# test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
# print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

# test_pred = []
# test_labels = []
# for i in range(len(test_mask)):
#     if test_mask[i]:
#         test_pred.append(pred[i])
#         test_labels.append(np.argmax(labels[i]))


# print_log("Test Precision, Recall and F1-Score...")
# print_log(metrics.classification_report(test_labels, test_pred, digits=4))
# print_log("Macro average Test Precision, Recall and F1-Score...")
# print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
# print_log("Micro average Test Precision, Recall and F1-Score...")
# print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))





# doc and word embeddings
# tmp = model.layer1.embedding.numpy()
# word_embeddings = tmp[train_size: adj.shape[0] - test_size]
# train_doc_embeddings = tmp[:train_size]  # include val docs
# test_doc_embeddings = tmp[adj.shape[0] - test_size:]

# print_log('Embeddings:')
# print_log('\rWord_embeddings:'+str(len(word_embeddings)))
# print_log('\rTrain_doc_embeddings:'+str(len(train_doc_embeddings))) 
# print_log('\rTest_doc_embeddings:'+str(len(test_doc_embeddings))) 
# print_log('\rWord_embeddings:') 
# print(word_embeddings)

# with open('./data/corpus/' + dataset + '_vocab.txt', 'r') as f:
#     words = f.readlines()

# vocab_size = len(words)
# word_vectors = []
# for i in range(vocab_size):
#     word = words[i].strip()
#     word_vector = word_embeddings[i]
#     word_vector_str = ' '.join([str(x) for x in word_vector])
#     word_vectors.append(word + ' ' + word_vector_str)

# word_embeddings_str = '\n'.join(word_vectors)
# with open('./data/' + dataset + '_word_vectors.txt', 'w') as f:
#     f.write(word_embeddings_str)



# doc_vectors = []
# doc_id = 0
# for i in range(train_size):
#     doc_vector = train_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1

# for i in range(test_size):
#     doc_vector = test_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1

# doc_embeddings_str = '\n'.join(doc_vectors)
# with open('./data/' + dataset + '_doc_vectors.txt', 'w') as f:
#     f.write(doc_embeddings_str)



