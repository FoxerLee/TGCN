from utils.inits import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(rate))


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero
        self.embedding = None

        self.weights_ = []
        for i in range(1):
            w = self.add_variable('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_variable('bias', [output_dim])




    def call(self, inputs, training=None):
        x, support_ = inputs

        # dropout
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)


        # convolve
        supports = list()
        for i in range(len(support_)):
            if not self.featureless: # if it has features x
                pre_sup = dot(x, self.weights_[i], sparse=self.is_sparse_inputs)
            else:
                pre_sup = self.weights_[i]

            support = dot(support_[i], pre_sup, sparse=True)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.bias
        
        self.embedding = output # for visualization
        return self.activation(output)
