import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """
    Softmax cross-entropy loss with masking.
    """
    
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#     print(loss)
#     print("=====")
    
    mask = tf.cast(mask, dtype=tf.float32)
#     print(mask)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """
    Accuracy with masking.
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
# import tensorflow as tf


# def masked_softmax_cross_entropy(preds, labels, mask):
#     """
#     Softmax cross-entropy loss with masking.
#     """
    
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#     mask = tf.gather(mask, [0], axis=1)
#     mask = tf.reshape(mask, [tf.shape(mask).numpy()[0]])
#     mask /= tf.reduce_mean(mask)
#     loss *= mask
#     return tf.reduce_mean(loss)


# def masked_accuracy(preds, labels, mask):
#     """
#     Accuracy with masking.
#     """
#     correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
#     accuracy_all = tf.cast(correct_prediction, tf.float32)
#     mask = tf.gather(mask, [0], axis=1)
#     mask = tf.reshape(mask, [tf.shape(mask).numpy()[0]])
#     mask /= tf.reduce_mean(mask)
#     accuracy_all *= mask
#     return tf.reduce_mean(accuracy_all)
