import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
import tf_metrics

num_classes = 16
pos_indices = range(16)
average = 'macro'

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def output_labels(preds, labels, mask):
    """Accuracy with masking."""
    '''
    y_true = tf.cast(tf.argmax(labels, 1),dtype=tf.float32)
    y_pred = tf.cast(tf.argmax(preds, 1),dtype=tf.float32)
    
    y_true = tf.boolean_mask(y_true,mask)
    y_pred = tf.boolean_mask(y_pred,mask)
    '''
    y_true = tf.boolean_mask(labels, mask)
    y_pred = tf.boolean_mask(preds, mask)
    y_true = tf.cast(tf.argmax(y_true, 1), dtype=tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, 1), dtype=tf.float32)


    return y_true,y_pred

def masked_precision(preds, labels, mask):

    y_true = tf.cast(tf.argmax(labels, 1),dtype=tf.int32)
    y_pred = tf.cast(tf.argmax(preds, 1),dtype=tf.int32)
    #y_true = tf.boolean_mask(y_true,mask)
    #y_pred = tf.boolean_mask(y_pred,mask)
    #print y_pred.get_shape(),y_true.get_shape()
    prec = tf_metrics.precision(y_true, y_pred, num_classes, pos_indices, average=average)
    return prec[0]


def masked_recall(preds, labels, mask):
    """Accuracy with masking."""
    y_true = tf.cast(tf.argmax(labels, 1),dtype=tf.int32)
    y_pred = tf.cast(tf.argmax(preds, 1),dtype=tf.int32)
    y_true = tf.boolean_mask(y_true,mask)
    y_pred = tf.boolean_mask(y_pred,mask)
    #print y_pred.get_shape(), y_true.get_shape()
    rec = tf_metrics.recall(y_true, y_pred, num_classes, pos_indices, average=average)
    return rec[0]

def masked_f1(preds, labels, mask):
    """Accuracy with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    y_true = tf.cast(tf.argmax(labels, 1), dtype=tf.float32)
    y_pred = tf.cast(tf.argmax(preds, 1), dtype=tf.float32)
    y_true *= mask
    y_pred *= mask
    #print y_pred.get_shape(), y_true.get_shape()
    f1, _ = tf_metrics.f1(y_true, y_pred, num_classes, pos_indices, average=average)
    return f1
