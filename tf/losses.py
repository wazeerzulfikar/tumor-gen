import tensorflow as tf

def mse_criterion(pred, target):
	return tf.reduce_mean(tf.square(pred-target))

def abs_criterion(pred, target):
	return tf.reduce_mean(tf.abs(pred-target))

def cross_entropy_criterion(pred, target):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target))