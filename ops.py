import tensorflow as tf 

def instance_norm(inputs, name='instance_norm'):
	with tf.variable_scope(name):
		depth = inputs.get_shape()[3]
		scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0,0.02,dtype=tf.float32))
		offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
		inv = tf.rsqrt(variance+1e-5)
		normalized = (inputs-mean)*inv
		return scale*normalized + offset


def conv_block(inputs, filters, kernel_size=4, strides=(2,2), padding='same',
	has_norm_layer=True, has_activation_layer=True, use_instance_norm=True, use_leaky_relu=False, name='conv_block'):

	with tf.variable_scope(name):
		x = tf.layers.conv2d(
			inputs=inputs,
			filters=filters,
			kernel_size=[kernel_size,kernel_size],
			strides=strides,
			padding=padding,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
			)

		if has_norm_layer:
			if use_instance_norm:
				x = instance_norm(x)
				# x = tf.contrib.instance_norm(x)
			else:
				x = tf.layers.batch_normalization(
					x,
					momemtum=0.9,
					epsilon=1e-5)

		if has_activation_layer:
			if use_leaky_relu:
				x = tf.nn.leaky_relu(x, alpha=0.3)
			else:
				x = tf.nn.relu(x)

		return x


def res_block(inputs, filters=32, use_dropout=False, name='res_block'):
	with tf.variable_scope(name):
		y = conv_block(inputs, filters, kernel_size=3, strides=(1,1), name='conv1')
		if use_dropout:
			y = tf.layers.dropout(y, 0.5)
		y = conv_block(y, filters, kernel_size=3, strides=(1,1), has_activation_layer=False, name='conv2')

		return inputs + y


def up_block(inputs, filters, kernel_size=3, use_conv2d_transpose=True, use_instance_norm=True, name='conv2d_transpose'):
	with tf.variable_scope(name):
		x = tf.layers.conv2d_transpose(
			inputs=inputs,
			filters=filters,
			kernel_size=[kernel_size, kernel_size], 
			strides=(2,2),
			padding='same',
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

		if use_instance_norm:
			x = instance_norm(x)
			# x = tf.contrib.instance_norm(x)
		else:		
			x = tf.layers.batch_normalization(
						x,
						momemtum=0.9,
						epsilon=1e-5)

		x = tf.nn.relu(x)

	return x


