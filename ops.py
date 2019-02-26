import tensorflow as tf 

class InstanceNormalization(tf.keras.layers.Layer):
	def __init__(self):
		super(InstanceNormalization, self).__init__()

	def build(self, input_shape):
		depth = int(input_shape[-1])
		self.scale = self.add_variable('scale', shape=[depth], initializer=tf.random_normal_initializer(1.0,0.02,dtype=tf.float32))
		self.offset = self.add_variable('offset', shape=[depth], initializer=tf.constant_initializer(0.0))

	def call(self, inputs):
		mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
		inv = tf.rsqrt(variance)
		normalized = (inputs-mean)*inv
		return self.scale*normalized + self.offset


def conv_block(inputs, filters, kernel_size=4, strides=(2,2), padding='same',
	has_norm_layer=True, has_activation_layer=True, use_instance_norm=True, use_leaky_relu=False):

	x = tf.keras.layers.Conv2D(
		filters=filters,
		kernel_size=kernel_size,
		strides=strides,
		padding=padding,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
		)(inputs)

	if has_norm_layer:
		if use_instance_norm:
			x = InstanceNormalization()(x)
		else:
			x = tf.keras.layers.BatchNormalization(
				momemtum=0.9,
				epsilon=1e-5
				)(x)

	if has_activation_layer:
		if use_leaky_relu:
			x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
		else:
			x = tf.keras.layers.ReLU()(x)

	return x


def res_block(inputs, filters=32, use_dropout=False):

	y = conv_block(inputs, filters, kernel_size=3, strides=(1,1))
	if use_dropout:
		y = tf.keras.layers.Dropout(0.5)(y)
	y = conv_block(y, filters, kernel_size=3, strides=(1,1), has_activation_layer=False)

	return tf.keras.layers.Add()([inputs, y])


def up_block(inputs, filters, kernel_size=3, strides=2, use_conv2d_transpose=True, use_instance_norm=True):

	x = tf.keras.layers.Conv2DTranspose(
		filters=filters,
		kernel_size=kernel_size, 
		strides=(strides,strides),
		padding='same',
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
		)(inputs)

	if use_instance_norm:
			x = InstanceNormalization()(x)
	else:
		x = tf.keras.layers.BatchNormalization(
			momemtum=0.9,
			epsilon=1e-5
			)(x)

	x = tf.keras.layers.ReLU()(x)

	return x


