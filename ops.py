import tensorflow as tf 
import tensorflow.keras.backend as K
# tf.enable_eager_execution()

class InstanceNormalization(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(InstanceNormalization, self).__init__(**kwargs)

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
				momentum=0.9,
				epsilon=1e-5
				)(x)

	if has_activation_layer:
		if use_leaky_relu:
			x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
		else:
			x = tf.keras.layers.ReLU()(x)

	return x


def res_block(inputs, filters=32, use_dropout=False):

	y = conv_block(inputs, filters, kernel_size=3, strides=(1,1), use_leaky_relu=False)
	if use_dropout:
		y = tf.keras.layers.Dropout(0.5)(y)
	y = conv_block(y, filters, kernel_size=3, strides=(1,1), has_activation_layer=False)

	return tf.keras.layers.Add()([y, inputs])


def up_block(inputs, filters, kernel_size=4, strides=2, use_conv2d_transpose=True, use_instance_norm=True):

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
			momentum=0.9,
			epsilon=1e-5
			)(x)

	x = tf.keras.layers.ReLU()(x)

	return x



def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions 
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]


class SelfAttention(tf.keras.Model):
  def __init__(self, number_of_filters):
    super(SelfAttention, self).__init__()
    
    self.f = tf.keras.layers.Conv2D(number_of_filters//8, 1,
                                     strides=1, padding='SAME', name="f_x",
                                     activation=None)
    
    self.g = tf.keras.layers.Conv2D(number_of_filters//8, 1,
                                     strides=1, padding='SAME', name="g_x",
                                     activation=None)
    
    self.h = tf.keras.layers.Conv2D(number_of_filters, 1,
                                     strides=1, padding='SAME', name="h_x",
                                     activation=None)
    
    self.gamma = tf.Variable(0., dtype=tf.float32, trainable=True, name="gamma")
    
  def call(self, x):

    f = self.f(x)
    g = self.g(x)
    h = self.h(x)
    
    f_flatten = hw_flatten(f)
    g_flatten = hw_flatten(g)
    h_flatten = hw_flatten(h)
    
    s = tf.matmul(g_flatten, f_flatten, transpose_b=True) # [B,N,C] * [B, N, C] = [B, N, N]

    b = tf.nn.softmax(s, axis=-1)
    o = tf.matmul(b, h_flatten)

    y = self.gamma * tf.reshape(o, tf.shape(x)) + x

    return y

class ReLU(tf.keras.layers.ReLU):

	def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):

		if type(max_value) is dict:
			max_value = max_value['value']
		if type(negative_slope) is dict:
			negative_slope = negative_slope['value']
		if type(threshold) is dict:
			threshold = threshold['value']

		super(ReLU, self).__init__(max_value=None, negative_slope=0, threshold=0, **kwargs)



