import tensorflow as tf
from ops import *

def resnet_generator(image, n_res_blocks=6, reuse=False, name='generator'):

	with tf.variable_scope(name, reuse=reuse):

		c0 = conv_block(image, 64, 7, (1,1), name='c0')
		c1 = conv_block(c0, 128, 3, (2,2), name='c1')
		c2 = conv_block(c1, 256, 3, (2,2), name='c2')

		x = c2

		for i in range(n_res_blocks):
			x = res_block(x, 256, 3, name='res_block_'+str(i))

		u1 = up_block(x, 128, 3, name='up1')
		u2 = up_block(u1, 64, 3, name='up2')

		out = tf.nn.tanh(tf.layers.conv2d(u2, 3, 7, strides=(1,1)))

		return out


def discriminator(image, nf=64, n_hidden_layers=3, reuse=False, name='discriminator'):

	with tf.variable_scope(name, reuse=reuse):

		x = conv_block(image, nf, has_activation_layer=False, name='conv')

		for i in range(n_hidden_layers):
			n_filters = nf*(2**(i+1))
			x = conv_block(x, n_filters, use_leaky_relu=True, name='conv'+str(i))

		out = conv_block(x, 1, strides=(1,1), has_activation_layer=False, name='output')

		return out