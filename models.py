import tensorflow as tf
from ops import *

def discriminator(image, nf=64, n_hidden_layers=3, reuse=False, name='discriminator'):

	with tf.variable_scope(name, reuse=reuse):

		x = conv_block(image, nf, strides=(2,2), has_norm_layer=False, use_leaky_relu=True, name='conv')

		for i in range(n_hidden_layers):
			n_filters = nf*(2**(i+1))
			x = conv_block(x, n_filters, strides=(2,2), use_leaky_relu=True, name='conv'+str(i))

		out = conv_block(x, 1, strides=(1,1), has_norm_layer=False, has_activation_layer=False, name='output')

		return out


def resnet_generator(image, nf=64, n_res_blocks=9, channels=3, reuse=False, name='generator'):

	with tf.variable_scope(name, reuse=reuse):

		c0 = conv_block(image, nf, 7, strides=(1,1), name='c0')
		c1 = conv_block(c0, nf*2, 3, strides=(2,2), name='c1')
		c2 = conv_block(c1, nf*4, 3, strides=(2,2), name='c2')
		x = c2

		for i in range(n_res_blocks):
			x = res_block(x, nf*4, 3, name='res_block_'+str(i))

		u1 = up_block(x, nf*2, 3, name='up1')
		u2 = up_block(u1, nf, 3, name='up2')

		out = tf.nn.tanh(tf.layers.conv2d(u2, channels, 7, strides=(1,1), padding='same'))

		return out


def unet_generator(image, nf=64, channels=3, reuse=False, name='generator'):

	with tf.variable_scope(name, reuse=reuse):

		e1 = conv_block(image, nf, use_instance_norm=True, use_leaky_relu=True, name='e1')
		e2 = conv_block(e1, nf*2, use_instance_norm=True, use_leaky_relu=True, name='e2')
		e3 = conv_block(e2, nf*4, use_instance_norm=True, use_leaky_relu=True, name='e3')
		e4 = conv_block(e3, nf*8, use_instance_norm=True, use_leaky_relu=True, name='e4')
		e5 = conv_block(e4, nf*8, use_instance_norm=True, use_leaky_relu=True, name='e5')
		e6 = conv_block(e5, nf*8, use_instance_norm=True, use_leaky_relu=True, name='e6')
		e7 = conv_block(e6, nf*8, use_instance_norm=True, use_leaky_relu=True, name='e7')
		e8 = conv_block(e7, nf*8, use_instance_norm=True, use_leaky_relu=True, name='e8')

		d1 = up_block(e8, nf*8, name='d1')
		d1 = tf.nn.dropout(d1, 0.5)
		d1 = tf.concat([d1, e7], axis=3)

		d2 = up_block(d1, nf*8, name='d2')
		d2 = tf.nn.dropout(d2, 0.5)
		d2 = tf.concat([d2, e6], axis=3)

		d3 = up_block(d2, nf*8, name='d3')
		d3 = tf.nn.dropout(d3, 0.5)
		d3 = tf.concat([d3, e5], axis=3)

		d4 = up_block(d3, nf*8, name='d4')
		d4 = tf.nn.dropout(d4, 0.5)
		d4 = tf.concat([d4, e4], axis=3)

		d5 = up_block(d4, nf*8, name='d5')
		d5 = tf.nn.dropout(d5, 0.5)
		d5 = tf.concat([d5, e3], axis=3)

		d6 = up_block(d5, nf*8, name='d6')
		d6 = tf.nn.dropout(d6, 0.5)
		d6 = tf.concat([d6, e2], axis=3)

		d7 = up_block(d6, nf*8, name='d7')
		d7 = tf.nn.dropout(d7, 0.5)
		d7 = tf.concat([d7, e1], axis=3)

		out = tf.nn.tanh(tf.layers.conv2d_transpose(d7, channels, 3, 2, padding='same', name='out'))

		return out


