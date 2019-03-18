import tensorflow as tf
from ops import *

def create_discriminator(inputs, nf=64, n_hidden_layers=3, attention=False, name='discriminator'):

	# inputs = tf.keras.layers.Input(shape=img_shape)

	x = conv_block(inputs, nf, strides=(2,2), has_norm_layer=False, use_leaky_relu=True)

	for i in range(n_hidden_layers):
		n_filters = nf*(2**(i+1))
		x = conv_block(x, n_filters, strides=(2,2), use_leaky_relu=True)
		if i==1 and attention:
			x = SelfAttention(n_filters)(x)

	pred = tf.keras.layers.Conv2D(1, 1, strides=(1,1), padding='same')(x)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)


def resnet_generator(inputs, nf=64, n_res_blocks=9, channels=3, name='generator'):

	# inputs = tf.keras.layers.Input(shape=img_shape)

	c0 = conv_block(inputs, nf, 7, strides=(1,1), use_leaky_relu=False)
	c1 = conv_block(c0, nf*2, 3, strides=(2,2), use_leaky_relu=False)
	c2 = conv_block(c1, nf*4, 3, strides=(2,2), use_leaky_relu=False)
	x = c2

	for i in range(n_res_blocks):
		x = res_block(x, nf*4, 3)

	u1 = up_block(x, nf*2, 3)
	u2 = up_block(u1, nf, 3)

	pred = tf.keras.layers.Conv2D(channels, 7, strides=(1,1), padding='same', activation='tanh')(u2)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred])


def unet(inputs, nf=64, channels=3, reuse=False, name='generator'):

	c1 = conv_block(inputs, nf, 3, strides=2, use_leaky_relu=True)
	c2 = conv_block(c1, nf*2, 3, strides=2, use_leaky_relu=True)
	c3 = conv_block(c2, nf*4, 3, strides=2, use_leaky_relu=True)
	c4 = conv_block(c3, nf*8, 3, strides=2, use_leaky_relu=True)

	u1 = up_block(c4, nf*4, 3, strides=2)
	u1 = tf.keras.layers.Concatenate(axis=3)([u1,c3])
	u2 = up_block(u1, nf*2, 3, strides=2)
	u2 = tf.keras.layers.Concatenate(axis=3)([u1,c2])
	u3 = up_block(u2, nf, 3, strides=2)
	u3 = tf.keras.layers.Concatenate(axis=3)([u3,c1])

	return tf.keras.models.Model(inputs, u3)



