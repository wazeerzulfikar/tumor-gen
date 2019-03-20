import tensorflow as tf
import tensorflow.keras.backend as K
# tf.enable_eager_execution()

def create_discriminator3d(inputs, nf=64, n_hidden_layers=2, attention=False, name='discriminator'):

	# inputs = tf.keras.layers.Input(shape=img_shape)

	x = conv_block(inputs, nf, 7, strides=(4,4,4), n_dims=3, has_norm_layer=False, use_leaky_relu=True)

	for i in range(n_hidden_layers):
		n_filters = nf*(2**(i+1))
		x = conv_block(x, n_filters, strides=(4,4,4), n_dims=3, use_leaky_relu=True)
		if i==1 and attention:
			x = SelfAttention(n_filters)(x)

	pred = tf.keras.layers.Conv3D(1, 1, strides=(1,1,1), n_dims=3, padding='same')(x)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)


def resnet_generator3d(inputs, nf=64, n_res_blocks=9, channels=3, name='generator'):

	# inputs = tf.keras.layers.Input(shape=img_shape)

	c0 = conv_block(inputs, nf, 7, strides=(4,4,4), n_dims=3, use_leaky_relu=False)
	c1 = conv_block(c0, nf*2, 3, strides=(2,2,2), n_dims=3, use_leaky_relu=False)
	c2 = conv_block(c1, nf*4, 3, strides=(2,2,2), n_dims=3, use_leaky_relu=False)
	x = c2

	for i in range(n_res_blocks):
		x = res_block(x, nf*4, 3, n_dims=3)

	u1 = up_block(x, nf*2, 5, strides=(4,4,4), n_dims=3)
	u2 = up_block(u1, nf, 3, strides=(2,2,2), n_dims=3)

	pred = tf.keras.layers.Conv3DTranspose(channels, 5, strides=(2,2,2), padding='same', activation='tanh')(u2)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred])



