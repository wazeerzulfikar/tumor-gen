import tensorflow as tf
from keras import layers
from keras.models import Model
# from keras_contrib.layers import InstanceNormalization
import keras.backend as K
import os
import time

def batchnorm():

    return layers.BatchNormalization(momentum=0.9, axis=3, epsilon=1e-5)


def conv_block(x, filters, kernel_size=4, strides=(2,2), padding='same',
	has_norm_layer=True, use_instance_norm=False, has_activation_layer=True, use_leaky_relu=False):

	x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)

	if has_norm_layer:
		if use_instance_norm:
			x = InstanceNormalization(axis=1)(x)
		else:
			x = batchnorm()(x)

	if has_activation_layer:
		if use_leaky_relu:
			x = layers.LeakyReLU(alpha=0.3)(x)
		else:
			x = layers.Activation('relu')(x)

	return x


def discriminator(image_size=32,nf=64,n_hidden_layers=3,load_path=None):

	inputs = layers.Input(shape=(image_size,image_size,3))

	x = conv_block(inputs, nf, kernel_size=4, has_norm_layer=False, use_leaky_relu=True)

	for i in range(n_hidden_layers):
		nf = 2*nf
		x = conv_block(x, nf, kernel_size=4, use_instance_norm=True)

	outputs = layers.Conv2D(1, (4,4), activation='sigmoid')(x)

	model = Model(inputs=inputs, outputs=outputs)

	if load_path not None:
		model.load_weights(load_path)

	return model


def res_block(x, filters=256, use_dropout=False):

	y = conv_block(x, filters, 3, strides=(1,1))
	if use_dropout:
		y = Dropout(0.5)(y)
	y = conv_block(y, filters, 3, strides=(1,1), has_activation_layer=False)

	return layers.Add()([x,y])


def up_block(x, filters, kernel_size, use_conv2d_transpose=True):

	if use_conv2d_transpose:
		x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(x)
		x = batchnorm()(x)
		x = layers.Activation('relu')(x)

	else:
		x = UpSampling2D()(x)
		x = conv_block(x, filters, 1)

	return x


def resnet_generator(image_size=32, n_res_blocks=6, load_path=None):

	inputs = layers.Input(shape=(image_size,image_size,3))

	x = conv_block(inputs, 64, 7, (1,1))
	x = conv_block(x, 128, 3, (2,2))
	x = conv_block(x, 256, 3, (2,2))

	for i in range(n_res_blocks):
		x = res_block(x)

	x = up_block(x, 128, 3)
	x = up_block(x, 256, 3)

	outputs = layers.Conv2D(3, (7,7), activation='sigmoid', padding='same')(x)

	model = Model(inputs=inputs, outputs=outputs)

	if load_path not None:
		model.load_weights(load_path)

	return model, inputs, outputs


# def unet_generator(image_size=32, ):