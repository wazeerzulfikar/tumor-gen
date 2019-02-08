import tensorflow as tf
import numpy as np
import os
from models import *
from losses import *
from utils import *

image_size = 256
batch_size = 1
channels = 3
l1_lambda = 10
n_epochs = 100
save_step = 10
sess = tf.Session()

use_lsgan = True

if use_lsgan:
	gan_criterion = mse_criterion
else:
	gan_criterion = cross_entropy_criterion

dataset_path = '/data'

real_data = tf.placeholder(tf.float32, [None, image_size, image_size, channels+channels],name='real_A_B')

real_A = real_data[:,:,:,:channels]
real_B = real_data[:,:,:,channels:]

fake_B = resnet_generator(real_A, reuse=False, name='generator_A2B')
fake_A = resnet_generator(real_B, reuse=False, name='generator_B2A')

recon_B = resnet_generator(fake_A, reuse=True, name='generator_A2B')
recon_A = resnet_generator(fake_B, reuse=True, name='generator_B2A')

d_predict_fake_A = discriminator(fake_A, reuse=False, name='discriminator_A')
d_predict_fake_B = discriminator(fake_B, reuse=False, name='discriminator_B')

cycle_loss = abs_criterion(recon_A, real_A) + abs_criterion(recon_B, real_B)

g_a2b_loss = gan_criterion(d_predict_fake_B, tf.ones_like(d_predict_fake_B)) + l1_lambda * cycle_loss
g_b2a_loss = gan_criterion(d_predict_fake_A, tf.ones_like(d_predict_fake_A)) + l1_lambda * cycle_loss
g_loss = g_a2b_loss + g_b2a_loss - (l1_lambda*cycle_loss)

d_predict_real_A = discriminator(real_A, reuse=True, name='discriminator_A')
d_predict_real_B = discriminator(real_B, reuse=True, name='discriminator_B')

fake_A_sample = tf.placeholder(tf.float32, [None, image_size, image_size, channels], name='fake_A_sample')
fake_B_sample = tf.placeholder(tf.float32, [None, image_size, image_size, channels], name='fake_B_sample')

d_predict_fake_A_sample = discriminator(fake_A_sample, reuse=True, name='discriminator_A')
d_predict_fake_B_sample = discriminator(fake_A_sample, reuse=True, name='discriminator_B')

d_a_labels = tf.concat([tf.ones_like(d_predict_real_A), tf.zeros_like(d_predict_fake_A_sample)], axis=-1)
d_a_loss = gan_criterion(tf.concat([d_predict_real_A, d_predict_fake_A_sample], axis=-1), d_a_labels)

d_b_labels = tf.concat([tf.ones_like(d_predict_real_B), tf.zeros_like(d_predict_fake_B_sample)], axis=-1)
d_b_loss = gan_criterion(tf.concat([d_predict_real_B, d_predict_fake_B_sample], axis=-1), d_b_labels)

d_loss = d_a_loss + d_b_loss

d_vars = [var for var in tf.trainable_vars() if 'discriminator' in var.name]
g_vars = [var for var in tf.trainable_vars() if 'generator' in var.name]

