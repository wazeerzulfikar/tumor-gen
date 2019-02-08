import tensorflow as tf
import numpy as np
import os
import time
from models import *
from losses import *
from utils import *

image_size = 128
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

d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

### train

d_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_vars)
g_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_vars)

init_op = tf.global_variables_initializer()

sess.run(init_op)
saver = tf.train.Saver()

writer = tf.summary.FileWriter('/output/logs', sess.graph)

trainA = normalize(load_dataset(os.path.join(dataset_path, 'trainA'), image_size))
trainB = normalize(load_dataset(os.path.join(dataset_path, 'trainB'), image_size))
testA = load_dataset(os.path.join(dataset_path, 'testA'), image_size)
testB = load_dataset(os.path.join(dataset_path, 'testB'), image_size)

print('Number of training samples are {} and {}'.format(len(trainA), len(trainB)))
print('Number of testing samples are {} and {}'.format(len(testA), len(testB)))

train_A_generator = batch_generator(trainA, batch_size)
train_B_generator = batch_generator(trainB, batch_size)

n_batches = min(len(trainA), len(trainB))//batch_size
pool = ImagePool()


for epoch in range(n_epochs):
	start = time.time()

	for batch in range(n_batches):
		# print('batch ',batch)
		batch_A = next(train_A_generator)
		batch_B = next(train_B_generator)

		batch_A_B = np.concatenate((batch_A, batch_B), axis=3)

		fake_A_image, fake_B_image , _ = sess.run(
			[fake_A, fake_B, g_optimizer],
			feed_dict={real_data: batch_A_B})

		[fake_A_image, fake_B_image] = pool([fake_A_image, fake_B_image])

		_ = sess.run(
			[d_optimizer],
			feed_dict={real_data: batch_A_B, fake_A_sample: fake_A_image, fake_B_sample: fake_B_image})

	print('Epoch {}, time taken {}'.format(epoch, time.time()-start))

	if epoch%save_step==0:
		saver.save(sess, os.path.join('/output/saved_models', 'cyclegan.model'), global_step=epoch)
		fake_A_image, fake_B_image = sess.run(
			[fake_A, fake_B],
			feed_dict={real_data: batch_A_B})
		save_generator_output(batch_A_B[:,:,:,:channels], fake_B_image, epoch=epoch, direction='A2B')
		save_generator_output(batch_A_B[:,:,:,channels:], fake_A_image, epoch=epoch, direction='B2A')



