import tensorflow as tf
import numpy as np
import os
import time
from models import *
from losses import *
from utils import *

image_size = 256
batch_size = 1
channels = 3
l1_lambda = 10
n_epochs = 101
save_step = 5
tf_records_filename_trainA = '/data/horse2zebra_trainA.tfrecords'
tf_records_filename_trainB = '/data/horse2zebra_trainB.tfrecords'

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
sess = tf.Session(config=tfconfig)

use_lsgan = True

if use_lsgan:
	gan_criterion = mse_criterion
else:
	gan_criterion = cross_entropy_criterion

dataset_path = '/data/horse2zebra'

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

print(len(d_vars))
print(len(g_vars))

### train

d_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_vars)
g_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_vars)

init_op = tf.global_variables_initializer()

sess.run(init_op)
saver = tf.train.Saver()

writer = tf.summary.FileWriter('/output/logs', sess.graph)

testA = load_dataset(os.path.join(dataset_path, 'testA'), image_size)
testB = load_dataset(os.path.join(dataset_path, 'testB'), image_size)

pool = ImagePool()

trainA = tf.data.TFRecordDataset(tf_records_filename_trainA).map(parse_record)
trainA = trainA.shuffle(1000).repeat(None).batch(batch_size)
trainA_iter = trainA.make_one_shot_iterator()

trainB = tf.data.TFRecordDataset(tf_records_filename_trainB).map(parse_record)
trainB = trainB.shuffle(1000).repeat(None).batch(batch_size)
trainB_iter = trainB.make_one_shot_iterator()

trainA_next = trainA_iter.get_next()
trainB_next = trainB_iter.get_next()

for epoch in range(n_epochs):
	start = time.time()

	while True:

		try:
			batch_A = sess.run(trainA_next)[0]
			batch_B = sess.run(trainB_next)[0]
		except tf.errors.OutOfRangeError:
			# Epoch is over
			break

		if(batch_A.shape[3]!=3 or batch_B.shape[3]!=3):
			continue

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
		test_idx = np.random.choice(min(len(testA),len(testB)), size=16, replace=False)
		test_batch = normalize(np.concatenate((testA[test_idx],testB[test_idx]), axis=3))
		fake_A_image, fake_B_image = sess.run(
			[fake_A, fake_B],
			feed_dict={real_data: test_batch})
		save_generator_output(test_batch[:,:,:,:channels], fake_B_image, epoch=epoch, direction='A2B')
		save_generator_output(test_batch[:,:,:,channels:], fake_A_image, epoch=epoch, direction='B2A')



