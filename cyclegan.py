import tensorflow as tf
import numpy as np
import os
import time
from models import *
from losses import *
from utils import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true', help='Run test script')
args = parser.parse_args()

image_size = 256
batch_size = 1
channels = 1
l1_lambda = 10
n_epochs = 31
save_step = 1
continue_train = False
learning_rate = 2e-4
epoch_step = 100
load = True
pool = ImagePool()

tf_records_filename_trainA = '/data/tfrecords/no-tumor.tfrecords'
tf_records_filename_trainB = '/data/tfrecords/tumor.tfrecords'

sess = tf.keras.backend.get_session()

use_lsgan = True

if use_lsgan:
	gan_criterion = mse_criterion
else:
	gan_criterion = cross_entropy_criterion

# Model instantiation

img_input_shape = (image_size, image_size, channels)

real_A = tf.keras.layers.Input(shape=img_input_shape, name='real_A')
real_B = tf.keras.layers.Input(shape=img_input_shape, name='real_B')

discriminator_A = create_discriminator(real_A, nf=64, name='discriminator_A')
discriminator_B = create_discriminator(real_B, nf=64, name='discriminator_B')

generator_A2B = resnet_generator(real_A, channels=channels, nf=64, name='generator_A2B')
generator_B2A = resnet_generator(real_B, channels=channels, nf=64, name='generator_B2A')

# Loss function

fake_B = generator_A2B(real_A)
fake_A = generator_B2A(real_B)

recon_B = generator_A2B(fake_A)
recon_A = generator_B2A(fake_B)

pred_A_g = discriminator_A(fake_A)
pred_B_g = discriminator_B(fake_B)

fake_A_sample = tf.keras.layers.Input(shape=img_input_shape, name='fake_A_sample')
fake_B_sample = tf.keras.layers.Input(shape=img_input_shape, name='fake_B_sample')

pred_A_real = discriminator_A(real_A)
pred_B_real = discriminator_B(real_B)

pred_A_fake = discriminator_A(fake_A_sample)
pred_B_fake = discriminator_B(fake_B_sample)

reconstructions = tf.keras.layers.Concatenate(axis=0, name='reconstructions')([recon_A, recon_B])
d_predictions = tf.keras.layers.Concatenate(axis=0, name='d_predictions')([pred_A_g, pred_B_g])
real = tf.keras.layers.Concatenate(axis=0, name='real')([real_A, real_B])

g_loss = gan_criterion(tf.ones_like(d_predictions), d_predictions) + 10*cycle_loss(real, reconstructions)

d_A_loss = gan_criterion(tf.concat([tf.ones_like(pred_A_real), tf.zeros_like(pred_A_fake)],0), tf.concat([pred_A_real, pred_A_fake],0))
d_B_loss = gan_criterion(tf.concat([tf.ones_like(pred_B_real), tf.zeros_like(pred_B_fake)],0), tf.concat([pred_B_real, pred_B_fake],0))
d_loss = 0.5*(d_A_loss+d_B_loss)

lr = tf.placeholder(tf.float32, None, name='learning_rate')

update_g_a2b = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=generator_A2B.trainable_weights)
update_g_b2a = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=generator_B2A.trainable_weights)
update_d_a = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=discriminator_A.trainable_weights)
update_d_b = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=discriminator_B.trainable_weights)

def get_internal_updates(model):
    # get all internal update ops (like moving averages) of a model
    inbound_nodes = model.inbound_nodes
    input_tensors = []
    for ibn in inbound_nodes:
        input_tensors+= ibn.input_tensors
    updates = [model.get_updates_for(i) for i in input_tensors]
    return updates

other_parameter_updates = [get_internal_updates(m) for m in [generator_A2B,generator_B2A,discriminator_A,discriminator_B]]

g_train = [update_g_a2b, update_g_b2a]
d_train = [update_d_a, update_d_b]

pool = ImagePool()

tf_parser = lambda x: parse_mri_record(x, only_slice=True, axis=2)

trainA = tf.data.TFRecordDataset(tf_records_filename_trainA, compression_type='GZIP').map(parse_mri_random_slice, num_parallel_calls=batch_size)
trainA = trainA.shuffle(100).repeat(None).batch(batch_size)
trainA_iter = trainA.make_initializable_iterator()

trainB = tf.data.TFRecordDataset(tf_records_filename_trainB, compression_type='GZIP').map(parse_mri_slice, num_parallel_calls=batch_size)
trainB = trainB.shuffle(100).batch(batch_size)
trainB_iter = trainB.make_initializable_iterator()

trainA_next = trainA_iter.get_next()
trainB_next = trainB_iter.get_next()

testA = tf.data.TFRecordDataset(tf_records_filename_trainA, compression_type='GZIP').map(parse_mri_random_slice, num_parallel_calls=batch_size)
testA = testA.shuffle(64).batch(16)
testA_iter = testA.make_initializable_iterator()

testB = tf.data.TFRecordDataset(tf_records_filename_trainB, compression_type='GZIP').map(parse_mri_slice, num_parallel_calls=batch_size)
testB = testB.shuffle(64).batch(16)
testB_iter = testB.make_initializable_iterator()

testA_next = testA_iter.get_next()
testB_next = testB_iter.get_next()

sess.run(tf.global_variables_initializer())
learning_phase = tf.keras.backend.learning_phase()

if load:
	generator_A2B.load_weights('/output/saved_models/generator_A2B.h5')
	generator_B2A.load_weights('/output/saved_models/generator_B2A.h5')
	discriminator_A.load_weights('/output/saved_models/discriminator_A.h5')
	discriminator_B.load_weights('/output/saved_models/discriminator_B.h5')

for epoch in range(n_epochs):
	start = time.time()

	sess.run(trainA_iter.initializer)
	sess.run(trainB_iter.initializer)

	sess.run(testA_iter.initializer)
	sess.run(testB_iter.initializer)
	count = 0
	loss_d_total = 0
	loss_g_total = 0

	lr_value = learning_rate if epoch < epoch_step else learning_rate*(n_epochs-epoch)/(n_epochs-epoch_step)

	while True:
		try:
			batch_A = sess.run(trainA_next)
			batch_B = sess.run(trainB_next)
			count+=1

		except tf.errors.OutOfRangeError:
			# Epoch is over
			break

		fake_A_image, fake_B_image, _, loss_g = sess.run([fake_A, fake_B, g_train, g_loss], 
			feed_dict={real_A: batch_A, real_B: batch_B, learning_phase: True, lr: lr_value})

		fake_A_image, fake_B_image = pool([fake_A_image, fake_B_image])

		_, loss_d = sess.run([d_train, d_loss], 
			feed_dict={real_A: batch_A, real_B: batch_B, fake_A_sample: fake_A_image,
			fake_B_sample: fake_B_image, learning_phase: True, lr: lr_value})

		loss_d_total+=loss_d
		loss_g_total+=loss_g


	print(loss_d_total/count, loss_g_total/count)

	print('Epoch {}, time taken {}'.format(epoch, time.time()-start))

	batch_testA = sess.run(testA_next)
	batch_testB = sess.run(testB_next)

	fake_A_image, fake_B_image = sess.run([fake_A, fake_B], 
		feed_dict={real_A: batch_testA, real_B: batch_testB, learning_phase: False})

	save_generator_output(np.squeeze(batch_testA), np.squeeze(fake_B_image), direction='A2B')
	save_generator_output(np.squeeze(batch_testB), np.squeeze(fake_A_image), direction='B2A')

	if epoch%save_step==0:
		generator_A2B.save('/output/saved_models/generator_A2B.h5')
		generator_B2A.save('/output/saved_models/generator_B2A.h5')
		discriminator_A.save('/output/saved_models/discriminator_A.h5')
		discriminator_B.save('/output/saved_models/discriminator_B.h5')
		save_generator_output(np.squeeze(batch_testA), np.squeeze(fake_B_image), epoch=epoch, direction='A2B')
		save_generator_output(np.squeeze(batch_testB), np.squeeze(fake_A_image) ,epoch=epoch, direction='B2A')




