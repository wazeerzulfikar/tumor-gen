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

print(args.test)

image_size = 256
batch_size = 1
channels = 1
l1_lambda = 10
n_epochs = 301
save_step = 5
continue_train = False
test = args.test
checkpoint_dir = '/output/saved_models'

tf_records_filename_trainA = '/data/tfrecords/no-tumor.tfrecords'
tf_records_filename_trainB = '/data/tfrecords/tumor.tfrecords'

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
sess = tf.Session(config=tfconfig)

use_lsgan = True

if use_lsgan:
	gan_criterion = mse_criterion
else:
	gan_criterion = cross_entropy_criterion

adam = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

img_input_shape = (image_size, image_size, channels)

discriminator_A = create_discriminator(img_input_shape, nf=32, name='discriminator_A')
discriminator_B = create_discriminator(img_input_shape, nf=32, name='discriminator_B')

generator_A2B = resnet_generator(img_input_shape, channels=channels, name='generator_A2B')
generator_B2A = resnet_generator(img_input_shape, channels=channels, name='generator_B2A')

# Discriminator Stuff

discriminator_A.compile(optimizer=adam, loss=gan_criterion)
discriminator_B.compile(optimizer=adam, loss=gan_criterion)

# Generator Stuff

real_A = tf.keras.layers.Input(shape=img_input_shape, name='real_A')
real_B = tf.keras.layers.Input(shape=img_input_shape, name='real_B')

fake_B = generator_A2B(real_A)
fake_A = generator_B2A(real_B)

recon_B = generator_A2B(fake_A)
recon_A = generator_B2A(fake_B)

frozen_discriminator_A = tf.keras.models.Model(inputs=discriminator_A.inputs, outputs=discriminator_A.outputs)
frozen_discriminator_A.trainable = False

frozen_discriminator_B = tf.keras.models.Model(inputs=discriminator_B.inputs, outputs=discriminator_B.outputs)
frozen_discriminator_B.trainable = False

pred_A = frozen_discriminator_A(fake_A)
pred_B = frozen_discriminator_B(fake_B)

reconstructions = tf.keras.layers.Concatenate(axis=0, name='reconstructions')([recon_A, recon_B])
d_predictions = tf.keras.layers.Concatenate(axis=0, name='d_predictions')([pred_A, pred_B])

generator_trainer = tf.keras.models.Model(inputs=[real_A, real_B], outputs=[reconstructions, d_predictions])

losses = {
	'reconstructions' : cycle_loss,
	'd_predictions' : gan_criterion
}

loss_weights = {
	'reconstructions' : 10.0,
	'd_predictions' : 1.0
}

generator_trainer.compile(optimizer=adam, loss=losses, loss_weights=loss_weights)

pool = ImagePool()

trainA = tf.data.TFRecordDataset(tf_records_filename_trainA, compression_type='GZIP').map(parse_mri_record, num_parallel_calls=batch_size)
trainA = trainA.batch(batch_size)
trainA_iter = trainA.make_initializable_iterator()

trainB = tf.data.TFRecordDataset(tf_records_filename_trainB, compression_type='GZIP').map(parse_mri_record, num_parallel_calls=batch_size)
trainB = trainB.batch(batch_size)
trainB_iter = trainB.make_initializable_iterator()

trainA_next = trainA_iter.get_next()
trainB_next = trainB_iter.get_next()

if not test:

	for epoch in range(n_epochs):
		start = time.time()

		sess.run(trainA_iter.initializer)
		sess.run(trainB_iter.initializer)
		count = 0

		while True:

			count +=1
			print(count)
			print(time.time()-start)

			try:
				batch_A = sess.run(trainA_next)
				batch_B = sess.run(trainB_next)
			except tf.errors.OutOfRangeError:
				# Epoch is over
				break

			fake_A_image = generator_A2B.predict(batch_A)
			fake_B_image = generator_B2A.predict(batch_B)
			print('predict', time.time()-start)

			[fake_A_image, fake_B_image] = pool([fake_A_image, fake_B_image])

			label_shape = [batch_size]+list(discriminator_A.output_shape[1:])
			labels = tf.concat([tf.ones(label_shape), tf.zeros(label_shape)],axis=0)

			d_A_loss = discriminator_A.train_on_batch(x=tf.concat([batch_A, fake_A_image], axis=0), y=labels)
			d_B_loss = discriminator_B.train_on_batch(x=tf.concat([batch_B, fake_B_image], axis=0), y=labels)

			print('d A loss ', d_A_loss)
			print('d B loss ', d_B_loss)

			g_loss = generator_trainer.train_on_batch(x=[batch_A, batch_B], 
				y=[tf.concat([batch_A, batch_B], axis=0), tf.concat([tf.ones(label_shape), tf.ones(label_shape)], axis=0)])

			print('g loss ', g_loss)

		print('Epoch {}, time taken {}'.format(epoch, time.time()-start))

		if epoch%save_step==0:
			saver.save(sess, os.path.join('/output/saved_models', 'cyclegan.model'), global_step=epoch)
			# test_idx = np.random.choice(min(len(testA),len(testB)), size=16, replace=False)
			# test_batch = norm(np.concatenate((testA[test_idx],testB[test_idx]), axis=3))
			# fake_A_image, fake_B_image = sess.run(
			# 	[fake_A, fake_B],
			# 	feed_dict={real_data: test_batch})
			# save_generator_output(test_batch[:,:,:,:channels], fake_B_image, epoch=epoch, direction='A2B')
			# save_generator_output(test_batch[:,:,:,channels:], fake_A_image, epoch=epoch, direction='B2A')

			# save_generator_output(batch_A_B[:,:,:,:channels], fake_B_image, epoch=epoch, direction='A2B')
			# save_generator_output(batch_A_B[:,:,:,channels:], fake_A_image, epoch=epoch, direction='B2A')

else:

	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

	print('model at {} loaded'.format(ckpt_name))

	saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

	test_A = tf.placeholder(tf.float32, [None, image_size, image_size, channels], name='test_A')
	test_B = tf.placeholder(tf.float32, [None, image_size, image_size, channels], name='test_B')

	output_test_B = resnet_generator(test_A, channels=channels, reuse=True, name='generator_A2B')
	output_test_A = resnet_generator(test_B, channels=channels, reuse=True, name='generator_B2A')

	sess.run(trainA_iter.initializer)
	sess.run(trainB_iter.initializer)

	for i in range(5):
		batch_A = sess.run(trainA_next)
		batch_B = sess.run(trainB_next)

		fake_B_image = sess.run(output_test_B, feed_dict={test_A: batch_A})
		fake_A_image = sess.run(output_test_A, feed_dict={test_B: batch_B})

		save_mri_image(batch_A[0], '/output/generated/{}_real_A.nii.gz'.format(i))
		save_mri_image(fake_B_image[0], '/output/generated/{}_fake_B_image.nii.gz'.format(i))

		save_mri_image(batch_B[0], '/output/generated/{}_real_B.nii.gz'.format(i))
		save_mri_image(fake_A_image[0], '/output/generated/{}_fake_A_image.nii.gz'.format(i))

		print(i,' Done')





