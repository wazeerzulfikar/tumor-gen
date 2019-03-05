from models import *
from utils import *
from ops import *
from losses import *
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true', help='Run test script')
args = parser.parse_args()

print(args.test)

image_size = 256
channels = 1
z_dim = 256
batch_size = 16
n_epochs = 301
save_step = 20
gan_criterion = mse_criterion
test = args.test
load_model_path = '/output/saved_models'
load_epoch = 100
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/output/logs')

sess = tf.keras.backend.get_session()

def create_generator(input_shape, out_channels=3, name='generator'):

	inputs = tf.keras.layers.Input(shape=input_shape)

	project = tf.keras.layers.Dense(8*8*256, activation='tanh')(inputs)

	h = tf.keras.layers.Reshape((8, 8, 256))(project)

	h0 = up_block(h, 128, 5, strides=2)
	h1 = up_block(h0, 64, 5, strides=2)
	h2 = up_block(h1, 32, 5, strides=2)
	h3 = up_block(h2, 16, 5, strides=2)
	pred = tf.keras.layers.Conv2DTranspose(out_channels, 5, strides=(2,2), padding='same', activation='tanh')(h3)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)

z_input_shape = (z_dim,)
img_input_shape = (image_size, image_size, channels)

z = tf.keras.layers.Input(shape=z_input_shape, name='z')
real = tf.keras.layers.Input(shape=img_input_shape, name='real')

generator = create_generator(z_input_shape, out_channels=channels, name='generator')
discriminator = create_discriminator(img_input_shape, nf=32, name='discriminator')

# combined = tf.keras.models.Model(generator.inputs, discriminator.outputs)
combined = tf.keras.models.Sequential()
combined.add(generator)
combined.add(discriminator)

fake = generator(z)
d_pred_fake = discriminator(fake)
# d_pred_fake = combined(z)
d_pred_real = discriminator(real)

d_loss = gan_criterion(tf.concat([tf.ones_like(d_pred_real), tf.zeros_like(d_pred_fake)],0), tf.concat([d_pred_real, d_pred_fake],0))
g_loss = gan_criterion(tf.ones_like(d_pred_fake), d_pred_fake)

d_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=discriminator.trainable_variables)
g_optimizer = tf.train.AdamOptimizer(learning_rate=4e-4, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=generator.trainable_variables)

# d_grad = d_optimizer.compute_gradients(d_loss, discriminator.trainable_weights)
# g_grad = g_optimizer.compute_gradients(g_loss, generator.trainable_weights)
# d_update = d_optimizer.apply_gradients(d_grad)
# g_update = g_optimizer.apply_gradients(g_grad)

print(len(discriminator.trainable_variables))
print(len(generator.trainable_variables))
# print(generator.trainable_variables)
print('='*100)
# print(discriminator.trainable_variables)

def get_internal_updates(model):
    # get all internal update ops (like moving averages) of a model
    inbound_nodes = model.inbound_nodes
    input_tensors = []
    for ibn in inbound_nodes:
        input_tensors+= ibn.input_tensors
    updates = [model.get_updates_for(i) for i in input_tensors]
    return updates

other_parameter_updates = [get_internal_updates(m) for m in [generator, discriminator]]

train_step = [g_optimizer, d_optimizer]
losses = [d_loss, g_loss]

tfrecords_filename = '/data/tfrecords/tumor.tfrecords'

dset = tf.data.TFRecordDataset(tfrecords_filename, compression_type='GZIP').map(parse_mri_record, num_parallel_calls=batch_size)
dset = dset.shuffle(10*batch_size).batch(batch_size)
dset_iter = dset.make_initializable_iterator()
dset_next = dset_iter.get_next()

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(discriminator.summary())
print(generator.summary())
generator.compile(loss='mae',optimizer='adam')

x_batch = np.random.normal(0, 1, size=(batch_size , z_dim))
learning_phase = tf.keras.backend.learning_phase()

if not test:

	for e in range(n_epochs):

		print('Epoch ',e)
		start = time.time()
		count = 0

		sess.run(dset_iter.initializer)

		firsta = generator.predict(x_batch)
		# print(firsta[0])
		# batch_real = sess.run(dset_next)

		while True:
			# print(count)
			# count+=1

			try:
				batch_real = sess.run(dset_next)
			except tf.errors.OutOfRangeError:
				break

			if len(batch_real)!=batch_size:
				break

			batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

			# label_shape = [batch_size]+list(discriminator.output_shape[1:])
			# labels = tf.concat([tf.ones(label_shape), tf.zeros(label_shape)],axis=0)

			# if count%1==0:
			# 	fakes = generator.predict(batch_z)
			# 	d_loss = discriminator.train_on_batch(x=tf.concat([batch_real, fakes],axis=0), y=labels)
			# 	print('d loss ', d_loss)

			# g_loss = combined.train_on_batch(x=batch_z, y=tf.ones(label_shape))

			# print('g loss', g_loss)

			_, loss_ = sess.run([train_step, losses], 
				feed_dict={z: batch_z, real: batch_real, learning_phase: True})
			# print(loss_)
			# break


		# print(loss_)
		# fake = generator.predict(x_batch)
		# print('-'*100)
		# print(fake[0])
		# print('-'*100)
		# print(firsta[0]==fake[0])
		print(time.time()-start)

		for i in range(10):
			batch_test = np.random.uniform(-1, 1, size=(batch_size , z_dim))
			# print(batch_test)

			fake_image = generator.predict(batch_test)

			img = Image.fromarray(denormalize(fake_image[0]).reshape((256,256)))
			img.save('/output/generated/{}.jpg'.format(i))


		if e%save_step==0:
			discriminator.save(os.path.join('/output/saved_models', 'discriminator_{}.h5'.format(e)))
			generator.save(os.path.join('/output/saved_models', 'generator_{}.h5'.format(e)))

			
else:

	latest_model = sorted(glob.glob(os.path.join(load_model_path, 'generator*')), key=lambda x: int(x.replace('.h5','').split('_')[-1]))[-1]

	generator.load_weights(os.path.join(load_model_path, latest_model))

	for i in range(10):
		batch_x = np.random.uniform(-1, 1, size=(batch_size , z_dim))

		fake = generator.predict(batch_x)

		img = Image.fromarray(denormalize(fake[0]).reshape((256,256)))
		img.save('/output/generated/a_{}.jpg'.format(i))
