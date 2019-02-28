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
z_dim = 100
batch_size = 4
n_epochs = 301
save_step = 10
gan_criterion = mse_criterion
test = args.test
load_model_path = '/output/saved_models'
load_epoch = 100
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/output/logs')

sess = tf.Session()

def create_generator(input_shape, out_channels=3, name='generator'):

	inputs = tf.keras.layers.Input(shape=input_shape)

	project = tf.keras.layers.Dense(16*16*128)(inputs)

	h = tf.keras.layers.Reshape((16, 16, 128))(project)

	h0 = up_block(h, 64, 3, strides=2)
	h1 = up_block(h0, 32, 3, strides=2)
	h2 = up_block(h1, 16, 3, strides=2)
	# h3 = up_block(h2, 8, 3, strides=2)
	pred = tf.keras.layers.Conv2DTranspose(out_channels, 3, strides=(2,2), padding='same', activation='tanh')(h2)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)

z_input_shape = (z_dim,)
img_input_shape = (image_size, image_size, channels)

z = tf.keras.layers.Input(shape=z_input_shape, name='z')

generator = create_generator(z_input_shape, out_channels=channels, name='generator')
discriminator = create_discriminator(img_input_shape, nf=32, name='discriminator')

adam = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

discriminator.compile(optimizer=adam, loss=gan_criterion)

n_disc_trainable = len(discriminator.trainable_weights)
n_gen_trainable = len(generator.trainable_weights)

frozen_discriminator = tf.keras.models.Model(discriminator.inputs, discriminator.outputs)
frozen_discriminator.trainable = False

fake_image = generator(z)
d_pred = frozen_discriminator(fake_image)

combined = tf.keras.models.Model(inputs=[z], outputs=[d_pred])
combined.compile(optimizer=adam, loss=gan_criterion)

tfrecords_filename = '/data/tfrecords/tumor.tfrecords'

dset = tf.data.TFRecordDataset(tfrecords_filename, compression_type='GZIP').map(parse_mri_record, num_parallel_calls=batch_size)
dset = dset.batch(batch_size)
dset_iter = dset.make_initializable_iterator()
dset_next = dset_iter.get_next()

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(discriminator.summary())
print(generator.summary())

print()

if not test:

	for e in range(n_epochs):

		print('Epoch ',e)
		start = time.time()
		count = 0

		sess.run(dset_iter.initializer)

		while True:

			try:
				batch_real = sess.run(dset_next)
			except tf.errors.OutOfRangeError:
				break

			if len(batch_real)!=batch_size:
				break

			batch_z = np.random.normal(0, 1, size=(batch_size , z_dim))

			label_shape = [batch_size]+list(discriminator.output_shape[1:])
			labels = tf.concat([tf.ones(label_shape), tf.zeros(label_shape)],axis=0)

			if count%1==0:
				fakes = generator.predict(batch_z)
				d_loss = discriminator.train_on_batch(x=tf.concat([batch_real, fakes],axis=0), y=labels)
				print('d loss ', d_loss)

			g_loss = combined.train_on_batch(x=batch_z, y=tf.ones(label_shape))

			print('g loss', g_loss)

			for i in range(10):
				batch_test = np.random.normal(0, 1, size=(batch_size , z_dim))

				fake = generator.predict(batch_test)

				img = Image.fromarray(denormalize(fake[0]).reshape((256,256)))
				img.save('/output/generated/{}.jpg'.format(i))

		print(time.time()-start)

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
