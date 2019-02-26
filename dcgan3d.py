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
z_dim = 512
batch_size = 4
n_epochs = 101
save_step = 5
gan_criterion = mse_criterion
test = args.test
load_model_path = '/output/saved_models'
load_epoch = 100
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/output/logs')

sess = tf.Session()
	
class InstanceNormalization3D(tf.keras.layers.Layer):
	def __init__(self):
		super(InstanceNormalization3D, self).__init__()

	def build(self, input_shape):
		depth = int(input_shape[-1])
		self.scale = self.add_variable('scale', shape=[depth], initializer=tf.random_normal_initializer(1.0,0.02,dtype=tf.float32))
		self.offset = self.add_variable('offset', shape=[depth], initializer=tf.constant_initializer(0.0))

	def call(self, inputs):
		mean, variance = tf.nn.moments(inputs, axes=[1,2,3], keep_dims=True)
		inv = tf.rsqrt(variance)
		normalized = (inputs-mean)*inv
		return self.scale*normalized + self.offset


def conv_block3d(inputs, filters, kernel_size=4, strides=(2,2,2), padding='same',
	has_norm_layer=True, has_activation_layer=True, use_instance_norm=True, use_leaky_relu=False):

	x = tf.keras.layers.Conv3D(
		filters=filters,
		kernel_size=kernel_size,
		strides=strides,
		padding=padding,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
		)(inputs)

	if has_norm_layer:
		if use_instance_norm:
			x = InstanceNormalization3D()(x)
		else:
			x = tf.keras.layers.BatchNormalization(
				momemtum=0.9,
				epsilon=1e-5
				)(x)

	if has_activation_layer:
		if use_leaky_relu:
			x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
		else:
			x = tf.keras.layers.ReLU()(x)

	return x


def up_block3d(inputs, filters, kernel_size=3, strides=2, use_conv2d_transpose=True, use_instance_norm=True):

	x = tf.keras.layers.Conv3DTranspose(
		filters=filters,
		kernel_size=kernel_size, 
		strides=(strides,strides,strides),
		padding='same',
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
		)(inputs)

	if use_instance_norm:
			x = InstanceNormalization3D()(x)
	else:
		x = tf.keras.layers.BatchNormalization(
			momemtum=0.9,
			epsilon=1e-5
			)(x)

	x = tf.keras.layers.ReLU()(x)

	return x

def create_generator3d(input_shape, out_channels=3, name='generator'):

	inputs = tf.keras.layers.Input(shape=input_shape)

	project = tf.keras.layers.Dense(4*4*4*32)(inputs)

	h = tf.keras.layers.Reshape((4, 4, 4, 32))(project)

	h0 = up_block3d(h, 16, 3, strides=4)
	h1 = up_block3d(h0, 8, 3, strides=4)
	h2 = up_block3d(h1, 4, 3, strides=2)
	# h3 = up_block(h2, 8, 3, strides=2)
	pred = tf.keras.layers.Conv3DTranspose(out_channels, 3, strides=(2,2,2), padding='same', activation='tanh')(h2)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)


def create_discriminator3d(img_shape, nf=64, n_hidden_layers=3, reuse=False, name='discriminator'):

	inputs = tf.keras.layers.Input(shape=img_shape)

	x = conv_block3d(inputs, nf, strides=(2,2,2), has_norm_layer=False, use_leaky_relu=True)

	for i in range(n_hidden_layers):
		n_filters = nf*(2**(i+1))
		x = conv_block3d(x, n_filters, strides=(2,2,2), use_leaky_relu=True)

	pred = tf.keras.layers.Conv3D(1, 3, strides=(1,1,1), padding='same', activation='sigmoid')(x)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)


z_input_shape = (z_dim,)
img_input_shape = (image_size, image_size, image_size, channels)

z = tf.keras.layers.Input(shape=z_input_shape, name='z')

generator = create_generator3d(z_input_shape, out_channels=channels, name='generator')
discriminator = create_discriminator3d(img_input_shape, nf=32, name='discriminator')

adam = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

discriminator.compile(optimizer=adam, loss=gan_criterion)

frozen_discriminator = tf.keras.models.Model(discriminator.inputs, discriminator.outputs)
frozen_discriminator.trainable = False

fake_image = generator(z)
d_pred = frozen_discriminator(fake_image)

combined = tf.keras.models.Model(inputs=[z], outputs=[d_pred])
combined.compile(optimizer=adam, loss=gan_criterion)

tfrecords_filename = '/data/tfrecords/tumor.tfrecords'

dset = tf.data.TFRecordDataset(tfrecords_filename, compression_type='GZIP').map(lambda x : parse_mri_record(x, only_slice=False), num_parallel_calls=batch_size)
dset = dset.shuffle(batch_size).batch(batch_size)
dset_iter = dset.make_initializable_iterator()
dset_next = dset_iter.get_next()

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(discriminator.summary())
print(generator.summary())

if not test:

	for e in range(n_epochs):

		print('Epoch ',e)
		start = time.time()
		count = 0

		sess.run(dset_iter.initializer)

		while True:

			print(count)
			count+=1
			print(time.time()-start)

			try:
				batch_real = sess.run(dset_next)
			except tf.errors.OutOfRangeError:
				break

			if len(batch_real)!=batch_size or count==15:
				break

			batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

			fakes = generator.predict(batch_z)
			label_shape = [batch_size]+list(discriminator.output_shape[1:])
			labels = tf.concat([tf.ones(label_shape), tf.zeros(label_shape)],axis=0)
			discriminator.train_on_batch(x=tf.concat([batch_real, fakes],axis=0), y=labels)

			combined.train_on_batch(x=batch_z, y=tf.ones(label_shape))

		print(time.time()-start)

		if e%save_step==0:
			discriminator.save(os.path.join('/output/saved_models', 'discriminator_{}.h5'.format(e)))
			generator.save(os.path.join('/output/saved_models', 'generator_{}.h5'.format(e)))

			for i in range(5):
				batch_test = np.random.uniform(-1, 1, size=(batch_size , z_dim))

				fake = generator.predict(batch_test)

				save_mri_image(fake[0], '/output/generated/{}.nii.gz'.format(i))
else:

	latest_model = sorted(glob.glob(os.path.join(load_model_path, 'generator*')), key=lambda x: int(x.split('_')[-1]))[-1]

	generator.load_weights(os.path.join(load_model_path, latest_model))

	for i in range(10):
		batch_x = np.random.uniform(-1, 1, size=(batch_size , z_dim))

		fake = generator.predict(batch_x)

		img = Image.fromarray(denormalize(fake[0]).reshape((256,256)))
		img.save('/output/generated/{}.jpg'.format(i))
