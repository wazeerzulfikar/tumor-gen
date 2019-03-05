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
z_dim = 2048
batch_size = 2
n_epochs = 101
save_step = 5
gan_criterion = mse_criterion
test = args.test
load_model_path = '/output/saved_models'
load_epoch = 100
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/output/logs')

sess = tf.keras.backend.get_session()
	
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
	has_norm_layer=True, has_activation_layer=True, use_instance_norm=False, use_leaky_relu=False):

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
				momentum=0.9,
				epsilon=1e-5
				)(x)

	if has_activation_layer:
		if use_leaky_relu:
			x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
		else:
			x = tf.keras.layers.ReLU()(x)

	return x


def up_block3d(inputs, filters, kernel_size=3, strides=2, use_conv2d_transpose=True, use_instance_norm=False):

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
			momentum=0.9,
			epsilon=1e-5
			)(x)

	x = tf.keras.layers.ReLU()(x)

	return x

def create_generator3d(input_shape, out_channels=3, name='generator'):

	inputs = tf.keras.layers.Input(shape=input_shape)

	project = tf.keras.layers.Dense(4*4*4*128)(inputs)

	h = tf.keras.layers.Reshape((4, 4, 4, 128))(project)

	h0 = up_block3d(h, 64, 5, strides=4)
	h1 = up_block3d(h0, 32, 5, strides=2)
	h2 = up_block3d(h1, 16, 5, strides=2)
	h3 = up_block3d(h2, 8, 5, strides=2)
	pred = tf.keras.layers.Conv3DTranspose(out_channels, 5, strides=(2,2,2), padding='same', activation='tanh')(h3)

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
real = tf.keras.layers.Input(shape=img_input_shape, name='real')

generator = create_generator3d(z_input_shape, out_channels=channels, name='generator')
discriminator = create_discriminator3d(img_input_shape, nf=32, name='discriminator')

fake = generator(z)
d_pred_fake = discriminator(fake)
d_pred_real = discriminator(real)

d_loss = gan_criterion(tf.concat([tf.ones_like(d_pred_real), tf.zeros_like(d_pred_fake)],0), tf.concat([d_pred_real, d_pred_fake],0))
g_loss = gan_criterion(tf.ones_like(d_pred_fake), d_pred_fake)

d_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=discriminator.trainable_variables)
g_optimizer = tf.train.AdamOptimizer(learning_rate=2e-3, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=generator.trainable_variables)

# d_grad = d_optimizer.compute_gradients(d_loss, discriminator.trainable_weights)
# g_grad = g_optimizer.compute_gradients(g_loss, generator.trainable_weights)
# d_update = d_optimizer.apply_gradients(d_grad)
# g_update = g_optimizer.apply_gradients(g_grad)

print(len(discriminator.trainable_variables))
print(len(generator.trainable_variables))
print('='*100)

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

dset = tf.data.TFRecordDataset(tfrecords_filename, compression_type='GZIP').map(lambda x: parse_mri_record(x, only_slice=False), num_parallel_calls=batch_size)
dset = dset.shuffle(100).batch(batch_size)
dset_iter = dset.make_initializable_iterator()
dset_next = dset_iter.get_next()

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(discriminator.summary())
print(generator.summary())

learning_phase = tf.keras.backend.learning_phase()

if not test:

	for e in range(n_epochs):

		print('Epoch ',e)
		start = time.time()
		count = 0

		sess.run(dset_iter.initializer)

		while True:
			# print(count)
			count+=1

			try:
				batch_real = sess.run(dset_next)
			except tf.errors.OutOfRangeError:
				break

			if len(batch_real)!=batch_size:
				break

			batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

			_, loss_ = sess.run([train_step, losses], 
				feed_dict={z: batch_z, real: batch_real, learning_phase: True})
			# print(loss_)

		print(time.time()-start)

		for i in range(10):
			batch_test = np.random.uniform(-1, 1, size=(batch_size , z_dim))
			fake_image = sess.run(fake, feed_dict={z: batch_test, learning_phase: False})
			# print(fake_image.shape)

			save_mri_image(fake_image[0],'/output/generated/{}.nii.gz'.format(i))


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
