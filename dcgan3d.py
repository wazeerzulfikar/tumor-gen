from models import *
from utils import *
from ops import *
from losses import *
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true', help='Run test script')
args = parser.parse_args()

image_size = 128
channels = 1
z_dim = 1024
batch_size = 1
n_epochs = 101
save_step = 5
gan_criterion = mse_criterion
test = args.test
load_model_path = '/output/saved_models'
load_epoch = 100
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/output/logs')

sess = tf.keras.backend.get_session()

def create_generator3d(inputs, out_channels=3, name='generator'):

	project = tf.keras.layers.Dense(16*16*16*64, activation='relu')(inputs)

	x = tf.keras.layers.Reshape((16, 16, 16, 64))(project)

	x = up_block(x, 32, 5, n_dims=3, strides=2)
	x = up_block(x, 16, 5, n_dims=3, strides=2)
	x = up_block(x, 8, 5, n_dims=3, strides=2)
	pred = tf.keras.layers.Conv3D(out_channels, 3, strides=1, padding='same', activation='tanh')(x)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)


def create_discriminator3d(inputs, nf=64, n_hidden_layers=3, reuse=False, name='discriminator'):

	x = conv_block(inputs, nf, strides=(2,2,2), n_dims=3, has_norm_layer=False, use_leaky_relu=True)

	for i in range(n_hidden_layers):
		n_filters = nf*(2**(i+1))
		x = conv_block(x, n_filters, strides=(2,2,2), n_dims=3, use_leaky_relu=True)

	pred = tf.keras.layers.Conv3D(1, 3, strides=(1,1,1), padding='same', activation='sigmoid')(x)

	return tf.keras.models.Model(inputs=[inputs], outputs=[pred], name=name)


z_input_shape = (z_dim,)
img_input_shape = (image_size, image_size, image_size, channels)

z = tf.keras.layers.Input(shape=z_input_shape, name='z')
real = tf.keras.layers.Input(shape=img_input_shape, name='real')

generator = create_generator3d(z, out_channels=channels, name='generator')
discriminator = create_discriminator3d(real, nf=32, name='discriminator')

fake = generator(z)
d_pred_fake = discriminator(fake)
d_pred_real = discriminator(real)

d_loss = gan_criterion(tf.concat([tf.ones_like(d_pred_real), tf.zeros_like(d_pred_fake)],0), tf.concat([d_pred_real, d_pred_fake],0))
g_loss = gan_criterion(tf.ones_like(d_pred_fake), d_pred_fake)

d_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=discriminator.trainable_variables)
g_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=generator.trainable_variables)

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
losses = [g_loss, d_loss]

tfrecords_filename = '/data/tfrecords/no-tumor-128.tfrecords'

dset = tf.data.TFRecordDataset(tfrecords_filename, compression_type='GZIP').map(lambda x: parse_mri_record(x, only_slice=False), num_parallel_calls=batch_size)
dset = dset.shuffle(100).batch(batch_size)
dset_iter = dset.make_initializable_iterator()
dset_next = dset_iter.get_next()

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(discriminator.summary())
print(generator.summary())

learning_phase = tf.keras.backend.learning_phase()

# discriminator.load_weights(os.path.join('/output', 'discriminator.h5'))
# generator.load_weights(os.path.join('/output', 'generator.h5'))


for e in range(n_epochs):

	start = time.time()
	count = 0

	sess.run(dset_iter.initializer)

	losses_total = [0, 0]

	while True:
		# print(count)
		# print(time.time()-start)

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
		losses_total[0]+=loss_[0]
		losses_total[1]+=loss_[1]

	print(time.time()-start)
	losses_total = [i/count for i in losses_total]
	print('Epoch {} : g loss {}, d loss {}'.format(e, losses_total[0], losses_total[1]))

	for i in range(10):
		batch_test = np.random.uniform(-1, 1, size=(batch_size , z_dim))
		fake_image = sess.run(fake, feed_dict={z: batch_test, learning_phase: False})
		# print(fake_image.shape)

		save_mri_image(fake_image[0],'/output/generated/{}.nii.gz'.format(i))

	discriminator.save(os.path.join('/output', 'discriminator.h5'))
	generator.save(os.path.join('/output', 'generator.h5'))

	with open('/output/logs.txt','a+') as f:
		f.write('Epoch {} : g loss {}, d loss {} \n'.format(e, losses_total[0], losses_total[1]))
			
