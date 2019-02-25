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
checkpoint_dir = '/output/saved_models'

sess = tf.Session()

def create_generator(z, z_dim=100, channels=3, reuse=False, name='generator'):

	with tf.variable_scope(name, reuse=reuse):

		project = tf.layers.dense(z, 16*16*128)

		h = tf.reshape(project, [-1, 16, 16, 128])

		h0 = up_block(h, 64, 3, strides=2, name='h0')
		h1 = up_block(h0, 32, 3, strides=2, name='h1')
		h2 = up_block(h1, 16, 3, strides=2, name='h2')
		# h3 = up_block(h2, 16, 3, strides=2, name='h3')
		out = tf.nn.tanh(tf.layers.conv2d_transpose(h2, channels, 3, strides=(2,2), padding='same'))

		return out


z = tf.placeholder(tf.float32, [None, z_dim], name='z')
fake = tf.placeholder(tf.float32, [None, image_size, image_size, channels], name='fake')
real = tf.placeholder(tf.float32, [None, image_size, image_size, channels], name='real')

g_image = create_generator(z, z_dim=z_dim, channels=channels, reuse=False, name='generator')

d_pred_real = discriminator(real, reuse=False, name='discriminator')
d_pred_generator = discriminator(g_image, reuse=True, name='discriminator')
d_pred_fake = discriminator(fake, reuse=True, name='discriminator')

d_real_loss = gan_criterion(d_pred_real, tf.ones_like(d_pred_real))
d_fake_loss = gan_criterion(d_pred_fake, tf.zeros_like(d_pred_fake))

d_loss = d_fake_loss + d_real_loss

g_loss = gan_criterion(d_pred_generator, tf.ones_like(d_pred_generator))

d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

print(len(d_vars))
print(len(g_vars))

d_optim = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_vars)

tfrecords_filename = '/data/tfrecords/tumor.tfrecords'

dset = tf.data.TFRecordDataset(tfrecords_filename, compression_type='GZIP').map(parse_mri_record, num_parallel_calls=batch_size)
dset = dset.shuffle(1).batch(batch_size)
dset_iter = dset.make_initializable_iterator()
dset_next = dset_iter.get_next()

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
sess.run(init_op)

if not test:

	for e in range(n_epochs):

		print('Epoch ',e)
		start = time.time()

		sess.run(dset_iter.initializer)

		while True:
			try:
				batch_real = sess.run(dset_next)
			except tf.errors.OutOfRangeError:
				break

			batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

			_, fake_image = sess.run([g_optim, g_image], feed_dict={z: batch_z})

			sess.run(d_optim, feed_dict={real: batch_real, fake: fake_image})

		print(time.time()-start)

		if e%save_step==0:
			saver.save(sess, os.path.join('/output/saved_models', 'cyclegan.model'), global_step=e)

else:

	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

	print('model at {} loaded'.format(ckpt_name))

	saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

	test = tf.placeholder(tf.float32, [None, z_dim], name='test')

	g_image = create_generator(test, z_dim=z_dim, channels=channels, reuse=True, name='generator')

	for i in range(10):
		batch_x = np.random.uniform(-1, 1, size=(batch_size , z_dim))

		fake = sess.run(g_image, feed_dict={test: batch_x})

		img = Image.fromarray(denormalize(fake[0]).reshape((256,256)))
		img.save('/output/generated/{}.jpg'.format(i))

		# save_mri_image(fake[0], '/output/generated/{}.nii.gz'.format(i))

