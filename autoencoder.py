import tensorflow as tf

from utils import *
from models import *
from losses import *
from ops import *

n_epochs = 100
batch_size = 1
image_size = 256

def create_model(inputs):

	nf=16

	x = conv_block(inputs, 16, 3, strides=1, n_dims=2, use_leaky_relu=False, has_norm_layer=False)
	x = conv_block(x, 16, 3, strides=2, n_dims=2, use_leaky_relu=False, use_instance_norm=False)

	x = conv_block(x, 32, 3, strides=1, n_dims=2, use_leaky_relu=False, has_norm_layer=False)
	x = conv_block(x, 32, 3, strides=2, n_dims=2, use_leaky_relu=False, use_instance_norm=False)

	x = conv_block(x, 64, 3, strides=1, n_dims=2, use_leaky_relu=False, use_instance_norm=False)
	x = conv_block(x, 64, 3, strides=2, n_dims=2, use_leaky_relu=False, use_instance_norm=False)

	x = conv_block(x, 128, 3, strides=1, n_dims=2, use_leaky_relu=False, use_instance_norm=False)
	x = conv_block(x, 128, 3, strides=2, n_dims=2, use_leaky_relu=False, use_instance_norm=False)

	x = up_block(x, 64, 3, n_dims=2, use_instance_norm=False)
	x = up_block(x, 32, 3, n_dims=2, use_instance_norm=False)
	x = up_block(x, 16, 3, n_dims=2, use_instance_norm=False)
	x = up_block(x, 8, 3, n_dims=2, use_instance_norm=False)

	pred = tf.keras.layers.Conv2D(1, 3, activation='tanh', padding='same')(x)

	return tf.keras.models.Model(inputs=inputs, outputs=pred)

inp = tf.keras.layers.Input(shape=(image_size, image_size, 1))
model = create_model(inp)
model.compile(loss='mae', metrics=['acc'], optimizer='adam')

tf_record_parser = lambda x: (parse_mri_record(x, only_slice=True, image_size=image_size, axis=2),parse_mri_record(x, only_slice=True, image_size=image_size, axis=2)) 
tf_records_filename = '/data/tfrecords/tumor.tfrecords'

dset = tf.data.TFRecordDataset(tf_records_filename, compression_type='GZIP').map(tf_record_parser, num_parallel_calls=batch_size)
dset = dset.shuffle(100).repeat(None).batch(batch_size)
dset_iter = dset.make_one_shot_iterator()
dset_next = dset_iter.get_next()

# model = tf.keras.models.load_model('/output/autoencoder_mae.h5', custom_objects={'ReLU': ReLU})
print(model.summary())

# for e in range(n_epochs):
tboard = tf.keras.callbacks.TensorBoard(log_dir='/output/logs')
model.fit(dset, epochs=10, steps_per_epoch=800, callbacks=[tboard])

model.save('/output/autoencoder_mae.h5')
sess = tf.keras.backend.get_session()
model = tf.keras.models.load_model('/output/autoencoder_mae.h5', custom_objects={'ReLU': ReLU})

for j in range(5):
	test = sess.run(dset_next)[0]
	images = model.predict(test)
	# print(images.shape)
	for e,i in enumerate(images):
		Image.fromarray(np.squeeze(denormalize(test[e]))).save('/output/generated/{}_real.jpg'.format(j))
		Image.fromarray(np.squeeze(denormalize(i))).save('/output/generated/{}.jpg'.format(j))

		# save_mri_image(test[e], '/output/generated/{}_real.nii.gz'.format(j))
		# save_mri_image(i, '/output/generated/{}.nii.gz'.format(j))

