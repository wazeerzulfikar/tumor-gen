import tensorflow as tf
import os
import random
import numpy as np
import glob
from tqdm import tqdm
from utils import *

image_size = 256

# dataset_path = '../sample_datasets/horse2zebra/trainA'
# filenames = glob.glob(os.path.join(dataset_path, '*.jpg'))

# def _bytes_feature(value):
# 	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _int64_feature(value):
# 	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'horse2zebra.tfrecords'

# writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# for f in tqdm(filenames):

# 	img = read_image(f, image_size)
# 	img_raw = img.tostring()

# 	example = tf.train.Example(features=tf.train.Features(feature={
# 		'image_size': _int64_feature(image_size),
# 		'image_raw': _bytes_feature(img_raw)
# 		}))

# 	writer.write(example.SerializeToString())

# writer.close()

recon_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for record in record_iterator:

	example = tf.train.Example()
	example.ParseFromString(record)

	img_string = example.features.feature['image_raw'].bytes_list.value[0]
	img = np.fromstring(img_string, dtype=np.uint8).reshape((image_size, image_size, 3))

	plt.imshow(img)
	plt.show()
	