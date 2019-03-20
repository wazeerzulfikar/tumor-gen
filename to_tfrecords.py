import tensorflow as tf
import os
import random
import numpy as np
import glob
# from tqdm import tqdm
import nibabel as nib
from utils import *
from PIL import Image

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


dataset_path = '/data/no-tumor-conformed'
save_path = '/data/tfrecords/no-tumor.tfrecords'

filenames = glob.glob(os.path.join(dataset_path,'*.nii.gz'))

options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)

with tf.python_io.TFRecordWriter(save_path, options=options) as writer:

	for e,f in enumerate(filenames):
		if 'label' in f:
			continue
		print(e)
		img = nib.load(f)
		img = nib.as_closest_canonical(img).get_fdata(caching='unchanged', dtype=np.float32)
		feature = {
			'volume': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.ravel().tostring()]))
		}
		mri_example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(mri_example.SerializeToString())
