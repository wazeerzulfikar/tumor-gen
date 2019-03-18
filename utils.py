from PIL import Image
import glob
import os
import numpy as np
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
import copy


def normalize(img):
	return tf.divide(tf.subtract(img,tf.constant(127.5)),tf.constant(127.5))

def norm(img):
	return (img-127.5)/127.5

def z_score_normalize(img):
	mean, variance = tf.nn.moments(img, axes=[1,2], keep_dims=True)
	stddev = tf.sqrt(variance)
	return (img-mean)/stddev

def denormalize(img):
	return ((img*127.5)+127.5).clip(0,255).astype('uint8')


def read_image(path, image_size=32):
	img = Image.open(path).convert('RGB')
	img = img.resize((image_size, image_size))
	return np.array(img)

def read_both_images(path, image_size=32):
	img_A = normalize(read_image(path[0], image_size))
	img_B = normalize(read_image(path[1], image_size))

	return np.concatenate((img_A, img_B), axis=2)


def load_dataset(path, image_size=32):
	images = []

	files = glob.glob(os.path.join(path,'*.jpg'))

	for f in files:

		img = read_image(f, image_size)

		if np.random.randint(0,2):
			img = np.fliplr(img)

		images.append(img)

	return np.array(images)


def plot_batch(batch):
	n_elements = len(batch)
	nrows = int(math.sqrt(n_elements))
	ncols = n_elements//nrows
	fig, axarr = plt.subplots(nrows=nrows, ncols=ncols)

	for i in range(nrows):
		for j in range(ncols):
			axarr[i,j].imshow(batch[i*ncols+j])

	plt.show()


def save_generator_output(batch_x, batch_y, epoch='', direction='A2B',path='/output/generated'):

	image_size = batch_x.shape[1]

	batch_x = denormalize(batch_x)
	batch_y = denormalize(batch_y)

	output_array = np.vstack([np.hstack((img1, img2)) for img1, img2 in zip(batch_x, batch_y)])
	output = Image.fromarray(output_array)
	output.save(os.path.join(path,'generated_{}_{}_{}.jpg').format(direction, image_size, epoch))


def batch_generator(data, batch_size=16):
	while True:
		random.shuffle(data)
		for i in range(0,len(data)-batch_size,batch_size):
			batch = data[i:i+batch_size]
			yield batch


class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def parse_record(data_record):

	features = {
		'image_size': tf.FixedLenFeature([], tf.int64),
		'image_raw': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)
	}

	sample = tf.parse_single_example(data_record, features)

	image = tf.cast(tf.io.decode_raw(sample['image_raw'], tf.uint8), tf.float32)
	image = normalize(image)
	image = tf.reshape(image, shape=(256,256,3))
	label = tf.cast(sample['label'], tf.int32)

	return image


def parse_mri_record(record, only_slice=True, axis=0, image_size=128):

    features = {
        'volume': tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(record, features=features)
    img = tf.io.decode_raw(example['volume'], tf.float32)
    img = normalize(img)
    volume = tf.reshape(img, shape=(image_size, image_size, image_size))
    if not only_slice:
    	return tf.expand_dims(volume,3)
    slice_index = image_size//2
    if axis==0:
    	return_slice = volume[slice_index, :, :]
    if axis==1:
    	return_slice = volume[:, slice_index, :]
    if axis==2:
    	return_slice = volume[:, :, slice_index]
    return tf.expand_dims(return_slice, 2)


def parse_mri_random_slice(record, only_slice=True, axis=2):

    features = {
        'volume': tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(record, features=features)
    img = tf.io.decode_raw(example['volume'], tf.float32)
    img = normalize(img)
    volume = tf.reshape(img, shape=(256, 256, 256))
    if not only_slice:
    	return tf.expand_dims(volume,3)
    slice_index = int(np.random.uniform(96, 192))
    if axis==0:
    	return_slice = volume[slice_index, :, :]
    if axis==1:
    	return_slice = volume[:, slice_index, :]
    if axis==2:
    	return_slice = volume[:, :, slice_index]
    return tf.expand_dims(return_slice, 2)


def parse_mri_slice(record):

    features = {
        'slice': tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(record, features=features)
    img = tf.io.decode_raw(example['slice'], tf.float32)
    img = normalize(img)
    img = tf.reshape(img, shape=(256,256))
    return tf.expand_dims(img, 2)


def save_mri_image(img_array, filename):

	denormalized_img_array = denormalize(img_array)
	mri = nib.Nifti1Image(denormalized_img_array, np.eye(4))
	nib.save(mri, filename)


