from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import math


def normalize(img):
	return (img-127.5)/127.5


def denormalize(img):
	return ((img*127.5)+127.5).clip(0,255).astype('uint8')


def load_dataset(path, image_size=32):
	images = []

	files = glob.glob(os.path.join(path,'*.jpg'))

	for f in files:
		img = Image.open(f).convert('RGB')
		img = img.resize((image_size, image_size))
		img = np.array(img)

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


def save_generator_output(batch_x, generator, epoch='', direction='A2B',path='/output/generated'):

	generated_images = generator.predict(normalize(batch_x))

	image_size = batch_x.shape[1]
	batch_y = denormalize(generated_images)

	output_array = np.vstack([np.hstack((img1, img2)) for img1, img2 in zip(batch_x, batch_y)])
	output = Image.fromarray(output_array)
	output.save(os.path.join(path,'generated_{}_{}_{}.jpg').format(direction, image_size, epoch))


def batch_generator(data, batch_size=16):
	while True:
		random.shuffle(data)
		for i in range(0,len(data)-batch_size,batch_size):
			batch = data[i:i+batch_size]
			yield batch


# import nibabel as nib
# def read_mri(file_path):
# 	mri = nib.load(file_path)
# 	data = mr.get_data()
# 	mri.uncache()
# 	return data

