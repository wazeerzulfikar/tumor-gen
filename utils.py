import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(path, image_size=32):
	images = []

	files = glob.glob(os.path.join(path,'*.jpg'))

	for f in files:
		img = cv2.imread(f)
		img = cv2.cvtColor(cv2.resize(img, (image_size, image_size)), cv2.COLOR_BGR2RGB)
		images.append(img)

	return np.array(images)

def plot_batch(batch):

	n_elements = len(batch)
	nrows = int(sqrt(n_elements))
	ncols = n_elements//nrows
	fig, axarr = plt.subplots(nrows=nrows, ncols=ncols)

	for i in range(n_rows):
		for j in range(n_cols):
			axarr[i,j].imshow(batch[i*ncols+j])

	plt.show()
