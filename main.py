from utils import *
from train import generators, trainers
import random
import numpy as np
import matplotlib.pyplot as plt

image_size = 64
batch_size = 16
n_epochs = 1
save_step = 20


trainA = load_dataset('../sample_datasets/horse2zebra/trainA', image_size)
trainB = load_dataset('../sample_datasets/horse2zebra/trainB', image_size)
testA = load_dataset('../sample_datasets/horse2zebra/testA', image_size)
testB = load_dataset('../sample_datasets/horse2zebra/testB', image_size)

print('Number of training samples are {} and {}'.format(len(trainA), len(trainB)))
print('Number of testing samples are {} and {}'.format(len(testA), len(testB)))

generator_trainer, discriminator_A_trainer, discriminator_B_trainer = trainers
generator_A2B, generator_B2A = generators

print(generator_trainer)
print(generator_A2B)

def batch_generator(data, batch_size=16):
	while True:
		random.shuffle(data)
		for i in range(0,len(data),batch_size):
			batch = data[i:i+batch_size]
			yield batch

train_A_generator = batch_generator(trainA)
train_B_generator = batch_generator(trainB)

dummy_labels = np.zeros((batch_size, 1))

for epoch in range(n_epochs):
	print('Epoch {}'.format(epoch+1))

	for i in range(len(trainA)//batch_size):

		real_A = next(train_A_generator)
		real_B = next(train_B_generator)

		print(real_A.shape)

		fake_A = generator_A2B.predict(real_B)
		fake_B = generator_B2A.predict(real_A)

		# plt.imshow(fake_A[0])
		# plt.show()

		generator_trainer.train_on_batch([real_A, real_B], dummy_labels)

		discriminator_A_trainer.train_on_batch([real_A, fake_A], dummy_labels)

		discriminator_B_trainer.train_on_batch([real_B, fake_B], dummy_labels)

		print('done')

		break

	if epoch%save_step==0:
		generator_A2B.save('saved_models/generator_A2B.h5')
		generator_B2A.save('saved_models/generator_B2A.h5')

	
