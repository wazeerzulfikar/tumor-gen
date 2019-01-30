from utils import *
from train import generators, trainers, discriminators
import random
import numpy as np
# import matplotlib.pyplot as plt
import time
from keras.models import load_model

image_size = 256
batch_size = 8
n_epochs = 201
save_step = 40

trainA = load_dataset('/horse2zebra/trainA', image_size)
trainB = load_dataset('/horse2zebra/trainB', image_size)
testA = load_dataset('/horse2zebra/testA', image_size)
testB = load_dataset('/horse2zebra/testB', image_size)

print('Number of training samples are {} and {}'.format(len(trainA), len(trainB)))
print('Number of testing samples are {} and {}'.format(len(testA), len(testB)))

generator_trainer, discriminator_A_trainer, discriminator_B_trainer = trainers
generator_A2B, generator_B2A = generators
discriminator_A, discriminator_B = discriminators

print(generator_trainer)
print(generator_A2B)

def batch_generator(data, batch_size=16):
	while True:
		random.shuffle(data)
		for i in range(0,len(data)-batch_size,batch_size):
			batch = data[i:i+batch_size]
			yield batch

train_A_generator = batch_generator(trainA,batch_size)
train_B_generator = batch_generator(trainB,batch_size)

dummy_labels = np.zeros((batch_size, 1))

for epoch in range(n_epochs):
	print('Epoch {}'.format(epoch+1))
	start=time.time()

	for i in range(len(trainA)//batch_size):

		real_A = next(train_A_generator)
		real_B = next(train_B_generator)

		fake_A = generator_A2B.predict(real_B)
		fake_B = generator_B2A.predict(real_A)

		# plt.imshow(fake_A[0])
		# plt.show()

		generator_trainer.train_on_batch([real_A, real_B], dummy_labels)

		discriminator_A_trainer.train_on_batch([real_A, fake_A], dummy_labels)

		discriminator_B_trainer.train_on_batch([real_B, fake_B], dummy_labels)

		# print('Batch done')

	print('Time taken : '+str(time.time()-start))

	if epoch>0 and epoch%save_step==0:
		generator_A2B.save('/saved_models/generator_A2B_{}.h5'.format(epoch))
		generator_B2A.save('/saved_models/generator_B2A_{}.h5'.format(epoch))
		discriminator_A.save('/saved_models/discriminator_A_{}.h5'.format(epoch))
		discriminator_B.save('/saved_models/discriminator_B_{}.h5'.format(epoch))
