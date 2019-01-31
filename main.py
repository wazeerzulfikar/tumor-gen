from utils import *
from train import generators, trainers, discriminators
import random
import numpy as np
import time
from keras.models import load_model

image_size = 256
batch_size = 1
n_epochs = 301
save_step = 50

trainA = normalize(load_dataset('/data/horse2zebra/trainA', image_size))
trainB = normalize(load_dataset('/data/horse2zebra/trainB', image_size))
testA = load_dataset('/data/horse2zebra/testA', image_size)
testB = load_dataset('/data/horse2zebra/testB', image_size)

print('Number of training samples are {} and {}'.format(len(trainA), len(trainB)))
print('Number of testing samples are {} and {}'.format(len(testA), len(testB)))

generator_trainer, discriminator_A_trainer, discriminator_B_trainer = trainers
generator_A2B, generator_B2A = generators
discriminator_A, discriminator_B = discriminators

train_A_generator = batch_generator(trainA,batch_size)
train_B_generator = batch_generator(trainB,batch_size)

dummy_labels = np.zeros((batch_size, 1))

for epoch in range(0,n_epochs):

	print('Epoch {}'.format(epoch+1))
	start=time.time()

	for i in range(len(trainA)//batch_size):

		real_A = next(train_A_generator)
		real_B = next(train_B_generator)

		fake_A = generator_A2B.predict(real_B)
		fake_B = generator_B2A.predict(real_A)

		generator_trainer.train_on_batch([real_A, real_B], dummy_labels)

		discriminator_A_trainer.train_on_batch([real_A, fake_A], dummy_labels)

		discriminator_B_trainer.train_on_batch([real_B, fake_B], dummy_labels)

	print('Time taken : '+str(time.time()-start))

	if epoch%save_step==0:

		generator_A2B.save('/output/saved_models/generator_A2B_{}_{}.h5'.format(image_size,epoch))
		generator_B2A.save('/output/saved_models/generator_B2A_{}_{}.h5'.format(image_size,epoch))
		discriminator_A.save('/output/saved_models/discriminator_A_{}_{}.h5'.format(image_size,epoch))
		discriminator_B.save('/output/saved_models/discriminator_B_{}_{}.h5'.format(image_size,epoch))

		save_generator_output(testA[np.random.choice(len(testA), size=16, replace=False)], generator_A2B, epoch=epoch, direction='A2B')
		save_generator_output(testB[np.random.choice(len(testB), size=16, replace=False)], generator_B2A, epoch=epoch, direction='B2A')
