from keras.layers import Lambda
from models import *
from losses import *
import os

image_size = 256

load = False
load_from = 50
load_path = '/saved_models'

if load:
	discriminator_A = discriminator(image_size, load_path=os.path.join(load_path, 'discriminator_A_{}.h5'.format(load_from)))
	discriminator_B = discriminator(image_size, load_path=os.path.join(load_path, 'discriminator_B_{}.h5'.format(load_from)))

	generator_A2B, real_A, fake_B = resnet_generator(image_size, load_path=os.path.join(load_path, 'generator_A2B_{}.h5'.format(load_from)))
	generator_B2A, real_B, fake_A = resnet_generator(image_size, load_path=os.path.join(load_path, 'generator_B2A_{}.h5'.format(load_from)))

else:
	discriminator_A = discriminator(image_size)
	discriminator_B = discriminator(image_size)

	generator_A2B, real_A, fake_B = resnet_generator(image_size)
	generator_B2A, real_B, fake_A = resnet_generator(image_size)

print(discriminator_A.summary())
print('-'*150)
print(generator_A2B.summary())

recon_B = generator_A2B(fake_A)
discriminator_A_fake = discriminator_A(fake_A)

recon_A = generator_B2A(fake_B)
discriminator_B_fake = discriminator_B(fake_B)

lambda_layer_inputs = [discriminator_B_fake, recon_A, real_A, discriminator_A_fake, recon_B, real_B]

for l in generator_A2B.layers:
	l.trainable=True
for l in generator_B2A.layers:
	l.trainable=True
for l in discriminator_A.layers:
	l.trainable=False
for l in discriminator_B.layers:
	l.trainable=False

generator_trainer = Model(inputs=[real_A, real_B],outputs=Lambda(G_loss)(lambda_layer_inputs))
generator_trainer.compile('adam', loss='mae')


discriminator_A_real = discriminator_A(real_A)
fake_A_input = layers.Input(shape=(image_size,image_size,3))
discriminator_A_fake = discriminator_A(fake_A_input)

lambda_layer_inputs = [discriminator_A_real, discriminator_A_fake]

for l in generator_A2B.layers:
	l.trainable=False
for l in generator_B2A.layers:
	l.trainable=False
for l in discriminator_A.layers:
	l.trainable=True
for l in discriminator_B.layers:
	l.trainable=False

discriminator_A_trainer = Model(inputs=[real_A, fake_A_input], outputs=Lambda(D_loss)(lambda_layer_inputs))
discriminator_A_trainer.compile('adam', loss='mae')


discriminator_B_real = discriminator_B(real_B)
fake_B_input = layers.Input(shape=(image_size, image_size, 3))
discriminator_B_fake = discriminator_B(fake_B_input)

lambda_layer_inputs = [discriminator_B_real, discriminator_B_fake]

for l in generator_A2B.layers:
	l.trainable=False
for l in generator_B2A.layers:
	l.trainable=False
for l in discriminator_A.layers:
	l.trainable=False
for l in discriminator_B.layers:
	l.trainable=True

discriminator_B_trainer = Model(inputs=[real_B, fake_B_input], outputs=Lambda(D_loss)(lambda_layer_inputs))
discriminator_B_trainer.compile('adam', loss='mae')

trainers = [generator_trainer, discriminator_A_trainer, discriminator_B_trainer]
generators = [generator_A2B, generator_B2A]
discriminators = [discriminator_A, discriminator_B]



