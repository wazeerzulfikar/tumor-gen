import keras.backend as K
import numpy as np

def GAN_criterion(output, target, use_mse=True):
	if use_mse:
		diff = output-target
		dims = list(range(1,K.ndim(diff)))
		return K.expand_dims((K.mean(diff**2, dims)), 0)
	else:
		return K.mean(K.log(output+K.epsilon())*target + K.log(output+K.epsilon())*(1-target), axis=-1)


def cycle_criterion(recon, real):
	# return K.mean(K.abs(recon - real), axis=-1)
	diff = K.abs(recon-real)
	dims = list(range(1,K.ndim(diff)))
	return K.expand_dims((K.mean(diff, dims)), 0)


def G_loss(G_tensors, lambda_val=10):

	D_A_fake, recon_A, G_A_input, D_B_fake, recon_B, G_B_input = G_tensors

	G_A_loss = GAN_criterion(D_A_fake, K.ones_like(D_A_fake))
	cycle_A_loss = cycle_criterion(recon_A, G_A_input)

	G_B_loss = GAN_criterion(D_B_fake, K.ones_like(D_B_fake))
	cycle_B_loss = cycle_criterion(recon_B, G_B_input)

	loss = G_A_loss + G_B_loss + lambda_val * (cycle_A_loss + cycle_B_loss)

	return loss


def D_loss(D_tensors):

	D_real, D_fake = D_tensors

	D_real_loss = GAN_criterion(D_real, K.ones_like(D_real))

	D_fake_loss = GAN_criterion(D_fake, K.zeros_like(D_fake))

	loss = 0.5*(D_real_loss+D_fake_loss)

	return loss