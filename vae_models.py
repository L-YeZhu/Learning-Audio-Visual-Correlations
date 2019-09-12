"""
Created by Ye ZHU, based on the code provided by Spurra
last modified on Sep 12, 2019
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import torchvision
from torchvision import transforms
import torch.optim as optim
import math
import torch.utils.models_zoo as models_zoo
import matplotlib.pyplot as plt



class visual_encoder(nn.Module):
	def __init__(self, z_dim):
		super(visual_encoder, self).__init__()


		## Input: channel nb of input features
		# Cin = 512 * 7 * 7
		self.conv_blocks = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(), # 512 * 9 * 9
            nn.Conv2d(512, 512, 2, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(), # 512 * 10 * 10
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(), # 1024 * 5 * 5
			)

		## Output size
		self.conv_out_dim = 1024 * 5 * 5
		self.lin_lay = nn.Linear(self.conv_out_dim, z_dim)
		# print('lin_lay.shape:\n', self.lin_lay.shape)


		def forward(self,x):
			out_conv = self.conv_blocks(x)
			in_lay = out_conv.view(-1, self.conv_out_dim)
			out_lay = self.lin_lay(in_lay)
			return out_lay


class visual_decoder(nn.Module):
	def __init__(self, z_dim):
		super(visual_decoder,self).__init__()
		self.lin_lay = nn.Sequential(
			nn.Linear(z_dim, 1024 * 5 * 5),
			nn.BatchNorm1d(1024 * 5 * 5),
			nn.ReLU()
			)
		self.conv_blocks = nn.Sequential(
			nn.ConvTranspose2d(1024, 512, 4, 2, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(), # 512 * 10 * 10
			nn.ConvTranspose2d(512, 512, 2, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(), # 512 * 9 * 9
			nn.ConvTranspose2d(512, 512, 1, 1, 1),
			)
		# OUT : 512 * 7 * 7

		def forward(self, x):
			out_lay = self.lin_lay(x)
			in_conv = out_lay.view(-1, 1024, 5, 5)
			out_conv = self.conv_blocks(in_conv)
			return out_conv



class audio_encoder(nn.Module):
	def __init__(self, z_dim, in_dim):
		super(audio_encoder, self).__init__()
		in_dim = torch.FloatTensor(in_dim)
		self.in_size = in_dim.prod()
		self.lin_lays = nn.Sequential(
			nn.Linear(self.in_size, 256),
			nn.ReLU(),
			nn.Linear(256,512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, z_dim)
			)

		def forward(self, x):
			return slef.lin_lays(x.view(-1, self.in_size))



class audio_decoder(nn.Module):
	def __init__(self, z_dim, in_dim):
		super(audio_decoder,self).__init__()
		self.in_dim = torch.FloatTensor(in_dim)
		self.in_size = self.in_dim.prod()
		self.lin_lays = nn.Sequential(
			nn.Linear(self.in_size, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512,256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Linear(256, self.in_size)
			)

		def forward(self, x):
			out_lay = self.lin_lay(x)
			return out_lays.view(-1, self.in_dim[0], self.in_dim[1])


class VAE(nn.Module):

	"""
	Variational Autoencoder module for audio-visual cross-embedding
 
    """



	def __init__(self, z_dim, encoder, decoder):
		super(VAE, self).__init__()
		self.z_dim = z_dim
		self.encoder = encoder
		self.decoder = decoder

	def encode(self, x):
		h_i = self.encoder(x)
		# split to mu and logvar for reparameterize
		return h_i[:, self.z_dim:], h_i[:, :self.z_dim]

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mu(std).add_(mu)
		else:
			return mu

	def forward(self, x, vae_decoder=None):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)

		# if no seperate decoder is specifiedm use own
		if not vae_decoder:
			dec = self.decoder
		else:
			dec = vae_decoder.decoder

		return dec(z), mu, logvar




def loss_KL(mu, logvar):
	kl_loss = -0.5 * (1 + log(logvar * logvar) - mu * mu - logvar * logvar)
	return kl_loss


if __name__ == '__main__':

	input_dim = 512 * 7 * 7
	batch_size = 10
	epoch_nb = 10
	z_dim = 100

	visual_dataloader = torch.utils.data.Dataloader(visual_feature, batch_size = batch_size, shuffle = False)
	audio_dataloader = torch.utils.data.Dataloader(audio_feature, batch_size = batch_size, shuffle = False)

	loss_MSE = nn.MSELoss()

	cross_VAE = VAE(z_dim, visual_encoder, audio_decoder) 
	optimizer = optim.Adam(cross_VAE.parameter(), lr = 0.001)

	for epoch in range(epoch_nb):
		for i, data in enumerate(visual_dataloader,0):
			input_visual_data = data
			input_visual_data = Variable(input_visual_data.resize_(batch_size, input_dim))
			optimizer.zero_grad()
			audio_reconstruct = cross_VAE(input_data)
			loss = loss_MSE(audio_gt,audio_reconstruct) + loss_KL(cross_VAE.mu, cross_VAE.logvar)
			loss.backward()
			optimizer.step()
			l = loss.data[0]

		print(epoch, l)



