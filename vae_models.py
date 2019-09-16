"""
Created by Ye ZHU, based on the code provided by Spurra
last modified on Sep 16, 2019
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import torchvision
from torchvision import transforms
import torch.optim as optim
import math
import torch.utils.data
import matplotlib.pyplot as plt
import h5py
import torch.nn.functional as F




# class visual_decoder(nn.Module):
# 	def __init__(self, z_dim):
# 		super(visual_decoder,self).__init__()
# 		self.lin_lay = nn.Sequential(
# 			nn.Linear(z_dim, 1024 * 5 * 5),
# 			nn.BatchNorm1d(1024 * 5 * 5),
# 			nn.ReLU()
# 			)

# 		self.conv_blocks = nn.Sequential(
# 			nn.ConvTranspose2d(1024, 512, 4, 2, 1),
# 			nn.BatchNorm2d(512),
# 			nn.ReLU(), # 512 * 10 * 10
# 			nn.ConvTranspose2d(512, 512, 2, 1, 1),
# 			nn.BatchNorm2d(512),
# 			nn.ReLU(), # 512 * 9 * 9
# 			nn.ConvTranspose2d(512, 512, 1, 1, 1),
# 			)
# 		# OUT : 512 * 7 * 7

# 	def forward(self, x):
# 		out_lay = self.lin_lay(x)
# 		in_conv = out_lay.view(-1, 1024, 5, 5)
# 		out_conv = self.conv_blocks(in_conv)
# 		return out_conv

class visual_encoder(nn.Module):
	def __init__(self, input_dim, hidden_dim, latent_dim):
		super(visual_encoder,self).__init__()

		self.lin_lay = nn.Linear(input_dim, hidden_dim)
		self.mu = nn.Linear(hidden_dim, latent_dim)
		self.var = nn.Linear(hidden_dim, latent_dim)


	def forward(self, x):
		# x shape: [batch_size, input_dim]
		hidden = F.relu(self.lin_lay(x))
		# hidden shape: [batch_size, hidden_dim]

		# latent parameters
		mean = self.mu(hidden)
		# mean shape: [batch_size, latent_dim]

		log_var = self.var(hidden)
		# log_var shape: [batch_size, latent_dim]

		return mean, log_var


class audio_decoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim, output_dim):
		super(audio_decoder,self).__init__()

		self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
		self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		# x shape: [batch_size, latent_dim]
		x = F.relu(self.latent_to_hidden(x))
		# x shape: [batch_size, hidden_dim]
		generated_x = F.relu(self.hidden_to_out(x))
		# x shape: [batch_size, output_dim]

		return generated_x




# class audio_encoder(nn.Module):
# 	def __init__(self, z_dim):
# 		super(audio_encoder, self).__init__()

# 		self.lin_lays = nn.Sequential(
# 			nn.Linear(self.in_size, 256),
# 			nn.ReLU(),
# 			nn.Linear(256,512),
# 			nn.ReLU(),
# 			nn.Linear(512, 512),
# 			nn.ReLU(),
# 			nn.Linear(512, z_dim)
# 			)
# 		# OUT: 512 * 1 

# 	def forward(self, x):
# 		return self.lin_lays(x.view(-1, 512))




class VAE(nn.Module):

	"""
	Variational Autoencoder module for audio-visual cross-embedding
    """



	def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
		super(VAE, self).__init__()

		self.encoder = visual_encoder(input_dim, hidden_dim, latent_dim)
		self.decoder = audio_decoder(latent_dim, hidden_dim, output_dim)

	def forward(self, x):
		z_mu , z_var = self.encoder(x)

		#sample from the latent distribution and reparameterize
		std = torch.exp(z_var / 2)
		eps = torch.randn_like(std)
		x_sample = eps.mul(std).add_(z_mu)

		generated_x = self.decoder(x_sample)

		return generated_x, z_mu, z_var


	# def encode(self, x):
	# 	h_i = self.encoder(x)
	# 	print("h_i size:",h_i.shape)
	# 	# split to mu and logvar for reparameterize
	# 	return h_i[:, self.z_dim:], h_i[:, :self.z_dim]

	# def reparameterize(self, mu, logvar):
	# 	# if self.training:
	# 	std = logvar.mul(0.5).exp_()
	# 	eps = Variable(std.data.new(std.size()).normal_())
	# 	return eps.mu(std).add_(mu)
	# 	# else:
	# 	# 	return mu

	# def forward(self, x):

	# 	mu, logvar = self.encode(x)
	# 	z = self.reparameterize(mu, logvar)

	# 	# if no seperate decoder is specifiedm use own
	# 	#if not vae_decoder:
	# 	dec = self.decoder
	# 	#else:
	# 	#	dec = vae_decoder.decoder

	# 	return dec(z), mu, logvar




def calculate_loss(x, reconstructed_x, mu, logvar):
	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x,reconstructed_x)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return mse_loss + kl_loss



def train():
	# set the train mode
	cross_VAE.train()

	# loss of the epoch
	train_loss = 0
	# 4143 segment in total
	training_size = 4000


	for i in range(training_size):

		#Visual data input
		visual_data_input = visual_data[i,:,:,:,:]
		visual_data_input = np.transpose(visual_data_input,(0,3,1,2))
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		#visual_data_input = visual_data_input.to(device)
		visual_data_input = Variable(visual_data_input.resize_(batch_size, input_dim_visual))
		#print("visual_data_input size:", visual_data_input.size())

		#Audio data gt
		audio_data_gt = audio_data[i,:,:]
		audio_data_gt = audio_data_gt[:,np.newaxis,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		#audio_data_gt = audio_data_gt.to(device)
		audio_data_gt = Variable(audio_data_gt.resize_(batch_size,out_dim_audio))
		#print("audio_data_gt size:", audio_data_gt.size())

		optimizer.zero_grad()
		audio_reconstruct, mu, logvar = cross_VAE(visual_data_input)
		#print(mu, logvar)
		#print("audio_reconstruct:",audio_reconstruct)
		#print("audio_data_gt:",audio_data_gt)
		## what is data type for audio_reconstruct
		loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)
		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer.step()

	return train_loss


def test():
	#set the evaluation mode
	cross_VAE.eval()

	test_loss = 0
	# 4143 segment in total
	testing_size = 143

	with torch.no_grad():	

		for i in range(testing_size):
			i += 4000 
			#Visual data input
			visual_data_input = visual_data[i,:,:,:,:]
			visual_data_input = np.transpose(visual_data_input,(0,3,1,2))
			visual_data_input = torch.from_numpy(visual_data_input)
			visual_data_input = visual_data_input.float()
			visual_data_input = visual_data_input.cuda()
			#visual_data_input = visual_data_input.to(device)
			visual_data_input = Variable(visual_data_input.resize_(batch_size, input_dim_visual))
			#print("visual_data_input size:", visual_data_input.size())

			#Audio data gt
			audio_data_gt = audio_data[i,:,:]
			audio_data_gt = audio_data_gt[:,np.newaxis,:]
			audio_data_gt = torch.from_numpy(audio_data_gt)
			audio_data_gt = audio_data_gt.float()
			audio_data_gt = audio_data_gt.cuda()
			#audio_data_gt = audio_data_gt.to(device)
			audio_data_gt = Variable(audio_data_gt.resize_(batch_size,out_dim_audio))
			#print("audio_data_gt size:", audio_data_gt.size())

			optimizer.zero_grad()
			audio_reconstruct, mu, logvar = cross_VAE(visual_data_input)
			#print(mu, logvar)
			loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)
			test_loss += loss.item()

	return test_loss




if __name__ == '__main__':

	#evice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	input_dim_visual = 512 * 7 * 7
	hidden_dim = 512
	latent_dim = 100
	out_dim_audio = 128
	batch_size = 10
	epoch_nb = 10
	training_size = 4000
	testing_size = 143


	## Cross VAE Model
	# encoder = visual_encoder(input_dim_visual, hidden_dim, latent_dim)
	# decoder = audio_decoder(latent_dim, hidden_dim, out_dim_audio) 
	cross_VAE = VAE(input_dim_visual, hidden_dim, latent_dim, out_dim_audio) 
	cross_VAE = cross_VAE.cuda()
	#cross_VAE.train()
	optimizer = optim.Adam(cross_VAE.parameters(), lr = 0.0001)
	train_loss = 0


	# data processing
	file_name1 = 'audio_feature.h5'
	file_name2 = 'visual_feature.h5'
	f1 = h5py.File(file_name1, 'r')
	f2 = h5py.File(file_name2, 'r')

	f1_group_key = list(f1.keys())[0]
	f2_group_key = list(f2.keys())[0]

	audio_data = list(f1[f1_group_key])
	visual_data = list(f2[f2_group_key])
	audio_data = np.array(audio_data)
	visual_data = np.array(visual_data)


	for epoch in range(epoch_nb):

		train_loss = train()
		test_loss = test()

		train_loss /= training_size
		test_loss /= testing_size

		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')


