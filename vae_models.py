"""
Created by Ye ZHU, based on the code provided by Spurra
last modified on Sep 23, 2019
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class visual_encoder_conv(nn.Module):
	def __init__(self, hidden_dim, latent_dim, batch_size):
		super(visual_encoder_conv, self).__init__()

		# 512 * 7 * 7
		self.conv_blocks = nn.Sequential(
			nn.Conv2d(512, 512, 1, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(), # 512 * 9 * 9
			nn.Conv2d(512, 512, 2, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU() # 512 * 10 * 10
			# nn.BatchNorm2d(512, 1024, 3, 2, 1),
			# nn.ReLU(), # 1024 * 5 * 5
			)

		self.conv_out_dim = 512
		self.pooling = nn.AvgPool2d(10)
		self.lin_lay = nn.Linear(self.conv_out_dim, hidden_dim)
		self.mu = nn.Linear(hidden_dim, latent_dim)
		self.var = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		out_conv = self.conv_blocks(x)
		#print("out_conv size:", out_conv.size()) # 10 * 512 * 10 * 10
		in_lay = self.pooling(out_conv)
		#print("in_lay:",in_lay.size())
		in_lay = in_lay.view(batch_size, self.conv_out_dim)
		#print("in_lay:",in_lay.size())
		hidden = self.lin_lay(in_lay)
		mean = self.mu(hidden)
		log_var = self.var(hidden) 

		return mean, log_var





class visual_encoder_linear(nn.Module):
	def __init__(self, input_dim, hidden_dim, latent_dim):
		super(visual_encoder_linear,self).__init__()

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

	def __init__(self, hidden_dim, latent_dim, output_dim, batch_size):
		super(VAE, self).__init__()
		# self.encoder = encoder(hidden_dim, latent_dim, batch_size)
		# self.decoder = decoder(latent_dim, hidden_dim, output_dim)

		#self.encoder = visual_encoder_linear(input_dim, hidden_dim, latent_dim)
		self.encoder = visual_encoder_conv(hidden_dim, latent_dim, batch_size)
		self.decoder = audio_decoder(latent_dim, hidden_dim, output_dim)

	def forward(self, x):
		z_mu , z_var = self.encoder(x)

		#sample from the latent distribution and reparameterize
		std = torch.exp(z_var / 2)
		eps = torch.randn_like(std)
		x_sample = eps.mul(std).add_(z_mu)

		generated_x = self.decoder(x_sample)

		return generated_x, z_mu, z_var





def calculate_loss(x, reconstructed_x, mu, logvar):
	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x,reconstructed_x)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return mse_loss + kl_loss
	#return kl_loss



def maxpooling(x):
	m = nn.MaxPool2d(7)
	return m(x)

def avgpooling(x):
	m = nn.AvgPool2d(7)
	return m(x)



def train_visual2audio():
	# set the train mode
	cross_VAE.train()

	# loss of the epoch
	train_loss = 0
	# 4143 segment in total
	training_size = 3500


	for i in range(training_size):

		#Visual data input
		visual_data_input = visual_data[i,:,:,:,:]
		visual_data_input = np.transpose(visual_data_input,(0,3,1,2))
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_classes = torch.tensor(np.array([1]))
		visual_classes = visual_classes.float()
		visual_classes = visual_classes.cuda()

		#print(visual_data_input.size())
		#visual_data_input = avgpooling(visual_data_input)
		#print(visual_data_input.size())
		#visual_data_input = Variable(visual_data_input.resize_(batch_size, input_dim_visual))
		visual_data_input = Variable(visual_data_input)
		visual_classes = Variable(visual_classes)
		#print("visual_data_input size:", visual_data_input.size())

		#Audio data gt
		audio_data_gt = audio_data[i,:,:]
		audio_data_gt = audio_data_gt[:,np.newaxis,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_classes = torch.tensor(np.array([2]))
		audio_classes = audio_classes.float()
		audio_classes = audio_classes.cuda()
		audio_data_gt = Variable(audio_data_gt.resize_(batch_size,out_dim_audio))
		audio_classes = Variable(audio_classes)
		#print("audio_data_gt size:", audio_data_gt.size())

		optimizer.zero_grad()
		audio_reconstruct, mu, logvar = cross_VAE(visual_data_input)

		# loss
		loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)
		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer.step()

	return train_loss



def train_audio2visual():
	# set the train mode
	cross_VAE.train()

	# loss of the epoch
	train_loss = 0
	# 4143 segment in total
	training_size = 3500


	for i in range(training_size):

		#Visual data input
		visual_data_input = visual_data[i,:,:,:,:]
		visual_data_input = np.transpose(visual_data_input,(0,3,1,2))
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_classes = torch.tensor(np.array([1]))
		visual_classes = visual_classes.float()
		visual_classes = visual_classes.cuda()

		#print(visual_data_input.size())
		#visual_data_input = avgpooling(visual_data_input)
		#print(visual_data_input.size())
		#visual_data_input = Variable(visual_data_input.resize_(batch_size, input_dim_visual))
		visual_data_input = Variable(visual_data_input)
		visual_classes = Variable(visual_classes)
		#print("visual_data_input size:", visual_data_input.size())

		#Audio data gt
		audio_data_gt = audio_data[i,:,:]
		audio_data_gt = audio_data_gt[:,np.newaxis,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_classes = torch.tensor(np.array([2]))
		audio_classes = audio_classes.float()
		audio_classes = audio_classes.cuda()
		audio_data_gt = Variable(audio_data_gt.resize_(batch_size,out_dim_audio))
		audio_classes = Variable(audio_classes)
		#print("audio_data_gt size:", audio_data_gt.size())

		optimizer.zero_grad()
		visual_reconstruct, mu, logvar = cross_VAE(audio_data_gt)

		# loss
		loss = calculate_loss(visual_data_input, visual_reconstruct,  mu, logvar)
		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer.step()

	return train_loss


def test_visual2audio():
	#set the evaluation mode
	cross_VAE.eval()

	test_loss = 0
	# 4143 segment in total
	testing_size = 643

	with torch.no_grad():	

		for i in range(testing_size):
			i += 3500
			#Visual data input
			visual_data_input = visual_data[i,:,:,:,:]
			visual_data_input = np.transpose(visual_data_input,(0,3,1,2))
			visual_data_input = torch.from_numpy(visual_data_input)
			visual_data_input = visual_data_input.float()
			visual_data_input = visual_data_input.cuda()
			#visual_data_input = avgpooling(visual_data_input)
			#visual_data_input = Variable(visual_data_input.resize_(batch_size, input_dim_visual))
			visual_data_input = Variable(visual_data_input)
			#print("visual_data_input size:", visual_data_input.size())

			#Audio data gt
			audio_data_gt = audio_data[i,:,:]
			audio_data_gt = audio_data_gt[:,np.newaxis,:]
			audio_data_gt = torch.from_numpy(audio_data_gt)
			audio_data_gt = audio_data_gt.float()
			audio_data_gt = audio_data_gt.cuda()
			audio_data_gt = Variable(audio_data_gt.resize_(batch_size,out_dim_audio))
			#print("audio_data_gt size:", audio_data_gt.size())

			optimizer.zero_grad()
			audio_reconstruct, mu, logvar = cross_VAE(visual_data_input)
			#print(mu, logvar)

			# loss, without back propagation
			loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)
			test_loss += loss.item()

	return test_loss



def test_audio2visual():
	#set the evaluation mode
	cross_VAE.eval()

	test_loss = 0
	# 4143 segment in total
	testing_size = 643

	with torch.no_grad():	

		for i in range(testing_size):
			i += 3500
			#Visual data input
			visual_data_input = visual_data[i,:,:,:,:]
			visual_data_input = np.transpose(visual_data_input,(0,3,1,2))
			visual_data_input = torch.from_numpy(visual_data_input)
			visual_data_input = visual_data_input.float()
			visual_data_input = visual_data_input.cuda()
			#visual_data_input = avgpooling(visual_data_input)
			#visual_data_input = Variable(visual_data_input.resize_(batch_size, input_dim_visual))
			visual_data_input = Variable(visual_data_input)
			#print("visual_data_input size:", visual_data_input.size())

			#Audio data gt
			audio_data_gt = audio_data[i,:,:]
			audio_data_gt = audio_data_gt[:,np.newaxis,:]
			audio_data_gt = torch.from_numpy(audio_data_gt)
			audio_data_gt = audio_data_gt.float()
			audio_data_gt = audio_data_gt.cuda()
			audio_data_gt = Variable(audio_data_gt.resize_(batch_size,out_dim_audio))
			#print("audio_data_gt size:", audio_data_gt.size())

			optimizer.zero_grad()
			visual_reconstruct, mu, logvar = cross_VAE(audio_data_gt)
			#print(mu, logvar)

			# loss, without back propagation
			loss = calculate_loss(visual_data_input, visual_reconstruct,  mu, logvar)
			test_loss += loss.item()

	return test_loss


if __name__ == '__main__':

	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	input_dim_visual = 512 * 1 * 1
	hidden_dim = 100
	latent_dim = 80
	out_dim_audio = 128
	batch_size = 10
	epoch_nb = 20
	training_size = 3500
	testing_size = 643


	## Cross VAE Model
	# encoder = visual_encoder(input_dim_visual, hidden_dim, latent_dim)
	# decoder = audio_decoder(latent_dim, hidden_dim, out_dim_audio) 
	cross_VAE = VAE(hidden_dim, latent_dim, out_dim_audio, batch_size) 
	cross_VAE = cross_VAE.cuda()
	#optimizer = optim.Adam(cross_VAE.parameters(), lr = 0.0001)


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

		# if epoch < 15:
		# 	optimizer = optim.Adam(cross_VAE.parameters(), lr = 0.0001)
		# else:
		# 	optimizer = optim.Adam(cross_VAE.parameters(), lr = 0.00001)

		optimizer = optim.Adam(cross_VAE.parameters(), lr = 0.0001)

		if epoch < 10 :
			print(f'Cross training: from visual to audio:')

			train_loss = 0
			# Cross training: Visual to audio
			train_loss = train_visual2audio()
			test_loss1 = test_visual2audio()
			test_loss2 = test_audio2visual()

			train_loss /= training_size
			test_loss1 /= testing_size
			test_loss2 /= testing_size

			#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
			print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss1: {test_loss1:.2f}, Test Loss2: {test_loss2: .2f}')
			#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

		else:
			print(f'Cross training: from audio to visual:')

			train_loss = 0
			# Cross training: Audio to visual
			train_loss = train_visual2audio()
			test_loss1 = test_visual2audio()
			test_loss2 = test_audio2visual()

			train_loss /= training_size
			test_loss1 /= testing_size
			test_loss2 /= testing_size

			#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss1: {test_loss:.2f}')
			print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss1: {test_loss1:.2f}, Test Loss2: {test_loss2: .2f}')



	## Visulization
	# model.eval()
	# z_list = None
	# l_list = []






