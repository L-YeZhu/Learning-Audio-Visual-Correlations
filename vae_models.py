"""
Created by Ye ZHU
last modified on Sep 30, 2019
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
from model_components import *



#data loader
with h5py.File('data/labels_closs.h5', 'r') as hf:
    closs_labels = hf['avadataset'][:]
with h5py.File('data/visual_feature_vec.h5', 'r') as hf:
    video_features = hf['avadataset'][:]
with h5py.File('data/audio_feature.h5', 'r') as hf:
    audio_features = hf['avadataset'][:]
with h5py.File('data/train_order_match.h5', 'r') as hf:
    train_l = hf['order'][:]
with h5py.File('data/val_order_match.h5', 'r') as hf:
    val_l = hf['order'][:]
with h5py.File('data/test_order_match.h5', 'r') as hf:
    test_l = hf['order'][:]

closs_labels = np.array(closs_labels) ## 4143 * 10
audio_features = np.array(audio_features)  ##  4143 * 10 * 128
video_features = np.array(video_features)  ##  4143 * 10 * 512
closs_labels = closs_labels.astype("float32")
audio_features = audio_features.astype("float32")
video_features = video_features.astype("float32")


##
x_audio_train = np.zeros((len(train_l)*10, 128))
x_video_train = np.zeros((len(train_l)*10, 512))
x_audio_val = np.zeros((len(val_l)*10, 128))
x_video_val = np.zeros((len(val_l)*10, 512))
x_audio_test = np.zeros((len(test_l)*10, 128))
x_video_test = np.zeros((len(test_l)*10, 512))
y_train      = np.zeros((len(train_l)*10))
y_val        = np.zeros((len(val_l)*10))
y_test       = np.zeros((len(test_l)*10))
##
for i in range(len(train_l)):
    id = train_l[i]
    for j in range(10):
        x_audio_train[10*i + j, :] = audio_features[id, j, :]
        x_video_train[10*i + j, :] = video_features[id, j, :]
        y_train[10*i + j] = closs_labels[id, j]

for i in range(len(val_l)):
    id = val_l[i]
    for j in range(10):
        x_audio_val[10 * i + j, :] = audio_features[id, j, :]
        x_video_val[10 * i + j, :] = video_features[id, j, :]
        y_val[10 * i + j] = closs_labels[id, j]

for i in range(len(test_l)):
    id = test_l[i]
    for j in range(10):
        x_audio_test[10 * i + j, :] = audio_features[id, j, :]
        x_video_test[10 * i + j, :] = video_features[id, j, :]
        y_test[10 * i + j] = closs_labels[id, j]


print("data loading finished!")





# class visual_encoder_conv(nn.Module):
# 	def __init__(self, hidden_dim, latent_dim, batch_size):
# 		super(visual_encoder_conv, self).__init__()

# 		# 512 * 7 * 7
# 		self.conv_blocks = nn.Sequential(
# 			nn.Conv2d(512, 512, 1, 1, 1),
# 			nn.BatchNorm2d(512),
# 			nn.ReLU(), # 512 * 9 * 9
# 			nn.Conv2d(512, 512, 2, 1, 1),
# 			nn.BatchNorm2d(512),
# 			nn.ReLU() # 512 * 10 * 10
# 			# nn.BatchNorm2d(512, 1024, 3, 2, 1),
# 			# nn.ReLU(), # 1024 * 5 * 5
# 			)

# 		self.conv_out_dim = 512
# 		self.pooling = nn.AvgPool2d(10)
# 		self.lin_lay = nn.Linear(self.conv_out_dim, hidden_dim)
# 		self.mu = nn.Linear(hidden_dim, latent_dim)
# 		self.var = nn.Linear(hidden_dim, latent_dim)

# 	def forward(self, x):
# 		out_conv = self.conv_blocks(x)
# 		#print("out_conv size:", out_conv.size()) # 10 * 512 * 10 * 10
# 		in_lay = self.pooling(out_conv)
# 		#print("in_lay:",in_lay.size())
# 		in_lay = in_lay.view(batch_size, self.conv_out_dim)
# 		#print("in_lay:",in_lay.size())
# 		hidden = self.lin_lay(in_lay)
# 		mean = self.mu(hidden)
# 		log_var = self.var(hidden) 

# 		return mean, log_var



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


def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x,reconstructed_x,2)
	return dis


def calculate_loss(x, reconstructed_x, mu, logvar):
	# norm = nn.BatchNorm1d(128).cuda()
	# x = norm(x)
	# reconstructed_x = norm(reconstructed_x)
	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x,reconstructed_x)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return mse_loss + kl_loss*0.1, kl_loss, mse_loss
	#return kl_loss

def calculate_loss_a2v(x_audio, x_video, reconstructed_video , mu, logvar):
	# norm = nn.BatchNorm1d(128).cuda()
	# x = norm(x)
	# reconstructed_x = norm(reconstructed_x)
	# _ , mu1, logvar1 = vae_audio(x_audio)
	# z1 = vae_audio.reparameterize(mu1, logvar1)
	# _, mu2, logvar2 = vae_video(x_video)
	# z2 = vae_video.reparameterize(mu2, logvar2)
	# latent_loss = euclidean_dis(z1,z2)
	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x_video,reconstructed_video)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return  mse_loss + kl_loss, kl_loss, mse_loss


def calculate_loss_v2a(x_video, x_audio, reconstructed_audio, mu, logvar):
	# norm = nn.BatchNorm1d(128).cuda()
	# x = norm(x)
	# reconstructed_x = norm(reconstructed_x)
	# _ , mu1, logvar1 = vae_video(x_video)
	# z1 = vae_video.reparameterize(mu1, logvar1)
	# _ , mu2, logvar2 = vae_audio(x_audio)
	# z2 = vae_audio.reparameterize(mu2, logvar2)
	# latent_loss = euclidean_dis(z1, z2)
	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x_audio,reconstructed_audio)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return  mse_loss + kl_loss*0.08, kl_loss, mse_loss


def avgpooling(x):
	m = nn.AvgPool2d(7)
	return m(x)


def pre_linear(x):
	m = nn.Linear(512,128).cuda()
	return F.relu(m(x))



def train_audio():
	vae_audio.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("audio_data_gt size:", audio_data_gt.size)

		optimizer_audio.zero_grad()
		audio_reconstruct, mu, logvar = vae_audio(audio_data_gt)

		# loss
		#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar, out_dim_audio)
		loss, kl, mse = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		#print(train_loss)
		optimizer_audio.step()

	return train_loss, kl_loss, mse_loss


def train_audio_test():
	vae_audio.train()
	train_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("audio_data_gt size:", audio_data_gt.size)

		optimizer_audio.zero_grad()
		audio_reconstruct, mu, logvar = vae_audio(audio_data_gt)

		# loss
		#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar, out_dim_audio)
		loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)

		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer_audio.step()

	return train_loss


def train_video_test():
	vae_video.train()
	vae_audio.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	training_size = len(train_l)

	for i in range(training_size):
		visual_data_input = visual_data[i,:,:,:,:]
		visual_data_input = np.transpose(visual_data_input,(0,3,1,2))
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = avgpooling(visual_data_input)
		visual_data_input = Variable(visual_data_input.resize_(batch_size, input_dim_visual))
		optimizer_video.zero_grad()
		video_reconstruct, mu, logvar = vae_audio(visual_data_input)

		# loss
		loss, kl, mse = calculate_loss(visual_data_input, video_reconstruct,  mu, logvar)

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		#print(train_loss)
		optimizer_video.step()

	return train_loss, kl_loss



def train_video():
	vae_video.train()
	vae_audio.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10
		# visual data for traning
		visual_data_input = x_video_train[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)
		#visual_data_input = pre_linear(visual_data_input)

		optimizer_video.zero_grad()
		video_reconstruct, mu, logvar = vae_video(visual_data_input)

		# loss
		loss, kl, mse = calculate_loss(visual_data_input, video_reconstruct,  mu, logvar)

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		#print(train_loss)
		#optimizer_audio.step()
		optimizer_video.step()

	return train_loss, kl_loss, mse_loss



def train_v2a():
	vae_video.train()
	vae_audio.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10
		# visual data for traning
		visual_data_input = x_video_train[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)
		#visual_data_input = pre_linear(visual_data_input)
		#print("visual_data_input size:", visual_data_input.size)

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("audio_data_gt size:", audio_data_gt.size)

		optimizer_video.zero_grad()
		optimizer_audio.zero_grad()
		audio_reconstruct, mu, logvar = vae_video(visual_data_input, vae_audio)
		#audio_reconstruct, mu, logvar = vae_audio(visual_data_input)

		# loss
		#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)
		loss1, kl, mse = calculate_loss_v2a(visual_data_input, audio_data_gt, audio_reconstruct,  mu, logvar)

		loss1.backward()
		train_loss += loss1.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		#print(train_loss)
		#optimizer_audio.step()
		optimizer_video.step()

		### RE-UPDATE audio_encode ####
		#audio_reconstruct2, mu2, logvar2 = vae_audio(audio_reconstruct)
		#loss2, kl, mse = calculate_loss(audio_reconstruct, audio_reconstruct2, mu2, logvar2)
		#loss2.backward()
		#optimizer_audio.step()
		#optimizer_video.step()

	return train_loss, kl_loss, mse_loss


def train_a2v():
	vae_audio.train()
	vae_video.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10
		# visual data for traning
		visual_data_input = x_video_train[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)
		#visual_data_input = pre_linear(visual_data_input)
		#print("training visual_data_input size:", visual_data_input.size)

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("training audio_data_gt size:", audio_data_gt.size)

		optimizer_audio.zero_grad()
		optimizer_video.zero_grad()
		visual_reconstruct, mu, logvar = vae_audio(audio_data_gt,vae_video)
		#visual_reconstruct, mu, logvar = vae_audio(audio_data_gt)


		#loss = calculate_loss(visual_data_input, visual_reconstruct,  mu, logvar)
		loss1, kl, mse = calculate_loss_a2v(audio_data_gt, visual_data_input, visual_reconstruct,  mu, logvar)

		loss1.backward()
		train_loss += loss1.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		#print("testing point 2 - loss:", train_loss)
		#print(train_loss)
		#optimizer_video.step()
		optimizer_audio.step()


		### RE-UPDATE video_encode ####
		#visual_reconstruct2, mu2, logvar2 = vae_video(visual_reconstruct)
		#loss2, kl, mse = calculate_loss(visual_reconstruct, visual_reconstruct2, mu2, logvar2)
		#loss2.backward()
		#optimizer_video.step()
		#optimizer_video.step()

	return train_loss, kl_loss, mse_loss


#######################################################################################################


def caluculate_loss_generaldec(x_visual, x_audio, x_reconstruct, mu, logvar, epoch):
	loss_MSE = nn.MSELoss()
	x_input = torch.cat((x_visual, x_audio), 1)
	#bs = x_reconstruct.size(0)
	mse_loss = loss_MSE(x_input, x_reconstruct)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	if epoch < 7:
		final_loss = mse_loss + kl_loss*0.1
	else:
		final_loss = mse_loss + kl_loss*0.1
	#bce_loss = F.binary_cross_entropy(x_reconstruct, x_input, size_average = False)
	return final_loss, kl_loss, mse_loss


def train_video_generaldec(epoch):
	#vae_audio.train()
	vae_video.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	training_size = len(train_l)
	#optimizer_video.zero_grad()

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10
		visual_data_input = x_video_train[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)

		optimizer_video.zero_grad()
		# x_reconstruct should be in size of 512 + 128
		if epoch == 0:
			x_reconstruct, mu, logvar = vae_video(visual_data_input)
		else:
			x_reconstruct, mu, logvar = vae_video(visual_data_input,vae_audio)

		loss, kl, mse = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch)

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()

		optimizer_video.step()
		#optimizer_audio.step()

	return train_loss, kl_loss, mse_loss


def train_audio_generaldec(epoch):
	vae_audio.train()
	#vae_video.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10
		visual_data_input = x_video_train[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)

		optimizer_audio.zero_grad()
		## x_reconstruct should be in size of 512 + 128, same decoder is used 
		x_reconstruct, mu, logvar = vae_audio(audio_data_gt, vae_video)
		loss, kl, mse = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch)		

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()

		optimizer_audio.step()
		#optimizer_video.step()

	return train_loss, kl_loss, mse_loss


def val_video_generaldec():
	vae_audio.eval()
	vae_video.eval()
	val_loss = 0
	kl_loss = 0
	mse_loss = 0
	val_size = len(val_l)

	for video_id in range(val_size):
		s = video_id * 10
		e = s + 10
		visual_data_input = x_video_val[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

		audio_data_gt = x_audio_val[s:e,:]		
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)

		#optimizer_video.zero_grad()
		x_reconstruct, mu, logvar = vae_video(visual_data_input, vae_audio)
		loss, kl, mse = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch)

		val_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()

	return val_loss, kl_loss, mse_loss


def val_audio_generaldec():
	vae_audio.eval()
	vae_video.eval()
	val_loss = 0
	kl_loss = 0
	mse_loss = 0
	val_size = len(val_l)

	for video_id in range(val_size):
		s = video_id * 10
		e = s + 10
		visual_data_input = x_video_val[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

		audio_data_gt = x_audio_val[s:e,:]		
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)

		#optimizer_audio.zero_grad()
		x_reconstruct, mu, logvar = vae_audio(audio_data_gt, vae_video)
		loss, kl, mse = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch)

		val_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()

	return val_loss, kl_loss, mse_loss




# def val_v2a():
# 	cross_vae_v2a.eval()
# 	val_loss = 0
# 	val_size = len(val_l)

# 	with torch.no_grad():

# 		for video_id in range(val_size):
# 			s = video_id * 10
# 			e = s + 10
# 			# visual data for testing
# 			visual_data_input = x_video_val[s:e,:]
# 			visual_data_input = torch.from_numpy(visual_data_input)
# 			visual_data_input = visual_data_input.float()
# 			visual_data_input = visual_data_input.cuda()
# 			visual_data_input = Variable(visual_data_input)
# 			#print(visual_data_input.size())
# 			visual_data_input = pre_linear(visual_data_input)	
# 			#print("testing visual_data_input size:", visual_data_input.size)	

# 			#audio data gt
# 			audio_data_gt = x_audio_val[s:e,:]
# 			audio_data_gt = torch.from_numpy(audio_data_gt)
# 			audio_data_gt = audio_data_gt.float()
# 			audio_data_gt = audio_data_gt.cuda()
# 			audio_data_gt = Variable(audio_data_gt)
# 			#print("testing audio_data_gt size:", audio_data_gt.size)	


# 			optimizer.zero_grad()
# 			audio_reconstruct, mu, logvar = cross_vae_v2a(visual_data_input)
# 			# print(audio_data_gt.size())
# 			# print(audio_reconstruct.size())
# 			loss = calculate_loss(visual_data_input, audio_data_gt, audio_reconstruct,  mu, logvar)
# 			distance_loss = loss.item()
# 			distance_euc = euclidean_dis(audio_data_gt, audio_reconstruct)
# 			val_loss += loss.item()

# 	return val_loss


# def val_a2v():
# 	cross_vae_a2v.eval()
# 	val_loss = 0
# 	val_size = len(val_l)

# 	with torch.no_grad():

# 		for video_id in range(val_size):
# 			s = video_id * 10
# 			e = s + 10
# 			# visual data for testing
# 			visual_data_input = x_video_val[s:e,:]
# 			visual_data_input = torch.from_numpy(visual_data_input)
# 			visual_data_input = visual_data_input.float()
# 			visual_data_input = visual_data_input.cuda()
# 			visual_data_input = Variable(visual_data_input)
# 			visual_data_input = pre_linear(visual_data_input)	
# 			#print("testing visual_data_input size:", visual_data_input.size)	

# 			#audio data gt
# 			audio_data_gt = x_audio_val[s:e,:]
# 			audio_data_gt = torch.from_numpy(audio_data_gt)
# 			audio_data_gt = audio_data_gt.float()
# 			audio_data_gt = audio_data_gt.cuda()
# 			audio_data_gt = Variable(audio_data_gt)
# 			#print("testing audio_data_gt size:", audio_data_gt.size)	

# 			optimizer.zero_grad()
# 			visual_reconstruct, mu, logvar = cross_vae_a2v(audio_data_gt)

# 			loss = calculate_loss(audio_data_gt, visual_data_input, visual_reconstruct,  mu, logvar)
# 			distance_loss = loss.item()
# 			distance_euc = euclidean_dis(visual_data_input, visual_reconstruct)
# 			val_loss += loss.item()

# 	return val_loss





def tsne():
	vae_audio.eval()
	vae_video.eval()
	z_list = None
	l_list = []
	testing_size = len(test_l)

	with torch.no_grad():
		for video_id in range(testing_size * 2):
			s = video_id * 1
			e = s + 1
			visual_data_input = x_video_test[s:e,:]
			visual_data_input = torch.from_numpy(visual_data_input)
			visual_data_input = visual_data_input.float()
			visual_data_input = visual_data_input.cuda()
			visual_data_input = Variable(visual_data_input)
			visual_classes = torch.tensor(np.array([1]))
			visual_classes = visual_classes.float()
			visual_classes = visual_classes.cuda()
			visual_classes = Variable(visual_classes)
			#visual_data_input = pre_linear(visual_data_input)


			audio_data_gt = x_audio_val[s:e,:]
			audio_data_gt = torch.from_numpy(audio_data_gt)
			audio_data_gt = audio_data_gt.float()
			audio_data_gt = audio_data_gt.cuda()
			audio_data_gt = Variable(audio_data_gt)
			audio_classes = torch.tensor(np.array([0]))
			audio_classes = audio_classes.float()
			audio_classes = audio_classes.cuda()
			audio_classes = Variable(audio_classes)

			optimizer_audio.zero_grad()
			optimizer_video.zero_grad()

			audio_reconstruct, mu1, logvar1 = vae_audio(audio_data_gt)
			z_audio = vae_audio.reparameterize(mu1, logvar1)
			if video_id == 0:
					z_list = z_audio
					l_list = audio_classes
			else:
					z_list = torch.cat((z_list, z_audio), 0)
					l_list = torch.cat((l_list, audio_classes), 0)

			video_reconstruct, mu2, logvar2 = vae_video(visual_data_input, vae_audio)
			z_video = vae_video.reparameterize(mu2, logvar2)

			z_list = torch.cat((z_list, z_video), 0)
			l_list = torch.cat((l_list, visual_classes), 0)

	z_list = z_list.data.cpu().numpy()
	l_list = l_list.cpu().numpy()
	#print("l_list:",l_list)
	X_reduced = TSNE(n_components=2, random_state=0).fit_transform(z_list)
	point_count = 0
	for i in range(len(l_list)):
		if (l_list[i] == 0):
			plt.scatter(X_reduced[i,0], X_reduced[i,1], c = 'red')
			point_count += 1
		elif (l_list[i] == 1):
			plt.scatter(X_reduced[i,0], X_reduced[i,1], c = 'green')
			point_count +=1

	print("point_count:", point_count)
	#plt.legend()
	plt.show()



# def cross_tsne():
# 	vae_audio.eval()
# 	vae_video.eval()
# 	z_list = None
# 	l_list = []
# 	testing_size = len(test_l)

# 	with torch.no_grad():
# 		for video_id in range(testing_size):
# 			s = video_id * 10
# 			e = s + 10
# 			visual_data_input = x_video_test[s:e,:]
# 			visual_data_input = torch.from_numpy(visual_data_input)
# 			visual_data_input = visual_data_input.float()
# 			visual_data_input = visual_data_input.cuda()
# 			visual_data_input = Variable(visual_data_input)
# 			visual_classes = torch.tensor(np.array([1]))
# 			visual_classes = visual_classes.float()
# 			visual_classes = visual_classes.cuda()
# 			visual_classes = Variable(visual_classes)


# 			audio_data_gt = x_audio_val[s:e,:]
# 			audio_data_gt = torch.from_numpy(audio_data_gt)
# 			audio_data_gt = audio_data_gt.float()
# 			audio_data_gt = audio_data_gt.cuda()
# 			audio_data_gt = Variable(audio_data_gt)
# 			audio_classes = torch.tensor(np.array([0]))
# 			audio_classes = audio_classes.float()
# 			audio_classes = audio_classes.cuda()
# 			audio_classes = Variable(audio_classes)

# 			optimizer_audio.zero_grad()
# 			optimizer_video.zero_grad()

# 			video_reconstruct, mu1, logvar1 = vae_audio(audio_data_gt, vae_video)
# 			z_a2v = vae_audio.reparameterize(mu1, logvar1)
# 			if video_id == 0:
# 					z_list = z_a2v
# 					l_list = audio_classes
# 			else:
# 					z_list = torch.cat((z_list, z_a2v), 0)
# 					l_list = torch.cat((l_list, audio_classes), 0)

# 			audio_reconstruct, mu2, logvar2 = vae_video(visual_data_input, vae_audio)
# 			z_v2a = vae_video.reparameterize(mu2, logvar2)

# 			z_list = torch.cat((z_list, z_v2a), 0)
# 			l_list = torch.cat((l_list, visual_classes), 0)

# 	z_list = z_list.data.cpu().numpy()
# 	l_list = l_list.cpu().numpy()
# 	#print("l_list:",l_list)
# 	X_reduced = TSNE(n_components=2, random_state=0).fit_transform(z_list)
# 	point_count = 0
# 	for i in range(len(l_list)):
# 		if (l_list[i] == 0):
# 			plt.scatter(X_reduced[i,0], X_reduced[i,1], c = 'red')
# 			point_count += 1
# 		elif (l_list[i] == 1):
# 			plt.scatter(X_reduced[i,0], X_reduced[i,1], c = 'green')
# 			point_count +=1

# 	print("point_count:", point_count)
# 	#plt.legend()
# 	plt.show()




if __name__ == '__main__':

	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	input_dim_visual = 512
	latent_dim = 100
	#out_dim_audio = 128
	batch_size = 10
	epoch_nb = 10
	training_size = len(train_l)
	testing_size = len(test_l)
	val_size = len(val_l)



	## Cross VAE Model
	audio_encode = audio_encoder(latent_dim)
	audio_decode = audio_decoder(latent_dim)
	video_encode = visual_encoder(latent_dim)
	video_decode = visual_decoder(latent_dim)
	general_decode = general_decoder(latent_dim)
	#vae_audio = VAE_varenc(latent_dim, audio_encode, general_decode)
	#vae_video = VAE_varenc(latent_dim, video_encode, general_decode)
	vae_audio = VAE(latent_dim, audio_encode, general_decode)
	vae_video = VAE(latent_dim, video_encode, general_decode)
	#vae_video = VAE(latent_dim, video_encode, video_decode)
	vae_audio.cuda()
	vae_video.cuda()

	optimizer_audio = optim.Adam(vae_audio.parameters(), lr = 0.00001)
	optimizer_video = optim.Adam(vae_video.parameters(), lr = 0.00001)



	# 	# data processing
	# file_name1 = 'audio_feature.h5'
	# file_name2 = 'visual_feature.h5'
	# f1 = h5py.File(file_name1, 'r')
	# f2 = h5py.File(file_name2, 'r')

	# f1_group_key = list(f1.keys())[0]
	# f2_group_key = list(f2.keys())[0]

	# audio_data = list(f1[f1_group_key])
	# visual_data = list(f2[f2_group_key])
	# audio_data = np.array(audio_data)
	# visual_data = np.array(visual_data)


	for epoch in range(epoch_nb):

		# if (epoch < 5):
		# 	optimizer_audio = optim.Adam(vae_audio.parameters(), lr = 0.0001)
		# 	optimizer_video = optim.Adam(vae_video.parameters(), lr = 0.0001)
		# else:
		# 	optimizer_audio = optim.Adam(vae_audio.parameters(), lr = 0.00001)
		# 	optimizer_video = optim.Adam(vae_video.parameters(), lr = 0.00001)

		#if epoch > 10:

		print(f'Training with general decoder --- video to all:')
		train_loss = 0
		train_loss, kl_loss, mse_loss = train_video_generaldec(epoch)
		train_loss /= training_size
		kl_loss /= training_size
		mse_loss /= training_size
		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
		print("KL and MSE loss:", kl_loss, mse_loss)

	#else:

		print(f'Training with general decoder --- audio to all:')
		train_loss = 0
		train_loss, kl_loss, mse_loss = train_audio_generaldec(epoch)
		train_loss /= training_size
		kl_loss /= training_size
		mse_loss /= training_size
		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
		print("KL and MSE loss:", kl_loss, mse_loss) 		

	tsne()


	print(f'Val with general decoder --- audio to all:')
	val_loss = 0
	val_loss, kl_loss, mse_loss = val_audio_generaldec()
	val_loss /= val_size
	kl_loss /= val_size
	mse_loss /= val_size
	print(f'Val Loss for audio input: {train_loss:.2f}')
	print("KL and MSE loss:", kl_loss, mse_loss) 	


	print(f'Val with general decoder --- video to all:')
	val_loss = 0
	val_loss, kl_loss, mse_loss = val_video_generaldec()
	val_loss /= val_size
	kl_loss /= val_size
	mse_loss /= val_size
	print(f'Val Loss for video input: {train_loss:.2f}')
	print("KL and MSE loss:", kl_loss, mse_loss) 



		# total_loss = 0

# # 	#if epoch %2 == 0 :
# 		print(f'Single modality training: from audio to audio:')

# 		train_loss = 0
# 		# Cross training: Visual to audio
# 		train_loss, kl_loss, mse_loss = train_audio()
# 		# test_loss1 = test_visual2audio()
# 		# test_loss2 = test_audio2visual()

# 		train_loss /= training_size
# 		kl_loss /= training_size
# 		mse_loss /= training_size
# 		#total_loss += train_loss

# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss1: {test_loss1:.2f}, Test Loss2: {test_loss2: .2f}')
# 		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
# 		print("KL and MSE loss:", kl_loss, mse_loss)


# # 		# print(f'Cross modality training: from audio to video:')
# # 		# train_loss = 0
# # 		# train_loss = train_a2v()
# # 		# train_loss /= training_size
# # 		# print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
# # ###############################################################################################################################

# # 	#else:
# 		print(f'Single modality training: from video to video:')

# 		train_loss = 0
# 		# Cross training: Audio to visual
# 		train_loss, kl_loss, mse_loss = train_video()
# 		# test_loss1 = test_visual2audio()
# 		# test_loss2 = test_audio2visual()

# 		train_loss /= training_size
# 		kl_loss /= training_size
# 		mse_loss /= training_size
# 		#total_loss += train_loss

# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss1: {test_loss:.2f}')
# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss1: {test_loss1:.2f}, Test Loss2: {test_loss2: .2f}')
# 		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
# 		print("KL and MSE loss:", kl_loss, mse_loss)

# 		print(f'Cross modality training: from audio to video:')
# 		train_loss = 0
# 		train_loss, kl_loss, mse_loss = train_a2v()
# 		train_loss /= training_size
# 		kl_loss /= training_size
# 		mse_loss /= training_size
# 		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
# 		print("KL and MSE loss:", kl_loss, mse_loss)

# 		print(f'Cross modality training: from video to audio:')
# 		train_loss = 0
# 		train_loss, kl_loss, mse_loss = train_v2a()
# 		train_loss /= training_size
# 		kl_loss /= training_size
# 		mse_loss /= training_size
# 		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
# 		print("KL and MSE loss:", kl_loss, mse_loss)


		#print(f'Epoch {epoch}, --------Total Loss: {total_loss:.2f} -------')
		# print(f'Cross modality training: from audio to video:')
		# train_loss = 0
		# train_loss = train_a2v()
		# train_loss /= training_size
		# print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

		# print(f'Cross modality training: from video to audio:')
		# train_loss = 0
		# train_loss = train_v2a()
		# train_loss /= training_size
		# print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

	# ## Visulization
	# torch.save(vae_audio.state_dict(), 'vae_audio_cross_latentloss.pkl')
	# torch.save(vae_video.state_dict(), 'vae_video_cross_latentloss.pkl')
	#tsne()
	#cross_tsne()






