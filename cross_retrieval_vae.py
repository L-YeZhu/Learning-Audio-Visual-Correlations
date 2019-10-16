"""
Created by Ye ZHU
last modified on Oct 10, 2019
"""

import os
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
from model_components_longseq import *



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
x_audio_train = np.zeros((len(train_l), 128*10))
x_video_train = np.zeros((len(train_l), 512*10))
x_audio_val = np.zeros((len(val_l), 128*10))
x_video_val = np.zeros((len(val_l), 512*10))
x_audio_test = np.zeros((len(test_l), 128*10))
x_video_test = np.zeros((len(test_l), 512*10))
y_train      = np.zeros((len(train_l), 10))
y_val        = np.zeros((len(val_l), 10))
y_test       = np.zeros((len(test_l), 10))
##
for i in range(len(train_l)):
    id = train_l[i]
    for j in range(10):
        s_a = j * 128
        s_v = j * 512
        e_a = s_a + 128
        e_v = s_v + 512
        x_audio_train[i, s_a:e_a] = audio_features[id, j, :]
        x_video_train[i, s_v:e_v] = video_features[id, j, :]
        y_train[i, j] = closs_labels[id, j]

for i in range(len(val_l)):
    id = val_l[i]
    for j in range(10):
        s_a = j * 128
        s_v = j * 512
        e_a = s_a + 128
        e_v = s_v + 512
        x_audio_val[i, s_a:e_a] = audio_features[id, j, :]
        x_video_val[i, s_v:e_v] = video_features[id, j, :]
        y_val[i, j] = closs_labels[id, j]

for i in range(len(test_l)):
    id = test_l[i]
    for j in range(10):
        s_a = j * 128
        s_v = j * 512
        e_a = s_a + 128
        e_v = s_v + 512
        x_audio_test[i, s_a:e_a] = audio_features[id, j, :]
        x_video_test[i, s_v:e_v] = video_features[id, j, :]
        y_test[i, j] = closs_labels[id, j]


print("data loading finished!")


#############################################################################



def calculate_loss(x, reconstructed_x, mu, logvar):
	# norm = nn.BatchNorm1d(1280).cuda()
	# x = norm(x)
	# reconstructed_x = norm(reconstructed_x)
	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x,reconstructed_x)
	kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()))

	return mse_loss + kl_loss*0.1



def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x,reconstructed_x,2)
	return dis


def train_audio():
	vae_audio.train()
	train_loss = 0
	training_size = len(train_l)

	for video_id in range(int(training_size/9)):
		s = video_id * 9
		e = s + 9

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
		loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)

		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer_audio.step()

	return train_loss



def train_video():
	vae_video.train()
	train_loss = 0
	training_size = len(train_l)

	for video_id in range(int(training_size/9)):
		s = video_id * 9
		e = s + 9
		# visual data for traning
		visual_data_input = x_video_train[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

		optimizer_video.zero_grad()
		video_reconstruct, mu, logvar = vae_video(visual_data_input)

		# loss
		loss = calculate_loss(visual_data_input, video_reconstruct,  mu, logvar)

		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer_video.step()

	return train_loss



def train_v2a():
	vae_video.train()
	vae_audio.train()
	train_loss = 0
	training_size = len(train_l)

	for video_id in range(int(training_size/9)):
		s = video_id * 9
		e = s + 9
		# visual data for traning
		visual_data_input = x_video_train[s:e,:] ### 10 * 5120
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)
		#print(visual_data_input.size())

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:] ### 10 * 1280
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("audio_data_gt size:", audio_data_gt.size)


		#print("video_xi size:", video_xi.size())
		optimizer_video.zero_grad()
		optimizer_audio.zero_grad()
		audio_reconstruct, mu, logvar = vae_video(visual_data_input, vae_audio)

		# loss
		#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar, out_dim_audio)
		loss = calculate_loss(audio_data_gt, audio_reconstruct, mu, logvar)

		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer_video.step()
		optimizer_audio.step()

	return train_loss



def train_a2v():
	vae_audio.train()
	vae_video.train()
	train_loss = 0
	training_size = len(train_l)

	for video_id in range(int(training_size/9)):
		s = video_id * 9
		e = s + 9
		# visual data for traning
		visual_data_input = x_video_train[s:e,:]
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)
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
		video_reconstruct, mu, logvar = vae_audio(audio_data_gt, vae_video)

		# loss
		loss = calculate_loss(visual_data_input, video_reconstruct,  mu, logvar)
		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer_audio.step()
		optimizer_video.step()

	return train_loss




def cross_retrieval_v2a(x_v, x_a):
	vae_video.eval()
	vae_audio.eval()
	loss_MSE = nn.MSELoss()
	reconstructed_audio, mu1, logvar1 = vae_video(x_v, vae_audio)
	#reconstructed_x2, mu2, logvar2 = cross_VAE(x_out)
	#distance_loss = calculate_loss(x_out, reconstructed_x1, mu1, logvar1)
	distance_euc = euclidean_dis(x_a, reconstructed_audio)
	distance_mse = loss_MSE(x_a,reconstructed_audio)
	return distance_euc.item()






if __name__ == '__main__':

	# input_dim_visual = 128 * 10
	# hidden_dim = 100 * 10
	latent_dim = 128 * 10
	# out_dim_audio = 128 * 10
	batch_size = 9
	epoch_nb = 5
	n_classes = 20
	training_size = len(train_l)
	testing_size = len(test_l)
	val_size = len(val_l)
	#print(training_size/3)

	audio_encode = audio_encoder(latent_dim)
	audio_decode = audio_decoder(latent_dim)
	video_encode = visual_encoder(latent_dim)
	video_decode = visual_decoder(latent_dim)
	vae_audio = VAE(latent_dim, audio_encode, audio_decode)
	vae_video = VAE(latent_dim, video_encode, video_decode)
	vae_audio.cuda()
	vae_video.cuda()

	optimizer_audio = optim.Adam(vae_audio.parameters(), lr = 0.00001)
	optimizer_video = optim.Adam(vae_video.parameters(), lr = 0.00001)

	for epoch in range(epoch_nb):
		total_loss = 0
		print(f'Cross modality training: from audio to video:')
		train_loss = 0
		train_loss = train_a2v()
		train_loss /= training_size
		total_loss += train_loss
		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

		print(f'Cross modality training: from video to audio:')
		train_loss = 0
		train_loss = train_v2a()
		train_loss /= training_size
		total_loss += train_loss
		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

		print(f'Single modality training: from audio to audio:')	
		train_loss = 0
		train_loss = train_audio()
		train_loss /= training_size
		total_loss += train_loss
		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

		print(f'Single modality training: from video to video:')	
		train_loss = 0
		train_loss = train_video()
		train_loss /= training_size
		total_loss += train_loss
		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')



	# torch.save(vae_audio.state_dict(), 'vae_audio_cross_retrieval.pkl')
	# torch.save(vae_video.state_dict(), 'vae_video_cross_retrieval.pkl')
	# print("Training completed!")

###### Testing process ######

	# #cross_VAE = torch.load('VAE_retrieval_sequence.pkl')
	print("Testing begins!")
	print("Testing samples:", testing_size)

	audio_encode = audio_encoder(latent_dim)
	audio_decode = audio_decoder(latent_dim)
	video_encode = visual_encoder(latent_dim)
	video_decode = visual_decoder(latent_dim)
	vae_audio = VAE(latent_dim, audio_encode, audio_decode)
	vae_video = VAE(latent_dim, video_encode, video_decode)
	vae_audio.cuda()
	vae_video.cuda()

	vae_audio.load_state_dict(torch.load('vae_audio_cross_retrieval.pkl'))
	vae_video.load_state_dict(torch.load('vae_video_cross_retrieval.pkl'))
	print("Model load completed!")

	count_num = 0
	audio_count = 0
	video_count = 0
	video_acc = 0
	audio_acc = 0
	pos_len = 0


	#### Read event labels for test set #####
	f = open("data/testSet.txt", 'r')
	dataset = f.readlines()  # all good videos and the duration is 10s
	xx = 0


	#### given the video, find the corresponding audio ####
	for i in range(testing_size):
		score = []
		visual_data_input = x_video_test[i,:]
		visual_data_input = np.expand_dims(visual_data_input, axis=0)
		#print(visual_data_input.shape)
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

		for j in range(testing_size):
			audio_data_gt = x_audio_test[j, :]
			audio_data_gt = np.expand_dims(audio_data_gt, axis=0)
			audio_data_gt = torch.from_numpy(audio_data_gt)
			audio_data_gt = audio_data_gt.float()
			audio_data_gt = audio_data_gt.cuda()
			audio_data_gt = Variable(audio_data_gt)

			s = cross_retrieval_v2a(visual_data_input, audio_data_gt)
			score.append(s)
			#print(score)

		score = np.array(score).astype('float32')
		print(score)
		print("Testing number:", i )
		min_id = int(np.argmin(score))
		print(min_id)
		if min_id == i:
			audio_acc += 1

	retrieval_acc_v2a = audio_acc * 100 / testing_size
	print("retrieval_acc_v2a:", retrieval_acc_v2a)

	# #### given the audio, find the corresponding video ####
	# for i in range(testing_size):
	# 	score = []
	# 	audio_data_gt = x_audio_test[i, :]
	# 	audio_data_gt = torch.from_numpy(audio_data_gt)
	# 	audio_data_gt = audio_data_gt.float()
	# 	audio_data_gt = audio_data_gt.cuda()
	# 	audio_data_gt = Variable(audio_data_gt)

	# 	for j in range(testing_size):
	# 		visual_data_input = x_video_test[j,:]
	# 		visual_data_input = torch.from_numpy(visual_data_input)
	# 		visual_data_input = visual_data_input.float()
	# 		visual_data_input = visual_data_input.cuda()
	# 		visual_data_input = Variable(visual_data_input)
	# 		visual_data_input = pre_linear(visual_data_input)

	# 		s = cross_retrieval(audio_data_gt, visual_data_input)
	# 		score.append(s)
	# 		#print(score)

	# 	score = np.array(score).astype('float32')
	# 	print("Testing number:", i )
	# 	min_id = int(np.argmin(score))
	# 	print(min_id)
	# 	print(score)
	# 	if min_id == i:
	# 		video_acc += 1

	# retrieval_acc_a2v = video_acc * 100 / testing_size
	# print("retrieval_acc_v2a:", retrieval_acc_a2v)
