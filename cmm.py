"""
Created by Ye ZHU
last modified on Sep 30, 2019
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
import random
random.seed(3344)


##data loader
#load train data
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


# print("closs_labels size:", closs_labels.shape)
# print(closs_labels[1,:])
# print(closs_labels[10,:])
# print(closs_labels[100,:])
# print(closs_labels[1000,:])
# # print(sum(closs_labels))
# # print("visual_features size:", video_features.shape) 
# # print("audio_features size:", audio_features.shape)
# # print("train_l:", train_l.shape)
# # print("val_l:", val_l.shape)
# print("test_l", test_l)
# print(len(train_l))


################ VAE model #################
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



class audio_decoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim, output_dim):
		super(audio_decoder,self).__init__()

		self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
		self.hidden_to_out = nn.Linear(hidden_dim, output_dim)
		self.norm = nn.BatchNorm1d(128)

	def forward(self, x):
		# x shape: [batch_size, latent_dim]
		x = F.relu(self.latent_to_hidden(x))
		# x shape: [batch_size, hidden_dim]
		generated_x = F.relu(self.hidden_to_out(x))
		# x shape: [batch_size, output_dim]

		return generated_x


class VAE(nn.Module):

	"""
	Variational Autoencoder module for audio-visual cross-embedding
    """

	def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
		super(VAE, self).__init__()
		# self.encoder = encoder(hidden_dim, latent_dim, batch_size)
		# self.decoder = decoder(latent_dim, hidden_dim, output_dim)

		self.encoder = visual_encoder_linear(input_dim, hidden_dim, latent_dim)
		#self.encoder = visual_encoder_conv(hidden_dim, latent_dim, batch_size)
		self.decoder = audio_decoder(latent_dim, hidden_dim, output_dim)


	def reparametrize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std) + mu
		else:
			return mu




	def forward(self, x):
		z_mu , z_var = self.encoder(x)

		#sample from the latent distribution and reparameterize
		std = torch.exp(z_var / 2)
		eps = torch.randn_like(std)
		x_sample = eps.mul(std).add_(z_mu)

		generated_x = self.decoder(x_sample)

		return generated_x, z_mu, z_var



def calculate_loss(x, reconstructed_x, mu, logvar):
	# norm = nn.BatchNorm1d(128).cuda()
	# x = norm(x)
	# reconstructed_x = norm(reconstructed_x)
	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x,reconstructed_x)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	euc_loss = torch.dist(x,reconstructed_x,2)
	return mse_loss + kl_loss


def euclidean_dis(x, reconstructed_x):
	# x = x.cpu()
	# x = x.data.numpy()
	# reconstructed_x = reconstructed_x.cpu()
	# reconstructed_x = reconstructed_x.data.numpy()
	# dis = np.linalg.norm(x - reconstructed_x)
	dis = torch.dist(x,reconstructed_x,2)
	return dis

def pre_linear(x):
	m1 = nn.Linear(512,128).cuda()
	m2 = nn.BatchNorm1d(128).cuda()
	#return F.relu(m2(m1(x)))
	return F.relu(m1(x))


def compute_accuracy(predictions, labels):
    c = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            c += 1
    return c / len(predictions)


def compute_precision(predictions, labels):
    c = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and predictions[i] == labels[i]:
            c += 1
    return c



def train_v2a():
	cross_VAE.train()
	train_loss = 0
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
		visual_data_input = pre_linear(visual_data_input)
		#print("visual_data_input size:", visual_data_input.size)

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("audio_data_gt size:", audio_data_gt.size)

		optimizer.zero_grad()
		audio_reconstruct, mu, logvar = cross_VAE(visual_data_input)

		# loss
		#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar, out_dim_audio)
		loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)
		#distance = euclidean_dis(audio_data_gt, audio_reconstruct)
		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer.step()

	return train_loss


def train_a2v():
	cross_VAE.train()
	train_loss = 0
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
		visual_data_input = pre_linear(visual_data_input)
		#print("training visual_data_input size:", visual_data_input.size)

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:]
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("training audio_data_gt size:", audio_data_gt.size)

		optimizer.zero_grad()
		visual_reconstruct, mu, logvar = cross_VAE(audio_data_gt)

		loss = calculate_loss(visual_data_input, visual_reconstruct,  mu, logvar)
		#distance = euclidean_dis(visual_data_input, visual_reconstruct)
		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer.step()

	return train_loss


def val_v2a():
	cross_VAE.eval()
	val_loss = 0
	val_size = len(val_l)

	with torch.no_grad():

		for video_id in range(val_size):
			s = video_id * 10
			e = s + 10
			# visual data for testing
			visual_data_input = x_video_val[s:e,:]
			visual_data_input = torch.from_numpy(visual_data_input)
			visual_data_input = visual_data_input.float()
			visual_data_input = visual_data_input.cuda()
			visual_data_input = Variable(visual_data_input)
			#print(visual_data_input.size())
			visual_data_input = pre_linear(visual_data_input)	
			#print("testing visual_data_input size:", visual_data_input.size)	

			#audio data gt
			audio_data_gt = x_audio_val[s:e,:]
			audio_data_gt = torch.from_numpy(audio_data_gt)
			audio_data_gt = audio_data_gt.float()
			audio_data_gt = audio_data_gt.cuda()
			audio_data_gt = Variable(audio_data_gt)
			#print("testing audio_data_gt size:", audio_data_gt.size)	

			optimizer.zero_grad()
			audio_reconstruct, mu, logvar = cross_VAE(visual_data_input)
			# print(audio_data_gt.size())
			# print(audio_reconstruct.size())
			loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar)
			distance_loss = loss.item()
			distance_euc = euclidean_dis(audio_data_gt, audio_reconstruct)
			val_loss += loss.item()

	return val_loss


def val_a2v():
	cross_VAE.eval()
	val_loss = 0
	val_size = len(val_l)

	with torch.no_grad():

		for video_id in range(val_size):
			s = video_id * 10
			e = s + 10
			# visual data for testing
			visual_data_input = x_video_val[s:e,:]
			visual_data_input = torch.from_numpy(visual_data_input)
			visual_data_input = visual_data_input.float()
			visual_data_input = visual_data_input.cuda()
			visual_data_input = Variable(visual_data_input)
			visual_data_input = pre_linear(visual_data_input)	
			#print("testing visual_data_input size:", visual_data_input.size)	

			#audio data gt
			audio_data_gt = x_audio_val[s:e,:]
			audio_data_gt = torch.from_numpy(audio_data_gt)
			audio_data_gt = audio_data_gt.float()
			audio_data_gt = audio_data_gt.cuda()
			audio_data_gt = Variable(audio_data_gt)
			#print("testing audio_data_gt size:", audio_data_gt.size)	

			optimizer.zero_grad()
			visual_reconstruct, mu, logvar = cross_VAE(audio_data_gt)

			loss = calculate_loss(visual_data_input, visual_reconstruct,  mu, logvar)
			distance_loss = loss.item()
			distance_euc = euclidean_dis(visual_data_input, visual_reconstruct)
			val_loss += loss.item()

	return val_loss


def test_cmm(input_feature, output_feature):
	cross_VAE.eval()
	input_feature = input_feature.float()
	output_feature = output_feature.float()
	with torch.no_grad():
	#distance= 0
		reconstructed_x, mu, logvar = cross_VAE(input_feature)
		loss1 = calculate_loss(output_feature, reconstructed_x, mu, logvar)
		distance_loss = loss1.item()
		loss2 = euclidean_dis(output_feature, reconstructed_x)
		distance_euc = loss2.item()
	return distance_euc



if __name__ == '__main__':

	input_dim_visual = 512 * 1 * 1
	hidden_dim = 100
	latent_dim = 80
	out_dim_audio = 128
	batch_size = 10
	epoch_nb = 20
	training_size = len(train_l)
	testing_size = len(test_l)
	val_size = len(val_l)

	# cross_VAE = VAE(out_dim_audio, hidden_dim, latent_dim, out_dim_audio) 
	# cross_VAE = cross_VAE.cuda()

	# for epoch in range(epoch_nb):


	# 	optimizer = optim.Adam(cross_VAE.parameters(), lr = 0.00001)

	# 	if epoch %2 == 0 :
	# 		print(f'Cross training: from visual to audio:')
	# 		train_loss = 0
	# 		train_loss = train_v2a()
	# 		val_loss1 = val_v2a()
	# 		val_loss2 = val_a2v()
	# 		train_loss /= training_size
	# 		val_loss1 /= val_size
	# 		val_loss2 /= val_size
	# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
	# 		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss1: {val_loss1:.2f}, Val Loss2: {val_loss2: .2f}')
	# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

	# 	else:
	# 		print(f'Cross training: from audio to visual:')
	# 		train_loss = 0
	# 		# Cross training: Audio to visual
	# 		train_loss = train_a2v()
	# 		val_loss1 = val_v2a()
	# 		val_loss2 = val_a2v()
	# 		train_loss /= training_size
	# 		val_loss1 /= val_size
	# 		val_loss2 /= val_size
	# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss1: {test_loss:.2f}')
	# 		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss1: {val_loss1:.2f}, Val Loss2: {val_loss2: .2f}')

	# torch.save(cross_VAE,'VAE_cmm3.pkl')
	# print("Training completed!")


###### Testing, codes modified based on AVE - cmm_test.py ######


	cross_VAE = torch.load('VAE_cmm3.pkl')

	print("Testing begins!")

	N = y_test.shape[0]
	s = 200
	e = s + 10
	count_num = 0
	audio_count = 0
	video_count = 0
	video_acc = 0
	audio_acc = 0
	pos_len = 0

	f = open("data/Annotations.txt", 'r')
	dataset = f.readlines()  # all good videos and the duration is 10s
	xx = 0
	for video_id in range(int(N / 10)):
	    s = video_id * 10
	    e = s + 10
	    x_test = y_test[s: e]
	    if np.sum(x_test) == 10:
	        continue
	    count_num += 1
	    xx += np.sum(x_test)
	    nb = np.argwhere(x_test == 1)

	    seg = np.zeros(len(nb)).astype('int8')
	    for i in range(len(nb)):
	        seg[i] = nb[i][0]

	    l = len(seg)

	    x_test_video_feature = x_video_test[s:e, :]    
	    x_test_video_feature = torch.from_numpy(x_test_video_feature)
	    x_test_video_feature = x_test_video_feature.float()
	    x_test_video_feature = x_test_video_feature.cuda()
	    x_test_video_feature = Variable(x_test_video_feature)
	    x_test_video_feature = pre_linear(x_test_video_feature)

	    x_test_audio_feautre = x_audio_test[s:e, :]
	    x_test_audio_feautre = torch.from_numpy(x_test_audio_feautre)
	    x_test_audio_feautre = x_test_audio_feautre.float()
	    x_test_audio_feautre = x_test_audio_feautre.cuda()
	    x_test_audio_feautre = Variable(x_test_audio_feautre)


	    # print("x_test_video_feature:", x_test_video_feature[0:1,:])
	    # print("x_test_audio_feautre:", x_test_audio_feautre[0:1,:])


	    # given audio clip
	    score = []
	    for nn_count in range(10 - l + 1):
	        s = 0
	        for i in range(l):
	        	s += test_cmm(x_test_video_feature[nn_count + i:nn_count + i + 1, :], x_test_audio_feautre[seg[i:i + 1], :])
	        score.append(s)
	    score = np.array(score).astype('float32')
	    id = int(np.argmin(score))
	    pred_vid = np.zeros(10)
	    for i in range(id, id + int(l)):
	        pred_vid[i] = 1

	    if np.argmin(score) == seg[0]:
	        audio_count += 1
	    video_acc += compute_precision(x_test, pred_vid)
	    # calculate single accuracy
	    ind = np.where(x_test - pred_vid == 0)[0]
	    acc_v = len(ind)    

	    # given video clip
	    score = []
	    for nn_count in range(10 - l + 1):
	        s = 0
	        for i in range(l):
	            s += test_cmm(x_test_video_feature[seg[i:i + 1], :], x_test_audio_feautre[nn_count + i:nn_count + i + 1, :])
	        score.append(s)
	    score = np.array(score).astype('float32')

	    if np.argmin(score) == seg[0]:
	        video_count += 1
	    pred_aid = np.zeros(10)
	    id = int(np.argmin(score))
	    for i in range(id, id + int(l)):
	        pred_aid[i] = 1
	    audio_acc += compute_precision(x_test, pred_aid)
	    pos_len += len(seg)

	    # calculate single accuracy
	    ind = np.where(x_test - pred_aid == 0)[0]
	    acc_a = len(ind)
	    # print('num:{}, {}'.format(video_id, dataset[test_l[video_id]].rstrip('\n')))
	    # print('vid_input: ', x_test, 'pred:', pred_vid, 'correct: ', acc_v)
	    # print('aud_input: ', x_test, 'pred:', pred_aid, 'correct: ', acc_a)

	print(video_count * 100 / count_num, audio_count * 100 / count_num)

