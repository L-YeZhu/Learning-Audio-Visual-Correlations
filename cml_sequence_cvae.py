"""
Created by Ye ZHU
last modified on Oct 8, 2019
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
# print("x_audio_train size:", x_audio_train.shape)
# print("x_video_train size:", x_video_train.shape)
# print("y_train size:", y_train.shape)




################ VAE model #################
class visual_encoder_linear(nn.Module):
	def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
		super(visual_encoder_linear,self).__init__()

		self.lin_lay = nn.Linear(input_dim + n_classes, hidden_dim)
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
	def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
		super(audio_decoder,self).__init__()

		self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
		self.hidden_to_out = nn.Linear(hidden_dim, output_dim)
		self.norm = nn.BatchNorm1d(128)

	def forward(self, x):
		# x shape: [batch_size, latent_dim]
		x = F.relu(self.latent_to_hidden(x))
		# x shape: [batch_size, hidden_dim]
		generated_x = F.relu(self.hidden_to_out(x))
		# x shape: [batch_size, output_dim]

		return generated_x


class CVAE(nn.Module):

	"""
	Variational Autoencoder module for audio-visual cross-embedding
    """

	def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, n_classes):
		super(CVAE, self).__init__()
		# self.encoder = encoder(hidden_dim, latent_dim, batch_size)
		# self.decoder = decoder(latent_dim, hidden_dim, output_dim)

		self.encoder = visual_encoder_linear(input_dim, hidden_dim, latent_dim, n_classes)
		#self.encoder = visual_encoder_conv(hidden_dim, latent_dim, batch_size)
		self.decoder = audio_decoder(latent_dim, hidden_dim, output_dim, n_classes)


	def reparametrize(self, mu, logvar):
		std = torch.exp(logvar / 2)
		eps = torch.randn_like(std)
		z = eps.mul(std).add_(mu)
		return z
		# if self.training:
		# 	std = logvar.mul(0.5).exp_()
		# 	eps = Variable(std.data.new(std.size()).normal_())
		# 	return eps.mul(std) + mu
		# else:
		# 	return mu

	def forward(self, x, y):
		x = torch.cat((x,y), dim = 1)
		z_mu , z_var = self.encoder(x)

		#sample from the latent distribution and reparameterize
		std = torch.exp(z_var / 2)
		eps = torch.randn_like(std)
		x_sample = eps.mul(std).add_(z_mu)

		z = torch.cat((x_sample, y), dim = 1 )

		generated_x = self.decoder(z)

		return generated_x, z_mu, z_var



def calculate_loss(xi, xt, reconstructed_xt, mu, logvar):
	# norm = nn.BatchNorm1d(128).cuda()
	# x = norm(x)
	# reconstructed_x = norm(reconstructed_x)
	loss_MSE = nn.MSELoss()
	#loss_MSE = nn.MSELoss(reduce = False, size_average = False)
	mse_loss = loss_MSE(xt,reconstructed_xt)
	#mse_loss = mse_loss.mean(1)
	#kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim = 1)
	kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()))

	# reconstructed_x1, mu1, logvar1 = cross_CVAE(xi)
	# reconstructed_x2, mu2, logvar2 = cross_CVAE(xt)

	# latent_loss = euclidean_dis(mu1, mu2)
	# loss = mse_loss + kl_loss + latent_loss

	return mse_loss + kl_loss


def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x,reconstructed_x,2)
	return dis

def pre_linear(x):
	m1 = nn.Linear(5120,1280).cuda()
	m2 = nn.BatchNorm1d(1280).cuda()
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


def idx2onehot(idx, n = 2):
	assert idx.shape[1] == 1
	assert torch.max(idx).item() < n
	onehot = torch.zeros(idx.size(0),n)
	onehot.scatter_(1, idx.data, 1)
	return onehot



def train_v2a():
	cross_CVAE.train()
	train_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 3
		e = s + 3
		# visual data for traning
		visual_data_input = x_video_train[s:e,:] ### 10 * 5120
		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		#print("visual_data_input size:", visual_data_input.size())
		visual_data_input = Variable(visual_data_input)
		visual_data_input = pre_linear(visual_data_input)
		#print(visual_data_input.size())

		#audio data gt
		audio_data_gt = x_audio_train[s:e,:] ### 10 * 1280
		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)
		#print("audio_data_gt size:", audio_data_gt.size)

		### syn_labels info for training
		syn_label = y_train[s:e,:] ### 10 * 10
		syn_label = torch.from_numpy(syn_label)
		#print("syn size:", syn_label.size())
		syn_label = syn_label.long()
		syn_label = idx2onehot(syn_label.view(-1,1))
		# print(syn_label.size())
		# print(syn_label)
		syn_label = syn_label.view(-1,20)
		#print(syn_label.size())
		# print(syn_label)
		syn_label = syn_label.cuda()
		syn_label = Variable(syn_label)
		#syn_label = idx2onehot(syn_label.view(-1,1))
		# optimizer.zero_grad()


		#print("video_xi size:", video_xi.size())
		optimizer.zero_grad()
		audio_reconstruct, mu, logvar = cross_CVAE(visual_data_input, syn_label)

		# loss
		#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar, out_dim_audio)
		loss = calculate_loss(visual_data_input, audio_data_gt, audio_reconstruct, mu, logvar)
		#print(loss)

		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer.step()

	return train_loss


def train_a2v():
	cross_CVAE.train()
	train_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 3
		e = s + 3
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

		syn_label = y_train[s:e]
		syn_label = torch.from_numpy(syn_label)
		syn_label = syn_label.long()
		syn_label = idx2onehot(syn_label.view(-1,1))
		syn_label = syn_label.view(-1,20)
		# print(syn_label.size())
		# print(syn_label)
		syn_label = syn_label.cuda()
		syn_label = Variable(syn_label)



		optimizer.zero_grad()
		video_reconstruct, mu, logvar = cross_CVAE(audio_data_gt,syn_label)


		# loss
		loss = calculate_loss(audio_data_gt, visual_data_input, video_reconstruct,  mu, logvar)
		loss.backward()
		train_loss += loss.item()
		#print(train_loss)
		optimizer.step()

	return train_loss


def val_v2a():
	cross_CVAE.eval()
	val_loss = 0
	val_size = len(val_l)

	with torch.no_grad():

		for video_id in range(val_size):
			s = video_id * 3
			e = s + 3
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


			syn_label = y_val[s:e]
			syn_label = torch.from_numpy(syn_label)
			syn_label = syn_label.long()
			syn_label = idx2onehot(syn_label.view(-1,1))
			syn_label = syn_label.view(-1,20)
			# print(syn_label.size())
			# print(syn_label)
			syn_label = syn_label.cuda()
			syn_label = Variable(syn_label)

			optimizer.zero_grad()
			audio_reconstruct, mu, logvar = cross_CVAE(visual_data_input, syn_label)

			# loss
			#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar, out_dim_audio)
			loss = calculate_loss(visual_data_input, audio_data_gt, audio_reconstruct, mu, logvar)
			#print(loss)
			val_loss += loss.item()


	return val_loss


def val_a2v():
	cross_CVAE.eval()
	val_loss = 0
	val_size = len(val_l)

	with torch.no_grad():

		for video_id in range(val_size):
			s = video_id * 3
			e = s + 3
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

			syn_label = y_val[s:e]
			syn_label = torch.from_numpy(syn_label)
			syn_label = syn_label.long()
			syn_label = idx2onehot(syn_label.view(-1,1))
			syn_label = syn_label.view(-1,20)
			# print(syn_label.size())
			# print(syn_label)
			syn_label = syn_label.cuda()
			syn_label = Variable(syn_label)


			optimizer.zero_grad()
			video_reconstruct, mu, logvar = cross_CVAE(audio_data_gt, syn_label)

			# loss
			#loss = calculate_loss(audio_data_gt, audio_reconstruct,  mu, logvar, out_dim_audio)
			loss = calculate_loss(audio_data_gt, visual_data_input, video_reconstruct, mu, logvar)
			#print(loss)
			val_loss += loss.item()

	return val_loss


def test_cmm(input_feature, syn_label_prev, output_feature, syn_label):
	cross_CVAE.eval()
	input_feature = input_feature.float()
	output_feature = output_feature.float()
	#distance= 0
	reconstructed_x1, mu1, logvar1 = cross_CVAE(input_feature, syn_label_prev)
	z1 = cross_CVAE.reparametrize(mu1,logvar1)
	reconstructed_x2, mu2, logvar2 = cross_CVAE(output_feature, syn_label)
	z2 = cross_CVAE.reparametrize(mu2,logvar2)
	loss_MSE = nn.MSELoss()
	#loss_MSE = nn.MSELoss(reduce = False, size_average = False)
	mse_loss = loss_MSE(input_feature,reconstructed_x2)
	kl_loss1 = -0.5 * torch.sum((1 + logvar1 - mu1.pow(2) - logvar1.exp()))
	kl_loss2 = -0.5 * torch.sum((1 + logvar2 - mu2.pow(2) - logvar2.exp()))
	#print("mu:",mu1)
	#print("logvar:",logvar1)
	# loss1 = calculate_loss(input_feature, output_feature, reconstructed_x1, mu1, logvar1)
	# distance_loss = loss1.item()
	# loss2 = euclidean_dis(output_feature, reconstructed_x1)
	loss2 = euclidean_dis(z1, z2)
	distance_euc1 = loss2.item()
	loss3 = euclidean_dis(reconstructed_x2, input_feature)
	distance_euc2 = loss3.item()
	return kl_loss1.item() - kl_loss2.item()




if __name__ == '__main__':

	input_dim_visual = 128 * 10
	hidden_dim = 100 * 10
	latent_dim = 80 * 10
	out_dim_audio = 128 * 10
	batch_size = 3
	epoch_nb = 10
	n_classes = 20
	training_size = len(train_l)
	testing_size = len(test_l)
	val_size = len(val_l)

	# cross_CVAE = CVAE(input_dim_visual, hidden_dim, latent_dim, out_dim_audio, n_classes) 
	# cross_CVAE = cross_CVAE.cuda()

	# for epoch in range(epoch_nb):


	# 	optimizer = optim.Adam(cross_CVAE.parameters(), lr = 0.00001)

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
	# 		#print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')


	# torch.save(cross_CVAE,'CVAE_cml_sequence.pkl')
	# print("Training completed!")


###### Testing, codes modified based on AVE - cmm_test.py ######

	cross_CVAE = torch.load('CVAE_cml_sequence.pkl')
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

	    syn_label1 = np.ones(1)
	    syn_label1 = torch.from_numpy(syn_label1)
	    syn_label1 = syn_label1.long()
	    syn_label1 = idx2onehot(syn_label1.view(-1,1))
	    syn_label1 = syn_label1.cuda()
	    syn_label1 = Variable(syn_label1)


	    syn_label0 = np.zeros(1)
	    syn_label0 = torch.from_numpy(syn_label0)
	    syn_label0 = syn_label0.long()
	    syn_label0 = idx2onehot(syn_label0.view(-1,1))
	    syn_label0 = syn_label0.cuda()
	    syn_label0 = Variable(syn_label0)

	    # print("x_test_video_feature:", x_test_video_feature[0:1,:])
	    # print("x_test_audio_feautre:", x_test_audio_feautre[0:1,:])


	    #given audio clip
	    score = []
	    for nn_count in range(10 - l + 1):
	        s1 = 0
	        s0 = 0
	        s = 0
	        for i in range(l):
	        	s0 += test_cmm(x_test_video_feature[nn_count + i:nn_count + i + 1, :], syn_label0, x_test_audio_feautre[seg[i:i + 1], :], syn_label1)
	        	s1 += test_cmm(x_test_video_feature[nn_count + i:nn_count + i + 1, :], syn_label1, x_test_audio_feautre[seg[i:i + 1], :], syn_label1)
	        	s += min(s0,s1)
	        score.append(s0)
	    score = np.array(score).astype('float32')
	    #print(score)
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


	    #given video clip
	    score = []
	    for nn_count in range(10 - l + 1):
	        s0 = 0
	        s1 = 0
	        s = 0
	        for i in range(l):
	            s1 += test_cmm(x_test_video_feature[seg[i:i + 1], :], syn_label1, x_test_audio_feautre[nn_count + i:nn_count + i + 1, :], syn_label1)
	            s0 += test_cmm(x_test_video_feature[seg[i:i + 1], :], syn_label1, x_test_audio_feautre[nn_count + i:nn_count + i + 1, :], syn_label0)
	            s += min(s0,s1)
	        score.append(s)
	    score = np.array(score).astype('float32')
	    #print(score)

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

	    # score = []
	    # for nn_count in range(10 - l + 1):
	    # 	s0 = test_cmm(x_test_video_feature[seg[0:l], :], syn_label1, x_test_audio_feautre[nn_count:nn_count + l, :], syn_label1)
	    # 	s1 = test_cmm(x_test_video_feature[seg[0:l], :], syn_label1, x_test_audio_feautre[nn_count:nn_count + l, :], syn_label0)
	    # 	s = min(s0,s1)
	    # 	score.append(s)
	    # score = np.array(score).astype('float32')
	    # print(score)

	    # if np.argmin(score) == seg[0]:
	    #     video_count += 1
	    # pred_aid = np.zeros(10)
	    # id = int(np.argmin(score))
	    # for i in range(id, id + int(l)):
	    #     pred_aid[i] = 1
	    # audio_acc += compute_precision(x_test, pred_aid)
	    # pos_len += len(seg)
	    # # calculate single accuracy
	    # ind = np.where(x_test - pred_aid == 0)[0]
	    # acc_a = len(ind)	    	


	    #print('num:{}, {}'.format(video_id, dataset[test_l[video_id]].rstrip('\n')))
	    # print('vid_input: ', x_test, 'pred:', pred_vid, 'correct: ', acc_v)
	    # print('aud_input: ', x_test, 'pred:', pred_aid, 'correct: ', acc_a)

	print(video_count * 100 / count_num, audio_count * 100 / count_num)