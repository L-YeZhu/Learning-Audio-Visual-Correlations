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
with h5py.File('data/labels.h5', 'r') as hf:
	labels = hf['avadataset'][:]

closs_labels = np.array(closs_labels) ## 4143 * 10
audio_features = np.array(audio_features)  ##  4143 * 10 * 128
video_features = np.array(video_features)  ##  4143 * 10 * 512
closs_labels = closs_labels.astype("float32")
audio_features = audio_features.astype("float32")
video_features = video_features.astype("float32")
labels = np.array(labels)


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
class_train = np.zeros((len(train_l)*10))
class_val = np.zeros((len(val_l)*10))
class_test = np.zeros((len(test_l)*10))



##
for i in range(len(train_l)):
    id = train_l[i]
    for j in range(10):
        x_audio_train[10*i + j, :] = audio_features[id, j, :]
        x_video_train[10*i + j, :] = video_features[id, j, :]
        y_train[10*i + j] = closs_labels[id, j]
        class_temp = np.array(np.nonzero(labels[id,j,:]))
        class_train[10*i + j] = class_temp[0,0]

for i in range(len(val_l)):
    id = val_l[i]
    for j in range(10):
        x_audio_val[10 * i + j, :] = audio_features[id, j, :]
        x_video_val[10 * i + j, :] = video_features[id, j, :]
        y_val[10 * i + j] = closs_labels[id, j]
        class_temp = np.array(np.nonzero(labels[id,j,:]))
        class_val[10*i + j] = class_temp[0,0]

for i in range(len(test_l)):
    id = test_l[i]
    for j in range(10):
        x_audio_test[10 * i + j, :] = audio_features[id, j, :]
        x_video_test[10 * i + j, :] = video_features[id, j, :]
        y_test[10 * i + j] = closs_labels[id, j]
        class_temp = np.array(np.nonzero(labels[id,j,:]))
        class_test[10*i + j] = class_temp[0,0]

print("data loading finished!")



def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x,reconstructed_x,2)
	return dis

def avgpooling(x):
	m = nn.AvgPool2d(7)
	return m(x)

def caluculate_loss_generaldec(x_visual, x_audio, x_reconstruct, mu, logvar, epoch):
	loss_MSE = nn.MSELoss()
	x_input = torch.cat((x_visual, x_audio), 1)
	#bs = x_reconstruct.size(0)
	mse_loss = loss_MSE(x_input, x_reconstruct)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	_ , mu1, logvar1 = vae_audio(x_audio)
	z1 = vae_audio.reparameterize(mu1, logvar1)
	_, mu2, logvar2 = vae_video(x_visual)
	z2 = vae_video.reparameterize(mu2, logvar2)
	latent_loss = euclidean_dis(z1,z2)
	if epoch < 10:
		final_loss = mse_loss + kl_loss*0.1 + latent_loss
	else:
		final_loss = mse_loss + kl_loss*0.01 + latent_loss
	return final_loss, kl_loss, mse_loss, latent_loss

def train_video_generaldec(epoch):
	#vae_audio.train()
	vae_video.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	latent_loss = 0
	training_size = len(train_l)
	#optimizer_video.zero_grad()


	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10
		visual_data_input = x_video_train[s:e,:]
		audio_data_gt = x_audio_train[s:e,:]
		event_label = y_train[s:e]
		bg_index = np.where(event_label == 0)[0]
		visual_data_input = np.delete(visual_data_input, bg_index, axis=0)
		audio_data_gt = np.delete(audio_data_gt, bg_index, axis=0)


		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

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

		loss, kl, mse, latent = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch)

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		latent_loss += latent.item()

		optimizer_video.step()
		#optimizer_audio.step()

	return train_loss, kl_loss, mse_loss, latent_loss


def train_audio_generaldec(epoch):
	vae_audio.train()
	#vae_video.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	latent_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10
		visual_data_input = x_video_train[s:e, :]
		audio_data_gt = x_audio_train[s:e, :]
		event_label = y_train[s:e]
		bg_index = np.where(event_label == 0)[0]
		visual_data_input = np.delete(visual_data_input, bg_index, axis=0)
		audio_data_gt = np.delete(audio_data_gt, bg_index, axis=0)		

		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)


		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)

		optimizer_audio.zero_grad()
		## x_reconstruct should be in size of 512 + 128, same decoder is used 
		x_reconstruct, mu, logvar = vae_audio(audio_data_gt, vae_video)
		loss, kl, mse, latent = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch)		

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		latent_loss += latent.item()

		optimizer_audio.step()
		#optimizer_video.step()

	return train_loss, kl_loss, mse_loss, latent_loss



def train_generaldec(epoch):
	vae_audio.train()
	vae_video.train()
	train_loss = 0
	kl_loss = 0
	mse_loss = 0
	latent_loss = 0
	training_size = len(train_l)

	for video_id in range(training_size):
		s = video_id * 10
		e = s + 10

		visual_data_input = x_video_train[s:e,:]
		audio_data_gt = x_audio_train[s:e,:]
		event_label = y_train[s:e]
		bg_index = np.where(event_label == 0)[0]
		visual_data_input = np.delete(visual_data_input, bg_index, axis=0)
		audio_data_gt = np.delete(audio_data_gt, bg_index, axis=0)


		visual_data_input = torch.from_numpy(visual_data_input)
		visual_data_input = visual_data_input.float()
		visual_data_input = visual_data_input.cuda()
		visual_data_input = Variable(visual_data_input)

		audio_data_gt = torch.from_numpy(audio_data_gt)
		audio_data_gt = audio_data_gt.float()
		audio_data_gt = audio_data_gt.cuda()
		audio_data_gt = Variable(audio_data_gt)

		optimizer_audio.zero_grad()
		optimizer_video.zero_grad()

		if epoch == 0:
			x_reconstruct_from_v, mu1, logvar1 = vae_video(visual_data_input)
		else:
			x_reconstruct_from_v, mu1, logvar1 = vae_video(visual_data_input,vae_audio)

		loss1, kl1, mse1, latent1 = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct_from_v, mu1, logvar1, epoch)

		x_reconstruct_from_a, mu2, logvar2 = vae_audio(audio_data_gt, vae_video)
		loss2, kl2, mse2, latent2 = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct_from_a, mu2, logvar2, epoch)

		loss = loss1 + loss2
		kl = kl1 + kl2
		mse = mse1 + mse2
		latent = latent1 + latent2

		loss.backward()
		train_loss += loss.item()
		kl_loss += kl.item()
		mse_loss += mse.item()
		latent_loss += mse.item()

		optimizer_video.step()
		optimizer_audio.step()

	return train_loss, kl_loss, mse_loss, latent_loss



if __name__ == '__main__':

	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	input_dim_visual = 512
	latent_dim = 100
	#out_dim_audio = 128
	batch_size = 10
	epoch_nb = 15
	training_size = len(train_l)
	testing_size = len(test_l)
	val_size = len(val_l)

	audio_encode = audio_encoder(latent_dim)
	video_encode = visual_encoder(latent_dim)
	general_decode = general_decoder(latent_dim)
	vae_audio = VAE(latent_dim, audio_encode, general_decode)
	vae_video = VAE(latent_dim, video_encode, general_decode)

	vae_audio.cuda()
	vae_video.cuda()

	optimizer_audio = optim.Adam(vae_audio.parameters(), lr = 0.00001)
	optimizer_video = optim.Adam(vae_video.parameters(), lr = 0.00001)


	for epoch in range(epoch_nb):

		train_loss = 0
		train_loss, kl_loss, mse_loss, latent_loss = train_generaldec(epoch)
		train_loss /= training_size
		kl_loss /= training_size
		mse_loss /= training_size
		latent_loss /= training_size
		print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')



	torch.save(vae_audio.state_dict(), 'msvae_a.pkl')
	torch.save(vae_video.state_dict(), 'msvae_v.pkl')
