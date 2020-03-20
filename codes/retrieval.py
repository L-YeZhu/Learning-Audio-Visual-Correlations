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
from model_components import *
from retrieval_components import *


####### Dataloader #########
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
labels = np.array(labels) ## 4143 * 10 * 29


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
        # class_train[10*i + j, :] = labels[id, j, :]
        class_temp = np.array(np.nonzero(labels[id,j,:]))
        class_train[10*i + j] = class_temp[0,0] 
        # class_train = 

for i in range(len(val_l)):
    id = val_l[i]
    for j in range(10):
        x_audio_val[10 * i + j, :] = audio_features[id, j, :]
        x_video_val[10 * i + j, :] = video_features[id, j, :]
        y_val[10 * i + j] = closs_labels[id, j]
        #class_val[10*i + j, :] = labels[id, j, :]
        class_temp = np.array(np.nonzero(labels[id,j,:]))
        class_val[10*i + j] = class_temp[0,0]

for i in range(len(test_l)):
    id = test_l[i]
    for j in range(10):
        x_audio_test[10 * i + j, :] = audio_features[id, j, :]
        x_video_test[10 * i + j, :] = video_features[id, j, :]
        y_test[10 * i + j] = closs_labels[id, j]
        #class_test[10*i + j, :] = labels[id, j, :]
        class_temp = np.array(np.nonzero(labels[id,j,:]))
        class_test[10*i + j] = class_temp[0,0]

def norm(x):
	m,n = x.shape
	#print(m,n)
	for i in range(m):
		x[i,:] /= np.max(np.abs(x[i,:]))
	return x


x_video_train = norm(x_video_train)
x_audio_train = norm(x_audio_train)
x_video_val = norm(x_video_val)
x_audio_val = norm(x_audio_val)
x_video_test = norm(x_video_test)
x_audio_test = norm(x_audio_test)


print("data loading finished!")


########################Database for retrieval###############################

retrieval_test_audio = np.zeros((len(test_l),128))
retrieval_test_visual = np.zeros((len(test_l),512))
retrieval_test_label = np.zeros(len(test_l))

for video_id in range(len(test_l)):
	s = video_id * 10
	e = s + 10

	visual_data_input = x_video_test[s:e,:]
	audio_data_gt = x_audio_test[s:e, :]
	class_label = class_test[s:e]
	nobg_index = np.where(class_label != 28)[0]	

	pick_id = np.random.randint(len(nobg_index))
	retrieval_test_visual[video_id,:] = visual_data_input[nobg_index[pick_id],:]
	retrieval_test_audio[video_id,:] = audio_data_gt[nobg_index[pick_id],:]
	retrieval_test_label[video_id] = class_label[nobg_index[pick_id]]

print("Retrieval Database created!")


################################################################################

def calculate_score_va(video_feature, audio_feature):
	vae_audio.eval()
	vae_video.eval()
	video_feature = video_feature.float()
	audio_feature = audio_feature.float()
	pair = torch.cat((video_feature, audio_feature),1)
	loss_MSE = nn.MSELoss()
	with torch.no_grad():
	#distance= 0
		reconstructed_audio, mu1, logvar1 = vae_video(video_feature)
		reconstructed_video, mu2, logvar2 = vae_audio(audio_feature)
		loss1 = euclidean_dis(mu1,mu2)
		loss2 = loss_MSE(reconstructed_audio, reconstructed_video)
	return loss1.item()*0.5 + loss2.item()*0.5


def calculate_score_vv(video_feature1, video_feature2):
	vae_audio.eval()
	vae_video.eval()
	video_feature1 = video_feature1.float()
	video_feature2 = video_feature2.float()
	loss_MSE = nn.MSELoss()
	#pair = torch.cat((video_feature, audio_feature),1)
	with torch.no_grad():
	#distance= 0
		reconstructed_v1, mu1, logvar1 = vae_video(video_feature1)
		reconstructed_v2, mu2, logvar2 = vae_video(video_feature2)
		loss1 = euclidean_dis(mu1,mu2)
		loss2 = loss_MSE(reconstructed_v1, reconstructed_v2)
	return loss1.item()*0.5 + loss2.item()*0.5


def calculate_score_av(audio_feature, video_feature):
	vae_audio.eval()
	vae_video.eval()
	video_feature = video_feature.float()
	audio_feature = audio_feature.float()
	pair = torch.cat((video_feature, audio_feature),1)
	loss_MSE = nn.MSELoss()
	with torch.no_grad():
	#distance= 0
		reconstructed_audio, mu1, logvar1 = vae_video(video_feature)
		reconstructed_video, mu2, logvar2 = vae_audio(audio_feature)
		loss1 = euclidean_dis(mu1,mu2)
		loss2 = loss_MSE(reconstructed_audio, reconstructed_video)
	return loss1.item()*0.5 + loss2.item()*0.5


def calculate_score_aa(video_feature1, video_feature2):
	vae_audio.eval()
	vae_video.eval()
	video_feature1 = video_feature1.float()
	video_feature2 = video_feature2.float()
	loss_MSE = nn.MSELoss()
	#pair = torch.cat((video_feature, audio_feature),1)
	with torch.no_grad():
	#distance= 0
		reconstructed_v1, mu1, logvar1 = vae_audio(video_feature1)
		reconstructed_v2, mu2, logvar2 = vae_audio(video_feature2)
		loss1 = euclidean_dis(mu1,mu2)
		loss2 = loss_MSE(reconstructed_v1, reconstructed_v2)
	return loss1.item()*0.5 + loss2.item()*0.5




def find_sub_min(arr, n):
    arr_ = arr
    for i in range(n):
        arr_ = arr
        arr_[np.argmin(arr_)] = np.max(arr)
        #arr = arr_
    return int(np.argmin(arr_))


def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x,reconstructed_x,2)
	return dis

def euc(x1,x2):
	dis = torch.dist(x1,x2,2)
	return dis.item()



def msvae_mrr():
	vae_video.eval()
	vae_audio.eval()
	testing_size = len(test_l)
	rr_vv = 0
	rr_va = 0
	rr_av = 0
	rr_aa = 0

	f = open("data/Annotations.txt", 'r')
	dataset = f.readlines() 

	###given the visual input###
	for video_id in range(testing_size):
		print('num:{}, {}'.format(video_id, dataset[test_l[video_id]].rstrip('\n')))
		v_input = retrieval_test_visual[video_id,:]
		v_input = np.expand_dims(v_input, axis=0)
		v_input = torch.from_numpy(v_input)
		v_input = v_input.float()
		v_input = v_input.cuda()
		v_input = Variable(v_input)
		v_label = retrieval_test_label[video_id]

		score_v = np.zeros(testing_size)
		score_a = np.zeros(testing_size)
		for i in range(testing_size):
			visual_target = retrieval_test_visual[i,:]
			visual_target = np.expand_dims(visual_target, axis=0)
			audio_target = retrieval_test_audio[i,:]
			audio_target = np.expand_dims(audio_target, axis=0)
			visual_target = torch.from_numpy(visual_target)
			visual_target = visual_target.cuda()
			visual_target = Variable(visual_target)
			audio_target = torch.from_numpy(audio_target)
			audio_target = audio_target.float()
			audio_target = audio_target.cuda()
			audio_target = Variable(audio_target)
			score_v[i] = calculate_score_vv(v_input, visual_target)
			score_a[i] = calculate_score_va(v_input, audio_target)


		score_v = np.delete(score_v, video_id)
		score_index_v = np.argsort(score_v)
		#print(score_index_v)
		score_index_a = np.argsort(score_a)


		for j in range(testing_size-1):
			if v_label == retrieval_test_label[score_index_v[j]]: 
				rr_vv += 1/(j+1)
				print("found first index for vv:",j+1)
				break
		print("top5 retrieved for vv:", v_label, retrieval_test_label[score_index_v[0:5]], score_index_v[0:5])
		for j in range(testing_size):
			if v_label == retrieval_test_label[score_index_a[j]]:
				rr_va += 1/(j+1)
				print("found first index for va:",j+1)
				break
		print("top5 retrieved for va:", v_label, retrieval_test_label[score_index_a[0:5]], score_index_a[0:5])

	print("v-v:",rr_vv/testing_size)
	print("v-a:",rr_va/testing_size)
	vv_mrr = rr_vv / testing_size
	va_mrr = rr_va / testing_size

#########################################################################
	###given the aduio input###
	for audio_id in range(testing_size):
		print('num:{}, {}'.format(audio_id, dataset[test_l[audio_id]].rstrip('\n')))
		a_input = retrieval_test_audio[audio_id,:]
		a_input = np.expand_dims(a_input, axis=0)
		#print(v_input.shape)
		a_input = torch.from_numpy(a_input)
		a_input = a_input.float()
		a_input = a_input.cuda()
		a_input = Variable(a_input)
		a_label = retrieval_test_label[audio_id]

		score_v = np.zeros(testing_size)
		score_a = np.zeros(testing_size)
		# top_v = np.zeros(5)
		# top_a = np.zeros(5)
		for i in range(testing_size):
			visual_target = retrieval_test_visual[i,:]
			visual_target = np.expand_dims(visual_target, axis=0)
			audio_target = retrieval_test_audio[i,:]
			audio_target = np.expand_dims(audio_target, axis=0)
			visual_target = torch.from_numpy(visual_target)
			visual_target = visual_target.float()
			visual_target = visual_target.cuda()
			visual_target = Variable(visual_target)
			audio_target = torch.from_numpy(audio_target)
			audio_target = audio_target.cuda()
			audio_target = Variable(audio_target)
			score_v[i] = calculate_score_av(a_input, visual_target)
			score_a[i] = calculate_score_aa(a_input, audio_target)


		score_a = np.delete(score_a, audio_id)
		score_index_a = np.argsort(score_a)


		for j in range(testing_size-1):
			if a_label == retrieval_test_label[score_index_v[j]]:
				rr_av += 1/(j+1)
				print("found first index for av:",j+1)
				break
		print("top5 retrieved for av:", a_label, retrieval_test_label[score_index_v[0:5]], score_index_v[0:5])
		for j in range(testing_size):
			if a_label == retrieval_test_label[score_index_a[j]]:
				rr_aa += 1/(j+1)
				print("found first index for aa:",j+1)
				break
		print("top5 retrieved for aa:", a_label, retrieval_test_label[score_index_a[0:5]], score_index_a[0:5])

	print("a-v:",rr_av/testing_size)
	print("a-a:",rr_aa/testing_size)
	av_mrr = rr_av / testing_size
	aa_mrr = rr_aa / testing_size

	return vv_mrr, va_mrr, av_mrr, aa_mrr







if __name__ == '__main__':


	######################## MS-VAE #######################
	latent_dim = 100
	training_size = len(train_l)
	testing_size = len(test_l)
	val_size = len(val_l)
	k = 8
	epoch_nb = 15

	audio_encode = audio_encoder(latent_dim)
	audio_decode = general_decoder(latent_dim)
	video_encode = visual_encoder(latent_dim)
	video_decode = general_decoder(latent_dim)
	vae_audio = VAE(latent_dim, audio_encode, audio_decode)
	vae_video = VAE(latent_dim, video_encode, video_decode)
	vae_audio.cuda()
	vae_video.cuda()

	vae_audio.load_state_dict(torch.load('msvae_audio_nobg1.pkl'))
	vae_video.load_state_dict(torch.load('msvae_video_nobg1.pkl'))


	vv_mrr, va_mrr, av_mrr, aa_mrr = msvae_mrr()

	print(vv_mrr, va_mrr, av_mrr, aa_mrr)
	