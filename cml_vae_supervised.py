"""
Created by Ye ZHU
last modified on Oct 27, 2019
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
from model_components import *
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


def norm(x):
	m,n = x.shape
	#print(m,n)
	for i in range(m):
		x[i,:] /= np.max(np.abs(x[i,:]))
	return x

print("data loading finished!")



####################################

def calculate_loss(x, reconstructed_x, mu, logvar):

	loss_MSE = nn.MSELoss()
	mse_loss = loss_MSE(x,reconstructed_x)
	#kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim = 1)
	kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()))
	return mse_loss + kl_loss


def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x,reconstructed_x,2)
	return dis


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


def CLF_loss(z_audio, z_video):
    z = torch.cat((z_audio, z_video),1)
    clf_out = clf_module(z)
    print("clf_out:", clf_out)
    # clf_out_audio = clf_module(z_audio)
    # clf_out_video = clf_module(z_video)
    # # print(clf_out_audio)
    # # print(clf_out_video)
    criterion = nn.BCELoss()
    clf_loss_0 = criterion(clf_out, event_label_0)
    clf_loss_1 = criterion(clf_out, event_label_1)
    print("clf_loss_0:", clf_loss_0)
    print("clf_loss_1:", clf_loss_1)
    # clf_loss1 = criterion(clf_out_audio, event_label)
    # clf_loss2 = criterion(clf_out_video, event_label)
    # #return clf_out_audio + clf_out_video
    #print("clf1:",clf_loss1)
    #print("clf2:",clf_loss2)
    return clf_out



def test_cmm_a2v(video_feature, audio_feature):
	vae_audio.eval()
	vae_video.eval()
	clf_module.eval()
	video_feature = video_feature.float()
	audio_feature = audio_feature.float()
	with torch.no_grad():
	#distance= 0
		reconstructed_audio, mu1, logvar1 = vae_video(video_feature,vae_audio)
		reconstructed_video, mu2, logvar2 = vae_audio(audio_feature,vae_video)
		#print("mu:",mu1)
		#print("logvar:",logvar1)
		#loss1 = calculate_loss(output_feature, reconstructed_x, mu, logvar)
		#distance_loss = loss1.item()
		#loss1 = euclidean_dis(audio_feature, reconstructed_audio)
		#loss = calculate_loss(audio_feature, reconstructed_audio, mu1, logvar1)
		z1 = vae_video.reparameterize(mu1,logvar1)
		z2 = vae_audio.reparameterize(mu2,logvar2)
		loss1 = euclidean_dis(z1,z2)
		loss2 = euclidean_dis(reconstructed_audio, reconstructed_video)
		loss3 = CLF_loss(z1,z2)
		#distance_euc = loss2.item()
	return loss1.item()*0.5  + loss2.item()*0.5 + abs(loss3.item())*10


def test_cmm_v2a(video_feature, audio_feature):
	vae_audio_v2a.eval()
	vae_video_v2a.eval()
	video_feature = video_feature.float()
	audio_feature = audio_feature.float()
	with torch.no_grad():
	#distance= 0
		reconstructed_audio, mu1, logvar1 = vae_video_v2a(video_feature, vae_audio_v2a)
		reconstructed_video, mu2, logvar2 = vae_audio_v2a(audio_feature, vae_video_v2a)
		#print(reconstructed_audio.size())
		#print("mu:",mu1)
		#print("logvar:",logvar1)
		#loss1 = calculate_loss(output_feature, reconstructed_x, mu, logvar)
		#distance_loss = loss1.item()
		#loss1 = euclidean_dis(video_feature, reconstructed_video)
		#loss = calculate_loss(video_feature, reconstructed_video, mu2, logvar2)
		z1 = vae_video_v2a.reparameterize(mu1,logvar1)
		z2 = vae_audio_v2a.reparameterize(mu2,logvar2)
		loss1 = euclidean_dis(z1,z2)
		loss2 = euclidean_dis(reconstructed_audio, reconstructed_video)
		#distance_euc = loss2.item()
	return  loss1.item()* 0.5 + loss2.item()*0.5


#############################################################################

if __name__ == '__main__':

	latent_dim = 100
	training_size = len(train_l)
	testing_size = len(test_l)
	val_size = len(val_l)

	audio_encode = audio_encoder(latent_dim)
	audio_decode = general_decoder(latent_dim)
	video_encode = visual_encoder(latent_dim)
	video_decode = general_decoder(latent_dim)
	vae_audio = VAE(latent_dim, audio_encode, audio_decode)
	vae_video = VAE(latent_dim, video_encode, video_decode)
	vae_audio.cuda()
	vae_video.cuda()

	vae_audio.load_state_dict(torch.load('vae_audio_supervised_test2.pkl'))
	vae_video.load_state_dict(torch.load('vae_video_supervised_test2.pkl'))


	clf_module = classifier(latent_dim)
	clf_module.cuda()
	clf_module.load_state_dict(torch.load('clf_supervised_test2.pkl'))

	print("Model load completed!")

	event_label_0 = torch.tensor(np.array([0]))
	event_label_0 = event_label_0.float()
	event_label_0 = event_label_0.cuda()
	event_label_0 = Variable(event_label_0)

	event_label_1 = torch.tensor(np.array([1]))
	event_label_1 = event_label_1.float()
	event_label_1 = event_label_1.cuda()
	event_label_1 = Variable(event_label_1)


	x_video_train = norm(x_video_train)
	x_audio_train = norm(x_audio_train)
	x_video_val = norm(x_video_val)
	x_audio_val = norm(x_audio_val)
	x_video_test = norm(x_video_test)
	x_audio_test = norm(x_audio_test)

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
	    #x_test_video_feature = norm_video(x_test_video_feature)
	    x_test_video_feature = Variable(x_test_video_feature)

	    x_test_audio_feautre = x_audio_test[s:e, :]
	    x_test_audio_feautre = torch.from_numpy(x_test_audio_feautre)
	    x_test_audio_feautre = x_test_audio_feautre.float()
	    x_test_audio_feautre = x_test_audio_feautre.cuda()
	    #x_test_audio_feautre = norm_audio(x_test_audio_feautre)
	    x_test_audio_feautre = Variable(x_test_audio_feautre)

	    # print("x_test_video_feature:", x_test_video_feature[0:1,:])
	    # print("x_test_audio_feautre:", x_test_audio_feautre[0:1,:])


	    # given audio clip
	    score = []

	    # vae_audio.load_state_dict(torch.load('vae_audio_epoch1_a2v.pkl'))
	    # vae_video.load_state_dict(torch.load('vae_video_epoch1_a2v.pkl'))
	    # print("Model for A2V load completed!")
	    for nn_count in range(10 - l + 1):
	        s = 0
	        for i in range(l):
	        	s += test_cmm_a2v(x_test_video_feature[nn_count + i:nn_count + i + 1, :], x_test_audio_feautre[seg[i:i + 1], :])
	        score.append(s)
	    score = np.array(score).astype('float32')
	   # print(score)
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
	    # vae_audio.load_state_dict(torch.load('vae_audio_epoch1_v2a.pkl'))
	    # vae_video.load_state_dict(torch.load('vae_video_epoch1_v2a.pkl'))
	    # print("Model for V2A load completed!")
	    for nn_count in range(10 - l + 1):
	        s = 0
	        for i in range(l):
	            s += test_cmm_a2v(x_test_video_feature[seg[i:i + 1], :], x_test_audio_feautre[nn_count + i:nn_count + i + 1, :])
	        score.append(s)
	    score = np.array(score).astype('float32')
	   # print(score)
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

