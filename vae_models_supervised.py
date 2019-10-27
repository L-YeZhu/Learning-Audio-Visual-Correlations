"""
Created by Ye ZHU
last modified on Oct 27, 2019
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


def norm(x):
	m,n = x.shape
	#print(m,n)
	for i in range(m):
		x[i,:] /= np.max(np.abs(x[i,:]))
	return x

# x_video_train = norm(x_video_train)
# x_audio_train = norm(x_audio_train)
# x_video_val = norm(x_video_val)
# x_audio_val = norm(x_audio_val)
# x_video_test = norm(x_video_test)
# x_audio_test = norm(x_audio_test)


print("data loading finished!")


def euclidean_dis(x, reconstructed_x):
    dis = torch.dist(x,reconstructed_x,2)
    return dis


def CLF_loss(z_audio, z_video, target_label):
    z_total = torch.cat((z_audio, z_video),1)
    clf_out = clf_module(z_total)
    #clf_out_audio = clf_module(z_audio)
    #clf_out_video = clf_module(z_video)
    criterion = nn.BCELoss()
    clf_loss = criterion(clf_out, target_label)
    #clf_loss1 = criterion(clf_out_audio, target_label)
    #clf_loss2 = criterion(clf_out_video, target_label)
    return clf_loss




def caluculate_loss_generaldec(x_visual, x_audio, x_reconstruct, mu, logvar, epoch, event_label):
    loss_MSE = nn.MSELoss()
    x_input = torch.cat((x_visual, x_audio), 1)
    #bs = x_reconstruct.size(0)
    mse_loss = loss_MSE(x_input, x_reconstruct)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    _ , mu1, logvar1 = vae_audio(x_audio)
    z1 = vae_audio.reparameterize(mu1, logvar1)
    _, mu2, logvar2 = vae_video(x_visual)
    z2 = vae_video.reparameterize(mu2, logvar2)
    #z = torch.cat((z1, z2),1)
    clf_loss = CLF_loss(z1, z2, event_label)
    latent_loss = euclidean_dis(z1,z2)
    if epoch < 10:
        final_loss = mse_loss + kl_loss*0.1 + latent_loss + clf_loss*0
    else:
        final_loss = mse_loss + kl_loss*0.01 + latent_loss + clf_loss*0
    #bce_loss = F.binary_cross_entropy(x_reconstruct, x_input, size_average = False)
    #print(lambda1, lambda2)
    return final_loss, kl_loss, mse_loss, latent_loss, clf_loss


def train_video_generaldec(epoch):
    #vae_audio.train()
    vae_video.train()
    clf_module.train()
    train_loss = 0
    kl_loss = 0
    mse_loss = 0
    latent_loss = 0
    clf_loss = 0
    training_size = len(train_l)
    #optimizer_video.zero_grad()


    for video_id in range(training_size):
        s = video_id * 10
        e = s + 10
        visual_data_input = x_video_train[s:e,:]
        visual_data_input = torch.from_numpy(visual_data_input)
        visual_data_input = visual_data_input.float()
        visual_data_input = visual_data_input.cuda()
        #visual_data_input = norm_video(visual_data_input)
        visual_data_input = Variable(visual_data_input)

        audio_data_gt = x_audio_train[s:e,:]
        audio_data_gt = torch.from_numpy(audio_data_gt)
        audio_data_gt = audio_data_gt.float()
        audio_data_gt = audio_data_gt.cuda()
        #audio_data_gt = norm_audio(audio_data_gt)
        audio_data_gt = Variable(audio_data_gt)

        event_label = y_train[s:e]
        event_label = np.expand_dims(event_label, axis=1)
        event_label = torch.from_numpy(event_label)
        event_label = event_label.float()
        event_label = event_label.cuda()
        event_label = Variable(event_label)



        optimizer_video.zero_grad()
        optimizer_clf.zeros()
        # x_reconstruct should be in size of 512 + 128
        if epoch == 0:
            x_reconstruct, mu, logvar = vae_video(visual_data_input)
        else:
            x_reconstruct, mu, logvar = vae_video(visual_data_input,vae_audio)

        loss, kl, mse, latent, clf = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch, event_label)

        loss.backward()
        train_loss += loss.item()
        kl_loss += kl.item()
        mse_loss += mse.item()
        latent_loss += latent.item()
        clf_loss += clf.item()


        optimizer_video.step()
        optimizer_clf.step()
        #optimizer_audio.step()

    return train_loss, kl_loss, mse_loss



def train_audio_generaldec(epoch):
    vae_audio.train()
    clf_module.train()
    #vae_video.train()
    train_loss = 0
    kl_loss = 0
    mse_loss = 0
    latent_loss = 0
    clf_loss = 0
    training_size = len(train_l)

    for video_id in range(training_size):
        s = video_id * 10
        e = s + 10
        visual_data_input = x_video_train[s:e,:]
        visual_data_input = torch.from_numpy(visual_data_input)
        visual_data_input = visual_data_input.float()
        visual_data_input = visual_data_input.cuda()
        #visual_data_input = norm_video(visual_data_input)
        visual_data_input = Variable(visual_data_input)

        audio_data_gt = x_audio_train[s:e,:]
        audio_data_gt = torch.from_numpy(audio_data_gt)
        audio_data_gt = audio_data_gt.float()
        audio_data_gt = audio_data_gt.cuda()
        #audio_data_gt = norm_audio(audio_data_gt)
        audio_data_gt = Variable(audio_data_gt)


        event_label = y_train[s:e]
        event_label = np.expand_dims(event_label, axis=1)
        event_label = torch.from_numpy(event_label)
        event_label = event_label.float()
        event_label = event_label.cuda()
        event_label = Variable(event_label)

        optimizer_audio.zero_grad()
        optimizer_clf.zero_grad()
        ## x_reconstruct should be in size of 512 + 128, same decoder is used 
        x_reconstruct, mu, logvar = vae_audio(audio_data_gt, vae_video)
        loss, kl, mse = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct, mu, logvar, epoch, event_label)      

        loss.backward()
        train_loss += loss.item()
        kl_loss += kl.item()
        mse_loss += mse.item()

        optimizer_audio.step()
        optimizer_clf.step()
        #optimizer_video.step()

    return train_loss, kl_loss, mse_loss


def train_generaldec(epoch):
    vae_audio.train()
    vae_video.train()
    clf_module.train()
    train_loss = 0
    kl_loss = 0
    mse_loss = 0
    latent_loss = 0
    clf_loss = 0

    training_size = len(train_l)

    for video_id in range(training_size):
        s = video_id * 10
        e = s + 10

        visual_data_input = x_video_train[s:e,:]
        visual_data_input = torch.from_numpy(visual_data_input)
        visual_data_input = visual_data_input.float()
        visual_data_input = visual_data_input.cuda()
        #visual_data_input = norm_video(visual_data_input)
        visual_data_input = Variable(visual_data_input)

        audio_data_gt = x_audio_train[s:e,:]
        audio_data_gt = torch.from_numpy(audio_data_gt)
        audio_data_gt = audio_data_gt.float()
        audio_data_gt = audio_data_gt.cuda()
        #audio_data_gt = norm_audio(audio_data_gt)
        audio_data_gt = Variable(audio_data_gt)


        event_label = y_train[s:e]
        event_label = np.expand_dims(event_label, axis=1)
        event_label = torch.from_numpy(event_label)
        event_label = event_label.float()
        event_label = event_label.cuda()
        event_label = Variable(event_label)



        optimizer_audio.zero_grad()
        optimizer_video.zero_grad()
        optimizer_clf.zero_grad()

        if epoch == 0:
            x_reconstruct_from_v, mu1, logvar1 = vae_video(visual_data_input)
        else:
            x_reconstruct_from_v, mu1, logvar1 = vae_video(visual_data_input,vae_audio)

        loss1, kl1, mse1, latent1, clf1 = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct_from_v, mu1, logvar1, epoch, event_label)

        x_reconstruct_from_a, mu2, logvar2 = vae_audio(audio_data_gt, vae_video)
        loss2, kl2, mse2, latent2, clf2 = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct_from_a, mu2, logvar2, epoch, event_label)

        loss = loss1 + loss2
        kl = kl1 + kl2
        mse = mse1 + mse2
        latent = latent1 + latent2
        clf = clf1 + clf2

        loss.backward()
        train_loss += loss.item()
        kl_loss += kl.item()
        mse_loss += mse.item()
        latent_loss += latent.item()
        clf_loss += clf.item()

        optimizer_video.step()
        optimizer_audio.step()
        optimizer_clf.step()

    return train_loss, kl_loss, mse_loss, latent_loss, clf_loss



# def train_clf(epoch):
#     clf_module.train()
#     clf_loss = 0
    



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

    ## Cross VAE Model
    audio_encode = audio_encoder(latent_dim)
    #audio_decode = audio_decoder(latent_dim)
    video_encode = visual_encoder(latent_dim)
    #video_decode = visual_decoder(latent_dim)
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

    ################# Classificatuon module ######################
    clf_module = classifier(latent_dim)
    clf_module.cuda()
    optimizer_clf = optim.Adam(clf_module.parameters(), lr = 0.00001)

    for epoch in range(epoch_nb):

        print(f'Training with general decoder --- video to all:')
        train_loss = 0
        train_loss, kl_loss, mse_loss, latent_loss, clf_loss = train_generaldec(epoch)
        train_loss /= training_size
        kl_loss /= training_size
        mse_loss /= training_size
        latent_loss /= training_size
        clf_loss /= training_size
        print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
        print("KL, MSE, LATENT AND CLF loss:", kl_loss, mse_loss, latent_loss, clf_loss)

    #else:

        # print(f'Training with general decoder --- audio to all:')
        # train_loss = 0
        # train_loss, kl_loss, mse_loss = train_audio_generaldec(epoch)
        # train_loss /= training_size
        # kl_loss /= training_size
        # mse_loss /= training_size
        # print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
        # print("KL and MSE loss:", kl_loss, mse_loss)      

        # if (epoch == 9):
        #   torch.save(vae_audio.state_dict(), 'vae_audio_epoch1_v2a.pkl')
        #   torch.save(vae_video.state_dict(), 'vae_video_epoch1_v2a.pkl')


    torch.save(vae_audio.state_dict(), 'vae_audio_supervised_test2.pkl')
    torch.save(vae_video.state_dict(), 'vae_video_supervised_test2.pkl')
    torch.save(clf_module.state_dict(),'clf_supervised_test2.pkl')
    #tsne()