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


class visual_encoder(nn.Module):
	def __init__(self, latent_dim):
		super(visual_encoder,self).__init__()

		self.lin_lays = nn.Sequential(
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
        )
		self.mu = nn.Linear(128, latent_dim)
		self.var = nn.Linear(128, latent_dim)


	def forward(self, x):
		# x shape: [batch_size, input_dim]
		hidden = self.lin_lays(x)
		# hidden shape: [batch_size, hidden_dim]
		# latent parameters
		mean = self.mu(hidden)

		log_var = self.var(hidden)

		return mean, log_var


class audio_encoder(nn.Module):
	def __init__(self, latent_dim):
		super(audio_encoder,self).__init__()

		self.lin_lays = nn.Sequential(
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
        )

		self.mu = nn.Linear(128, latent_dim)
		self.var = nn.Linear(128, latent_dim)


	def forward(self, x):
		# x shape: [batch_size, input_dim]
		hidden = self.lin_lays(x)
		mean = self.mu(hidden)
		log_var = self.var(hidden)

		return mean, log_var



class general_decoder(nn.Module):
	def __init__(self, latent_dim):
		super(general_decoder, self).__init__()

		self.lin_lays = nn.Sequential(
			nn.Linear(latent_dim, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 640),
			nn.ReLU(),
			nn.Linear(640, 640),
			nn.ReLU(),
			nn.Linear(640, 640),
			)
		
	def forward(self, x):
		generated_x = self.lin_lays(x)
		return generated_x