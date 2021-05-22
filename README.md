## Learning Audio-Visual Correlations from Variational Cross-Modal Generations

This is the code implementation for the ICCASP2021 paper [Learning Audio-Visual Correlations from Variational Cross-Modal Generations](https://arxiv.org/pdf/2102.03424.pdf). In this work, we propose a Variational Autencoder with Multiple encoders and a Shared decoder (MS-VAE) framework for processing the data from visual and audio modalities. We use the [AVE dataset](https://github.com/YapengTian/AVE-ECCV18) for experiments, and thank the authors of the previous work for sharing their codes and data.

<p align="center">
<img src="https://github.com/L-YeZhu/Learning-Audio-Visual-Correlations/blob/master/fig1.png" width="500">
  </p>

#### 1. We implement the project using
Python 3.6 <br />
Pytorch 1.2

#### 2. Training and pre-trained models
Please download the audio and visual features from [here](https://github.com/YapengTian/AVE-ECCV18), and place the data files in the data folder. Note that we use the features for CML task for experiments. <br />
To train the model, run the <code>msvae.py</code>. <br />
For the cross-modal localization task, run the <code>cml.py</code>. <br />
For the cross-modal retrieval task, run the <code>retrieval.py</code>. <br />
The pre-trained models are also available for download: [audio](https://drive.google.com/file/d/1uEMsmd70xucCTeaC3EcsXqOU79tq7DhP/view?usp=sharing) and [visual](https://drive.google.com/file/d/17nmKWUX-nXByadPU5sgUeIqqrHUI5FBk/view?usp=sharing).


#### 3. Citation
Please consider citing our paper if you find it useful.
```
@InProceedings{zhu2021learning,    
  author = {Zhu, Ye and Wu, Yu and Latapie, Hugo and Yang, Yi and Yan, Yan},    
  title = {Learning Audio-Visual Correlations from Variational Cross-Modal Generation},    
  booktitle = {International Conference on Acoustics, Speech, and Signal Processing(ICCASP)},    
  year = {2021} 
  }
```

