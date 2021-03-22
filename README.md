## Learning Audio-Visual Correlations from Variational Cross-Modal Generations

This is the code implementation for the ICCASP2021 paper [Learning Audio-Visual Correlations from Variational Cross-Modal Generations](https://arxiv.org/pdf/2102.03424.pdf). In this work, we propose a Variational Autencoder with Multiple encoders and a Shared decoder (MS-VAE) framework for processing the data from visual and audio modalities. We use the [AVE dataset](https://github.com/YapengTian/AVE-ECCV18) for experiments, and thank the authors of the previous work for sharing their codes and data.

#### 1. We implement the project using
Python 3.6 <br />
Pytorch 1.2

#### 2. Please download the audio and visual features from [here](https://github.com/YapengTian/AVE-ECCV18), and place the data files in the data folder. 
To train the model, run the <code>msvae.py</code>. <br />
For the cross-modal localization task, run the <code>cml.py</code>. <br />
For the cross-modal retrieval task, run the <code>retrieval.py</code>. <br />

#### 3. Citation
Please consider citing our paper if you find it useful.
```
@InProceedings{zhu2020describing,    
  author = {Zhu, Ye and Wu, Yu and Latapie, Hugo and Yang, Yi and Yan, Yan},    
  title = {Learning Audio-Visual Correlations from Variational Cross-Modal Generation},    
  booktitle = {International Conference on Acoustics, Speech, and Signal Processing(ICCASP)},    
  year = {2021} 
  }
```

