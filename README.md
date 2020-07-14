# Concurrently Extrapolating and Interpolating Networks for Continuous Model Generation

## 1. Supplementary Material for "Continuous Model Generation of Image Smoothing Operators Based on Deep Convolutional Neural Networks" can be downloaded from [this website](https://github.com/mdcnn/CEIN/blob/master/elsarticle-template.pdf) or [BaiDu with codes：j6gu](https://pan.baidu.com/s/1oanRGfzdq_x2DhMTnRndJA)

### Generalizability of DSA module for image super-resolution
### The codes for the networks of DAS-improved RNAN and RNAN（Our implementation of image super-resolution is achieved based on [the RNAN source codes](https://github.com/yulunzhang/RNAN)，so you need to put the  python code of our DAS-improved RNAN （rnan_das.py） into the RNAN framework when you wish to train a model with your dataset）

### Some visual comparisons


<p align='center'>  
  <img src='Fig/Fig8.png' width='1200'/>
Fig. 8 The visual comparison of image smoothing results when using different network architectures.
 

<p align='center'>  
  <img src='Fig/Fig9.png' width='1200'/>
Fig. 9 The visual comparison of image smoothing results when using different network architectures.


<p align='center'>  
  <img src='Fig/Fig10.png' width='1200'/>
Fig. 10 The visualization of smoothed images (in the first row) predicted with models generated by our CEI and the residuals (in the second row) between these images and the input image, when only a set of specific-effect label images generated is given for training.
  

<p align='center'>  
  <img src='Fig/Fig11.png' width='1200'/>
Fig. 11 The visualization of smooth transition produced by the proposed CEI when the training datasets of Van Gogh's style and Cezanne's style are given for training. (a) Input images, (b, d, f) the generated images using our CEI, (c) generated Van Gogh's style images, (e) the generated Cezanne's style images
  
  
<p align='center'>  
  <img src='Fig/Fig12.png' width='1200'/>
Fig. 12 The visualization of smooth transition produced by the proposed CEI when the training dataset of only Van Gogh's style is given for training. (a) Input images, (b, d, e, f) the generated images using our CEI, (c) the generated Van Gogh's style images
