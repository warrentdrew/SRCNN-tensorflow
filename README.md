# SRCNN-tensorflow
## Overview
Imaplementation of SRCNN algorithm. The original Matlab and Caffe from official website can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

## Files
* Test/: Testing dataset
* Train/: Training dataset
* main_process.py: Main training/testing file
* srcnn_model.py: Define srcnn model class
* utils.py: Data preprocessing functions
* preprocess.m: Preprocess the trainong/testing images including image cropping & normalization (**Matlab code**)
* modcrop.m: Crop the suitable size for the specific scale(**Matlab code**)

## Usage
For training, python main_process.py 
For testing, python main_process.py --is_train False --extract_stride 21
## Result
**TODO**

## References
* Original paper: [Image Super-Resolution Using Deep Convolutional Networks](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)
	* Some of the codes are based on this repository that the image preprocessing implemented by scipy.



 