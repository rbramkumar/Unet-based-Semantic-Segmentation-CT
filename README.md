# Unet-based-Semantic-Segmantation-CT

This is a Keras implementation of semantic segmentation of CT images for extraocular muscles that was used for our work here: (link to paper). The intent with using Keras was to enable training/reuse using any back-end i.e., Tensorflow, Caffe, PyTorch etc. We evaluate the 2D Unet architecture proposed by Ronneberger et al. (https://arxiv.org/pdf/1505.04597.pdf), and V-net architecture by Milletari et al. (https://arxiv.org/pdf/1606.04797.pdf), and also an custom implementation specific to Medical imaging by using ensemble 2D segmentations to come up with the volumetric segmentation. 



### Credits



