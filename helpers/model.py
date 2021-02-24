import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.initializers import *
from keras.utils import plot_model
from keras import backend as K
import time
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance

# This function will help us store the computation time as well (since our experiment also hypothesises on time taken)
class TimeHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.times = []

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)
		
time_callback = TimeHistory()



def dice_coefficient(y_true, y_pred, smooth=1.):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1-dice_coefficient(y_true, y_pred)

##############  WEIGHTED BINARY CROSS ENTROPY LOSS  #######################

def create_weighted_binary_crossentropy(zero_weight, one_weight):

	def weighted_binary_crossentropy(y_true, y_pred):
		b_ce = K.binary_crossentropy(y_true, y_pred)
		weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
		weighted_b_ce = weight_vector * b_ce
		return K.mean(weighted_b_ce)
	return weighted_binary_crossentropy
# Set class weights
weighted_binary_crossentropy = create_weighted_binary_crossentropy(zero_weight=0.1, one_weight=0.9)

########################  DICE LOSS  #######################################

def dice_coefficient(y_true, y_pred, smooth=1.):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1-dice_coefficient(y_true, y_pred)



#####################  DICE + WBCE LOSS #####################################
def dice_wbce_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    smooth=1.

    # Calculate wbce
    w = weighted_binary_crossentropy(y_true, y_pred)
    d = dice_coef_loss(y_true, y_pred)

    return w+d



    

##############  FOCAL LOSS (FOCAL TVERSKY) #######################
#https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
# Reference https://arxiv.org/abs/1810.07842
def tversky(y_true, y_pred):
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_neg = K.sum(y_true_pos * (1-y_pred_pos))
	false_pos = K.sum((1-y_true_pos)*y_pred_pos)
	alpha = 0.7
	return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
	return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
	pt_1 = tversky(y_true, y_pred)
	gamma = 0.75
	return K.pow((1-pt_1), gamma)



#################### NEEDS WORK #####################
def calc_dist_map(seg):
	res = np.zeros_like(seg)
	posmask = seg.astype(np.bool)

	if posmask.any():
		negmask = ~posmask
		res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

	return res

def calc_dist_map_batch(y_true):
	y_true_numpy = y_true.numpy()
	return np.array([calc_dist_map(y)
					 for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true, y_pred):
	y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
									 inp=[y_true],
									 Tout=tf.float32)
	multipled = y_pred * y_true_dist_map
	return K.mean(multipled)

def dice_and_surface(y_pred, y_true):
	return dice_coef_loss(y_pred, y_true) + surface_loss(y_true, y_pred)
#####################################################




#initializer = RandomNormal(mean=0., stddev=1.)
#initializer = GlorotNormal(seed=1)
initializer = GlorotUniform(seed=1)

def unet2d(input_size = (256,256,1), n_classes=4):
	
	inputs = Input(input_size)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
	drop4 = Dropout(0.2)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv5)
	drop5 = Dropout(0.2)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
	conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
	
	conv10 = Conv2D(n_classes, 1, activation = 'sigmoid')(conv9)

	model = Model(inputs = inputs, outputs = conv10)
	
	#model.summary()

	return model


############################### 2D UNET with Batch Norm ######################

def unet_2d_model_deep4(n_classes=1, im_sz=64, n_channels=1, n_filters_start=16, growth_factor=2, upconv=True):
	
	n_filters = n_filters_start
	inputs = Input((im_sz, im_sz, n_channels))
	bn1   = BatchNormalization()(inputs)
	conv1 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False, data_format="channels_last")(bn1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
	pool1 = Dropout(rate = 0.2)(pool1)

	n_filters *= growth_factor
	conv2 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)
	pool2 = Dropout(rate = 0.2)(pool2)

	n_filters *= growth_factor
	conv3 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)
	pool3 = Dropout(rate = 0.2)(pool3)

	n_filters *= growth_factor
	conv4 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)
	pool4 = Dropout(rate = 0.2)(pool4)
	
	
	n_filters *= growth_factor
	conv5 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv5)
	
	
	n_filters //= growth_factor
	if upconv:
		up6 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv5), conv4])
	else:
		up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
	conv6 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv6)
	conv6 = Dropout(rate = 0.2)(conv6)

	n_filters //= growth_factor
	if upconv:
		up7 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv6), conv3])
	else:
		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
	conv7 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv7)
	conv7 = Dropout(rate = 0.2)(conv7)

	n_filters //= growth_factor
	if upconv:
		up8 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv7), conv2])
	else:
		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
	conv8 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv8)
	conv8 = Dropout(rate = 0.2)(conv8)

	n_filters //= growth_factor
	if upconv:
		up9 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv8), conv1])
	else:
		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
	conv9 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up9)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv9)
	conv9 = Dropout(rate = 0.2)(conv9)
	
	
	conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid', kernel_initializer = initializer, use_bias=False)(conv9)

	model = Model(inputs=inputs, outputs=conv10)
	
	return model


