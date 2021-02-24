import numpy as np
import keras
import os
import random
import gc
from helpers.data import *

class DataGenerator(keras.utils.Sequence):
	
	'Generates data for Keras'
	'IMPORTANT NOTE: SCAN IDs go from 1-total inclusive but indexes go from 0-(total-1) since we use np.arange'
	def __init__(self, 
				 TRAINING_DATA_DIR, 
				 scan_IDs, 
				 num_scans_in_batch = 4, 
				 num_patches_in_batch=20, 
				 dim=(64,64), 
				 view_plane='coronal',
				 muscles='all',
				 window_width=250,
				 n_channels=1, 
				 n_classes=4, 
				 shuffle=True,
				 validation_flag=1):
		
		'Initialization'
		self.TRAINING_DATA_DIR = TRAINING_DATA_DIR
		self.scan_IDs = scan_IDs # SCAN IDS should start from 1-total inclusive
		self.num_scans_in_batch=num_scans_in_batch
		self.num_patches_in_batch=num_patches_in_batch
		self.dim = dim
		self.view_plane=view_plane
		self.muscles=muscles
		self.window_width=window_width
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.validation_flag=validation_flag
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		'In this case we move 1 scan after another - pick num_scans_in_batch and generate patches'
		return int(len(self.scan_IDs)-self.num_scans_in_batch+1)

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index : (index+self.num_scans_in_batch)]
		#print("Indexes")
		#print(indexes)

		# Find list of IDs
		scan_IDs_temp = [self.scan_IDs[k] for k in indexes]
		#print("scan_ID_temp")
		#print(scan_IDs_temp)

		# Generate data
		#print("\nGenerating data for batch: "+str(scan_IDs_temp))
		X, y = self.__data_generation(scan_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.scan_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, scan_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		
		# Initialization
		X = np.empty((self.num_patches_in_batch, *self.dim, self.n_channels))
		y = np.empty((self.num_patches_in_batch, *self.dim, self.n_channels))
		#print("Data generator triggered")

		scans=[]
		masks=[]

		for i, ID in enumerate(scan_IDs_temp):
			#print(i, ID)            
							
			scan = np.load(os.path.join(self.TRAINING_DATA_DIR, str("SCAN_"+str(ID)+".npy")))
			mask = np.load(os.path.join(self.TRAINING_DATA_DIR, str("MASK_"+str(ID)+".npy")))
		
			scans.append(scan)
			masks.append(mask)
			#print("Scans imported")

		# Generate data
		if len(self.dim)==2:
			if self.validation_flag==0:
				scans, masks = get_2d_cor_patches(scans, masks, self.num_patches_in_batch//8, sz=self.dim[0], scale_factor=(80,120), intensity_factor=(80,120))
				scans, masks = augment_2d_patches(scans, masks)
				scans, masks = turn_2d_patches_list_to_numpy(scans, masks, self.n_classes)
			if self.validation_flag==1:
				scans, masks = get_2d_cor_patches(scans, masks, self.num_patches_in_batch//2, sz=self.dim[0])
				scans, masks = turn_2d_patches_list_to_numpy(scans, masks, self.n_classes)
				
		# Window operation 
		scans = window_image(scans, window_center = 0, window_width=self.window_width)
		
		X=scans.astype(np.float32)
		y=masks.astype(np.float32)
		
		# Conservatively release memory to delete the variables we don't need anymore
		# The scans will be written over in the next step (batch) anyway
		del scans, masks
		gc.collect()
		
		return X, y