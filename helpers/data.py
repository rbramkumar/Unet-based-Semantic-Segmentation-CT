import numpy as np
import keras
import random
import os
from scipy.ndimage import rotate, zoom
import skimage.transform

def one_hot_encode_mask(mask):
	mask_medial = np.zeros(mask.shape) 
	mask_lateral = np.zeros(mask.shape) 
	mask_superior = np.zeros(mask.shape) 
	mask_inferior = np.zeros(mask.shape) 

	mask_medial[mask == 1] = 1
	mask_lateral[mask == 2] = 1
	mask_superior[mask == 3] = 1
	mask_inferior[mask == 4] = 1

	mask_oh = np.stack((mask_medial, mask_lateral, mask_superior, mask_inferior), axis = -1)
	return mask_oh

# Function to window image
def window_image(image, window_center, window_width):
	img_min = window_center - window_width // 2
	img_max = window_center + window_width // 2
	window_image = image.copy()
	window_image[window_image < img_min] = img_min
	window_image[window_image > img_max] = img_max
	
	return window_image

## HELPER FUNCTIONS FOR DATA GENERATOR

####################################### 2D data section #################################

#Find 1 random patch, from both img and mask
def get_2d_cor_rand_patch(img, mask, sz=64):
	"""
	:param img: ndarray with shape (x_sz, y_sz, z_sz)
	:param mask: ndarray with shape (x_sz, y_sz, z_sz)
	:param sz: size of random patch
	:return: patch with shape (x_sz_p, z_sz_p)
	"""
	
	# For coronal patch only x and z 
	assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[2] > sz  and img.shape[0:2] == mask.shape[0:2]

	# Get first and last non-zero index in xz (coronal planes) and exclude 5mm of edge cases
	y_range = np.nonzero([np.count_nonzero(mask[:,i,:]) for i in range(mask.shape[1])])
	ymin=np.min(y_range)
	ymax=np.max(y_range)
	
	xc = random.randint(0, img.shape[0] - sz)
	yc = random.randint(ymin+5, ymax-5)
	zc = random.randint(0, img.shape[2] - sz)
		
	patch_img  =  img[xc:(xc + sz), yc, zc:(zc + sz)]
	patch_mask = mask[xc:(xc + sz), yc, zc:(zc + sz)]

	return patch_img, patch_mask


# Input X,Y and return random patches, with and without EYES
def get_2d_cor_patches(scans, masks, num_patches_required = 100, sz = 64, scale_factor=(100,101), intensity_factor=(100,101)):

	eye_muscle_threshold=150
	
	X = []
	y = []
	
	while len(X) < num_patches_required*0.8:

		i = random.randint(0,len(scans)-1)
		try:
			scale=np.random.randint(scale_factor[0],scale_factor[1])/100
			intensity=np.random.randint(intensity_factor[0],intensity_factor[1])/100
			
			scaled_patch_dim=int(scale*sz)
			
			random_X, random_Y = get_2d_cor_rand_patch(img = scans[i], mask = masks[i], sz = scaled_patch_dim)
			
			if np.count_nonzero(random_Y>0)>eye_muscle_threshold:
				random_X=zoom(random_X, zoom = sz/random_X.shape[0], order=3)
				random_Y=zoom(random_Y, zoom = sz/random_Y.shape[0], order=0)
				X.append(random_X*intensity)
				y.append(random_Y)
		except: 
			pass
	
	while len(X) < num_patches_required:
		i = random.randint(0,len(scans)-1)
		try:
			scale=np.random.randint(scale_factor[0],scale_factor[1])/100
			intensity=np.random.randint(intensity_factor[0],intensity_factor[1])/100
			scaled_patch_dim=int(scale*sz)

			random_X, random_Y = get_2d_cor_rand_patch(img = scans[i], mask = masks[i], sz = scaled_patch_dim)
			random_X=zoom(random_X, zoom = sz/random_X.shape[0], order=3)
			random_Y=zoom(random_Y, zoom = sz/random_Y.shape[0], order=0)
			X.append(random_X*intensity)
			y.append(random_Y)
		except:
			pass
				
	return X, y


# Augmentation
# Augmentation
def augment_2d_patches(train_images, mask_images):
	
	rotangle1=random.randint(0,5)
	rotangle2=random.randint(5,20)
	rotangle3=random.randint(0,20)
	rotangle4=random.randint(0,20)

	#Augment 2D images and masks
	if len(train_images[0].shape)==2:
		
		tranposes=[]

		for i in train_images:
			
			img_patch_flip1=i[::-1,:]
			img_patch_flip2=i[:,::-1]
			img_patch_transpose=i.transpose([1,0])
			img_hybridaug1=img_patch_transpose[::-1,:]
			img_hybridaug2=img_patch_transpose[:,::-1]
			img_hybridaug3=img_patch_flip1[:,::-1]
			img_patch_rotateangle1 = skimage.transform.rotate(i, angle=rotangle1)#,mode='reflect')
			img_patch_rotateangle2 = skimage.transform.rotate(i, angle=rotangle2)#,mode='reflect')

			tranposes.append(img_patch_flip1)
			tranposes.append(img_patch_flip2)
			tranposes.append(img_patch_transpose)
			tranposes.append(img_hybridaug1)
			tranposes.append(img_hybridaug2)
			tranposes.append(img_hybridaug3)
			tranposes.append(img_patch_rotateangle1)
			tranposes.append(img_patch_rotateangle2)

		train_images=tranposes

		tranposes=[]

		for i in mask_images:
			
			mask_patch_flip1=i[::-1,:]
			mask_patch_flip2=i[:,::-1]
			mask_patch_transpose=i.transpose([1,0])
			mask_hybridaug1=mask_patch_transpose[::-1,:]
			mask_hybridaug2=mask_patch_transpose[:,::-1]
			mask_hybridaug3=mask_patch_flip1[:,::-1]
			mask_patch_rotateangle1 = skimage.transform.rotate(i, angle=rotangle1)#,mode='reflect')
			mask_patch_rotateangle2 = skimage.transform.rotate(i, angle=rotangle2)#,mode='reflect')

			tranposes.append(mask_patch_flip1)
			tranposes.append(mask_patch_flip2)
			tranposes.append(mask_patch_transpose)
			tranposes.append(mask_hybridaug1)
			tranposes.append(mask_hybridaug2)
			tranposes.append(mask_hybridaug3)
			tranposes.append(mask_patch_rotateangle1)
			tranposes.append(mask_patch_rotateangle2)

		mask_images=tranposes    
			
	return train_images, mask_images

# List to Numpy
def turn_2d_patches_list_to_numpy(x_imgs, y_imgs, num_classes):
		
	x_imgs=np.asarray(x_imgs)
	x_imgs=x_imgs.astype(np.float32)
	
	x_imgs = np.reshape(x_imgs,(x_imgs.shape[0],x_imgs.shape[1],x_imgs.shape[2],1))

	if num_classes == 1:
		y_imgs=np.asarray(y_imgs)
		y_imgs = np.reshape(y_imgs,(y_imgs.shape[0],y_imgs.shape[1],y_imgs.shape[2],1))
		y_imgs[y_imgs > 0] = 1

	elif num_classes == 4:
		y_imgs=np.asarray(y_imgs)

		y_medial = np.zeros((y_imgs.shape[0], y_imgs.shape[1], y_imgs.shape[2]))
		y_lateral = np.zeros((y_imgs.shape[0],y_imgs.shape[1],y_imgs.shape[2]))
		y_superior = np.zeros((y_imgs.shape[0],y_imgs.shape[1],y_imgs.shape[2]))
		y_inferior = np.zeros((y_imgs.shape[0],y_imgs.shape[1],y_imgs.shape[2]))

		y_medial[y_imgs == 1] = 1
		y_lateral[y_imgs == 2] = 1
		y_superior[y_imgs == 3] = 1
		y_inferior[y_imgs == 4] = 1

		y_imgs = np.stack((y_medial, y_lateral, y_superior, y_inferior), axis = -1)

	return x_imgs, y_imgs