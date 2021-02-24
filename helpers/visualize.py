import matplotlib.pyplot as plt
import numpy as np


def reverse_one_hot_encode(encoded_mask, thres = 0.5):
	muscles=encoded_mask.shape[-1]
	predicted_mask=np.zeros((encoded_mask.shape[0],encoded_mask.shape[1],1))
	muscles=[1,2,3,4]
	for muscle in muscles:
		for i in range(encoded_mask.shape[0]):
			for j in range(encoded_mask.shape[1]):
								if encoded_mask[i,j,muscle-1]>thres:
										predicted_mask[i,j,0]=muscle

	return predicted_mask

def visualize_scan_mask(scan, mask, alpha=0.5):
	# Sagittal
	muscles=0
	for layer in range(mask.shape[0]):
		pix_array=mask[layer,:,:]
		m = np.count_nonzero(pix_array)
		if m > muscles:
			sagittal_layer=layer
			muscles=m
	plt.imshow(scan[sagittal_layer,:,:],cmap='gray')
	plt.imshow(mask[sagittal_layer,:,:],alpha=alpha)
	plt.show()

	# Coronal
	muscles=0
	for layer in range(mask.shape[1]):
		pix_array=mask[:,layer,:]
		m = np.count_nonzero(pix_array)
		if m > muscles:
			coronal_layer=layer
			muscles=m
	plt.imshow(scan[:,coronal_layer,:],cmap='gray')
	plt.imshow(mask[:,coronal_layer,:],alpha=alpha)
	plt.show()

	# Axial
	muscles=0
	for layer in range(mask.shape[2]):
		pix_array=mask[:,:,layer]
		m = np.count_nonzero(pix_array)
		if m > muscles:
			axial_layer=layer
			muscles=m
	plt.imshow(scan[:,:,axial_layer],cmap='gray')
	plt.imshow(mask[:,:,axial_layer],alpha=alpha)
	plt.show()

def visualize_scan(scan):
	# Sagittal
	sagittal_layer = scan.shape[0]//2
	plt.imshow(scan[sagittal_layer,:,:],cmap='gray')
	plt.show()

	# Coronal
	coronal_layer = scan.shape[1]//2
	plt.imshow(scan[:,coronal_layer,:],cmap='gray')
	plt.show()

	# Axial
	axial_layer = scan.shape[2]//2
	plt.imshow(scan[:,:,axial_layer],cmap='gray')
	plt.show()