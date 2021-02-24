
####################### Helper functions unit testing ##########################

ID = random.randint(0,100)
print(ID)

scan=np.load(os.path.join(data_folder, 'SCAN_'+str(ID)+'.npy'))
mask=np.load(os.path.join(data_folder, 'MASK_'+str(ID)+'.npy'))

print(scan.shape)
print(mask.shape)

visualize_scan_mask(scan, mask)

scan_ww = window_image(image=scan, window_center=0, window_width=250)
visualize_scan_mask(scan_ww, mask, alpha=0.5)
print(scan_ww.shape)
print(mask.shape)

a = 0
while a<100:
    sp, mp = get_2d_cor_rand_patch(scan_ww, mask, sz=64)
    a = np.count_nonzero(mp>0)    
print(a)
plt.imshow(sp, cmap='gray')
plt.imshow(mp, alpha=0.5)
plt.show()

if scan.shape[0:2]==mask.shape[0:2]:
    print("yes")

scans=[]
masks = []

scans.append(scan_ww)
masks.append(mask)

print(scans[0].shape)
print(masks[0].shape)

x, y = get_2d_cor_patches(scans, masks, num_patches_required = 100, sz = 64)

print(len(x))
print(len(y))

print(x[0].shape)
print(y[0].shape)

for i in range(5):
    ID=random.randint(0,99)
    print(ID)
    plt.imshow(x[ID], cmap='gray')
    plt.imshow(y[ID], alpha=0.5)
    plt.show()


ID=1
print(ID)
plt.imshow(x[ID], cmap='gray')
plt.imshow(y[ID], alpha=0.5)
plt.show()

train_images=[]
mask_images = []

train_images.append(x[1])
mask_images.append(y[1])

print(len(train_images))
print(len(mask_images))

x_aug, y_aug = augment_2d_patches(train_images, mask_images)
print(len(x_aug))
for i in range(len(x_aug)):
    plt.imshow(x_aug[i], cmap='gray')
    plt.imshow(y_aug[i], alpha=0.5)
    plt.show()


x, y = turn_2d_patches_list_to_numpy(x_aug, y_aug, num_classes=1)
print(x.shape)
print(y.shape)

plt.imshow(x[0,:,:,0], cmap='gray')
plt.imshow(y[0,:,:,0], alpha=0.5)
plt.show()



######################## CRF #####################################

im = np.random.randint(100, size=(64,64,3))/100
plt.imshow(im[:,:,0], cmap='gray')
plt.show()

mask = np.random.randint(5, size=(64,64))
print(np.unique(mask))
plt.imshow(mask)
plt.show()

colors, labels = np.unique(mask, return_inverse=True)
print(colors, labels)

image_size = mask.shape[:2]
print(image_size)

n_labels = len(set(labels.flat))
print(n_labels)
import pydensecrf.densecrf as dcrf
d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
print(d)
print(dir(d))

U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=True)
print(U)
d.setUnaryEnergy(U)
d.addPairwiseGaussian(sxy=(3,3), compat=3)
d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
print(d)

Q = d.inference(5) # 5 - num of iterations
MAP = np.argmax(Q, axis=0).reshape(image_size)
unique_map = np.unique(MAP)
for u in unique_map: # get original labels back
    np.putmask(MAP, MAP == u, colors[u])

print(MAP.shape)
plt.imshow(MAP)
plt.show()



############################### augmentation scale #######################################

data_folder = '../compiled_npys'
ID=np.random.randint(0,100)
scan=np.load(os.path.join(data_folder, 'SCAN_'+str(ID)+'.npy'))
mask=np.load(os.path.join(data_folder, 'MASK_'+str(ID)+'.npy'))
print(scan.shape)
print(mask.shape)

scans = []
masks = []
scans.append(scan)
masks.append(mask)
x, y = get_2d_cor_patches(scans, masks, 10, sz=80)
plt.imshow(x[0], cmap='gray')
plt.imshow(y[0], alpha=0.5)
plt.show()

print(np.unique(y[0]))
test=y[0]
plt.subplot(221)
plt.imshow(test==1)
plt.show()
plt.subplot(222)
plt.imshow(test==2)
plt.show()
plt.subplot(223)
plt.imshow(test==3)
plt.show()
plt.subplot(224)
plt.imshow(test==4)
plt.show()


from scipy.ndimage import zoom
import numpy as np

sz=64

x_test=np.reshape(x[0],(80,1,80))
y_test=np.reshape(y[0],(80,1,80))

print(x_test.shape)
print(y_test.shape)
print(np.unique(y_test))

scale=np.random.randint(80,120)/100
reverse_scale=1/scale
print("scale:"+str(scale))
print("inverse scale:"+str(reverse_scale))

scaled_patch_dim = int(scale*sz)
print("scaled_patch_dim:"+str(scaled_patch_dim))
patch_dim_recalculated=scaled_patch_dim*reverse_scale
print("patch_dim_recalculated:"+str(patch_dim_recalculated))


x_patch, y_patch = get_2d_cor_rand_patch(x_test,y_test,sz=scaled_patch_dim)
print(x_patch.shape)
print(y_patch.shape)

plt.imshow(x_patch,cmap='gray') 
plt.imshow(y_patch, alpha=0.5)
plt.show()


x_p = zoom(input = x_patch, zoom=64/x_patch.shape[0], order=3)
y_p = zoom(input = y_patch, zoom=64/x_patch.shape[1], order=0)
print(x_p.shape)
print(y_p.shape)
print(np.unique(y_p))
plt.imshow(x_p,cmap='gray')
plt.imshow(y_p, alpha=0.5)
plt.show()


'''


plt.imshow(x_patch,cmap='gray')
plt.show()
plt.imshow(y_patch)
plt.show()

x = zoom(input = x_patch, zoom=1/scale, order=3)
#y = zoom(input = y_patch, int(1/scale), mode='linear')
print(x.shape)
'''




######################################### coronal slice selection #######################################
ID=np.random.randint(0,100)
testscan=np.load(os.path.join(data_folder, 'SCAN_'+str(ID)+'.npy'))
testmask=np.load(os.path.join(data_folder, 'MASK_'+str(ID)+'.npy'))
print(testscan.shape)
print(testmask.shape)

visualize_scan_mask(testscan, testmask)


y_range = np.nonzero([np.count_nonzero(testmask[:,i,:]) for i in range(testmask.shape[1])])
ymin=np.min(y_range)
ymax=np.max(y_range)
print(ymin, ymax)
visualize_scan_mask(testscan[:,ymin+5:ymax-5,:], testmask[:,ymin+5:ymax-5,:])
