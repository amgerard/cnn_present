import os
from glob import glob
import sys
import numpy as np
from skimage.util import img_as_float
import SimpleITK as sitk

def combineImgs(pair):
	t1_path = pair[0]
	t2_path = pair[1]
	t1 = img_as_float(sitk.GetArrayFromImage(sitk.ReadImage(t1_path)))
	t2 = img_as_float(sitk.GetArrayFromImage(sitk.ReadImage(t2_path)))

	# transform image and return
	# return t1[...,None]
	return np.concatenate([t1[...,None],t2[...,None]],axis=3)

def combineMasks(path2Masks=''):
	
	if path2Masks == '':
		path2Masks = '/Users/amgerard/uiowa/research/nii/anatomicalModelOfNormalBrain/'
	
	# mask paths
	gry_path = os.path.join(path2Masks, 'phantom_1.0mm_normal_gry.nii')
	wht_path = os.path.join(path2Masks, 'phantom_1.0mm_normal_wht.nii')
	csf_path = os.path.join(path2Masks, 'phantom_1.0mm_normal_csf.nii')
	
	# mask tensors
	gry = sitk.GetArrayFromImage(sitk.ReadImage(gry_path)) 
	wht = sitk.GetArrayFromImage(sitk.ReadImage(wht_path)) 
	csf = sitk.GetArrayFromImage(sitk.ReadImage(csf_path)) 

	# transform image and return
	mask = np.zeros(gry.shape)
	mask[gry == 1] = 1.0
	mask[wht == 1] = 2.0
	mask[csf == 1] = 3.0
	return mask

def sample_patches(X,y):
	X_tmp = []
	y_tmp = []
	for im in range(X.shape[0]):
		for i in range(50,150,5):	
			for j in range(50,150,5):
				for k in range(50,150,5):
					X_tmp.append(X[im,i:i+24,j:j+24,k,:])			
					y_tmp.append(np.bincount(y[im,i+10:i+15,j+10:j+15,k].astype(int).flatten()).argmax())
	X_new = np.concatenate([im[None,...] for im in X_tmp])
	y = np.concatenate([im[None,...] for im in y_tmp])
	y_new = np.zeros([y.shape[0],4])
	y_new[np.arange(y.shape[0]),y] = 1
	return X_new,y_new

def get_data2():

	X,y,Xtst,ytst = get_data()
	X_new,y_new = sample_patches(X,y)
	Xtst_new,ytst_new = sample_patches(Xtst,ytst)
	return X_new,y_new,Xtst_new,ytst_new

#if __name__ == '__main__':
def get_data():
	
	# find all files that start with T1/T2 in path dir and sub-dirs
	path = '/Users/amgerard/uiowa/research/nii/'
	t1_paths = [y for x in os.walk(path) for y in glob(os.path.join(x[0], 'T1*.nii'))]
	t2_paths = [x.replace('T1','T2') for x in t1_paths]

	imgs = [combineImgs(x) for x in zip(t1_paths,t2_paths)]
	mask = combineMasks()

	imgs = np.vstack([im[None,...] for im in imgs if im.shape[0] == 181])
	masks_trn = np.concatenate([mask[None,...] for _ in range(25)])
	masks_tst = np.concatenate([mask[None,...] for _ in range(11)])
	return imgs[:25],masks_trn,imgs[25:],masks_tst
	#return imgs[:20],np.random.randint(2, size=20),imgs[20:],np.random.randint(2, size=16)
