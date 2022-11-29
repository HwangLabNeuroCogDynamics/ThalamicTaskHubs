# create thalamus by cortical FC mat
# this is to be run on the Argon cluster of uiowa
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, input_data, masking
import nilearn.image
from nilearn.image import resample_to_img, index_img
from scipy import stats, linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def generate_correlation_mat(x, y):
	"""Correlate each n with each m.
	Parameters
	----------
	x : np.array
	  Shape Num ROI N X Timepoints.
	y : np.array
	  Shape ROI num M X Timepoints.
	Returns
	-------
	np.array
	  N X M array in which each element is a correlation coefficient.
	"""
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
		raise ValueError('x and y must ' +
						 'have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x,
				 y.T) - n * np.dot(mu_x[:, np.newaxis],
								  mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pca_reg_fc(source_ts, target_ts):
	'''use PCA regression to calculate FC, inputs are time by ROI'''
	ts_len = target_ts.shape[0]
	roi_size = target_ts.shape[1]
	subcortical_size = source_ts.shape[1]
	n_comps = np.amin([ts_len, roi_size, subcortical_size]) // 20
	pca = PCA(n_comps)
	reduced_mat = pca.fit_transform(source_ts) # Time X components
	components = pca.components_
	regrmodel = LinearRegression()
	reg = regrmodel.fit(reduced_mat, target_ts) #cortex ts also time by ROI
	#project regression betas from component
	fcmat = pca.inverse_transform(reg.coef_).T #reshape to cortex

	return fcmat 


mdtb_dir = '/Shared/lss_kahwang_hpc/data/MDTB/'
tomoya_dir = '/Shared/lss_kahwang_hpc/data/Tomoya/'

mdtb_subs = ['sub-02','sub-04','sub-09','sub-14','sub-17','sub-19','sub-21','sub-24','sub-26','sub-28','sub-31','sub-03','sub-06','sub-12','sub-15','sub-18','sub-20','sub-22','sub-25','sub-27','sub-30']
tomoya_subs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']

Schaefer400 = nib.load('/Shared/lss_kahwang_hpc/ROIs/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
Schaefer400_masker = input_data.NiftiLabelsMasker(Schaefer400)
mni_thalamus_mask = nib.load("/Shared/lss_kahwang_hpc/ROIs//mni_atlas/MNI_thalamus_2mm.nii.gz")
mni_thalamus_masker = input_data.NiftiMasker(mni_thalamus_mask) 

mdtb_fcmats = np.zeros((2445, 400, len(mdtb_subs)))
for i, sub in enumerate(mdtb_subs):
	func_file = nib.load(mdtb_dir+"3dDeconvolve/" + sub + "/FIRmodel_errts_block1.nii.gz")
	
	cortical_ts = Schaefer400_masker.fit_transform(func_file)
	roi_ts = mni_thalamus_masker.fit_transform(func_file) #roi_ts = data[np.nonzero(mni_thalamus_mask.get_fdata())]

	roi_ts = np.delete(roi_ts, np.where(roi_ts.mean(axis=1)==0)[0], axis=0)
	cortical_ts = np.delete(cortical_ts, np.where(cortical_ts.mean(axis=1)==0)[0], axis=0)
	
	#fcmat = generate_correlation_mat(roi_ts.T, cortical_ts.T)
	fcmat = pca_reg_fc(roi_ts, cortical_ts)
	fcmat[np.isnan(fcmat)] = 0
	mdtb_fcmats[:,:,i] = fcmat

np.save(mdtb_dir + "analysis/mdtb_fcmats_block1", mdtb_fcmats)

mdtb_fcmats = np.zeros((2445, 400, len(mdtb_subs)))
for i, sub in enumerate(mdtb_subs):
	func_file = nib.load(mdtb_dir+"3dDeconvolve/" + sub + "/FIRmodel_errts_block2.nii.gz")
	
	cortical_ts = Schaefer400_masker.fit_transform(func_file)
	roi_ts = mni_thalamus_masker.fit_transform(func_file) #roi_ts = data[np.nonzero(mni_thalamus_mask.get_fdata())]

	roi_ts = np.delete(roi_ts, np.where(roi_ts.mean(axis=1)==0)[0], axis=0)
	cortical_ts = np.delete(cortical_ts, np.where(cortical_ts.mean(axis=1)==0)[0], axis=0)
	
	#fcmat = generate_correlation_mat(roi_ts.T, cortical_ts.T)
	fcmat = pca_reg_fc(roi_ts, cortical_ts)
	fcmat[np.isnan(fcmat)] = 0
	mdtb_fcmats[:,:,i] = fcmat

np.save(mdtb_dir + "analysis/mdtb_fcmats_block2", mdtb_fcmats)

tomoya_fcmats = np.zeros((2445, 400, len(tomoya_subs)))
for i, sub in enumerate(tomoya_subs):
	func_file = nib.load(tomoya_dir+"3dDeconvolve/" + sub + "/FIRmodel_errts_block1.nii.gz")
	
	cortical_ts = Schaefer400_masker.fit_transform(func_file)
	roi_ts = mni_thalamus_masker.fit_transform(func_file)#data[np.nonzero(mni_thalamus_mask.get_fdata())]
	
	roi_ts = np.delete(roi_ts, np.where(roi_ts.mean(axis=1)==0)[0], axis=0)
	cortical_ts = np.delete(cortical_ts, np.where(cortical_ts.mean(axis=1)==0)[0], axis=0)
	
	#fcmat = generate_correlation_mat(roi_ts, cortical_ts.T)
	fcmat = pca_reg_fc(roi_ts, cortical_ts)
	fcmat[np.isnan(fcmat)] = 0
	tomoya_fcmats[:,:,i] = fcmat

np.save(tomoya_dir + "analysis/tomoya_fcmats_block1", tomoya_fcmats)

tomoya_fcmats = np.zeros((2445, 400, len(tomoya_subs)))
for i, sub in enumerate(tomoya_subs):
	func_file = nib.load(tomoya_dir+"3dDeconvolve/" + sub + "/FIRmodel_errts_block2.nii.gz")
	
	cortical_ts = Schaefer400_masker.fit_transform(func_file)
	roi_ts = mni_thalamus_masker.fit_transform(func_file)#data[np.nonzero(mni_thalamus_mask.get_fdata())]
	
	roi_ts = np.delete(roi_ts, np.where(roi_ts.mean(axis=1)==0)[0], axis=0)
	cortical_ts = np.delete(cortical_ts, np.where(cortical_ts.mean(axis=1)==0)[0], axis=0)
	
	#fcmat = generate_correlation_mat(roi_ts, cortical_ts.T)
	fcmat = pca_reg_fc(roi_ts, cortical_ts)
	fcmat[np.isnan(fcmat)] = 0
	tomoya_fcmats[:,:,i] = fcmat

np.save(tomoya_dir + "analysis/tomoya_fcmats_block2", tomoya_fcmats)



