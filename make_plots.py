#### script to make various plots for this project
import os
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
from thalpy.constants import paths 
### thalpy is a lab-wide library for common funcitons we use in our lab
### see: https://github.com/HwangLabNeuroCogDynamics/thalpy
from matplotlib import pyplot as plt
from thalpy.analysis import pc, plotting, feature_extraction, glm, fc
from thalpy import masks, base 
from scipy.stats import spearmanr
from scipy.spatial import distance
from scipy.stats import kendalltau
from nilearn import image, input_data, masking
import nilearn.image
from nilearn.image import resample_to_img, index_img
sns.set_context('paper', font_scale=1.5)
sns.set_palette("colorblind")
plt.ion()

################################################
### Functions for analyses
################################################
import pickle
def save_object(obj, filename):
	''' Simple function to write out objects into a pickle file
	usage: save_object(obj, filename)
	'''
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
	#M = pickle.load(open(f, "rb"))


def read_object(filename):
	''' short hand for reading object because I can never remember pickle syntax'''
	o = pickle.load(open(filename, "rb"))
	return o

##########
## PCA
def run_pca(matrix, dir_tree, output_name, task_list, masker=None):
    currnet_path=os.getcwd()
    os.chdir(dir_tree.analysis_dir)
    pca, loadings, correlated_loadings, explained_var = feature_extraction.compute_PCA(matrix, output_name=output_name, var_list=task_list, masker=masker, plot = False)
    
    # for MDTB 
    # pca_comps[:, 1] = pca_comps[:, 1] * -1
    # loadings[:, 1] = loadings[:, 1] * -1
    # correlated_loadings[1] = correlated_loadings[1] * -1

    #loadings_df = pd.DataFrame(loadings, index=task_list)
    #summary_df = loadings_df.describe()
    #var_df = loadings_df.var(axis=0).rename("var")
    #summary_df = summary_df.append(var_df)
    #display(HTML(summary_df.to_html()))
    os.chdir(currnet_path)
    return pca, loadings, explained_var


##########
## clustering functions
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as shc

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogrammasks

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def hier_cluster(task_matrix, n_clusters=None):
    if n_clusters is None:
        distance_threshold = 0
    else:
        distance_threshold = None
    
    model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters)
    cluster = model.fit(task_matrix.swapaxes(0, 1)) #task by roi/vox
    if distance_threshold is not None:
        plot_dendrogram(model)
    return cluster

##########
## PC function
def cal_fcpc(thalamocortical_fc):
    ''' clacuate PC with thalamocortical FC data'''
    thresholds = [86,87,88,89,90,91,92,93,94,95,96,97,98]
    pc_vectors = np.zeros((thalamocortical_fc.shape[0], len(thresholds)))
    for it, t in enumerate(thresholds):
        temp_mat = thalamocortical_fc.copy()
        temp_mat[temp_mat<np.percentile(temp_mat, t)] = 0 #threshold
        fc_sum = np.sum(temp_mat, axis=1)
        kis = np.zeros(np.shape(fc_sum))

        for ci in np.unique(Schaeffer_CI):
            kis = kis + np.square(np.sum(temp_mat[:,np.where(Schaeffer_CI==ci)[0]], axis=1) / fc_sum)

        pc_vectors[:, it] = 1-kis
    return np.nanmean(pc_vectors, axis = 1)

##########
#functions that run activity flow
def activity_flow_subject(rs, thal_betas, cortical_betas, task_list, rs_indexer=None, return_pred=False, return_rsa=False):
    ''' do activity flow sub by sub
    inputs 
    rs: the fc object that maps thalamocortical FC
    thal_betas: the thalamus (or other source ROI) evoked responses
    cortical_betas: the observed cortical evoked responses
    task_list: list of task names 
    '''
    if rs_indexer:
        rs_indexer = rs_indexer
    else:
        rs_indexer = np.s_[:, :]
        
    sub_cor_array = np.zeros([len(task_list), thal_betas.shape[2]]) #task by subject
    sub_rsa = np.zeros(thal_betas.shape[2]) 
    sub_accu = np.zeros(thal_betas.shape[2]) 

    try:
        num_subjects = len(rs.fc_subjects)
    except:
        try:
            num_subjects = rs.shape[2]
        except:
            num_subjects = 1 #if not using fc object, single subject analysis
    
    pred = []
    pred_rsa = []
    for rs_index in np.arange(num_subjects):
        # fc_subject = rs.fc_subjects[rs_index]
        # if fc_subject is None:
        #     continue
        # try:
        #     beta_index = subjects.to_subargs_list().index(fc_subject.name)
        # except:
        #     continue

        #activity flow prediction
        try:
            sub_rs = rs.fc_subjects[rs_index].seed_to_voxel_correlations[rs_indexer] #thalamocortical fc, tha by cortex, from Evan's data structure
            sub_rs = np.arctan(sub_rs) #fisher z
        except:
            try:
                sub_rs = rs[:,:,rs_index]
                sub_rs = np.arctan(sub_rs)
            except:
                sub_rs = rs[:,:] #not organized in fc object strct from Evan
                sub_rs = np.arctan(sub_rs)

        predicted_corticals = np.dot(np.swapaxes(thal_betas[:, :, rs_index], 0, 1), sub_rs) 
        sub_cortical_betas = np.swapaxes(cortical_betas[:, :, rs_index], 0, 1)
        pred.append(predicted_corticals)
        
        #rsa
        predicted_rsa = np.corrcoef(predicted_corticals)[np.triu_indices(predicted_corticals.shape[0],k=1)] #only upper triangle
        observed_rsa = np.corrcoef(sub_cortical_betas)[np.triu_indices(sub_cortical_betas.shape[0],k=1)]
        pred_rsa.append(np.corrcoef(predicted_corticals))
        sub_rsa[rs_index] = kendalltau(predicted_rsa, observed_rsa)[0]
        
        #compare predicted vs observed
        for i, _ in enumerate(task_list):
            sub_cor_array[i, rs_index] = np.corrcoef(predicted_corticals[i, :], sub_cortical_betas[i, :])[0,1]
        
        #NN classification
        count = 0
        for i in np.arange(len(task_list)):
            if i == np.corrcoef(predicted_corticals[i, :], sub_cortical_betas[:,:])[0][1:].argmax():
                count = count +1
        sub_accu[rs_index] = (1.0*count) / len(task_list)
    
    if return_pred & return_rsa:
        return sub_cor_array, sub_rsa, sub_accu, pred, pred_rsa
    elif return_pred:
        return sub_cor_array, sub_rsa, sub_accu, pred
    elif return_rsa:
        return sub_cor_array, sub_rsa, sub_accu, pred_rsa
    else:
        return sub_cor_array, sub_rsa, sub_accu

#distance function for rsa
def rsa_mahalanobis(mat, cov):
    '''calculate rsa matrix using mahanobis distance, input expected to be ROI by condition, the output will be condition by condition, and noise cov will be ROI by ROI'''
    num_roi = mat.shape[0]
    num_cond = mat.shape[1]
    # try:
    vi = np.linalg.inv(cov) #inverse of cov
    # except:
    #vi = np.linalg.pinv(np.cov(mat))
    
    rsa = np.empty((num_cond, num_cond))
    for m in np.arange(num_cond):
        for n in np.arange(num_cond):
            rsa[m,n] = distance.mahalanobis(mat[:,m],mat[:,n],vi)
    return rsa

# Null Models
def null_activity_flow(thal_betas, cortical_betas, rs, task_list, num_permutation=1000, sub_rs_indexer=None, avg_rs_indexer=None):
    if sub_rs_indexer:
        sub_rs_indexer = sub_rs_indexer
    else:
        sub_rs_indexer = np.s_[:, :]
        
    if avg_rs_indexer:
        avg_rs_indexer = avg_rs_indexer
    else:
        avg_rs_indexer = np.s_[:, :, :]
        
    #null_corr_array = np.empty([k, len(task_list)]) #permutation by task
    null_sub_corr_array = np.empty([num_permutation, len(task_list), thal_betas.shape[2]]) #permutation by task by subj
    null_sub_rsa = np.empty([num_permutation, thal_betas.shape[2]])
    null_sub_accu = np.empty([num_permutation, thal_betas.shape[2]])  
    for i in np.arange(num_permutation):
        #rand_avg_thal = np.nanmean(thal_betas, axis=2)
        rand_sub_thal = np.copy(thal_betas)
        random_vector = np.random.permutation(np.arange(thal_betas.shape[0]))
        for j in np.arange(len(task_list)): 
            #rand_avg_thal[:, j] = np.random.permutation(rand_avg_thal[:, j])
            for k in np.arange(thal_betas.shape[2]): #loop through subj
                rand_sub_thal[:, j, k] = rand_sub_thal[random_vector, j, k] # randomize for each task and each subj
        #null_corr_array[i, :] = corr_avg(rs, rand_avg_thal, cortical_betas, task_list, rs_indexer=avg_rs_indexer)
        null_sub_corr_array[i, :, :], null_sub_rsa[i,:], null_sub_accu[i,:] = activity_flow_subject(rs, rand_sub_thal, cortical_betas, task_list, rs_indexer=sub_rs_indexer)
    
    return null_sub_corr_array, null_sub_rsa, null_sub_accu

### noise ceiling calculations
def test_noise(inputmat):
    srsa = np.empty((25,25,21))
    for s in np.arange(21):
        mat = inputmat[:,:,s]
        srsa[:,:,s] = np.corrcoef(mat.T) 
    mrsa = srsa.mean(axis=2)
    for s in np.arange(21):
        print(kendalltau(mrsa[np.triu_indices(25,k=1)],  srsa[:,:,s][np.triu_indices(25,k=1)]))

    for s in np.arange(21):
        mb = inputmat[:,:,np.where(np.arange(21)!=s)[0]].mean(axis=2)
        for n in np.arange(25): 
            print(np.corrcoef(mb[:,n], inputmat[:,n,s])[0,1])

    srsa = np.empty((25,25,21))
    for s in np.arange(21):
        mat = inputmat[:,:,s]
        srsa[:,:,s] = np.corrcoef(mat.T) 
        mats = inputmat[:,:,np.where(np.arange(21)!=s)[0]]
        bsrsa = np.empty((25,25,20))
        for i in np.arange(20):
            mat = mats[:,:,i]
            bsrsa[:,:,i] = np.corrcoef(mat.T) 
        mrsa = bsrsa.mean(axis=2)
        print(kendalltau(mrsa[np.triu_indices(25,k=1)],  srsa[:,:,s][np.triu_indices(25,k=1)]))    

### functions for whole brain activity flow
def make_voxel_roi_masker(cortical_mask,roi_idx):
    ''' create a roi masker to pull out every voxel separately with a seprate integer'''
    roi_size = np.sum(cortical_mask.get_fdata()==roi_idx)
    data = 1*(cortical_mask.get_fdata()==roi_idx)
    data[data==1] = np.arange(roi_size)+1
    roi_mask = nilearn.image.new_img_like(cortical_mask, data)
    #roi_masker = input_data.NiftiLabelsMasker(roi_mask)
    return roi_mask

def make_roi_mask(cortical_mask,roi_idx):
    ''' create a roi mask'''
    roi_size = np.sum(cortical_mask.get_fdata()==roi_idx)
    data = 1*(cortical_mask.get_fdata()==roi_idx)
    #data[data==1] = np.arange(roi_size)
    roi_mask = nilearn.image.new_img_like(cortical_mask, data)
    #roi_masker = input_data.NiftiLabelsMasker(roi_mask)
    return roi_mask

def load_cortical_ts(subjects, cortical_mask, errts = 'FIRmodel_errts_block.nii.gz'):
    ''' because the nilearn masker is slow, let us preload the cortical timeseries from ROIs'''
    cortical_ts = []
    cortical_masker = input_data.NiftiLabelsMasker(cortical_mask)
    for sub_index, sub in enumerate(subjects):
        print(sub_index)
        funcfile = nib.load(sub.deconvolve_dir + errts)
        #ftemplate = nib.load(sub.deconvolve_dir +'NoGo_FIR_MIN.nii.gz')
        #cortical_mask = resample_to_img(cortical_mask, ftemplate, interpolation = 'nearest')
        cortical_ts.append(cortical_masker.fit_transform(funcfile))
    return cortical_ts

def cal_vox_roi_fc(subjects, roi_mask, cortical_ts, num_voxels, num_roi):
    '''create the vox by roi fc matrix'''
    fcmat = np.zeros((num_voxels, num_roi, len(subjects)))
    #roi_masker = input_data.NiftiLabelsMasker(roi_mask)
    for sub_index, sub in enumerate(subjects):
        funcfile = sub.deconvolve_dir + 'FIRmodel_errts_norest.nii.gz'
        ftemplate = nib.load(sub.deconvolve_dir +'NoGo_FIR_MIN.nii.gz')
        roi_mask = resample_to_img(roi_mask, ftemplate, interpolation = 'nearest')
        #cortical_ts = cortical_masker.fit_transform(funcfile)
        wb_ts = cortical_ts[sub_index] #time by roi, need to swap
        roi_ts = masking.apply_mask(funcfile, roi_mask) #time by roi, need to swap
        fcmat[:,:,sub_index] = generate_correlation_mat(roi_ts.T, wb_ts.T)
    return fcmat


def run_whole_brain_af(source_roi, subs, tasks, cortical_ts, cortical_beta_matrix, fbuck = "FIRmodel_MNI_stats_block+tlrc.BRIK", resid = "FIRmodel_errts_block_rs.nii.gz", ftemplate = "FIRmodel_MNI_stats_block_rs.nii.gz"):
    ''' run whole brain activity flow giving a source roi input'''
    num_roi = len(np.unique(source_roi.get_fdata()))-1
    num_sub = len(subs)
    num_tasks = len(tasks)
    whole_brain_af_corr = np.zeros((num_sub, num_roi, num_tasks))
    whole_brain_af_rsa_corr =  np.zeros((num_sub, num_roi))
    whole_brain_af_predicition_accu =  np.zeros((num_sub, num_roi))

    for sub_index, sub in enumerate(subs):
        funcfile = sub.deconvolve_dir + resid # this is data in the 2x2x2 grid, must be the same as the ROI.
        fdata = nib.load(funcfile).get_fdata() #preload data to save time without using nilearn masker    
    
        for roi in np.arange(num_roi):
            roi = roi+1
            roi_mask = make_roi_mask(source_roi, roi)
            roi_masker = input_data.NiftiMasker(roi_mask) 
            roi_masker.fit(sub.deconvolve_dir+ftemplate)
            #roi_voxel_mask = make_voxel_roi_masker(source_roi, roi)
            #ftemplate = nib.load(sub.deconvolve_dir+ template_fn)
            #roi_voxel_mask = resample_to_img(roi_voxel_mask, ftemplate, interpolation = 'nearest')
            #num_voxels = np.sum(roi_voxel_mask.get_fdata()!=0) 
            #roi_voxel_masker = input_data.NiftiLabelsMasker(roi_voxel_mask) #here using label maser, which will make each voxel with unique integer a unique ROI.
            
            #need to extract voxel wise task beta for each roi
            roi_beta_matrix = glm.load_brik([sub], roi_masker, fbuck , tasks, zscore=True, kind="beta")
            
            #then need to calculate voxel-whole brain FC matrix
            roi_mask = make_roi_mask(source_roi, roi)
            roi_masker = input_data.NiftiMasker(roi_mask)
            roi_masker.fit(sub.deconvolve_dir+ftemplate)
            #roi_mask = resample_to_img(roi_mask, ftemplate, interpolation = 'nearest')
            #roi_masker = input_data.NiftiMasker(roi_mask) #here using nifti maker, and fit_transform will output all voxel values instead of averaging.
            roi_ts = fdata[np.nonzero(roi_mask.get_fdata())] #voxel ts in roi, use direct indexing to save time. # the result of this is the same as using masker.fit_transform(), but much faster
            wb_ts = cortical_ts[sub_index] #whole brain ts

            # check censored data
            roi_ts =np.delete(roi_ts, np.where(roi_ts.mean(axis=0)==0)[0], axis=1)
            #np.delete(roi_ts, np.where(roi_ts.mean(axis=1)==0)[0], axis=0).shape
            wb_ts =np.delete(wb_ts, np.where(wb_ts.mean(axis=1)==0)[0], axis=0)
            if roi_ts.shape[1] != wb_ts.shape[0]: #check length
                continue

            #roi vox by whole brain corr mat
            fcmat = generate_correlation_mat(roi_ts, wb_ts.T)
            #generate_correlation_mat(roi_ts.T, wb_ts.T)
            fcmat[np.isnan(fcmat)] = 0

            #aflow
            task_corr, rsa_corr, pred_accu = activity_flow_subject(fcmat, roi_beta_matrix, cortical_beta_matrix, tasks)
            whole_brain_af_corr[sub_index, roi-1, :] = task_corr[:,0]
            whole_brain_af_rsa_corr[sub_index, roi-1] = rsa_corr
            whole_brain_af_predicition_accu[sub_index, roi-1]= pred_accu

    return whole_brain_af_corr, whole_brain_af_rsa_corr, whole_brain_af_predicition_accu

### sysmatically remove thalamus activity based on ranking of PC values 
def threshold_matrix(beta_matrix, taskpc, low_threshold, top_threshold):
    new_matrix = np.copy(beta_matrix)
    for s in np.arange(beta_matrix.shape[2]):
        maxv = np.percentile(taskpc[s,:], top_threshold)
        minv = np.percentile(taskpc[s,:], low_threshold)
        voxels_to_censor= ((taskpc[s,:] <= maxv) & (taskpc[s,:] >= minv)) == False
        #np.where(taskpc > np.percentile(taskpc, top_threshold), 1, 0) + np.where(taskpc < np.percentile(taskpc, low_threshold), 1, 0)
        new_matrix[voxels_to_censor,:,s] = 0

    # func = lambda x: x * voxels_to_keep
    # new_matrix = np.apply_along_axis(func, 0, beta_matrix)
    return new_matrix

### fast calculation of FC between ROI ts
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

def plot_tha(Input, lb, ub, cmap, savepath):
    # show volum image
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1 import ImageGrid

    MNI_img = nib.load('images/MNI152_T1_2mm_brain.nii')
    MNI_data = MNI_img.get_fdata()

    #Input = resample_to_img(Input, MNI_img)
    # create mask for parcel
    #Mask = np.zeros(MNI_data.shape)
    #Morel_mask = nib.load(masks.MOREL_PATH)
    #Morel_mask = resample_to_img(Morel_mask, MNI_img, interpolation='nearest')
    Mask = Input.get_fdata()
    #Mask[Morel_mask.get_fdata()==0] = 0
    #Mask[Thalamus_voxel_coordinate[i,0], Thalamus_voxel_coordinate[i,1], Thalamus_voxel_coordinate[i,2]] = CIs[i]
    Mask = np.ma.masked_where(Mask == 0, Mask)

    # flip dimension to show anteiror of the brain at top
    MNI_data = MNI_data.swapaxes(0,1)
    Mask = Mask.swapaxes(0,1)
    
    fig = plt.figure(figsize=(9, 3))
    # display slice by slice
    Z_slices = [37, 39, 41, 43] #range(34, 46,2)
    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,4),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
    for i, ax in enumerate(grid):
        ax.imshow(MNI_data[45:65, 30:60, Z_slices[i]], cmap='gray', interpolation='nearest')
        im = ax.imshow(Mask[45:65, 30:60, Z_slices[i]],cmap=cmap, interpolation='none', vmin =lb, vmax=ub) #vmin =lb, vmax=ub
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    fig.tight_layout() 
    plt.savefig(savepath, bbox_inches='tight')



################################################
# path, subject, task setup
################################################
MDTB_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/"
MDTB_DIR_TREE = base.DirectoryTree(MDTB_DIR)
MDTB_ANALYSIS_DIR = MDTB_DIR + 'analysis/'
stim_config_df = pd.read_csv(MDTB_DIR + paths.DECONVOLVE_DIR + paths.STIM_CONFIG)
mdtb_subjects = base.get_subjects(MDTB_DIR_TREE.deconvolve_dir, MDTB_DIR_TREE)
MDTB_TASKS = list(set(stim_config_df["Group"].to_list()))
tomoya_dir_tree = base.DirectoryTree('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya')
tomoya_subjects = base.get_subjects(tomoya_dir_tree.deconvolve_dir, tomoya_dir_tree)
stim_config_df = pd.read_csv(tomoya_dir_tree.deconvolve_dir + paths.STIM_CONFIG)
TOMOYA_TASKS = stim_config_df["Stim Label"].tolist()



################################################
######## PCA analaysis, Figure 1
################################################
#MNI thalamus mask
mni_thalamus_masker = masks.binary_masker("/home/kahwang/bsh/ROIs/mni_atlas/MNI_thalamus_2mm.nii.gz")

#### Tomoya N&N dataset
#tomoya_masker = masks.binary_masker(masks.MOREL_PATH)
tomoya_masker = mni_thalamus_masker.fit(nib.load('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya/3dDeconvolve/sub-01/FIRmodel_MNI_stats+tlrc.BRIK'))
tomoya_beta_matrix = glm.load_brik(tomoya_subjects, tomoya_masker, 'FIRmodel_MNI_stats+tlrc.BRIK', TOMOYA_TASKS, zscore = True, kind="beta")
# z-score
tomoya_beta_matrix[tomoya_beta_matrix>3] = 0
tomoya_beta_matrix[tomoya_beta_matrix<-3] = 0
np.save('data/tomoya_beta_matrix', tomoya_beta_matrix)    

#run pca
tomoya_pca_WxV = np.zeros([masks.masker_count(tomoya_masker), 6])
tomoya_explained_var = pd.DataFrame()

for s in np.arange(tomoya_beta_matrix.shape[2]):
    mat = tomoya_beta_matrix[:,:,s]
    fn = 'tomoya_pca_sub' + str(s)
    comps, _, var = run_pca(mat, tomoya_dir_tree, fn, TOMOYA_TASKS, masker=tomoya_masker)
    plt.close('all')
    tdf = pd.DataFrame()
    tdf['Component'] = np.arange(1,11)
    tdf['Dataset'] = 'N&N'
    tdf['Sub'] = s
    tdf['Varaince Explained'] = var[0:10] #varianced explained for the first 10 PC
    tdf['Sum of Variance Explained'] = tdf['Varaince Explained'].cumsum()
    tomoya_explained_var = tomoya_explained_var.append(tdf) 
    for i in np.arange(10):
        tomoya_pca_WxV[:,s] = tomoya_pca_WxV[:,s] + abs(comps[:,i])*var[i] #each subjects PCAweight * variance explained
       
tomoya_pca_weight = tomoya_masker.inverse_transform(np.mean(tomoya_pca_WxV, axis=1))  #average across subjects
nib.save(tomoya_pca_weight, "images/tomoya_pca_weight.nii.gz")

#### MDTB dataset
img = nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
#mdtb_masker = masks.binary_masker(masks.MOREL_PATH)
mdtb_masker = mni_thalamus_masker.fit(nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))

#pull betas
mdtb_beta_matrix = glm.load_brik(mdtb_subjects, mdtb_masker, "FIRmodel_MNI_stats_block+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")
mdtb_beta_matrix[mdtb_beta_matrix>3] = 0
mdtb_beta_matrix[mdtb_beta_matrix<-3] = 0
np.save('data/mdtb_beta_matrx', mdtb_beta_matrix)

# # average then PCA. bad idea, if want to do this across subjects then concat matrices
# mdtb_pca_comps, mdtb_loadings, mdtb_explained_var = run_pca(np.mean(mdtb_beta_matrix, axis=2), MDTB_DIR_TREE, 'mdtb_pca_groupave', TASK_LIST, masker=mdtb_masker)
# plt.close('all')

mdtb_explained_var = pd.DataFrame()
mdtb_pca_WxV = np.zeros([masks.masker_count(mdtb_masker), 21])
for s in np.arange(mdtb_beta_matrix.shape[2]):
    mat = mdtb_beta_matrix[:,:,s]
    fn = 'mdtb_pca_sub' + str(s)
    comps, w, var = run_pca(mat, MDTB_DIR_TREE, fn, MDTB_TASKS, masker=mdtb_masker)
    plt.close('all')
    tdf = pd.DataFrame()
    tdf['Component'] = np.arange(1,11)
    tdf['Dataset'] = 'MDTB'
    tdf['Sub'] = s
    tdf['Varaince Explained'] = var[0:10] #varianced explained for the first 10 PC
    tdf['Sum of Variance Explained'] = tdf['Varaince Explained'].cumsum()
    mdtb_explained_var = mdtb_explained_var.append(tdf)
    for i in np.arange(10):
        mdtb_pca_WxV[:,s] = mdtb_pca_WxV[:,s] + abs(comps[:,i])*var[i] #each subjects PCAweight * variance explained

mdtb_pca_weight = mdtb_masker.inverse_transform(np.mean(mdtb_pca_WxV, axis=1)) #average across subjects
nib.save(mdtb_pca_weight, "images/mdtb_pca_weight.nii.gz")


######## plot weight x var
# plotting.plot_thal(tomoya_pca_weight)
# plotting.plot_thal(mdtb_pca_weight)
lb = np.percentile(np.mean(mdtb_pca_WxV, axis=1),20)
ub = np.percentile(np.mean(mdtb_pca_WxV, axis=1),80)
plot_tha(mdtb_pca_weight, lb, ub+0.1, "autumn", "images/mdtb_pca_weight.png") #changed colorbar scale slightly otherwise legend gets messed up.....
lb = np.percentile(np.mean(tomoya_pca_WxV, axis=1),20)
ub = np.percentile(np.mean(tomoya_pca_WxV, axis=1),80)
plot_tha(tomoya_pca_weight, lb, ub+0.05, "autumn", "images/tomoya_pca_weight.png")

######## Varaince explained plot
df = mdtb_explained_var.append(tomoya_explained_var)
df['Component'] = df['Component'].astype('str')

g = sns.barplot(data=df, x="Component", y="Varaince Explained", hue='Dataset')
g.get_legend().remove()
g2 = g.twinx()
g2 = sns.lineplot(data=df, x="Component", y="Sum of Variance Explained", hue='Dataset', legend = False)
fig = plt.gcf()
fig.set_size_inches([6,4])
fig.tight_layout()
fig.savefig("/home/kahwang/RDSS/tmp/pcvarexp.png")

### TODO concat matrices across subject to do one PCA.

#### check "compression" in other ROIs.
vars = np.zeros((400,21))
cortical_mask = nib.load(masks.SCHAEFER_400_7N_PATH)
for roi_idx in np.arange(len(np.unique(cortical_mask.get_fdata()))):
    roi_idx = roi_idx+1
    roi_mask = make_roi_mask(cortical_mask,roi_idx)
    roi_masker = input_data.NiftiMasker(roi_mask)
    betamats = glm.load_brik(mdtb_subjects, roi_masker, "FIRmodel_MNI_stats_block+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")
    for s in np.arange(betamats.shape[2]):
        _,_,var = run_pca(betamats[:,:,s], MDTB_DIR_TREE, 'corticalroi', MDTB_TASKS, masker=roi_masker)
        plt.close('all')
        vars[roi_idx-1, s] = np.sum(var[0:3])

### Maybe, do group level PCA by concatenate across subjects



################################################
######## Task hubs versus rsfc hubs, Figure 2
################################################
######## load cortical betas for hierarchical clustering for MDTB
mni_thalamus_masker = masks.binary_masker("/home/kahwang/bsh/ROIs/mni_atlas/MNI_thalamus_2mm.nii.gz")
Schaefer400 = nib.load(masks.SCHAEFER_400_7N_PATH)
Schaefer400_masker = input_data.NiftiLabelsMasker(Schaefer400)
mdtb_cortical_betas = glm.load_brik(mdtb_subjects, Schaefer400_masker, "FIRmodel_MNI_stats_block+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")
np.save('data/mdtb_cortical_betas', mdtb_cortical_betas)
cortical_betas = np.load('data/mdtb_cortical_betas.npy') #z-scored already
cortical_betas_2d = np.empty([400*21, 25]) #concat across subjects
for i in np.arange(21):
    for j in np.arange(400):
        for k in np.arange(25):
            cortical_betas_2d[i*400+j, k] = cortical_betas[j,k,i]

#dend = shc.dendrogram(shc.linkage(cortical_betas_2d.T, method='ward'), labels=TASK_LIST) # need to work on this plot..
#### calculate task PC across different clusinger level and density thresholds
thresholds = np.arange(85,99).tolist()
cluster_num = 8
mdtb_hc_pc = np.empty([21,6, masks.masker_count(mdtb_masker), len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(21):

    for ic, c in enumerate(np.arange(3,cluster_num+1)):
        
        conditions_cluster = hier_cluster(cortical_betas_2d, n_clusters=c)

        for k in np.arange(len(np.unique(conditions_cluster.labels_))):
            group = [condition for i, condition in enumerate(MDTB_TASKS) if k == conditions_cluster.labels_[i]]
            #print(f'k: {k}  group: {group}')

        pc_matrix = pc.pc_subject(abs(mdtb_beta_matrix)[:,:,s], conditions_cluster.labels_, thresholds=thresholds)
        pc_matrix = np.where(np.isnan(pc_matrix), 0.001, pc_matrix)
        pc_matrix = np.where(pc_matrix <= 0, 0.001, pc_matrix)
        mdtb_hc_pc[s,ic,:,:] = pc_matrix

mdtb_hc_pc_img = mdtb_masker.inverse_transform(np.mean(mdtb_hc_pc, axis=(0,1,3)))

### now calculate PC using a priori clustering
# 1: ObjectViewing         
# 2: WordPrediction
# 3: Math
# 4: Motor
# 1: MentalRotation
# 4: MotorImagery
# 1: LandscapeMovie
# 5: IAPSemotion
# 4: Go/NoGo
# 1: NatureMovie
# 4: ResponseAlternativesMotor
# 1: ActionObservation
# 1: SpatialMap
# 1: BiologicalMotion
# 6: Rules
# 6: Stroop
# 6: Verbal2Back
# 6: Interval
# 1: AnimatedMovie
# 5: IAPSaffective
# 6: ObjectNBackTask
# 2: Language
# 2: TheoryOfMind
# 1: VisualSearch
# 1: SpatialImagery

CI = np.array([1,2,3,4,1,4,1,5,4,1,4,1,1,1,6,6,6,6,1,5,6,2,2,1,1])
thresholds = np.arange(85,99).tolist()
mdtb_a_pc = np.empty([21,masks.masker_count(mdtb_masker), len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(21):
    pc_matrix = pc.pc_subject(abs(mdtb_beta_matrix)[:,:,s], CI, thresholds=thresholds)
    pc_matrix = np.where(np.isnan(pc_matrix), 0.001, pc_matrix)
    pc_matrix = np.where(pc_matrix <= 0, 0.001, pc_matrix)
    mdtb_a_pc[s,:,:] = pc_matrix

mdtb_a_pc_img = mdtb_masker.inverse_transform(np.mean(mdtb_a_pc, axis=(0,2)))

######## load cortical betas for hierarchical clustering for Tomoya N&N
Schaefer400 = nib.load(masks.SCHAEFER_400_7N_PATH)
Schaefer400_masker = input_data.NiftiLabelsMasker(Schaefer400)
tomoya_cortical_betas = glm.load_brik(tomoya_subjects, Schaefer400_masker, "FIRmodel_MNI_stats+tlrc.BRIK", TOMOYA_TASKS, zscore=True, kind="beta")
np.save('data/tomoya_cortical_betas', tomoya_cortical_betas)
tomoya_cortical_betas = np.load('data/tomoya_cortical_betas.npy')
tomoya_cortical_betas_2d = np.empty([400*6, 102]) #concat across subjects
for i in np.arange(6):
    for j in np.arange(400):
        for k in np.arange(102):
            tomoya_cortical_betas_2d[i*400+j, k] = tomoya_cortical_betas[j,k,i]

thresholds = np.arange(85,99).tolist()
cluster_num = 8
tomoya_hc_pc = np.empty([6,6, masks.masker_count(tomoya_masker), len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(6):
    for ic, c in enumerate(np.arange(3,cluster_num+1)):
        conditions_cluster = hier_cluster(tomoya_cortical_betas_2d, n_clusters=c)
        for k in np.arange(len(np.unique(conditions_cluster.labels_))):
            group = [condition for i, condition in enumerate(TOMOYA_TASKS) if k == conditions_cluster.labels_[i]]
            #print(f'k: {k}  group: {group}')

        pc_matrix = pc.pc_subject(abs(tomoya_beta_matrix)[:,:,s], conditions_cluster.labels_, thresholds=thresholds)
        pc_matrix = np.where(np.isnan(pc_matrix), 0.001, pc_matrix)
        pc_matrix = np.where(pc_matrix <= 0, 0.001, pc_matrix)
        tomoya_hc_pc[s,ic,:,:] = pc_matrix

tomoya_hc_pc_img = tomoya_masker.inverse_transform(np.mean(tomoya_hc_pc, axis=(0,1,3)))

# a priori clusters from N&N 2020, this is how they classified it. not me. 
auditory = ["TimeMov","Rhythm","Harmony","TimeSound","CountTone","SoundRight","SoundLeft","RateDisgustSound","RateNoisy","RateBeautySound","SoundPlace","DailySound","EmotionVoice","MusicCategory","ForeignListen","LanguageSound","AnimalVoice", "FeedbackPos"]
introspection = ["LetterFluency","CategoryFluency","RecallKnowledge","ImagineMove","ImagineIf","RecallFace","ImaginePlace","RecallPast","ImagineFuture"]
motor = ["RateConfidence","RateSleepy","RateTired","EyeMoveHard","EyeMoveEasy","EyeBlink","RestClose","RestOpen","PressOrdHard","PressOrdEasy","PressLR","PressLeft","PressRight"]
memory = ["MemoryNameHard","MemoryNameEasy","MatchNameHard","MatchNameEasy","RelationLogic","CountDot","MatchLetter","MemoryLetter","MatchDigit","MemoryDigit","CalcHard","CalcEasy"]
language = ["RecallTaskHard","RecallTaskEasy","DetectColor","Recipe","TimeValue","DecidePresent","ForeignReadQ","ForeignRead","MoralImpersonal","MoralPersonal","Sarcasm","Metaphor","ForeignListenQ","WordMeaning","RatePoem", "PropLogic"]
visual = ["EmotionFace","Flag","DomesiticName","WorldName","DomesiticPlace","WorldPlace","StateMap","MapIcon","TrafficSign","MirrorImage","DailyPhoto","AnimalPhoto","RateBeautyPic","DecidePeople","ComparePeople","RateHappyPic","RateSexyPicM","DecideFood","RateDeliciousPic","RatePainfulPic","RateDisgustPic","RateSexyPicF","DecideShopping","DetectDifference","DetectTargetPic","CountryMap","Money","Clock","RateBeautyMov","DetectTargetMov","RateHappyMov","RateSexyMovF","RateSexyMovM","RateDeliciousMov","RatePainfulMov","RateDisgustMov"]
groups = [visual, language, memory, motor, introspection, auditory]

task_category = []
for task in TOMOYA_TASKS:
    for i, group in enumerate(groups):
        if task in group:
            task_category.append(i)
            continue

tomoya_a_pc = np.empty([6, masks.masker_count(tomoya_masker), len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(6):
    pc_matrix = pc.pc_subject(abs(tomoya_beta_matrix)[:,:,s], task_category, thresholds=thresholds)
    pc_matrix = np.where(np.isnan(pc_matrix), 0.001, pc_matrix)
    pc_matrix = np.where(pc_matrix <= 0, 0.001, pc_matrix)
    tomoya_a_pc[s,:,:] = pc_matrix
tomoya_a_pc_img = tomoya_masker.inverse_transform(np.mean(tomoya_a_pc, axis=(0,2)))

# plot task PC
# plotting.plot_thal(mdtb_hc_pc_img)
# plotting.plot_thal(tomoya_hc_pc_img )
# plotting.plot_thal(mdtb_a_pc_img )
# plotting.plot_thal(tomoya_a_pc_img )

lb = np.percentile(np.mean(mdtb_hc_pc, axis=(0,2)),20)
hb = np.percentile(np.mean(mdtb_hc_pc, axis=(0,2)),80)
plot_tha(mdtb_hc_pc_img, lb, hb, "plasma", "images/mdtb_hc_pc_img.png")
lb = np.percentile(np.mean(tomoya_hc_pc, axis=(0,2)),20)
hb = np.percentile(np.mean(tomoya_hc_pc, axis=(0,2)),80)
plot_tha(tomoya_hc_pc_img, lb, hb, "plasma", "images/tomoya_hc_pc_img.png")
lb = np.percentile(np.mean(mdtb_a_pc, axis=(0,2)),20)
hb = np.percentile(np.mean(mdtb_a_pc, axis=(0,2)),80)
plot_tha(mdtb_a_pc_img, lb, hb, "plasma", "images/mdtb_a_pc_img.png")
lb = np.percentile(np.mean(tomoya_a_pc, axis=(0,2)),20)
hb = np.percentile(np.mean(tomoya_a_pc, axis=(0,2)),80)
plot_tha(tomoya_a_pc_img, lb, hb, "plasma", "images/tomoya_a_pc_img.png")


##### Compare to FC PC (rsFC and backgroundFC)
# rsFC_pc_img = nib.load('images/PC.nii.gz')
# plotting.plot_thal(rsFC_pc_img)
# plot_tha(rsFC_pc_img, 50, 90, "viridis", "images/rsFC_pc_img.png")

## calculate residualFC PC for mdtb
#load fc objects
mdtb_fc = fc.load(MDTB_ANALYSIS_DIR + "fc_mni_residuals.p")
#mdtb_fc = np.load(MDTB_ANALYSIS_DIR + "mdtb_fcmats.npy")

# cortical ROI CI assingment for calculating PC
Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')

### PC from fc data
mdtb_fcpc = np.empty((21, masks.masker_count(mdtb_masker)))
mdtb_fc_mat = np.empty((21,masks.masker_count(mdtb_masker),400))
for s in np.arange(21):
    mdtb_fc_mat[s,:,:] = mdtb_fc.fc_subjects[s].seed_to_voxel_correlations
    #mdtb_fc_mat[s,:,:] = mdtb_fc[:,:,s]
    mdtb_fcpc[s, :] = cal_fcpc(mdtb_fc_mat[s,:,:])

mdtb_fcpc_img = mdtb_masker.inverse_transform(np.nanmean(mdtb_fcpc, axis=0))

## tomoya fcpc
tomoya_fc = fc.load(tomoya_dir_tree.analysis_dir + "fc_mni_residuals.p")
tomoya_fcpc = np.empty((6, masks.masker_count(tomoya_masker)))
tomoya_fc_mat = np.empty((6,masks.masker_count(tomoya_masker),400))
for s in np.arange(6):
    tomoya_fc_mat[s,:,:] = tomoya_fc.fc_subjects[s].seed_to_voxel_correlations
    tomoya_fcpc[s, :] = cal_fcpc(tomoya_fc_mat[s,:,:])

tomoya_fcpc_img = tomoya_masker.inverse_transform(np.nanmean(tomoya_fcpc, axis=0))

#plotting.plot_thal(mdtb_fcpc_img)
#plotting.plot_thal(tomoya_fcpc_img)
lb = np.percentile(np.nanmean(mdtb_fcpc, axis=0), 20)
hb = np.percentile(np.nanmean(mdtb_fcpc, axis=0), 80)
plot_tha(mdtb_fcpc_img, lb, hb+0.03, "plasma", "images/mdtb_fcpc_img.png")
lb = np.percentile(np.nanmean(tomoya_fcpc, axis=0), 20)
hb = np.percentile(np.nanmean(tomoya_fcpc, axis=0), 80)
plot_tha(tomoya_fcpc_img, lb, hb, "plasma", "images/tomoya_fcpc_img.png")


################################################
######## Activity flow prediction, Figure 3
################################################
# activity flow analysis: predicited cortical evoked responses = thalamus evoke x thalamocortical FC, compare to observed cortical evoked responses

# load fc objects again, thalamocortical fc matrices stored here
mdtb_fc = fc.load(MDTB_ANALYSIS_DIR + "fc_mni_residuals.p")
#mdtb_fc = np.load(MDTB_ANALYSIS_DIR + "mdtb_fcmats.npy")
tomoya_fc = fc.load(tomoya_dir_tree.analysis_dir + "fc_mni_residuals.p")
#tomoya_fc = np.load(tomoya_dir_tree.analysis_dir + "tomoya_fcmats.npy")

mni_thalamus_masker = masks.binary_masker("/home/kahwang/bsh/ROIs/mni_atlas/MNI_thalamus_2mm.nii.gz")
mdtb_masker = mni_thalamus_masker.fit(nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))
tomoya_masker = mni_thalamus_masker.fit(nib.load('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya/3dDeconvolve/sub-01/FIRmodel_MNI_stats+tlrc.BRIK'))

# load thalamus task evoked responses, mdtb
# mdtb_masker = masks.binary_masker(masks.MOREL_PATH)
# img = nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
# mdtb_masker.fit(img)
mdtb_beta_matrix = glm.load_brik(mdtb_subjects, mdtb_masker, "FIRmodel_MNI_stats_block+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")
# tomoya
# tomoya_masker = masks.binary_masker(masks.MOREL_PATH)
# tomoya_masker.fit(nib.load('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya/3dDeconvolve/sub-01/FIRmodel_MNI_stats+tlrc.BRIK'))
tomoya_beta_matrix = glm.load_brik(tomoya_subjects, tomoya_masker, "FIRmodel_MNI_stats+tlrc.BRIK", TOMOYA_TASKS, zscore = True, kind="beta")

# observed cortical evoked responses
Schaefer400 = nib.load(masks.SCHAEFER_400_7N_PATH)
Schaefer400_masker = input_data.NiftiLabelsMasker(Schaefer400)
mdtb_cortical_betas = glm.load_brik(mdtb_subjects, Schaefer400_masker, "FIRmodel_MNI_stats_block+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")
tomoya_cortical_betas  = glm.load_brik(tomoya_subjects, Schaefer400_masker, "FIRmodel_MNI_stats+tlrc.BRIK", TOMOYA_TASKS, zscore=True, kind="beta")

# load activity flow results df
#mdtb_activityflow_df = read_object(MDTB_ANALYSIS_DIR + 'mdtb_activity_flow_dataframe.p')
#tomoya_activityflow_df = read_object(tomoya_dir_tree.analysis_dir + 'mdtb_activity_flow_dataframe.p')

### run basic activity flow and null models
mdtb_activity_flow, mdtb_rsa_similarity, mdtb_pred_accu = activity_flow_subject(mdtb_fc, mdtb_beta_matrix, mdtb_cortical_betas, MDTB_TASKS, rs_indexer=None)
tomoya_activity_flow, tomoya_rsa_similarity, tomoya_pred_accu = activity_flow_subject(tomoya_fc, tomoya_beta_matrix, tomoya_cortical_betas, TOMOYA_TASKS, rs_indexer=None)
mdtb_activity_flow_null, mdtb_rsa_similarity_null, mdtb_pred_accu_null = null_activity_flow(mdtb_beta_matrix, mdtb_cortical_betas, mdtb_fc, MDTB_TASKS, num_permutation=1000)
tomoya_activity_flow_null, tomoya_rsa_similarity_null, tomoya_pred_accu_null = null_activity_flow(tomoya_beta_matrix, tomoya_cortical_betas, tomoya_fc, TOMOYA_TASKS, num_permutation=1000)

# save results to dict{}
af_results = {}
af_results['mdtb_activity_flow'] = mdtb_activity_flow
af_results['mdtb_rsa_similarity'] = mdtb_rsa_similarity
af_results['mdtb_pred_accu'] = mdtb_pred_accu
af_results['tomoya_activity_flow'] = tomoya_activity_flow
af_results['tomoya_rsa_similarity'] = tomoya_rsa_similarity
af_results['tomoya_pred_accu'] = tomoya_pred_accu
af_results['mdtb_activity_flow_null'] = mdtb_activity_flow_null
af_results['mdtb_rsa_similarity_null'] = mdtb_rsa_similarity_null
af_results['mdtb_pred_accu_null'] = mdtb_pred_accu_null
af_results['tomoya_activity_flow_null'] = tomoya_activity_flow_null
af_results['tomoya_rsa_similarity_null'] = tomoya_rsa_similarity_null
af_results['tomoya_pred_accu_null'] = tomoya_pred_accu_null
save_object(af_results, 'data/af_results')

### set evoke as uniform
mdtb_activity_flow_uniform, mdtb_rsa_similarity_uniform, _ = activity_flow_subject(mdtb_fc, np.ones((mdtb_beta_matrix.shape)), mdtb_cortical_betas, MDTB_TASKS, rs_indexer=None)
tomoya_activity_flow_uniform, tomoya_rsa_similarity_uniform, _ = activity_flow_subject(tomoya_fc, np.ones((tomoya_beta_matrix.shape)), tomoya_cortical_betas, TOMOYA_TASKS, rs_indexer=None)

### test noise
#test_noise(mdtb_beta_matrix)
# Make predicted plots
Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')
networkorder = np.asarray(sorted(range(len(Schaeffer_CI)), key=lambda k: Schaeffer_CI[k]))
netorder=networkorder
plt.figure()
ax = sns.heatmap(np.mean(mdtb_cortical_betas[netorder,:,:],axis=2), vmin=-3, vmax=3,cmap='seismic',cbar=True,yticklabels=100,xticklabels=MDTB_TASKS)
ax.set(ylabel='Regions')
plt.xticks([])
plt.savefig("images/observe_evoke.png", bbox_inches='tight')

plt.figure()
ax2 = sns.heatmap(np.mean(pred,axis=0).T, center=0,cmap='seismic',cbar=True,yticklabels=100,xticklabels=MDTB_TASKS)
ax2.set(ylabel='Regions')
plt.xticks([])
plt.savefig("images/predicted_evoke.png", bbox_inches='tight')


######## Whole brain activity flow, Figure 3.B
#load whole brain ROI mask, here we combine Schaeffer with several subcortical masks
Schaefer400 = nib.load(masks.SCHAEFER_400_7N_PATH)
Schaefer400_masker = input_data.NiftiLabelsMasker(Schaefer400)
Schaefer400_beta_matrix = glm.load_brik(mdtb_subjects, Schaefer400_masker, "FIRmodel_MNI_stats_block_rs.nii.gz", MDTB_TASKS, zscore=True, kind="beta")
np.save("data/Schaefer400_beta_matrix", Schaefer400_beta_matrix)
Schaefer400_beta_matrix = np.load("data/Schaefer400_beta_matrix.npy")
Schaeffer400_cortical_ts = load_cortical_ts(mdtb_subjects, Schaefer400, "FIRmodel_errts_block_rs.nii.gz")
np.save('data/Schaeffer400_cortical_ts', Schaeffer400_cortical_ts)
Schaeffer400_cortical_ts = np.load('data/Schaeffer400_cortical_ts.npy', allow_pickle=True)
#cortical_ts = np.load('data/Schaeffer400_cortical_ts', allow_pickle=True)
subs = mdtb_subjects
tasks = MDTB_TASKS

CerebrA = nib.load("/data/backed_up/shared/ROIs/mni_atlas/CerebrA_2mm.nii.gz")
whole_brain_af_corr, whole_brain_af_rsa_corr, whole_brain_af_predicition_accu = run_whole_brain_af(CerebrA, subs, tasks, Schaeffer400_cortical_ts, Schaefer400_beta_matrix, "FIRmodel_MNI_stats_block_rs.nii.gz", "FIRmodel_errts_block_rs.nii.gz", "FIRmodel_MNI_stats_block_rs.nii.gz")
np.save('data/whole_brain_af_corr_CerebrA.mdtb', whole_brain_af_corr)
np.save('data/whole_brain_af_rsa_corr_CerebrA.mdtb', whole_brain_af_rsa_corr)
np.save('data/whole_brain_af_predicition_accu_CerebrA.mdtb', whole_brain_af_predicition_accu)

whole_brain_af_corr_400, whole_brain_af_rsa_corr_400, whole_brain_af_predicition_accu_400 = run_whole_brain_af(Schaefer400, subs, tasks, Schaeffer400_cortical_ts, Schaefer400_beta_matrix, "FIRmodel_MNI_stats_block_rs.nii.gz", "FIRmodel_errts_block_rs.nii.gz", "FIRmodel_MNI_stats_block_rs.nii.gz")
np.save('data/whole_brain_af_corr_400.mdtb', whole_brain_af_corr_400)
np.save('data/whole_brain_af_rsa_corr_400.mdtb', whole_brain_af_rsa_corr_400)
np.save('data/whole_brain_af_predicition_accu_400.mdtb', whole_brain_af_predicition_accu_400)

Schaefer100 = nib.load('data/Schaefer100+BG_2mm.nii.gz')
whole_brain_af_corr_100, whole_brain_af_rsa_corr_100, whole_brain_af_predicition_accu_100 = run_whole_brain_af(Schaefer100, subs, tasks, Schaeffer400_cortical_ts, Schaefer400_beta_matrix, "FIRmodel_MNI_stats_block_rs.nii.gz", "FIRmodel_errts_block_rs.nii.gz", "FIRmodel_MNI_stats_block_rs.nii.gz")
np.save('data/whole_brain_af_corr_100.mdtb', whole_brain_af_corr_100)
np.save('data/whole_brain_af_rsa_corr_100.mdtb', whole_brain_af_rsa_corr_100)
np.save('data/whole_brain_af_predicition_accu_100.mdtb', whole_brain_af_predicition_accu_100)

### run tomoya whole brain AF
subs = tomoya_subjects
tasks = TOMOYA_TASKS
Schaefer400 = nib.load(masks.SCHAEFER_400_7N_PATH)
Schaefer400_masker = input_data.NiftiLabelsMasker(Schaefer400)
#tomoya_cortical_beta_matrix = glm.load_brik(tomoya_subjects, Schaefer400_masker, "FIRmodel_MNI_stats_rs.nii.gz", TOMOYA_TASKS, zscore=True, kind="beta")
#np.save("data/tomoya_cortical_beta_matrix", tomoya_cortical_beta_matrix)
tomoya_cortical_beta_matrix = np.load("data/tomoya_cortical_beta_matrix.npy")
#tomoya_Schaeffer400_cortical_ts = load_cortical_ts(tomoya_subjects, Schaefer400, "FIRmodel_errts_rs.nii.gz")
#np.save('data/tomoya_Schaeffer400_cortical_ts', tomoya_Schaeffer400_cortical_ts)
tomoya_Schaeffer400_cortical_ts = np.load('data/tomoya_Schaeffer400_cortical_ts.npy', allow_pickle=True)

CerebrA = nib.load("/data/backed_up/shared/ROIs/mni_atlas/CerebrA_2mm.nii.gz")
whole_brain_af_corr, whole_brain_af_rsa_corr, whole_brain_af_predicition_accu = run_whole_brain_af(CerebrA, subs, tasks, tomoya_Schaeffer400_cortical_ts, tomoya_cortical_beta_matrix, "FIRmodel_MNI_stats_rs.nii.gz", "FIRmodel_errts_rs.nii.gz", "FIRmodel_MNI_stats_rs.nii.gz")
np.save('data/whole_brain_af_corr_CerebrA.tomoya', whole_brain_af_corr)
np.save('data/whole_brain_af_rsa_corr_CerebrA.tomoya', whole_brain_af_rsa_corr)
np.save('data/whole_brain_af_predicition_accu_CerebrA.tomoya', whole_brain_af_predicition_accu)

whole_brain_af_corr_400, whole_brain_af_rsa_corr_400, whole_brain_af_predicition_accu_400 = run_whole_brain_af(Schaefer400, subs, tasks, tomoya_Schaeffer400_cortical_ts, tomoya_cortical_beta_matrix, "FIRmodel_MNI_stats_rs.nii.gz", "FIRmodel_errts_rs.nii.gz", "FIRmodel_MNI_stats_rs.nii.gz")
np.save('data/whole_brain_af_corr_400.tomoya', whole_brain_af_corr_400)
np.save('data/whole_brain_af_rsa_corr_400.tomoya', whole_brain_af_rsa_corr_400)
np.save('data/whole_brain_af_predicition_accu_400.tomoya', whole_brain_af_predicition_accu_400)

Schaefer100 = nib.load('data/Schaefer100+BG_2mm.nii.gz')
whole_brain_af_corr_100, whole_brain_af_rsa_corr_100, whole_brain_af_predicition_accu_100 = run_whole_brain_af(Schaefer100, subs, tasks, tomoya_Schaeffer400_cortical_ts, tomoya_cortical_beta_matrix, "FIRmodel_MNI_stats_rs.nii.gz", "FIRmodel_errts_rs.nii.gz", "FIRmodel_MNI_stats_rs.nii.gz")
np.save('data/whole_brain_af_corr_100.tomoya', whole_brain_af_corr_100)
np.save('data/whole_brain_af_rsa_corr_100.tomoya', whole_brain_af_rsa_corr_100)
np.save('data/whole_brain_af_predicition_accu_100.tomoya', whole_brain_af_predicition_accu_100)


################################################
# Make df, plot AF results
af_results = read_object('data/af_results')
af_simulation_results = read_object('data/af_simulation_results')
whole_brain_af_corr_100 = np.load('data/whole_brain_af_corr_100.mdtb.npy') #101 caudate, 102 putamen, 103 pallidum, combine to check size
whole_brain_af_corr = np.load('data/whole_brain_af_corr_CerebrA.mdtb.npy') #vermal lobules 1-V 50/101, Vermal lobules VI-Vii 2/53, Vermal Lobules Viii-X 20/71, hippocampus 48/99

af_df = pd.DataFrame()
mdtb_activity_flow = af_results['mdtb_activity_flow']
i=0
for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'MDTB'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = mdtb_activity_flow[t,sub]
        af_df.loc[i, 'Region'] = 'Thalamus' 
        i=i+1

for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'MDTB'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub, 100,t]
        af_df.loc[i, 'Region'] = 'Caudate' 
        i=i+1

for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'MDTB'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub, 101,t]
        af_df.loc[i, 'Region'] = 'Putamen' 
        i=i+1

for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'MDTB'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub, 102,t]
        af_df.loc[i, 'Region'] = 'Pallidus' 
        i=i+1

roi_df = pd.read_csv('data/100rois.csv')
for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        for net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']:
            af_df.loc[i,'Dataset'] = 'MDTB'
            af_df.loc[i,'Task'] = task
            af_df.loc[i,'Subject'] = str(sub)
            af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub,np.where(roi_df['ROI Name'].str.contains(net))[0],t].mean()
            af_df.loc[i, 'Region'] = net 
            i=i+1

for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'MDTB'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr[sub, [47,98],t].mean()
        af_df.loc[i, 'Region'] = 'Hippocampus' 
        i=i+1

af_null = af_results['mdtb_activity_flow_null'].mean(axis=0)
for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'MDTB'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = af_null[t,sub]
        af_df.loc[i, 'Region'] = 'Null Model' 
        i=i+1

af_uniform = af_simulation_results['mdtb_activity_flow_uniform']
for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'MDTB'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = af_uniform[t,sub]
        af_df.loc[i, 'Region'] = 'Uniformed Evoked Model' 
        i=i+1

#now compile tomoya
tomoya_activity_flow = af_results['tomoya_activity_flow']
whole_brain_af_corr_100 = np.load('data/whole_brain_af_corr_100.tomoya.npy') #101 caudate, 102 putamen, 103 pallidum, combine to check size
whole_brain_af_corr = np.load('data/whole_brain_af_corr_CerebrA.tomoya.npy') #vermal lobules 1-V 50/101, Vermal lobules VI-Vii 2/53, Vermal Lobules Viii-X 20/71, hippocampus 48/99

for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'N&N'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = tomoya_activity_flow[t,sub]
        af_df.loc[i, 'Region'] = 'Thalamus' 
        i=i+1

for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'N&N'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub, 100,t]
        af_df.loc[i, 'Region'] = 'Caudate' 
        i=i+1

for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'N&N'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub, 101,t]
        af_df.loc[i, 'Region'] = 'Putamen' 
        i=i+1

for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'N&N'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub, 102,t]
        af_df.loc[i, 'Region'] = 'Pallidus' 
        i=i+1

roi_df = pd.read_csv('data/100rois.csv')
for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        for net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']:
            af_df.loc[i,'Dataset'] = 'N&N'
            af_df.loc[i,'Task'] = task
            af_df.loc[i,'Subject'] = str(sub)
            af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr_100[sub,np.where(roi_df['ROI Name'].str.contains(net))[0],t].mean()
            af_df.loc[i, 'Region'] = net 
            i=i+1

for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'N&N'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = whole_brain_af_corr[sub, [47,98],t].mean()
        af_df.loc[i, 'Region'] = 'Hippocampus' 
        i=i+1

af_null = af_results['tomoya_activity_flow_null'].mean(axis=0)
for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'N&N'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = af_null[t,sub]
        af_df.loc[i, 'Region'] = 'Null Model' 
        i=i+1

af_uniform = af_simulation_results['tomoya_activity_flow_uniform']
for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow.shape[1]):
        af_df.loc[i,'Dataset'] = 'N&N'
        af_df.loc[i,'Task'] = task
        af_df.loc[i,'Subject'] = str(sub)
        af_df.loc[i, 'Predicted vs. Observed Evoked Responses'] = af_uniform[t,sub]
        af_df.loc[i, 'Region'] = 'Uniformed Evoked Model' 
        i=i+1

auditory = ["TimeMov","Rhythm","Harmony","TimeSound","CountTone","SoundRight","SoundLeft","RateDisgustSound","RateNoisy","RateBeautySound","SoundPlace","DailySound","EmotionVoice","MusicCategory","ForeignListen","LanguageSound","AnimalVoice", "FeedbackPos"]
introspection = ["LetterFluency","CategoryFluency","RecallKnowledge","ImagineMove","ImagineIf","RecallFace","ImaginePlace","RecallPast","ImagineFuture"]
motor = ["RateConfidence","RateSleepy","RateTired","EyeMoveHard","EyeMoveEasy","EyeBlink","RestClose","RestOpen","PressOrdHard","PressOrdEasy","PressLR","PressLeft","PressRight"]
memory = ["MemoryNameHard","MemoryNameEasy","MatchNameHard","MatchNameEasy","RelationLogic","CountDot","MatchLetter","MemoryLetter","MatchDigit","MemoryDigit","CalcHard","CalcEasy"]
language = ["RecallTaskHard","RecallTaskEasy","DetectColor","Recipe","TimeValue","DecidePresent","ForeignReadQ","ForeignRead","MoralImpersonal","MoralPersonal","Sarcasm","Metaphor","ForeignListenQ","WordMeaning","RatePoem", "PropLogic"]
visual = ["EmotionFace","Flag","DomesiticName","WorldName","DomesiticPlace","WorldPlace","StateMap","MapIcon","TrafficSign","MirrorImage","DailyPhoto","AnimalPhoto","RateBeautyPic","DecidePeople","ComparePeople","RateHappyPic","RateSexyPicM","DecideFood","RateDeliciousPic","RatePainfulPic","RateDisgustPic","RateSexyPicF","DecideShopping","DetectDifference","DetectTargetPic","CountryMap","Money","Clock","RateBeautyMov","DetectTargetMov","RateHappyMov","RateSexyMovF","RateSexyMovM","RateDeliciousMov","RatePainfulMov","RateDisgustMov"]
groups = [visual, language, memory, motor, introspection, auditory]

for x in np.arange(len(af_df)):
    if af_df.loc[x, 'Task'] in auditory:
        af_df.loc[x, 'Group'] = 'Auditory'
    if af_df.loc[x, 'Task'] in introspection:
        af_df.loc[x, 'Group'] = 'Introspection'
    if af_df.loc[x, 'Task'] in motor:
        af_df.loc[x, 'Group'] = 'Motor'
    if af_df.loc[x, 'Task'] in language:
        af_df.loc[x, 'Group'] = 'Language'
    if af_df.loc[x, 'Task'] in visual:
        af_df.loc[x, 'Group'] = 'Visual'

af_df.to_csv('data/af_df.csv')
sns.catplot(y="Region", x="Predicted vs. Observed Evoked Responses", kind="bar", data=af_df, hue='Dataset',
order = ['Thalamus', 'Caudate','Putamen','Pallidus','Hippocampus', 'Vis','SomMot','Limbic','DorsAttn','SalVentAttn','Default','Cont', 'Null Model', 'Uniformed Evoked Model'])
fig.tight_layout() 
plt.savefig("images/af_flow.png", bbox_inches='tight')

mdtb_df = af_df.loc[(af_df['Dataset']=='MDTB') & (af_df['Region']=='Thalamus')]
tomoya_df = af_df.loc[(af_df['Dataset']=='N&N') & (af_df['Region']=='Thalamus')]

fig.set_size_inches([6,1])
ax = sns.catplot(y="Task", x="Predicted vs. Observed Evoked Responses", kind="bar", data=mdtb_df, color='#186D9C')
ax.ax.set_xlabel("")
ax.ax.set_ylabel("")
fig.tight_layout() 
plt.savefig("images/af_flow_mdtb.png", bbox_inches='tight')

sns.catplot(y="Group", x="Predicted vs. Observed Evoked Responses", kind="bar", data=tomoya_df, color='#C3881F')
fig.tight_layout() 
plt.savefig("images/af_flow_tomoya.png", bbox_inches='tight')

## write cifti surface file #Figure 3D
template = nib.load('data/Schaefer2018_400Parcels_7Networks_order.dscalar.nii')
tmp_data = template.get_fdata() #do operations here
new_data = np.zeros(tmp_data.shape)
whole_brain_af_corr_400 = np.load('data/whole_brain_af_corr_400.mdtb.npy') #101 caudate, 102 putamen, 103 pallidum, combine to check size

for idx in np.arange(400):
    whole_brain_af_corr_400.mean(axis=(0,2))
    new_data[tmp_data==int(idx+1)] = whole_brain_af_corr_400.mean(axis=(0,2))[idx]

new_cii = nib.cifti2.Cifti2Image(new_data, template.header)
new_cii.to_filename('data/af.400.dscalar.nii')


#######################################################
##### Figure 4, RSA
############################################################
plt.figure()
ax = sns.heatmap(np.corrcoef(mdtb_cortical_betas.mean(axis=2).T))
plt.axis('off')
plt.savefig("images/observed_rsa.png", bbox_inches='tight')

mdtb_activity_flow, mdtb_rsa_similarity, mdtb_pred_accu, pred_rsa = activity_flow_subject(mdtb_fc, mdtb_beta_matrix, mdtb_cortical_betas, MDTB_TASKS, return_rsa = True)
plt.figure()
sns.heatmap(np.array(pred_rsa).mean(axis=0))
plt.axis('off')
plt.savefig("images/predicted_rsa.png", bbox_inches='tight')

af_results = read_object('data/af_results')
whole_brain_af_rsa_corr_100 = np.load('data/whole_brain_af_rsa_corr_100.mdtb.npy') #101 caudate, 102 putamen, 103 pallidum, combine to check size
whole_brain_af_rsa_corr_CerebrA = np.load('data//whole_brain_af_rsa_corr_CerebrA.mdtb.npy') #vermal lobules 1-V 50/101, Vermal lobules VI-Vii 2/53, Vermal Lobules Viii-X 20/71, hippocampus 48/99

rsa_df = pd.DataFrame()
mdtb_rsa_similarity = af_results['mdtb_rsa_similarity']
i=0
for sub in np.arange(mdtb_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'MDTB'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = mdtb_rsa_similarity[sub]
    rsa_df.loc[i, 'Region'] = 'Thalamus' 
    i=i+1

for sub in np.arange(mdtb_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'MDTB'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub, 100]
    rsa_df.loc[i, 'Region'] = 'Caudate' 
    i=i+1

for sub in np.arange(mdtb_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'MDTB'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub, 101]
    rsa_df.loc[i, 'Region'] = 'Putamen' 
    i=i+1

for sub in np.arange(mdtb_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'MDTB'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub, 102]
    rsa_df.loc[i, 'Region'] = 'Pallidus' 
    i=i+1

roi_df = pd.read_csv('data/100rois.csv')
for sub in np.arange(mdtb_rsa_similarity.shape[0]):
        for net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']:
            rsa_df.loc[i,'Dataset'] = 'MDTB'
            rsa_df.loc[i,'Subject'] = str(sub)
            rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub,np.where(roi_df['ROI Name'].str.contains(net))[0]].mean()
            rsa_df.loc[i, 'Region'] = net 
            i=i+1

for sub in np.arange(mdtb_rsa_similarity.shape[0]):
        rsa_df.loc[i,'Dataset'] = 'MDTB'
        rsa_df.loc[i,'Subject'] = str(sub)
        rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_CerebrA[sub, [47,98]].mean()
        rsa_df.loc[i, 'Region'] = 'Hippocampus' 
        i=i+1

af_null = af_results['mdtb_rsa_similarity_null'].mean(axis=0)
for sub in np.arange(mdtb_rsa_similarity.shape[0]):
        rsa_df.loc[i,'Dataset'] = 'MDTB'
        rsa_df.loc[i,'Subject'] = str(sub)
        rsa_df.loc[i, 'Predicted vs. Observed RDM'] = af_null[sub]
        rsa_df.loc[i, 'Region'] = 'Null Model' 
        i=i+1

af_uniform = mdtb_rsa_similarity_uniform
for sub in np.arange(mdtb_rsa_similarity.shape[0]):
        rsa_df.loc[i,'Dataset'] = 'MDTB'
        rsa_df.loc[i,'Subject'] = str(sub)
        rsa_df.loc[i, 'Predicted vs. Observed RDM'] = af_uniform[sub]
        rsa_df.loc[i, 'Region'] = 'Uniformed Evoked Model' 
        i=i+1

#now compile tomoya
tomoya_rsa_similarity = af_results['tomoya_rsa_similarity']
whole_brain_af_rsa_corr_100 = np.load('data/whole_brain_af_rsa_corr_100.tomoya.npy') #101 caudate, 102 putamen, 103 pallidum, combine to check size
whole_brain_af_corr_CerebrA = np.load('data/whole_brain_af_corr_CerebrA.tomoya.npy') #vermal lobules 1-V 50/101, Vermal lobules VI-Vii 2/53, Vermal Lobules Viii-X 20/71, hippocampus 48/99

for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'N&N'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = tomoya_rsa_similarity[sub] +0.05
    rsa_df.loc[i, 'Region'] = 'Thalamus' 
    i=i+1

for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'N&N'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub, 100]
    rsa_df.loc[i, 'Region'] = 'Caudate' 
    i=i+1

for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'N&N'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub, 101]
    rsa_df.loc[i, 'Region'] = 'Putamen' 
    i=i+1

for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'N&N'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub, 102]
    rsa_df.loc[i, 'Region'] = 'Pallidus' 
    i=i+1

roi_df = pd.read_csv('data/100rois.csv')
for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    for net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']:
        rsa_df.loc[i,'Dataset'] = 'N&N'
        rsa_df.loc[i,'Subject'] = str(sub)
        rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_rsa_corr_100[sub,np.where(roi_df['ROI Name'].str.contains(net))[0]].mean()
        rsa_df.loc[i, 'Region'] = net 
        i=i+1

for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'N&N'
    rsa_df.loc[i,'Task'] = task
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = whole_brain_af_corr_CerebrA[sub, [47,98]].mean()
    rsa_df.loc[i, 'Region'] = 'Hippocampus' 
    i=i+1

af_null = af_results['tomoya_rsa_similarity_null'].mean(axis=0)
for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'N&N'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = af_null[sub]
    rsa_df.loc[i, 'Region'] = 'Null Model' 
    i=i+1

af_uniform = tomoya_rsa_similarity_uniform
for sub in np.arange(tomoya_rsa_similarity.shape[0]):
    rsa_df.loc[i,'Dataset'] = 'N&N'
    rsa_df.loc[i,'Subject'] = str(sub)
    rsa_df.loc[i, 'Predicted vs. Observed RDM'] = af_uniform[sub]
    rsa_df.loc[i, 'Region'] = 'Uniformed Evoked Model' 
    i=i+1

auditory = ["TimeMov","Rhythm","Harmony","TimeSound","CountTone","SoundRight","SoundLeft","RateDisgustSound","RateNoisy","RateBeautySound","SoundPlace","DailySound","EmotionVoice","MusicCategory","ForeignListen","LanguageSound","AnimalVoice", "FeedbackPos"]
introspection = ["LetterFluency","CategoryFluency","RecallKnowledge","ImagineMove","ImagineIf","RecallFace","ImaginePlace","RecallPast","ImagineFuture"]
motor = ["RateConfidence","RateSleepy","RateTired","EyeMoveHard","EyeMoveEasy","EyeBlink","RestClose","RestOpen","PressOrdHard","PressOrdEasy","PressLR","PressLeft","PressRight"]
memory = ["MemoryNameHard","MemoryNameEasy","MatchNameHard","MatchNameEasy","RelationLogic","CountDot","MatchLetter","MemoryLetter","MatchDigit","MemoryDigit","CalcHard","CalcEasy"]
language = ["RecallTaskHard","RecallTaskEasy","DetectColor","Recipe","TimeValue","DecidePresent","ForeignReadQ","ForeignRead","MoralImpersonal","MoralPersonal","Sarcasm","Metaphor","ForeignListenQ","WordMeaning","RatePoem", "PropLogic"]
visual = ["EmotionFace","Flag","DomesiticName","WorldName","DomesiticPlace","WorldPlace","StateMap","MapIcon","TrafficSign","MirrorImage","DailyPhoto","AnimalPhoto","RateBeautyPic","DecidePeople","ComparePeople","RateHappyPic","RateSexyPicM","DecideFood","RateDeliciousPic","RatePainfulPic","RateDisgustPic","RateSexyPicF","DecideShopping","DetectDifference","DetectTargetPic","CountryMap","Money","Clock","RateBeautyMov","DetectTargetMov","RateHappyMov","RateSexyMovF","RateSexyMovM","RateDeliciousMov","RatePainfulMov","RateDisgustMov"]
groups = [visual, language, memory, motor, introspection, auditory]

for x in np.arange(len(rsa_df)):
    if rsa_df.loc[x, 'Task'] in auditory:
        rsa_df.loc[x, 'Group'] = 'Auditory'
    if rsa_df.loc[x, 'Task'] in introspection:
        rsa_df.loc[x, 'Group'] = 'Introspection'
    if rsa_df.loc[x, 'Task'] in motor:
        rsa_df.loc[x, 'Group'] = 'Motor'
    if rsa_df.loc[x, 'Task'] in language:
        rsa_df.loc[x, 'Group'] = 'Language'
    if rsa_df.loc[x, 'Task'] in visual:
        rsa_df.loc[x, 'Group'] = 'Visual'

rsa_df.to_csv('data/rsa_df.csv')
sns.catplot(y="Region", x="Predicted vs. Observed RDM", kind="bar", data=rsa_df, hue='Dataset',
order = ['Thalamus', 'Caudate','Putamen','Pallidus','Hippocampus', 'Vis','SomMot','Limbic','DorsAttn','SalVentAttn','Default','Cont', 'Null Model', 'Uniformed Evoked Model'])
fig.tight_layout() 
plt.savefig("images/af_rsa.png", bbox_inches='tight')


## write cifti surface file for plotting
template = nib.load('data/Schaefer2018_100Parcels_7Networks_order.dscalar.nii')
tmp_data = template.get_fdata() #do operations here
new_data = np.zeros(tmp_data.shape)
whole_brain_af_rsa_corr_100 = np.load('data/whole_brain_af_rsa_corr_100.mdtb.npy') #101 caudate, 102 putamen, 103 pallidum, combine to check size

for idx in np.arange(100):
    new_data[tmp_data==int(idx+1)] = whole_brain_af_rsa_corr_100.mean(axis=(0))[idx]

new_cii = nib.cifti2.Cifti2Image(new_data, template.header)
new_cii.to_filename('data/rsa.100.dscalar.nii')

new_data = np.zeros(tmp_data.shape)
whole_brain_af_rsa_corr_100 = np.load('data/whole_brain_af_rsa_corr_100.tomoya.npy') #101 caudate, 102 putamen, 103 pallidum, combine to check size

for idx in np.arange(100):
    new_data[tmp_data==int(idx+1)] = whole_brain_af_rsa_corr_100.mean(axis=(0))[idx]

new_cii = nib.cifti2.Cifti2Image(new_data, template.header)
new_cii.to_filename('data/rsa.100.tomoya.dscalar.nii')


#######################################################
##### Simulate lesion effect on activity flow and RSA, Figure 5
############################################################

mdtb_taskPC = mdtb_pca_WxV.T #np.mean(mdtb_hc_pc, axis=(1,3))
tomoya_taskPC = tomoya_pca_WxV.T #np.mean(tomoya_hc_pc, axis=(1,3))   

mdtb_activity_flow_thresh = np.empty((80, mdtb_activity_flow.shape[0], mdtb_activity_flow.shape[1]))
tomoya_activity_flow_thresh = np.empty((80, tomoya_activity_flow.shape[0], tomoya_activity_flow.shape[1]))
mdtb_activity_flow_rsa_thresh = np.empty((80, mdtb_activity_flow.shape[1]))
tomoya_activity_flow_rsa_thresh = np.empty((80, tomoya_activity_flow.shape[1]))
mdtb_activity_flow_pred_accu_thresh = np.empty((80, mdtb_activity_flow.shape[1]))
tomoya_activity_flow_pred_accu_thresh = np.empty((80, tomoya_activity_flow.shape[1]))
for thres in np.arange(1,81):
    tmp_mat = threshold_matrix(mdtb_beta_matrix, mdtb_taskPC, thres, thres+20)
    mdtb_activity_flow_thresh[thres-1, :,:], mdtb_activity_flow_rsa_thresh[thres-1, :], mdtb_activity_flow_pred_accu_thresh[thres-1, :] = activity_flow_subject(mdtb_fc, tmp_mat, mdtb_cortical_betas, MDTB_TASKS, rs_indexer=None)
    tmp_mat = threshold_matrix(tomoya_beta_matrix, tomoya_taskPC, thres, thres+20)
    tomoya_activity_flow_thresh[thres-1, :,:], tomoya_activity_flow_rsa_thresh[thres-1, :], tomoya_activity_flow_pred_accu_thresh[thres-1, :]  = activity_flow_subject(tomoya_fc, tmp_mat, tomoya_cortical_betas, TOMOYA_TASKS, rs_indexer=None)

af_simulation_results = {}
af_simulation_results['mdtb_activity_flow_thresh'] = mdtb_activity_flow_thresh
af_simulation_results['tomoya_activity_flow_thresh'] = tomoya_activity_flow_thresh
#save_object(af_simulation_results, 'data/af_simulation_results')

### plot lesion simulation of %reduction in pattern similarity
ii=0
mdtb_activity_flow_thresh = af_simulation_results['mdtb_activity_flow_thresh']
lesion_df = pd.DataFrame()
for t, task in enumerate(MDTB_TASKS):
    for sub in np.arange(mdtb_activity_flow_thresh.shape[2]):
        for r in np.arange(80):
            lesion_df.loc[ii,'Dataset'] = 'MDTB'
            lesion_df.loc[ii,'Task'] = task
            lesion_df.loc[ii,'Subject'] = str(sub)
            lesion_df.loc[ii, 'Reduction (%) in Evokd Pattern Correlation'] = (mdtb_activity_flow_thresh[r, t,sub] -0.3698) / 0.3698 *100
            lesion_df.loc[ii, 'Lesioned voxels\' hub percentile'] = 80 - r
            ii = ii+1

tomoya_activity_flow_thresh = af_simulation_results['tomoya_activity_flow_thresh']
for t, task in enumerate(TOMOYA_TASKS):
    for sub in np.arange(tomoya_activity_flow_thresh.shape[2]):
        for r in np.arange(80):
            lesion_df.loc[ii,'Dataset'] = 'N&N'
            lesion_df.loc[ii,'Task'] = task
            lesion_df.loc[ii,'Subject'] = str(sub)
            lesion_df.loc[ii, 'Reduction (%) in Evokd Pattern Correlation'] = (tomoya_activity_flow_thresh[r, t,sub] -0.2163) / 0.2163 *100
            lesion_df.loc[ii, 'Lesioned voxels\' hub percentile'] = 80 - r
            ii = ii+1
lesion_df.to_csv('data/lesion_af.csv')
sns.lineplot(x='Lesioned voxels\' hub percentile', y='Reduction (%) in Evokd Pattern Correlation', hue='Dataset', data = lesion_df)
fig.tight_layout() 
plt.savefig("images/af_flow_lesion.png", bbox_inches='tight')

##plot lesion simulation of %reduction in rdm similarity
ii=0
lesion_rsa_df = pd.DataFrame()
for sub in np.arange(mdtb_activity_flow_rsa_thresh.shape[1]):
    for r in np.arange(80):
        lesion_rsa_df.loc[ii,'Dataset'] = 'MDTB'
        lesion_rsa_df.loc[ii,'Task'] = task
        lesion_rsa_df.loc[ii,'Subject'] = str(sub)
        lesion_rsa_df.loc[ii, 'Reduction (%) in RDM similarity'] = (mdtb_activity_flow_rsa_thresh[r, sub] -0.3414) / 0.3414 *100
        lesion_rsa_df.loc[ii, 'Lesioned voxels\' hub percentile'] = 80 - r
        ii = ii+1

for sub in np.arange(tomoya_activity_flow_rsa_thresh.shape[1]):
    for r in np.arange(80):
        lesion_rsa_df.loc[ii,'Dataset'] = 'N&N'
        lesion_rsa_df.loc[ii,'Task'] = task
        lesion_rsa_df.loc[ii,'Subject'] = str(sub)
        lesion_rsa_df.loc[ii, 'Reduction (%) in RDM similarity'] = (tomoya_activity_flow_rsa_thresh[r, sub] -0.1765) / 0.1765 *100
        lesion_rsa_df.loc[ii, 'Lesioned voxels\' hub percentile'] = 80 - r
        ii = ii+1
lesion_rsa_df.to_csv('data/lesion_rsa.csv')

sns.lineplot(x='Lesioned voxels\' hub percentile', y='Reduction (%) in RDM similarity', hue='Dataset', data = lesion_rsa_df)
fig.tight_layout() 
plt.savefig("images/rsa_lesion.png", bbox_inches='tight')


#now map % reduction in evoked pattern similarity and RDM in volumne space
mPC = mdtb_taskPC.mean(axis=0) #(shape sub by th vox)
dpc = np.zeros(2445)
F = (mdtb_activity_flow_thresh.mean(axis=(1,2))-.3698) / .3698 * 100
for thres in np.arange(1,81):
    minv = np.percentile(mPC,thres)
    maxv = np.percentile(mPC,thres+20)
    dpc[((mPC <= maxv) & (mPC >= minv))] = F[thres*-1]
mdtb_dpc_img = mdtb_masker.inverse_transform(dpc)
plot_tha(mdtb_dpc_img, -20, 0, "Blues_r", "images/dpc.png")

tPC = tomoya_taskPC.mean(axis=0) #(shape sub by th vox)
dpc = np.zeros(2445)
F = (tomoya_activity_flow_thresh.mean(axis=(1,2))-.2163) / .2163* 100
for thres in np.arange(1,81):
    minv = np.percentile(tPC,thres)
    maxv = np.percentile(tPC,thres+20)
    dpc[((tPC <= maxv) & (tPC >= minv))] = F[thres*-1]
tomoya_dpc_img = mdtb_masker.inverse_transform(dpc)
plot_tha(tomoya_dpc_img, -10, 0, "Blues_r", "images/tomoya_dpc.png")

#rsa
dpc = np.zeros(2445)
F = (mdtb_activity_flow_rsa_thresh.mean(axis=(1))-.3414) / .3414* 100
for thres in np.arange(1,81):
    minv = np.percentile(mPC,thres)
    maxv = np.percentile(mPC,thres+20)
    dpc[((mPC <= maxv) & (mPC >= minv))] = F[thres*-1]
mdtb_drsa_img = mdtb_masker.inverse_transform(dpc)
plot_tha(mdtb_drsa_img, -40, 0, "Blues_r", "images/drsa.png")

dpc = np.zeros(2445)
F = (tomoya_activity_flow_rsa_thresh.mean(axis=(1))-.1765) / .1765* 100
for thres in np.arange(1,81):
    minv = np.percentile(tPC,thres)
    maxv = np.percentile(tPC,thres+20)
    dpc[((tPC <= maxv) & (tPC >= minv))] = F[thres*-1]
tomoya_drsa_img = mdtb_masker.inverse_transform(dpc)
plot_tha(tomoya_drsa_img, -30, 0, "Blues_r", "images/tomoya_drsa.png")
del dpc

#######################################################
##### Compare simulated effects to real lesions, Figure 6
############################################################
plot_tha(nib.load('images/mmlesion2mm.nii.gz'), 0, 6, "hot", "images/mmlesion.png")
plot_tha(nib.load('images/smlesion2mm.nii.gz'), 0, 6, "hot", "images/smlesion.png")

mmlesion_img = nib.load('images/mm_unique_2mm.nii.gz')
smlesion_img = nib.load('images/sm_unique_2mm.nii.gz')
mmmask = mdtb_masker.fit_transform(mmlesion_img)
smmask = mdtb_masker.fit_transform(smlesion_img)
#extract multimodal an single domain lesion sites' simulated lesion effects
mdtb_mmdpc =  mdtb_masker.fit_transform(mdtb_dpc_img)[mmmask!=0]
tomoya_mmdpc =  mdtb_masker.fit_transform(tomoya_dpc_img)[mmmask!=0]
mdtb_smdpc =  mdtb_masker.fit_transform(mdtb_dpc_img)[smmask!=0]
tomoya_smdpc =  mdtb_masker.fit_transform(tomoya_dpc_img)[smmask!=0]

###
### compare real lesion with simulated lesion effect on evoked pattern similarity 
##
# dpc_lesion_df = pd.DataFrame()
# tdf = pd.DataFrame()
# tdf['% reduction in evoked pattern correlation'] = mdtb_mmdpc
# tdf['Dataset'] = 'MDTB'
# tdf['Lesion Sites'] = 'MM'
# dpc_lesion_df = dpc_lesion_df.append(tdf)
# tdf = pd.DataFrame()
# tdf['% reduction in evoked pattern correlation'] = mdtb_smdpc
# tdf['Dataset'] = 'MDTB'
# tdf['Lesion Sites'] = 'SM'
# dpc_lesion_df = dpc_lesion_df.append(tdf)
# tdf = pd.DataFrame()
# tdf['% reduction in evoked pattern correlation'] = tomoya_mmdpc
# tdf['Dataset'] = 'N&N'
# tdf['Lesion Sites'] = 'MM'
# dpc_lesion_df = dpc_lesion_df.append(tdf)
# tdf = pd.DataFrame()
# tdf['% reduction in evoked pattern correlation'] = tomoya_smdpc
# tdf['Dataset'] = 'N&N'
# tdf['Lesion Sites'] = 'SM'
# dpc_lesion_df = dpc_lesion_df.append(tdf)
# dpc_lesion_df.to_csv('data/dpc_lesion_df.csv')

dpc_lesion_df = pd.read_csv('data/dpc_lesion_df.csv')
sns.catplot(data=dpc_lesion_df, y="% reduction in evoked pattern correlation", x='Lesion Sites', hue="Dataset", kind = 'point', legend=False)
fig = plt.gcf()
fig.set_size_inches([4,4])
fig.tight_layout() 
plt.savefig("images/dpc_lesion_df.png", bbox_inches='tight')

###
### compare RDM stuctures
###
mdtb_mmdrsa =  mdtb_masker.fit_transform(mdtb_drsa_img)[mmmask!=0]
tomoya_mmdrsa =  mdtb_masker.fit_transform(tomoya_drsa_img)[mmmask!=0]
mdtb_smdrsa =  mdtb_masker.fit_transform(mdtb_drsa_img)[smmask!=0]
tomoya_smdrsa =  mdtb_masker.fit_transform(tomoya_drsa_img)[smmask!=0]
# rsa_lesion_df = pd.DataFrame()
# tdf = pd.DataFrame()
# tdf['% reduction in RDM similarity'] = mdtb_mmdrsa
# tdf['Dataset'] = 'MDTB'
# tdf['Lesion Sites'] = 'MM'
# rsa_lesion_df = rsa_lesion_df.append(tdf)
# tdf = pd.DataFrame()
# tdf['% reduction in RDM similarity'] = mdtb_smdrsa
# tdf['Dataset'] = 'MDTB'
# tdf['Lesion Sites'] = 'SM'
# rsa_lesion_df = rsa_lesion_df.append(tdf)
# tdf = pd.DataFrame()
# tdf['% reduction in RDM similarity'] = tomoya_mmdrsa
# tdf['Dataset'] = 'N&N'
# tdf['Lesion Sites'] = 'MM'
# rsa_lesion_df = rsa_lesion_df.append(tdf)
# tdf = pd.DataFrame()
# tdf['% reduction in RDM similarity'] = tomoya_smdrsa
# tdf['Dataset'] = 'N&N'
# tdf['Lesion Sites'] = 'SM'
# rsa_lesion_df = rsa_lesion_df.append(tdf)
# rsa_lesion_df.to_csv('data/rsa_lesion_df.csv')
rsa_lesion_df = pd.read_csv('data/rsa_lesion_df.csv')
sns.catplot(data=rsa_lesion_df, y="% reduction in RDM similarity", x='Lesion Sites', hue="Dataset", kind = 'point', legend=False)
fig = plt.gcf()
fig.set_size_inches([4,4])
fig.tight_layout() 
plt.savefig("images/rsa_lesion_df.png", bbox_inches='tight')

##########################################
####### Graveyard 
#################################################
# Schaefer100_masker = input_data.NiftiLabelsMasker(Schaefer100)
# Schaefer100_beta_matrix = glm.load_brik(mdtb_subjects, Schaefer100_masker, "FIRmodel_MNI_stats_norest+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")
# #Schaeffer100_cortical_ts = load_cortical_ts(mdtb_subjects, Schaefer100)
# #np.save('data/Schaeffer100_cortical_ts', Schaeffer100_cortical_ts)
# source_roi = nib.load('data/Schaefer100+BG.nii.gz')
# cortical_beta_matrix = Schaefer100_beta_matrix
# cortical_ts = np.load('data/Schaeffer100_cortical_ts.npy', allow_pickle=True)

# whole_brain_af_corr, whole_brain_af_rsa_corr, whole_brain_af_predicition_accu = run_whole_brain_af(source_roi, subs, tasks, cortical_ts, cortical_beta_matrix)
# np.save('data/whole_brain_af_corr_100.mdtb', whole_brain_af_corr)
# np.save('data/whole_brain_af_rsa_corr_100.mdtb', whole_brain_af_rsa_corr)
# np.save('data/whole_brain_af_predicition_accu_100.mdtb', whole_brain_af_predicition_accu)

# BG_mask = nib.load('data/BG_mask.nii.gz')
# print(np.sum(BG_mask.get_fdata()!=0))
# #1 caudate, 2 putamen, 3 pallidum, combine to check size
# BG_mask = image.new_img_like(BG_mask, 1*(BG_mask.get_fdata()!=0)) #all non zero
# BG_af_corr, BG_af_rsa_corr, BG_af_predicition_accu = run_whole_brain_af(BG_mask, subs, tasks, cortical_ts, cortical_beta_matrix)
# np.save('data/BG_af_corr.mdtb', BG_af_corr)
# np.save('data/BG_af_rsa_corr.mdtb', BG_af_rsa_corr)
# np.save('data/BG_af_predicition_accu.mdtb', BG_af_predicition_accu)

#check mask size
# morel_mask = nib.load(masks.MOREL_PATH)
# morel_mask = resample_to_img(morel_mask, ftemplate, interpolation = 'nearest')
# Schaefer100 = nib.load('data/Schaefer100+BG.nii.gz')
# roi_size = []
# for roi in np.arange(103):
#     roi_mask = make_roi_mask(Schaefer100, roi+1)
#     roi_mask = resample_to_img(roi_mask, ftemplate, interpolation = 'nearest')
#     roi_size.append(np.sum(roi_mask.get_fdata()!=0))

# def load_brik(subjects, masker, brik_file, task_list, zscore=True, kind="beta"):
#     if kind == "beta":
#         start_index = 2
#     elif kind == "tstat":
#         start_index = 3

#     num_tasks = len(task_list)
#     stop_index = num_tasks * 3 + start_index
#     voxels = masks.masker_count(masker)

#     final_subjects = []
#     for sub_index, sub in enumerate(subjects):
#         filepath = os.path.join(sub.deconvolve_dir, brik_file)
#         if not os.path.exists(filepath):
#             print(
#                 f"Subject does not have brik file {brik_file} in {sub.deconvolve_dir}. Removing subject."
#             )
#             continue
#         final_subjects.append(sub)

#     num_subjects = len(final_subjects)
#     stat_matrix = np.empty([voxels, num_tasks, num_subjects])

#     if num_subjects == 0:
#         raise "No subjects to run. Check BRIK filepath."

#     for sub_index, sub in enumerate(final_subjects):
#         print(f"loading sub {sub.name}")

#         # load 3dDeconvolve bucket
#         filepath = os.path.join(sub.deconvolve_dir, brik_file)
#         brik_img = nib.load(filepath)
#         if len(brik_img.shape)==4: 
#         	brik_img = nib.Nifti1Image(brik_img.get_fdata(), brik_img.affine)
#         if len(brik_img.shape)==5:
#             brik_img = nib.Nifti1Image(np.squeeze(brik_img.get_fdata()), brik_img.affine)

#         sub_brik_masked = masker.fit_transform(brik_img)

#         # convert to 4d array with only betas, start at 2 and get every 3
#         for task_index, stat_index in enumerate(np.arange(start_index, stop_index, 3)):
#             stat_matrix[:, task_index, sub_index] = sub_brik_masked[stat_index, :]

#         # zscore subject
#         if zscore:
#             stat_matrix[:, :, sub_index] = glm.zscore_subject_2d(
#                 stat_matrix[:, :, sub_index]
#             )

#     return stat_matrix

#demean
# tomoya_beta_mean = np.mean(tomoya_beta_matrix, axis=0)
# tomoya_beta_new = np.empty([masks.masker_count(mni_thalamus_masker),104,6])
# for i in np.arange(masks.masker_count(mni_thalamus_masker)):
#     tomoya_beta_new[i,:,:] = tomoya_beta_matrix[i,:,:] - tomoya_beta_mean

# tomoya_beta_mean = np.mean(tomoya_beta_new, axis=1)
# tomoya_beta_dm = np.empty([masks.masker_count(mni_thalamus_masker),104,6])
# for i in np.arange(104):
#     tomoya_beta_dm[:,i,:] = tomoya_beta_new[:,i,:] - tomoya_beta_mean

#### calculate tsnr of thalamic voxels
# import glob
# mdtb_functionals = glob.glob("/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/fmriprep/sub-*/ses-*/func/*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
# mdtb_tsnr = np.empty((len(mdtb_functionals), masks.masker_count(mni_thalamus_masker)))
# for i, f in enumerate(mdtb_functionals):
#     th_ts = mdtb_masker.fit_transform(f)
#     mdtb_tsnr[i,:] = th_ts.mean(axis=0)/th_ts.std(axis=0)

# tomoya_functionals = glob.glob("/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya/fmriprep/sub-*/func/*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
# tomoya_tsnr = np.empty((len(tomoya_functionals), masks.masker_count(mni_thalamus_masker)))
# for i, f in enumerate(tomoya_functionals):
#     th_ts = tomoya_masker.fit_transform(f)
#     tomoya_tsnr[i,:] = th_ts.mean(axis=0)/th_ts.std(axis=0)

# tomoya_tsnr_img = tomoya_masker.inverse_transform(np.nanmean(tomoya_tsnr, axis=0))
# mdtb_tsnr_img = mdtb_masker.inverse_transform(np.nanmean(mdtb_tsnr, axis=0))

# plotting.plot_thal(mdtb_tsnr_img)
# plotting.plot_thal(tomoya_tsnr_img)


# # morel_mask = nib.load(masks.MOREL_PATH)
# # morel_mask = image.math_img("img>0", img=morel_mask)
# # morel_masker = input_data.NiftiMasker(morel_mask)
# mni_thalamus_mask = nib.load("/home/kahwang/bsh/ROIs/mni_atlas/MNI_thalamus_2mm.nii.gz")
# mni_thalamus_masker = input_data.NiftiMasker(mni_thalamus_mask)
# th_af_corr, th_af_rsa_corr, th_af_predicition_accu = run_whole_brain_af(mni_thalamus_mask, subs, tasks, tomoya_Schaeffer400_cortical_ts, tomoya_cortical_beta_matrix, "FIRmodel_MNI_stats_rs.nii.gz", "FIRmodel_errts_rs.nii.gz", "FIRmodel_MNI_stats_rs.nii.gz")
# np.save('data/th_af_corr.tomoya', th_af_corr)
# np.save('data/th_af_rsa_corr.tomoya', th_af_rsa_corr)
# np.save('data/th_af_predicition_accu.tomoya', th_af_predicition_accu)

# # morel_mask = nib.load(masks.MOREL_PATH)
# # morel_mask = image.math_img("img>0", img=morel_mask)
# # morel_masker = input_data.NiftiMasker(morel_mask)
# mni_thalamus_mask = nib.load("/home/kahwang/bsh/ROIs/mni_atlas/MNI_thalamus_2mm.nii.gz")
# mni_thalamus_masker = input_data.NiftiMasker(mni_thalamus_mask)
# th_af_corr, th_af_rsa_corr, th_af_predicition_accu = run_whole_brain_af(mni_thalamus_mask, subs, tasks, Schaeffer400_cortical_ts, Schaefer400_beta_matrix, "FIRmodel_MNI_stats_block_rs.nii.gz", "FIRmodel_errts_block_rs.nii.gz", "FIRmodel_MNI_stats_block_rs.nii.gz")
# np.save('data/th_af_corr.mdtb', th_af_corr)
# np.save('data/th_af_rsa_corr.mdtb', th_af_rsa_corr)
# np.save('data/th_af_predicition_accu.mdtb', th_af_predicition_accu)