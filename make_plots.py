#### script to make various plots for this project
import pwd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('paper', font_scale=1.5)
sns.set_palette("colorblind")
import setup
from thalpy.constants import paths
from matplotlib import pyplot as plt
from thalpy.analysis import pc, plotting, feature_extraction, glm
from thalpy import masks, fc, base
import nibabel as nib
import os
from IPython.display import display, HTML
plt.ion()

# path setup
MDTB_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/"
MDTB_DIR_TREE = base.DirectoryTree(MDTB_DIR)
MDTB_ANALYSIS_DIR = MDTB_DIR + 'analysis/'

def run_pca(matrix, dir_tree, output_name, task_list, masker=None):
    currnet_path=os.getcwd()
    os.chdir(dir_tree.analysis_dir)
    pca, loadings, correlated_loadings, explained_var = feature_extraction.compute_PCA(matrix, output_name=output_name, var_list=task_list, masker=masker)
    
    # for MDTB 
    # pca_comps[:, 1] = pca_comps[:, 1] * -1
    # loadings[:, 1] = loadings[:, 1] * -1
    # correlated_loadings[1] = correlated_loadings[1] * -1

    loadings_df = pd.DataFrame(loadings, index=task_list)
    summary_df = loadings_df.describe()
    var_df = loadings_df.var(axis=0).rename("var")
    summary_df = summary_df.append(var_df)
    #display(HTML(summary_df.to_html()))
    os.chdir(currnet_path)
    return pca, loadings, explained_var

################################################
######## PCA analaysis, Figure 1
################################################
#### Tomoya N&N dataset
tomoya_dir_tree = base.DirectoryTree('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya')
subjects = base.get_subjects(tomoya_dir_tree.deconvolve_dir, tomoya_dir_tree)
stim_config_df = pd.read_csv(tomoya_dir_tree.deconvolve_dir + paths.STIM_CONFIG)
TOMOYA_TASKS = stim_config_df["Stim Label"].tolist()
tomoya_masker = masks.binary_masker(masks.MOREL_PATH)
tomoya_masker.fit(nib.load('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya/3dDeconvolve/sub-01/FIRmodel_MNI_stats.nii'))
tomoya_beta_matrix = glm.load_brik(subjects, tomoya_masker, 'FIRmodel_MNI_stats.nii', 2227, TOMOYA_TASKS, zscore = False)

#demean
# tomoya_beta_mean = np.mean(tomoya_beta_matrix, axis=0)
# tomoya_beta_new = np.empty([2227,104,6])
# for i in np.arange(2227):
#     tomoya_beta_new[i,:,:] = tomoya_beta_matrix[i,:,:] - tomoya_beta_mean

# tomoya_beta_mean = np.mean(tomoya_beta_new, axis=1)
# tomoya_beta_dm = np.empty([2227,104,6])
# for i in np.arange(104):
#     tomoya_beta_dm[:,i,:] = tomoya_beta_new[:,i,:] - tomoya_beta_mean
    
# z-score
tomoya_beta_matrix = glm.zscore_subject_2d(tomoya_beta_matrix)
tomoya_beta_matrix[tomoya_beta_matrix>3] = 0
tomoya_beta_matrix[tomoya_beta_matrix<-3] = 0

#run pca
averaged_sub_beta_tomoya = np.mean(tomoya_beta_matrix, axis=2)
tomoya_pca_comps, tomoya_loadings, tomoya_explained_ave_var = run_pca(averaged_sub_beta_tomoya, tomoya_dir_tree, 'tomoya_pca_groupave', TOMOYA_TASKS, masker=tomoya_masker)
plt.close('all')

#tomoya_explained_var = []
tomoya_pca_WxV = np.zeros([2227, 6])
tomoya_explained_var = pd.DataFrame()
#np.zeros([10,6])

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
stim_config_df = pd.read_csv(MDTB_DIR + paths.DECONVOLVE_DIR + paths.STIM_CONFIG)
CONDITIONS_LIST = stim_config_df["Stim Label"].tolist()

img = nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
mdtb_masker = masks.binary_masker(masks.MOREL_PATH)
mdtb_masker.fit(img)

TASK_LIST = list(set(stim_config_df["Group"].to_list()))
TASK_LIST.remove("Rest")
#load zscored betas
mdtb_beta_matrix = np.load(MDTB_ANALYSIS_DIR + 'mdtb_task_zscored.npy') #check with Evan on how this is generated, used different function than Tomoya?
mdtb_beta_matrix[mdtb_beta_matrix>3] = 0
mdtb_beta_matrix[mdtb_beta_matrix<-3] = 0

#PCA
mdtb_pca_comps, mdtb_loadings, mdtb_explained_var = run_pca(np.mean(mdtb_beta_matrix, axis=2), MDTB_DIR_TREE, 'mdtb_pca_groupave', TASK_LIST, masker=mdtb_masker)
plt.close('all')

mdtb_explained_var = pd.DataFrame()
mdtb_pca_WxV = np.zeros([2227, 21])
#mdtb_explained_var = np.zeros([10,21])
for s in np.arange(mdtb_beta_matrix.shape[2]):
    mat = mdtb_beta_matrix[:,:,s]
    fn = 'mdtb_pca_sub' + str(s)
    comps, _, var = run_pca(mat, MDTB_DIR_TREE, fn, TASK_LIST, masker=mdtb_masker)
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
plotting.plot_thal(tomoya_pca_weight)
plotting.plot_thal(mdtb_pca_weight)

######## Varaince explained plot
df = mdtb_explained_var.append(tomoya_explained_var)
df['Component'] = df['Component'].astype('str')

g = sns.barplot(data=df, x="Component", y="Varaince Explained", hue='Dataset')
g.get_legend().remove()
g2 = g.twinx()
g2 = sns.lineplot(data=df, x="Component", y="Sum of Variance Explained", hue='Dataset', legend = False)
fig = plt.gcf()
fig.set_size_inches([4,4])
fig.tight_layout()
fig.savefig("/home/kahwang/RDSS/tmp/pcvarexp.png")



################################################
######## Task hubs versus rsfc hubs, Figure 2
################################################

