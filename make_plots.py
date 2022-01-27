#### script to make various plots for this project
import os
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
from thalpy.constants import paths
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

# path setup
MDTB_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/"
MDTB_DIR_TREE = base.DirectoryTree(MDTB_DIR)
MDTB_ANALYSIS_DIR = MDTB_DIR + 'analysis/'
tomoya_dir_tree = base.DirectoryTree('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya')
subjects = base.get_subjects(tomoya_dir_tree.deconvolve_dir, tomoya_dir_tree)
stim_config_df = pd.read_csv(tomoya_dir_tree.deconvolve_dir + paths.STIM_CONFIG)
TOMOYA_TASKS = stim_config_df["Stim Label"].tolist()

import pickle
def read_object(filename):
	''' short hand for reading object because I can never remember pickle syntax'''
	o = pickle.load(open(filename, "rb"))
	return o


################################################
######## PCA analaysis, Figure 1
################################################

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


#### Tomoya N&N dataset
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
mdtb_subjects = base.get_subjects(MDTB_DIR_TREE.deconvolve_dir, MDTB_DIR_TREE)
MDTB_TASKS = list(set(stim_config_df["Group"].to_list()))
MDTB_TASKS.remove("Rest")

img = nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
mdtb_masker = masks.binary_masker(masks.MOREL_PATH)
mdtb_masker.fit(img)

TASK_LIST = list(set(stim_config_df["Group"].to_list()))
TASK_LIST.remove("Rest")

#pull betas
mdtb_masker = masks.binary_masker(masks.MOREL_PATH)
mdtb_masker.fit(img)
mdtb_beta_matrix = glm.load_brik(mdtb_subjects, mdtb_masker, "FIRmodel_MNI_stats_norest+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")

#load zscored betas
#mdtb_beta_matrix = np.load(MDTB_ANALYSIS_DIR + 'mdtb_task_zscored.npy') #check with Evan on how this is generated, used different function than Tomoya?
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

for i, item in enumerate(TASK_LIST):
   print(f"{i}: {item}")

######## load cortical betas for hierarchical clustering for MDTB
cortical_betas = np.load(MDTB_ANALYSIS_DIR + "beta_cortical.npy") #z-scored already
cortical_betas_2d = np.empty([400*21, 25]) #concat across subjects
for i in np.arange(21):
    for j in np.arange(400):
        for k in np.arange(25):
            cortical_betas_2d[i*400+j, k] = cortical_betas[j,k,i]

#dend = shc.dendrogram(shc.linkage(cortical_betas_2d.T, method='ward'), labels=TASK_LIST) # need to work on this plot..

#### calculate task PC across different clusinger level and density thresholds
thresholds = np.arange(85,99).tolist()
cluster_num = 8
mdtb_hc_pc = np.empty([21,6, 2227, len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(21):

    for ic, c in enumerate(np.arange(3,cluster_num+1)):
        
        conditions_cluster = hier_cluster(cortical_betas_2d, n_clusters=c)

        for k in np.arange(len(np.unique(conditions_cluster.labels_))):
            group = [condition for i, condition in enumerate(TASK_LIST) if k == conditions_cluster.labels_[i]]
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
mdtb_a_pc = np.empty([21,2227, len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(21):
    pc_matrix = pc.pc_subject(abs(mdtb_beta_matrix)[:,:,s], CI, thresholds=thresholds)
    pc_matrix = np.where(np.isnan(pc_matrix), 0.001, pc_matrix)
    pc_matrix = np.where(pc_matrix <= 0, 0.001, pc_matrix)
    mdtb_a_pc[s,:,:] = pc_matrix

mdtb_a_pc_img = mdtb_masker.inverse_transform(np.mean(mdtb_a_pc, axis=(0,2)))

######## load cortical betas for hierarchical clustering for Tomoya N&N
tomoya_cortical_betas = np.load(os.path.join(tomoya_dir_tree.analysis_dir, 'beta_corticals.npy'))
tomoya_cortical_betas_2d = np.empty([400*6, 104]) #concat across subjects
for i in np.arange(6):
    for j in np.arange(400):
        for k in np.arange(104):
            tomoya_cortical_betas_2d[i*400+j, k] = tomoya_cortical_betas[j,k,i]

thresholds = np.arange(85,99).tolist()
cluster_num = 8
tomoya_hc_pc = np.empty([6,6, 2227, len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(6):
    for ic, c in enumerate(np.arange(3,cluster_num+1)):
        conditions_cluster = hier_cluster(tomoya_cortical_betas_2d, n_clusters=c)
        for k in np.arange(len(np.unique(conditions_cluster.labels_))):
            group = [condition for i, condition in enumerate(TASK_LIST) if k == conditions_cluster.labels_[i]]
            #print(f'k: {k}  group: {group}')

        pc_matrix = pc.pc_subject(abs(tomoya_beta_matrix)[:,:,s], conditions_cluster.labels_, thresholds=thresholds)
        pc_matrix = np.where(np.isnan(pc_matrix), 0.001, pc_matrix)
        pc_matrix = np.where(pc_matrix <= 0, 0.001, pc_matrix)
        tomoya_hc_pc[s,ic,:,:] = pc_matrix

tomoya_hc_pc_img = tomoya_masker.inverse_transform(np.mean(tomoya_hc_pc, axis=(0,1,3)))

# apriori clusters from N&N 2020
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

tomoya_a_pc = np.empty([6, 2227, len(thresholds)]) #sub by clustsize by vox by threshold
for s in np.arange(6):
    pc_matrix = pc.pc_subject(abs(tomoya_beta_matrix)[:,:,s], task_category, thresholds=thresholds)
    pc_matrix = np.where(np.isnan(pc_matrix), 0.001, pc_matrix)
    pc_matrix = np.where(pc_matrix <= 0, 0.001, pc_matrix)
    tomoya_a_pc[s,:,:] = pc_matrix
tomoya_a_pc_img = tomoya_masker.inverse_transform(np.mean(tomoya_a_pc, axis=(0,2)))

# plot task PC
plotting.plot_thal(mdtb_hc_pc_img)
plotting.plot_thal(tomoya_hc_pc_img )
plotting.plot_thal(mdtb_a_pc_img )
plotting.plot_thal(tomoya_a_pc_img )

# plotting.plot_thal(tomoya_pca_weight)
# plotting.plot_thal(tomoya_hc_pc_img, )
# plotting.plot_thal(mdtb_pca_weight)
# plotting.plot_thal(mdtb_hc_pc_img, )


##### Compare to FC PC (rsFC and backgroundFC)
rsFC_pc = tomoya_masker.fit_transform('PC.nii.gz')
rsFC_pc_img = nib.load('PC.nii.gz')
plotting.plot_thal(rsFC_pc_img)

## calculate residualFC PC for mdtb
#load fc objects
mdtb_fc = fc.load(MDTB_ANALYSIS_DIR + "fc_task_residuals.p")

# cortical ROI CI assingment for calculating PC
Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')

def cal_fcpc(thalamocortical_fc):
    ''' clacuate PC with thalamocortical FC data'''
    thresholds = [86,87,88,89,90,91,92,93,94,95,96,97,98]
    pc_vectors = np.zeros((2227, len(thresholds)))
    for it, t in enumerate(thresholds):
        temp_mat = thalamocortical_fc.copy()
        temp_mat[temp_mat<np.percentile(temp_mat, t)] = 0 #threshold
        fc_sum = np.sum(temp_mat, axis=1)
        kis = np.zeros(np.shape(fc_sum))

        for ci in np.unique(Schaeffer_CI):
            kis = kis + np.square(np.sum(temp_mat[:,np.where(Schaeffer_CI==ci)[0]], axis=1) / fc_sum)

        pc_vectors[:, it] = 1-kis
    return np.nanmean(pc_vectors, axis = 1)

mdtb_fcpc = np.empty((21, 2227))
mdtb_fc_mat = np.empty((21,2227,400))
for s in np.arange(21):
    mdtb_fc_mat[s,:,:] = mdtb_fc.fc_subjects[s].seed_to_voxel_correlations
    mdtb_fcpc[s, :] = cal_fcpc(mdtb_fc_mat[s,:,:])

mdtb_fcpc_img = mdtb_masker.inverse_transform(np.nanmean(mdtb_fcpc, axis=0))
plotting.plot_thal(mdtb_fcpc_img)

## tomoya fcpc
tomoya_fc = fc.load(tomoya_dir_tree.analysis_dir + "fc_task_residuals.p")
tomoya_fcpc = np.empty((6, 2227))
tomoya_fc_mat = np.empty((6,2227,400))
for s in np.arange(6):
    tomoya_fc_mat[s,:,:] = tomoya_fc.fc_subjects[s].seed_to_voxel_correlations
    tomoya_fcpc[s, :] = cal_fcpc(tomoya_fc_mat[s,:,:])

tomoya_fcpc_img = tomoya_masker.inverse_transform(np.nanmean(tomoya_fcpc, axis=0))
plotting.plot_thal(tomoya_fcpc_img)

#### calculate tsnr of thalamic voxels
import glob
mdtb_functionals = glob.glob("/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/fmriprep/sub-*/ses-*/func/*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
mdtb_tsnr = np.empty((len(mdtb_functionals), 2227))
for i, f in enumerate(mdtb_functionals):
    th_ts = mdtb_masker.fit_transform(f)
    mdtb_tsnr[i,:] = th_ts.mean(axis=0)/th_ts.std(axis=0)

tomoya_functionals = glob.glob("/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya/fmriprep/sub-*/func/*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
tomoya_tsnr = np.empty((len(tomoya_functionals), 2227))
for i, f in enumerate(tomoya_functionals):
    th_ts = tomoya_masker.fit_transform(f)
    tomoya_tsnr[i,:] = th_ts.mean(axis=0)/th_ts.std(axis=0)

tomoya_tsnr_img = tomoya_masker.inverse_transform(np.nanmean(tomoya_tsnr, axis=0))
mdtb_tsnr_img = mdtb_masker.inverse_transform(np.nanmean(mdtb_tsnr, axis=0))


plotting.plot_thal(mdtb_tsnr_img)
plotting.plot_thal(tomoya_tsnr_img)


################################################
######## Activity flow prediction, Figure 3
################################################
# activity flow analysis: predicited cortical evoked responses = thalamus evoke x thalamocortical FC, compare to observed cortical evoked responses

# load fc objects again, thalamocortical fc matrices stored here
mdtb_fc = fc.load(MDTB_ANALYSIS_DIR + "fc_task_residuals.p")
tomoya_fc = fc.load(tomoya_dir_tree.analysis_dir + "fc_task_residuals.p")

# load thalamus task evoked responses, mdtb
# mtdb_subjects = base.get_subjects(MDTB_DIR_TREE.deconvolve_dir, MDTB_DIR_TREE)
# stim_config_df = pd.read_csv(MDTB_DIR + paths.DECONVOLVE_DIR + paths.STIM_CONFIG)
# img = nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
# mdtb_masker = masks.binary_masker(masks.MOREL_PATH)
# mdtb_masker.fit(img)
MDTB_TASKS = list(set(stim_config_df["Group"].to_list()))
MDTB_TASKS.remove("Rest")
# mtdb_task_matrix, mtdb_task_tstat_matrix, masker = setup.setup_mdtb('block', masker = mdtb_masker, is_setup_block=True, voxels=2227)
mdtb_beta_matrix = np.load(MDTB_ANALYSIS_DIR + 'mdtb_task_zscored.npy')

# tomoya
# tomoya_dir_tree = base.DirectoryTree('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya')
# tomoya_subjects = base.get_subjects(tomoya_dir_tree.deconvolve_dir, tomoya_dir_tree)
stim_config_df = pd.read_csv(tomoya_dir_tree.deconvolve_dir + paths.STIM_CONFIG)
tomoya_TASKS = stim_config_df["Stim Label"].tolist()
# tomoya_beta_matrix = glm.load_brik(tomoya_subjects, tomoya_masker, 'FIRmodel_MNI_stats.nii', 2227, TOMOYA_TASKS, zscore = True)
tomoya_beta_matrix = np.load(os.path.join(tomoya_dir_tree.analysis_dir, 'tomoya_beta.npy'))

# observed cortical evoked responses
mdtb_cortical_betas = np.load(MDTB_ANALYSIS_DIR + "beta_cortical.npy") #z-scored already
tomoya_cortical_betas = np.load(os.path.join(tomoya_dir_tree.analysis_dir, 'beta_corticals.npy'))

# load activity flow results df
mdtb_activityflow_df = read_object(MDTB_ANALYSIS_DIR + 'mdtb_activity_flow_dataframe.p')
tomoya_activityflow_df = read_object(tomoya_dir_tree.analysis_dir + 'mdtb_activity_flow_dataframe.p')

#functions that run activity flow
def activity_flow_subject(rs, thal_betas, cortical_betas, task_list, rs_indexer=None):
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
        num_subjects = 1 #if not using fc object, single subject analysis

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
            sub_rs = rs[:,:] #not organize in fc object strct from Evan
            sub_rs = np.arctan(sub_rs)

        predicted_corticals = np.dot(np.swapaxes(thal_betas[:, :, rs_index], 0, 1), sub_rs) 
        sub_cortical_betas = np.swapaxes(cortical_betas[:, :, rs_index], 0, 1)
        
        #rsa
        predicted_rsa = np.corrcoef(predicted_corticals)[np.triu_indices(predicted_corticals.shape[0],k=1)] #only upper triangle
        observed_rsa = np.corrcoef(sub_cortical_betas)[np.triu_indices(sub_cortical_betas.shape[0],k=1)]
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

    return sub_cor_array, sub_rsa, sub_accu

#distance function for rsa
def rsa_mahalanobis(mat):
    '''calculate rsa matrix using mahanobis distance, input expected to be ROI by condition, the output will be condition by condition, and noise cov will be ROI by ROI'''
    num_roi = mat.shape[1]
    num_cond = mat.shape[0]
    # try:
    vi = np.linalg.inv(np.cov(mat)) #inverse of cov
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

### run basic activity flow and null models
mdtb_activity_flow, mdtb_rsa_similarity, mdtb_pred_accu = activity_flow_subject(mdtb_fc, mdtb_beta_matrix, mdtb_cortical_betas, MDTB_TASKS, rs_indexer=None)
tomoya_activity_flow, tomoya_rsa_similarity, tomoya_pred_accu = activity_flow_subject(tomoya_fc, tomoya_beta_matrix, tomoya_cortical_betas, tomoya_TASKS, rs_indexer=None)
mdtb_activity_flow_null, mdtb_rsa_similarity_null, mdtb_pred_accu_null = null_activity_flow(mdtb_beta_matrix, mdtb_cortical_betas, mdtb_fc, MDTB_TASKS, num_permutation=1000)
tomoya_activity_flow_null, tomoya_rsa_similarity_null, tomoya_pred_accu_null = null_activity_flow(tomoya_beta_matrix, tomoya_cortical_betas, tomoya_fc, tomoya_TASKS, num_permutation=1000)

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
    tomoya_activity_flow_thresh[thres-1, :,:], tomoya_activity_flow_rsa_thresh[thres-1, :], tomoya_activity_flow_pred_accu_thresh[thres-1, :]  = activity_flow_subject(tomoya_fc, tmp_mat, tomoya_cortical_betas, tomoya_TASKS, rs_indexer=None)

### set evoke as uniform
mdtb_activity_flow_uniform, _, _ = activity_flow_subject(mdtb_fc, np.ones((mdtb_beta_matrix.shape)), mdtb_cortical_betas, MDTB_TASKS, rs_indexer=None)
tomoya_activity_flow_uniform, _, _ = activity_flow_subject(tomoya_fc, np.ones((tomoya_beta_matrix.shape)), tomoya_cortical_betas, tomoya_TASKS, rs_indexer=None)

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

test_noise(mdtb_beta_matrix)


### write all the outputs into dataframe for plotting, and plot.


################################################
######## Whole brain activity flow, Figure 4
################################################
#rs = fc.load(MDTB_ANALYSIS_DIR + "fc_task_residuals_900.p")
#cortical_betas = np.load(MDTB_ANALYSIS_DIR + "beta_corticals_900.npy")
#num_roi = 900

def make_voxel_roi_masker(cortical_mask,roi_idx):
    ''' create a roi masker to pull out every voxel separately with a seprate integer'''
    roi_size = np.sum(Schaefer400.get_fdata()==roi_idx)
    data = 1*(cortical_mask.get_fdata()==roi_idx)
    data[data==1] = np.arange(roi_size)+1
    roi_mask = nilearn.image.new_img_like(cortical_mask, data)
    #roi_masker = input_data.NiftiLabelsMasker(roi_mask)
    return roi_mask

def make_roi_mask(cortical_mask,roi_idx):
    ''' create a roi mask'''
    roi_size = np.sum(Schaefer400.get_fdata()==roi_idx)
    data = 1*(cortical_mask.get_fdata()==roi_idx)
    #data[data==1] = np.arange(roi_size)
    roi_mask = nilearn.image.new_img_like(cortical_mask, data)
    #roi_masker = input_data.NiftiLabelsMasker(roi_mask)
    return roi_mask

def load_cortical_ts(subjects, cortical_mask):
    ''' because the nilearn masker is slow, let us preload the cortical timeseries from ROIs'''
    cortical_ts = []
    cortical_masker = input_data.NiftiLabelsMasker(cortical_mask)
    for sub_index, sub in enumerate(subjects):
        print(sub_index)
        funcfile = nib.load(sub.deconvolve_dir + 'FIRmodel_errts_norest.nii.gz')
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

#load whole brain ROI mask, here we combine Schaeffer 400 with several subcortical masks
mdtb_subjects = base.get_subjects(MDTB_DIR_TREE.deconvolve_dir, MDTB_DIR_TREE)
#stim_config_df = pd.read_csv(MDTB_DIR + paths.DECONVOLVE_DIR + paths.STIM_CONFIG)
#MDTB_TASKS = list(set(stim_config_df["Group"].to_list()))
#MDTB_TASKS.remove("Rest")


#schaefer_masker_900 = input_data.NiftiLabelsMasker(nib.load('/data/backed_up/shared/ROIs/Schaefer2018_900Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'))
Schaefer400 = nib.load('/data/backed_up/shared/ROIs/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
#Schaefer400 = resample_to_img(Schaefer400, '/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/3dDeconvolve/sub-21/combined_mask+tlrc.BRIK', interpolation = 'nearest') #speed up, otherwise everytime masker is in operation it resamples itself, slow!
Schaefer400_masker = input_data.NiftiLabelsMasker(Schaefer400)
#Schaefer400_masker.fit('/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/3dDeconvolve/sub-21/combined_mask+tlrc.BRIK')
Schaefer400_beta_matrix = glm.load_brik(mdtb_subjects, Schaefer400_masker, "FIRmodel_MNI_stats_norest+tlrc.BRIK", MDTB_TASKS, zscore=True, kind="beta")
#Schaeffer400_cortical_ts = load_cortical_ts(mdtb_subjects, Schaefer400)
cortical_ts = read_object('data/Schaeffer400_cortical_ts')

num_roi = 400
num_sub = len(mdtb_subjects)
num_tasks = len(MDTB_TASKS)
whole_brain_af_corr = np.zeros((num_sub, num_roi, num_tasks))
whole_brain_af_rsa_corr =  np.zeros((num_sub, num_roi))
whole_brain_af_predicition_accu =  np.zeros((num_sub, num_roi))

for sub_index, sub in enumerate(mdtb_subjects):
    funcfile = sub.deconvolve_dir + 'FIRmodel_errts_norest.nii.gz'
    fdata = nib.load(funcfile).get_fdata() #preload data to save time without using nilearn masker    
 
    for roi in np.arange(num_roi):
        roi = roi+1
        roi_voxel_mask = make_voxel_roi_masker(Schaefer400, roi)
        ftemplate = nib.load(sub.deconvolve_dir+'NoGo_FIR_MIN.nii.gz')
        roi_voxel_mask = resample_to_img(roi_voxel_mask, ftemplate, interpolation = 'nearest')
        #num_voxels = np.sum(roi_voxel_mask.get_fdata()!=0) 
        roi_voxel_masker = input_data.NiftiLabelsMasker(roi_voxel_mask)
        #need to extract voxel wise task beta for each roi
        roi_beta_matrix = load_brik([sub], roi_voxel_masker, "FIRmodel_MNI_stats_norest.nii.gz" , MDTB_TASKS, zscore=True, kind="beta")
        
        #then need to calculate voxel-whole brain FC matrix
        roi_mask = make_roi_mask(Schaefer400, roi)
        roi_mask = resample_to_img(roi_mask, ftemplate, interpolation = 'nearest')
        roi_ts = fdata[np.nonzero(roi_mask.get_fdata())] #voxel ts in roi, use direct indexing to save time
        wb_ts = cortical_ts[sub_index] #whole brain ts

        # check censored data
        roi_ts =np.delete(roi_ts, np.where(roi_ts.mean(axis=0)==0)[0], axis=1)
        wb_ts =np.delete(wb_ts, np.where(wb_ts.mean(axis=1)==0)[0], axis=0)
        if roi_ts.shape[1] != wb_ts.shape[0]: #check length
            continue

        #roi vox by whole brain corr mat
        fcmat = generate_correlation_mat(roi_ts, wb_ts.T)
        fcmat[np.isnan(fcmat)] = 0

        #aflow
        task_corr, rsa_corr, pred_accu = activity_flow_subject(fcmat, roi_beta_matrix, Schaefer400_beta_matrix, MDTB_TASKS)
        whole_brain_af_corr[sub_index, roi-1, :] = task_corr[:,0]
        whole_brain_af_rsa_corr[sub_index, roi-1] = rsa_corr
        whole_brain_af_predicition_accu[sub_index, roi-1]= pred_accu

np.save('data/whole_brain_af_corr.mdtb', whole_brain_af_corr)
np.save('data/whole_brain_af_rsa_corr.mdtb', whole_brain_af_rsa_corr)
np.save('data/whole_brain_af_predicition_accu.mdtb', whole_brain_af_predicition_accu)


def load_brik(subjects, masker, brik_file, task_list, zscore=True, kind="beta"):
    if kind == "beta":
        start_index = 2
    elif kind == "tstat":
        start_index = 3

    num_tasks = len(task_list)
    stop_index = num_tasks * 3 + start_index
    voxels = masks.masker_count(masker)

    final_subjects = []
    for sub_index, sub in enumerate(subjects):
        filepath = os.path.join(sub.deconvolve_dir, brik_file)
        if not os.path.exists(filepath):
            print(
                f"Subject does not have brik file {brik_file} in {sub.deconvolve_dir}. Removing subject."
            )
            continue
        final_subjects.append(sub)

    num_subjects = len(final_subjects)
    stat_matrix = np.empty([voxels, num_tasks, num_subjects])

    if num_subjects == 0:
        raise "No subjects to run. Check BRIK filepath."

    for sub_index, sub in enumerate(final_subjects):
        print(f"loading sub {sub.name}")

        # load 3dDeconvolve bucket
        filepath = os.path.join(sub.deconvolve_dir, brik_file)
        brik_img = nib.load(filepath)
        if len(brik_img.shape)==4: 
        	brik_img = nib.Nifti1Image(brik_img.get_fdata(), brik_img.affine)
        if len(brik_img.shape)==5:
            brik_img = nib.Nifti1Image(np.squeeze(brik_img.get_fdata()), brik_img.affine)

        sub_brik_masked = masker.fit_transform(brik_img)

        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, stat_index in enumerate(np.arange(start_index, stop_index, 3)):
            stat_matrix[:, task_index, sub_index] = sub_brik_masked[stat_index, :]

        # zscore subject
        if zscore:
            stat_matrix[:, :, sub_index] = glm.zscore_subject_2d(
                stat_matrix[:, :, sub_index]
            )

    return stat_matrix




