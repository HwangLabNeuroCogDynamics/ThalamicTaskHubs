#### script to make various plots for this project
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('paper', font_scale=1.5)
sns.set_palette("colorblind")
import setup
from thalpy.constants import paths
from matplotlib import pyplot as plt
from thalpy.analysis import pc, plotting, feature_extraction
from thalpy import masks
from thalpy import fc
from thalpy import base
import nibabel as nib
from thalpy import glm
import os

# path setup
MDTB_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/"
MDTB_DIR_TREE = base.DirectoryTree(MDTB_DIR)
MDTB_ANALYSIS_DIR = MDTB_DIR + 'analysis/'

################################################
######## PCA analaysis
################################################
stim_config_df = pd.read_csv(MDTB_DIR + paths.DECONVOLVE_DIR + paths.STIM_CONFIG)
CONDITIONS_LIST = stim_config_df["Stim Label"].tolist()

img = nib.load(MDTB_DIR_TREE.fmriprep_dir + "sub-02/ses-a1/func/sub-02_ses-a1_task-a_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
mdtb_masker = masks.binary_masker(masks.MOREL_PATH)
mdtb_masker.fit(img)

TASK_LIST = list(set(stim_config_df["Group"].to_list()))
TASK_LIST.remove("Rest")

from thalpy.analysis import glm

tomoya_dir_tree = base.DirectoryTree('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya')
subjects = base.get_subjects(tomoya_dir_tree.deconvolve_dir, tomoya_dir_tree)
stim_config_df = pd.read_csv(tomoya_dir_tree.deconvolve_dir + paths.STIM_CONFIG)
TOMOYA_TASKS = stim_config_df["Stim Label"].tolist()
tomoya_masker = masks.binary_masker(masks.MOREL_PATH)
tomoya_masker.fit(nib.load('/mnt/nfs/lss/lss_kahwang_hpc/data/Tomoya/3dDeconvolve/sub-01/FIRmodel_MNI_stats.nii'))
tomoya_beta_matrix = glm.load_brik(subjects, tomoya_masker, 'FIRmodel_MNI_stats.nii', 2227, TOMOYA_TASKS, zscore = False)



################################################
######## Varaince explained plot
################################################
# plot PC variances explained
df1 = pd.DataFrame()
df1['Component'] = np.arange(1,11)
df1['Dataset'] = 'MDTB'
df1['Varaince Explained'] = np.array([0.49422765, 0.22373263, 0.07304692, 0.05084266, 0.03473523, 0.02276043, 0.01554752, 0.01323701, 0.01273611, 0.00932534])
df1['Sum of Variance Explained'] = df1['Varaince Explained'].cumsum()

df2 = pd.DataFrame()
df2['Component'] = np.arange(1,11)
df2['Dataset'] = 'N&N'
df2['Varaince Explained'] = np.array([0.24283326, 0.13583435, 0.1266729,  0.04636843, 0.03969473, 0.03290696, 0.02337343, 0.02231827, 0.0147305,  0.01280407])
df2['Sum of Variance Explained'] = df2['Varaince Explained'].cumsum()
df = df1.append(df2)
df['Component'] = df['Component'].astype('str')

g = sns.barplot(data=df, x="Component", y="Varaince Explained", hue='Dataset')
g.get_legend().remove()
g2 = g.twinx()
g2 = sns.lineplot(data=df, x="Component", y="Sum of Variance Explained", hue='Dataset', legend = False)
fig = plt.gcf()
fig.set_size_inches([4,3])
fig.tight_layout()
fig.savefig("/home/kahwang/RDSS/tmp/pcave.png")

3dcalc -a tomoya_pca_component_0.nii -b tomoya_pca_component_7.nii -c tomoya_pca_component_9.nii -d tomoya_pca_component_3.nii -e tomoya_pca_component_4.nii -f tomoya_pca_component_5.nii -g tomoya_pca_component_6.nii -expr "(abs(a) + abs(b)*3 + abs(c)*3 + abs(d) + abs(e) + abs(f) + abs(g)*10)/10" -prefix tomoya_pca_weight.nii.gz


