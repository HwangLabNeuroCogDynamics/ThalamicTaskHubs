import common
import basic_settings as bs
import numpy as np
from nilearn import image, plotting, input_data
import matplotlib.pyplot as plt
import nibabel as nib
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
import pandas as pd

DATASET_DIR = '/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/'
ANALYSIS_DIR = DATASET_DIR + 'analysis/'
stim_config_df = pd.read_csv(DATASET_DIR + bs.DECONVOLVE_DIR + bs.STIM_CONFIG)
TASK_LIST = stim_config_df['Stim Label'].tolist()
GROUP_LIST = set(stim_config_df['Group'].to_list())


def setup():
    """Loads 3dDeconvolve data into matrix with 3D [960, 43, 21]
    (Voxels, Tasks, Subjects). Returns this matrix and Nifti masker.

    Returns
    -------
    numpy array
        Matrix storing beta values [960 Voxels, 45 Tasks, 20 Subjects]

    numpy array
        Matrix storing tstat values [960 Voxels, 45 Tasks, 20 Subjects]

    Nifti1Masker
        Object storing mask to transpose matrix back to MRI space
    """
    # set directory tree and get subjects

    dir_tree = common.DirectoryTree(DATASET_DIR)
    subjects = common.get_subjects(dir_tree)

    # set number of subjects (four didn't run due to exclusion based on motion)
    # ,standard shape and affine
    numsub = len(subjects) - 3
    STD_SHAPE = [79, 94, 65]
    STD_AFFINE = nib.load(
        subjects[1].deconvolve_dir + 'Go_FIR_MIN.nii.gz').affine

    # create complete matrix with 3 dimensions: voxels, tasks, and subjects
    beta_matrix = np.zeros([960, 45, numsub])
    tstat_matrix = np.zeros([960, 45, numsub])

    # Get mask
    mask = nib.load(
        ANALYSIS_DIR + 'Thalamus_Morel_consolidated_mask_v3.nii.gz')
    mask = image.math_img("img>0", img=mask)
    # mask = image.resample_img(mask, target_affine=STD_AFFINE,
    #                          target_shape=STD_SHAPE, interpolation='nearest')
    masker = input_data.NiftiMasker(
        mask, target_affine=STD_AFFINE, target_shape=STD_SHAPE)
    masker.fit()

    subject_index = 0
    for sub in subjects:
        filepath = sub.deconvolve_dir + 'FIRmodel_MNI_stats+tlrc.BRIK'
        if not os.path.exists(filepath):
            continue
        print(f'loading sub {sub.name}')

        # load 3dDeconvolve bucket
        sub_fullstats_4d = nib.load(filepath)
        sub_fullstats_4d_data = sub_fullstats_4d.get_fdata()

        beta_task_matrix = np.zeros([960, 45])
        tstat_task_matrix = np.zeros([960, 45])
        group_matrix = np.zeros([960, 30])

        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, i in enumerate(np.arange(2, 203, 3)):
            beta_array = sub_fullstats_4d_data[:, :, :, i]
            beta_array = nib.Nifti1Image(beta_array, STD_AFFINE)
            beta_array = masker.transform(beta_array).flatten()

            if task_index < 45:
                beta_task_matrix[:, task_index] = beta_array
    #             else:
    #                 group_matrix[:, task_index - 43] = beta_array
        beta_matrix[:, :, subject_index] = beta_task_matrix

        # get tstat matrix
        for task_index, i in enumerate(np.arange(3, 203, 3)):
            tstat_array = sub_fullstats_4d_data[:, :, :, i]
            tstat_array = nib.Nifti1Image(tstat_array, STD_AFFINE)
            tstat_array = masker.transform(tstat_array).flatten()

            if task_index < 45:
                tstat_task_matrix[:, task_index] = tstat_array
    #             else:
    #                 group_matrix[:, task_index - 43] = beta_array

        tstat_matrix[:, :, subject_index] = tstat_task_matrix
        subject_index += 1

    return beta_matrix, tstat_matrix, masker

# def differentiate_tasks():


def cluster_sub(sub_matrix, k):
    sub_correlation_matrix = np.corrcoef(sub_matrix)
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(sub_correlation_matrix)

    # plus 1 is just for plotting purposes, labels with 0 show up not having color
    return model.labels_ + 1, model


def plot_clusters(sub_matrix):
    ks = range(2, 10)
    inertias = []
    for k in ks:
        cluster_atlas, model = cluster_sub(sub_matrix, k)
        plotting.plot_roi(cluster_atlas, cmap=plt.cm.get_cmap('tab10'))
        plotting.show()

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


def consensus_cluster(task_matrix, masker, label='beta'):
    # consensus clustering

    for k_cluster in range(2, 8):
        consensus_matrix = np.zeros([960, 960, task_matrix.shape[-1]])
        for i in range(task_matrix.shape[-1]):
            sub_cluster, model = cluster_sub(
                task_matrix[:, :, i], k_cluster)
            coassignment_matrix = np.zeros([960, 960])
            for j in range(960):
                for k in range(960):
                    if sub_cluster[j] == sub_cluster[k]:
                        coassignment_matrix[j][k] = 1
                    else:
                        coassignment_matrix[j][k] = 0
            consensus_matrix[:, :, i] = coassignment_matrix

        mean_matrix = consensus_matrix.mean(2)
        final_consensus_cluster, model = cluster_sub(mean_matrix, k_cluster)
        final_consensus_cluster = masker.inverse_transform(
            final_consensus_cluster)
        nib.save(final_consensus_cluster, ANALYSIS_DIR
                 + f'{label}_consensus_cluster_{k_cluster}.nii')


def compute_PCA(task_matrix, masker):
    PCA_matrix = normalize_task_matrix(task_matrix)

    # set pca to explain 95% of variance
    pca = PCA(.95)
    PCA_components = pca.fit_transform(PCA_matrix)

    loadings = pd.DataFrame(pca.components_.T, index=TASK_LIST)
    print(loadings)

    correlated_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    correlated_loadings = pd.DataFrame(
        correlated_loadings, index=TASK_LIST)
    print(correlated_loadings)

    # Plot the explained variances
    # features = range(pca.n_components_)
    # plt.bar(features, pca.explained_variance_ratio_, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(features)
    # plt.show()

    # print variance explained by each component
    print(pca.explained_variance_ratio_)

    # save each component into nifti form
    for index in range(10):
        # save PC back into nifti image and visualize
        comp_array = PCA_components[:, index]
        comp_array = masker.inverse_transform(comp_array)
        nib.save(comp_array, ANALYSIS_DIR +
                 'Averaged_PCA_component_' + str(index) + '.nii')
        # view = plotting.view_img(comp_array, threshold=3)
        # # In a Jupyter notebook, if ``view`` is the output of a cell, it will
        # # be displayed below the cell
        # view.open_in_browser()

    return PCA_components


def normalize_task_matrix(matrix):
    print(matrix.shape)
    # normalize each subject matrix and average across subjects
    std_matrix = stats.zscore(matrix, axis=2)
    std_matrix = matrix.mean(2)
    return std_matrix


def participation_coefficient_abs(task_matrix, masker, label='beta'):
    task_matrix = normalize_task_matrix(task_matrix)
    PC_matrix = np.zeros([task_matrix.shape[0]])

    # get PC value for each voxel
    for voxel_index in range(task_matrix.shape[0]):
        sum_PC = 0.0
        sum_task = np.absolute(task_matrix[voxel_index, :]).sum()
        # sum (kis / ki)^ 2 for each task
        for task_index in range(task_matrix.shape[1]):
            sum_PC += (task_matrix[voxel_index, task_index] / sum_task) ** 2
        PC = 1 - sum_PC
        PC_matrix[voxel_index] = PC

    print(PC_matrix)
    PC_matrix = masker.inverse_transform(PC_matrix)
    nib.save(PC_matrix, ANALYSIS_DIR + label +
             'participation_coefficient_abs' + '.nii')


def participation_coefficient_ex(task_matrix, masker, label='beta'):
    task_matrix = normalize_task_matrix(task_matrix)
    PC_matrix = np.zeros([task_matrix.shape[0]])

    # get PC value for each voxel
    for voxel_index in range(task_matrix.shape[0]):

        sum_PC = 0.0
        sum_task = np.absolute(task_matrix[voxel_index, :]).sum()
        # sum (kis / ki)^ 2 for each task
        for task_index in range(task_matrix.shape[1]):
            sum_PC += (task_matrix[voxel_index, task_index] / sum_task) ** 2
        PC = 1 - sum_PC
        PC_matrix[voxel_index] = PC

    print(PC_matrix)
    PC_matrix = masker.inverse_transform(PC_matrix)
    nib.save(PC_matrix, ANALYSIS_DIR + label +
             'participation_coefficient_ex' + '.nii')


def participation_coefficient_thr(task_matrix, masker, label='beta'):
    task_matrix = normalize_task_matrix(task_matrix)
    task_matrix = np.apply_along_axis(threshold_arr, 0, task_matrix)
    PC_matrix = np.zeros([task_matrix.shape[0]])

    # get PC value for each voxel
    for voxel_index in range(task_matrix.shape[0]):

        sum_PC = 0.0
        sum_task = task_matrix[voxel_index, :].sum()
        # sum (kis / ki)^ 2 for each task
        for task_index in range(task_matrix.shape[1]):
            sum_PC += (task_matrix[voxel_index,
                                   task_index] / sum_task) ** 2
        PC = 1 - sum_PC
        PC_matrix[voxel_index] = PC

    print(PC_matrix)
    PC_matrix = masker.inverse_transform(PC_matrix)
    nib.save(PC_matrix, ANALYSIS_DIR + label +
             '_participation_coefficient_thr' + '.nii')


def threshold_arr(arr):
    new_arr = np.zeros(arr.shape)
    for index in range(arr.shape[0]):
        element = arr[index]
        if element > 0:
            new_arr[index] = element
        else:
            new_arr[index] = 0
    print(new_arr)
    return new_arr


# PCA on each individual tasks
# greene et al - method : winner take all for each voxel, threshold

def greene_method(task_matrix, threshold=0.750):
    # z score standardization for each subject
    task_matrix = stats.zscore(task_matrix, axis=2)

    # initialize matrix to store winning task in each voxel for each subjects
    # Shape is [Voxels, Subjects]
    winner_take_all_list = [[None] * task_matrix.shape[2]] * 960

    # loop through each subject and each voxel to get winning task
    for subject_index in range(task_matrix.shape[2]):
        for voxel_index in range(task_matrix.shape[0]):
            task_array = task_matrix[voxel_index, :, subject_index]
            max_task_value = np.amax(task_array)
            max_index = np.where(task_array == max_task_value)
            # todo import task list and check if is in order of 3d brik
            if len(max_index[0]) > 1:
                print('more than one task')
                task_name = 'Multiple'
                is_specific = False
            else:
                print(max_index[0])
                task_name = TASK_LIST[max_index[0][0]]
                is_specific = determine_specificity(
                    task_array, max_task_value, max_index, threshold)

            winner_take_all_list[voxel_index][subject_index] = [
                max_task_value, task_name, is_specific]

    return winner_take_all_list


def determine_specificity(task_array, max_value, max_index, threshold):
    threshold_value = max_value * threshold
    for index, task_beta in enumerate(task_array):
        if max_index == index:
            continue
        if task_beta > threshold_value:
            return False

    return True


def group_tasks(task_matrix):
    print(GROUP_LIST)
    grouped_df = stim_config_df.groupby('Group')
    grouped_task_matrix = np.zeros(
        [task_matrix.shape[0], len(grouped_df), task_matrix.shape[2]])

    for subject_index in range(task_matrix.shape[2]):
        sub_dict = {}
        for task_index in range(task_matrix.shape[1]):
            print(task_index)
            group = stim_config_df.loc[task_index]['Group']
            if group in sub_dict:
                print(group)
                sub_dict[group] = sub_dict[group]
                + task_matrix[:, task_index, subject_index]
            else:
                sub_dict[group] = list(
                    task_matrix[:, task_index, subject_index])

        for index, group in enumerate(GROUP_LIST):
            averaged_group_task = np.mean(sub_dict[group], axis=0)
            grouped_task_matrix[:, index, subject_index] = averaged_group_task

    return grouped_task_matrix
