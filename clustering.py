import common as common
import basic_settings as bs
import numpy as np
from nilearn import image, plotting, input_data
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import brikhead
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
import pandas as pd
import time

DATASET_DIR = '/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/'
# DATASET_DIR = '/Volumes/lss_kahwang_hpc/data/MDTB/'
ANALYSIS_DIR = DATASET_DIR + 'analysis/'
os.chdir(ANALYSIS_DIR)
stim_config_df = pd.read_csv(DATASET_DIR + bs.DECONVOLVE_DIR + bs.STIM_CONFIG)
TASK_LIST = stim_config_df['Stim Label'].tolist()
GROUP_LIST = list(set(stim_config_df['Group'].to_list()))
STD_SHAPE = [79, 94, 65]
VOXELS = 960

# def load_deconvolve_brik(dataset_dir, num_voxels, num_tasks, stop_index, masker=None, start_index=2):
#     dir_tree = common.DirectoryTree(dataset_dir)
#     subjects = common.get_subjects(dir_tree.deconvolve_dir, dir_tree)
#     numsub = len(subjects)
    
#     beta_matrix = np.empty([num_voxels, num_tasks, numsub])
#     tstat_matrix = np.empty([num_voxels, num_tasks, numsub])
    
#     if masker=None:
        
    
#     for sub in subjects:
#         filepath = sub.deconvolve_dir + 'FIRmodel_MNI_stats_' + dataset_key + '+tlrc.BRIK'
#         if not os.path.exists(filepath):
#             print(sub.deconvolve_dir)
#             continue
#         print(f'loading sub {sub.name}')

#     # load 3dDeconvolve bucket
#         sub_fullstats_4d = nib.load(filepath)
#         sub_fullstats_4d_data = sub_fullstats_4d.get_fdata()

#         beta_task_matrix = np.empty([VOXELS, num_tasks])
#         tstat_task_matrix = np.empty([VOXELS, num_tasks])

#         # visualize each beta array to ensure affine is correct
#         # convert to 4d array with only betas, start at 2 and get every 3
#         for task_index, i in enumerate(np.arange(start_index, stop_index, 3)):
#             beta_array = sub_fullstats_4d_data[:, :, :, i]
#             beta_array = nib.Nifti1Image(beta_array, std_affine)
#             beta_array = masker.transform(beta_array).flatten()
#             beta_task_matrix[:, task_index] = beta_array

#         beta_matrix[:, :, subject_index] = beta_task_matrix

#         # get tstat matrix
#         for task_index, i in enumerate(np.arange(start_index + 1, stop_index, 3)):
#             tstat_array = sub_fullstats_4d_data[:, :, :, i]
#             tstat_array = nib.Nifti1Image(tstat_array, std_affine)
#             tstat_array = masker.transform(tstat_array).flatten()
#             tstat_task_matrix[:, task_index] = tstat_array

#         tstat_matrix[:, :, subject_index] = tstat_task_matrix
#         subject_index += 1

#     return beta_matrix, tstat_matrix, masker

def setup_blocked(subjects, numsub, masker, std_affine, dataset_key):
    # create complete matrix with 3 dimensions: voxels, tasks, and subjects
    num_tasks = 25
    beta_matrix = np.empty([VOXELS, num_tasks, numsub])
    tstat_matrix = np.empty([VOXELS, num_tasks, numsub])
    subject_index = 0
    for sub in subjects:
        filepath = sub.deconvolve_dir + 'FIRmodel_MNI_stats_' + dataset_key + '+tlrc.BRIK'
        if not os.path.exists(filepath):
            print(sub.deconvolve_dir)
            continue
        print(f'loading sub {sub.name}')

        # load 3dDeconvolve bucket
        sub_fullstats_4d = nib.load(filepath)
        sub_fullstats_4d_data = sub_fullstats_4d.get_fdata()

        beta_task_matrix = np.empty([VOXELS, num_tasks])
        tstat_task_matrix = np.empty([VOXELS, num_tasks])

        # visualize each beta array to ensure affine is correct
        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, i in enumerate(np.arange(2, 77, 3)):
            beta_array = sub_fullstats_4d_data[:, :, :, i]
            beta_array = nib.Nifti1Image(beta_array, std_affine)
            beta_array = masker.transform(beta_array).flatten()
            beta_task_matrix[:, task_index] = beta_array

        beta_matrix[:, :, subject_index] = beta_task_matrix

        # get tstat matrix
        for task_index, i in enumerate(np.arange(3, 77, 3)):
            tstat_array = sub_fullstats_4d_data[:, :, :, i]
            tstat_array = nib.Nifti1Image(tstat_array, std_affine)
            tstat_array = masker.transform(tstat_array).flatten()
            tstat_task_matrix[:, task_index] = tstat_array

        tstat_matrix[:, :, subject_index] = tstat_task_matrix
        subject_index += 1

    return beta_matrix, tstat_matrix, masker

def setup_many_tasks(subjects, numsub, masker, std_affine, dataset_key):
    num_tasks = 45
    # create complete matrix with 3 dimensions: voxels, tasks, and subjects
    beta_matrix = np.empty([VOXELS, num_tasks, numsub])
    tstat_matrix = np.empty([VOXELS, num_tasks, numsub])

    subject_index = 0
    for sub in subjects:
        filepath = sub.deconvolve_dir + 'FIRmodel_MNI_stats' + dataset_key + '+tlrc.BRIK'
        if not os.path.exists(filepath):
            continue
        print(f"loading sub {sub.name}")

        # load 3dDeconvolve bucket
        sub_fullstats_4d = nib.load(filepath)
        sub_fullstats_4d_data = sub_fullstats_4d.get_fdata()

        beta_task_matrix = np.empty([VOXELS, num_tasks])
        tstat_task_matrix = np.empty([VOXELS, num_tasks])

        # visualize each beta array to ensure affine is correct
        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, i in enumerate(np.arange(2, 203, 3)):
            beta_array = sub_fullstats_4d_data[:, :, :, i]
            beta_array = nib.Nifti1Image(beta_array, std_affine)
            beta_array = masker.transform(beta_array).flatten()

            if task_index < num_tasks:
                beta_task_matrix[:, task_index] = beta_array

        beta_matrix[:, :, subject_index] = beta_task_matrix

        # get tstat matrix
        for task_index, i in enumerate(np.arange(3, 203, 3)):
            tstat_array = sub_fullstats_4d_data[:, :, :, i]
            tstat_array = nib.Nifti1Image(tstat_array, std_affine)
            tstat_array = masker.transform(tstat_array).flatten()

            if task_index < 45:
                tstat_task_matrix[:, task_index] = tstat_array

        tstat_matrix[:, :, subject_index] = tstat_task_matrix
        subject_index += 1

    return beta_matrix, tstat_matrix, masker


def setup(dataset_key, is_setup_block=True):
    """Load data.

    Loads 3dDeconvolve data into matrix with 3D [VOXELS, 43, 21]
    (Voxels, Tasks, Subjects). Returns this matrix and Nifti masker.

    Returns
    -------
    numpy array
        Matrix storing beta values [VOXELS Voxels, 45 Tasks, 20 Subjects]

    numpy array
        Matrix storing tstat values [VOXELS Voxels, 45 Tasks, 20 Subjects]

    Nifti1Masker
        Object storing mask to transpose matrix back to MRI space
    """
    # set directory tree and get subjects

    dir_tree = common.DirectoryTree(DATASET_DIR)
    subjects = common.get_subjects(dir_tree.deconvolve_dir, dir_tree)

    # set number of subjects (four didn't run due to exclusion based on motion)
    # ,standard shape and affine
    numsub = len(subjects)
    std_affine = nib.load(
        subjects[1].deconvolve_dir + 'Go_FIR_MIN.nii.gz').affine
    # Get mask
    mask = nib.load('Thalamus_Morel_consolidated_mask_v3.nii.gz')
    mask = image.math_img("img>0", img=mask)
    # mask = image.resample_img(mask, target_affine=std_affine,
    #                          target_shape=STD_SHAPE, interpolation='nearest')

    masker = input_data.NiftiMasker(
        mask, target_affine=std_affine, target_shape=STD_SHAPE)
    masker.fit()

    if is_setup_block:
        return setup_blocked(subjects, numsub, masker, std_affine, dataset_key)
    else:
        return setup_many_tasks(subjects, numsub, masker, std_affine, dataset_key)


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
        consensus_matrix = np.empty([VOXELS, VOXELS, task_matrix.shape[-1]])
        for i in range(task_matrix.shape[-1]):
            print(i)
            sub_cluster, model = cluster_sub(
                task_matrix[:, :, i], k_cluster)
            coassignment_matrix = np.empty([VOXELS, VOXELS])
            for j in range(VOXELS):
                for k in range(VOXELS):
                    if sub_cluster[j] == sub_cluster[k]:
                        coassignment_matrix[j][k] = 1
                    else:
                        coassignment_matrix[j][k] = 0
            consensus_matrix[:, :, i] = coassignment_matrix

        mean_matrix = consensus_matrix.mean(2)
        final_consensus_cluster, model = cluster_sub(mean_matrix, k_cluster)
        final_consensus_cluster = masker.inverse_transform(
            final_consensus_cluster)
        nib.save(final_consensus_cluster,
                 f'{label}_consensus_cluster_{k_cluster}.nii')


def compute_PCA(task_matrix, masker, output_name):
    print(GROUP_LIST)
    if task_matrix.shape[1] == 45:
        task_list = TASK_LIST
    elif task_matrix.shape[1] == 25:
        if 'Rest' in GROUP_LIST:
            GROUP_LIST.remove('Rest')
        task_list = GROUP_LIST
    else:
        task_list = GROUP_LIST
    print(GROUP_LIST)
    # set pca to explain 95% of variance
    pca = PCA(.95)
    PCA_components = pca.fit_transform(task_matrix)
    # get and save loadings that represent variables contributions to components
    loadings = pd.DataFrame(pca.components_.T, index=task_list)
    loadings.to_csv(output_name + 'loadings.csv')
    print(loadings)

    # get and save correlated loadings that represent each the correlations
    # between variables and components
    correlated_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    correlated_loadings = pd.DataFrame(
        correlated_loadings, index=task_list)
    correlated_loadings.to_csv(
        output_name + 'correlated_loadings.csv')
    print(correlated_loadings)

    # print variance explained by each component
    print(pca.explained_variance_ratio_)
    with open(output_name + 'explained_variance_list.txt', "w") as outfile:
        outfile.write("\n".join(map(str, pca.explained_variance_ratio_)))

    # # Plot the explained variances
    # features = range(pca.n_components_)
    # plt.bar(features, pca.explained_variance_ratio_, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(features)
    # plt.show()

    # save each component into nifti form
    for index, component in enumerate(PCA_components):
        if index == 10:
            break

        # save PC back into nifti image and visualize
        comp_array = PCA_components[:, index]
        comp_array = masker.inverse_transform(comp_array)
        nib.save(
            comp_array, f'{output_name}PCA_component_{index}.nii')
        # view = plotting.view_img(comp_array, threshold=3)
        # view.open_in_browser()

    return PCA_components


def zscore(matrix, masker=None, type='2d', label=''):
    zscored_matrix = np.empty(matrix.shape)

    for subject_index in range(matrix.shape[-1]):
        if type == '2d':
            print('2d zscore')
            zscored_matrix[:, :, subject_index] = zscore_subject_2d(
                matrix[:, :, subject_index])
        elif type == 'task':
            print('task zscore')
            zscored_matrix[:, :, subject_index] = zscore_subject_task(
                matrix[:, :, subject_index])

    if masker:
        nifti_img = nib.Nifti1Image(zscored_matrix, std_affine)
        nib.save(nifti_img, type + '_zscore_' + label + '.nii')

    return zscored_matrix


def zscore_subject_task(matrix):
    zscored_matrix = np.empty(matrix.shape)

    for task_index in range(matrix.shape[1]):
        task_array = matrix[:, task_index]
        sample_mean = np.mean(task_array)
        std_dev = np.std(task_array)

        for i in range(len(task_array)):
            zscored_matrix[i, task_index] = (matrix[i,
                                                    task_index] - sample_mean) / std_dev

    return zscored_matrix


def zscore_subject_2d(matrix):
    # 2D matrix shape [voxels, tasks]
    zscored_matrix = np.empty(matrix.shape)

    # zscore across 2d (voxel, task) within subject
    sample_mean = np.mean(matrix)
    std_dev = np.std(matrix)
    print(sample_mean)
    print(std_dev)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            zscored_matrix[i, j] = (matrix[i, j] - sample_mean) / std_dev

    print(np.mean(zscored_matrix))
    return zscored_matrix


def participation_coefficient_abs(task_matrix, masker, label='beta'):
    PC_matrix = np.empty([task_matrix.shape[0]])

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
    nib.save(PC_matrix, label +
             'participation_coefficient_abs' + '.nii')


def participation_coefficient_normalize(task_matrix, masker, label='beta'):
    PC_matrix = np.empty([task_matrix.shape[0]])
    min = np.min(task_matrix)
    mean = np.mean(task_matrix)
    norm_matrix = task_matrix + np.absolute(min)

    # get PC value for each voxel
    for voxel_index in range(task_matrix.shape[0]):
        sum_PC = 0.0
        sum_task = norm_matrix[voxel_index, :].sum()
        print(sum_task)
        # sum (kis / ki)^ 2 for each task
        for task_index in range(task_matrix.shape[1]):
            print(norm_matrix[voxel_index, task_index])
            sum_PC += ((norm_matrix[voxel_index, task_index] / sum_task) ** 2)
        PC = 1 - sum_PC
        PC_matrix[voxel_index] = PC

    print(PC_matrix)
    PC_matrix = masker.inverse_transform(PC_matrix)
    nib.save(PC_matrix,  label + 'pc_normalize.nii')


def participation_coefficient_ex(task_matrix, masker, label='beta'):
    PC_matrix = np.empty([task_matrix.shape[0]])

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
    nib.save(PC_matrix, label + 'participation_coefficient_ex.nii')


def participation_coefficient_thr(task_matrix, masker, label='beta'):
    task_matrix = np.apply_along_axis(threshold_arr, 0, task_matrix)
    PC_matrix = np.empty([task_matrix.shape[0]])

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

    PC_matrix = masker.inverse_transform(PC_matrix)
    nib.save(PC_matrix,  label + '_participation_coefficient_thr.nii')

    return PC_matrix

def threshold_arr(arr):
    new_arr = np.empty(arr.shape)
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
def greene_method_sub(total_matrix, threshold=0.667):
    # initialize matrix to store winning task in each voxel for each subject
    # Shape is [Voxels, Subjects]
    winner_take_all_list = [[None] * task_matrix.shape[2]] * VOXELS

    for subject_index in range(task_matrix.shape[2]):
        list = greene_method(task_matrix[:, :, subject_index])


def greene_method(task_matrix, task_list=GROUP_LIST, threshold=0.99):
    # loop through each voxel to get winning task
    df = pd.DataFrame(index=range(VOXELS), columns=[
                      'Task Name', 'Max Task value', 'Is Specific'])

    for voxel_index in range(task_matrix.shape[0]):
        task_array = task_matrix[voxel_index, :]
        max_task_value = np.amax(task_array)
        max_index = np.where(task_array == max_task_value)

        # todo import task list and check if is in order of 3d brik
        if len(max_index) > 1:
            task_name = 'Multiple'
            is_specific = False
            print('hi')
        else:
            task_name = task_list[max_index[0][0]]
            is_specific = determine_specificity(
                task_array, max_task_value, max_index[0], threshold)
            print(is_specific)

        df.loc[voxel_index] = [task_name, max_task_value, is_specific]

    return df


def determine_specificity(task_array, max_value, max_index, threshold):
    threshold_value = max_value * threshold
    for index, task_beta in enumerate(task_array):
        if max_index == index:
            print('continue')
            continue
        if task_beta > threshold_value:
            print(task_beta)
            print(index)
            return False

    return True


def group_tasks(task_matrix):
    print(GROUP_LIST)
    grouped_df = stim_config_df.groupby('Group')
    grouped_task_matrix = np.empty(
        [task_matrix.shape[0], len(grouped_df), task_matrix.shape[2]])

    for subject_index in range(task_matrix.shape[2]):
        sub_dict = {}
        for task_index in range(task_matrix.shape[1]):
            group = stim_config_df.loc[task_index]['Group']
            if group in sub_dict:
                group_list = sub_dict[group]
                group_list.append(
                    task_matrix[:, task_index, subject_index])
                sub_dict[group] = group_list
            else:
                group_list = []
                group_list.append(task_matrix[:, task_index, subject_index])
                sub_dict[group] = group_list

        for index, group in enumerate(GROUP_LIST):
            stacked_arrays = np.stack(sub_dict[group])
            averaged_group_task = np.mean(stacked_arrays, axis=0)
            grouped_task_matrix[:, index, subject_index] = averaged_group_task

    return grouped_task_matrix


def sub_averaged_to_nii(matrix, masker, prefix):
    if matrix.shape[1] == len(TASK_LIST):
        task_list = TASK_LIST
    elif matrix.shape[1] == 25:
        GROUP_LIST.remove('Rest')
        task_list = GROUP_LIST
    else:
        task_list = GROUP_LIST

    for i in range(matrix.shape[1]):
        print(matrix[:, i].shape)
        array = masker.inverse_transform(matrix[:, i])
        suffix = 'task_' + task_list[i] + '.nii'
        nib.save(array, prefix + suffix.replace('/', '-'))


def pca_subjects(matrix, masker):
    dir_tree = common.DirectoryTree(DATASET_DIR)
    subjects = common.get_subjects(dir_tree.deconvolve_dir, dir_tree)
    
    for index, sub in enumerate(subjects):
        compute_PCA(matrix[:, :, index], masker, sub.name)
        
