import common
import numpy as np
from nilearn import image, plotting, input_data
import matplotlib.pyplot as plt
import nibabel as nib
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats

DATASET_DIR = '/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/'
ANALYSIS_DIR = DATASET_DIR + 'analysis/'


def setup():
    '''Loads 3dDeconvolve data into matrix with 3D [960, 43, 21]
    (Voxels, Tasks, Subjects). Returns this matrix and Nifti masker'''
    # set directory tree and get subjects

    dir_tree = common.DirectoryTree(DATASET_DIR)
    subjects = common.get_subjects(dir_tree)

    # set number of subjects (four didn't run due to exclusion based on motion), standard shape and affine
    numsub = len(subjects) - 4
    STD_SHAPE = [79, 94, 65]
    STD_AFFINE = nib.load(
        subjects[1].deconvolve_dir + 'Go_FIR_MIN.nii.gz').affine

    # create complete matrix with 3 dimensions: voxels, tasks, and subjects
    task_matrix = np.zeros([960, 43, numsub])

    # Get mask
    mask = nib.load(
        ANALYSIS_DIR + 'Thalamus_Morel_consolidated_mask_v3.nii.gz')
    mask = image.math_img("img>0", img=mask)
    # mask = image.resample_img(mask, target_affine=STD_AFFINE,
    #                           target_shape=STD_SHAPE, interpolation='nearest')
    masker = input_data.NiftiMasker(
        mask, target_affine=STD_AFFINE, target_shape=STD_SHAPE)
    masker.fit()

    subject_index = 0
    for sub in subjects:
        filepath = sub.deconvolve_dir + 'FIRmodel_MNI_stats+tlrc.BRIK'
        if not os.path.exists(filepath):
            continue

        # load 3dDeconvolve bucket
        sub_fullstats_4d = nib.load(filepath)
        sub_fullstats_4d_data = sub_fullstats_4d.get_fdata()

        subject_task_matrix = np.zeros([960, 43])
        group_matrix = np.zeros([960, 30])

        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, i in enumerate(np.arange(2, 197, 3)):
            beta_array = sub_fullstats_4d_data[:, :, :, i]
            beta_array = nib.Nifti1Image(beta_array, STD_AFFINE)
            beta_array = masker.transform(beta_array).flatten()

            if task_index < 43:
                subject_task_matrix[:, task_index] = beta_array
    #             else:
    #                 group_matrix[:, task_index - 43] = beta_array
        task_matrix[:, :, subject_index] = subject_task_matrix
        subject_index += 1

    return task_matrix, masker


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


def compute_PCA(task_matrix, masker):
    # generate z scores for task columns (tasks are 2nd dimension)
    PCA_matrix = stats.zscore(task_matrix, axis=1)

    #  average z scores across all subjects (subjects are 3rd dimenison)
    PCA_matrix = PCA_matrix.mean(2)

    # set pca to explain 95% of variance
    pca = PCA(.95)
    PCA_components = pca.fit_transform(PCA_matrix)

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


def consensus_cluster(task_matrix, masker):
    # consensus clustering
    k_clusters = 3

    consensus_matrix = np.zeros([960, 960, task_matrix.shape[-1]])
    for i in range(task_matrix.shape[-1]):
        sub_cluster, model = cluster_sub(
            task_matrix[:, :, i], k_clusters)
        coassignment_matrix = np.zeros([960, 960])
        for j in range(960):
            for k in range(960):
                if sub_cluster[j] == sub_cluster[k]:
                    coassignment_matrix[j][k] = 1
                else:
                    coassignment_matrix[j][k] = 0
        consensus_matrix[:, :, i] = coassignment_matrix

    mean_matrix = consensus_matrix.mean(2)
    final_consensus_cluster, model = cluster_sub(mean_matrix, k_clusters)
    final_consensus_cluster = masker.inverse_transform(final_consensus_cluster)
    nib.save(final_consensus_cluster, ANALYSIS_DIR + 'consensus_cluster_3.nii')
