from thalpy import base
from thalpy.constants import paths
from thalpy.analysis import masks

import numpy as np
import nibabel as nib
from nibabel import brikhead
import os
import pandas as pd
from nilearn import image, plotting, input_data
import glob2 as glob

MDTB_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/"

stim_config_df = pd.read_csv(MDTB_DIR + paths.DECONVOLVE_DIR + paths.STIM_CONFIG)
TASK_LIST = stim_config_df["Stim Label"].tolist()
GROUP_LIST = list(set(stim_config_df["Group"].to_list()))
MDTB_STD_SHAPE = [79, 94, 65]
MDTB_VOXELS = 960


def setup_blocked(subjects, numsub, masker, std_affine, dataset_key):
    # create complete matrix with 3 dimensions: voxels, tasks, and subjects
    num_tasks = 25
    beta_matrix = np.empty([MDTB_VOXELS, num_tasks, numsub])
    tstat_matrix = np.empty([MDTB_VOXELS, num_tasks, numsub])
    subject_index = 0
    for sub in subjects:
        filepath = (
            sub.deconvolve_dir + "FIRmodel_MNI_stats_" + dataset_key + "+tlrc.BRIK"
        )
        if not os.path.exists(filepath):
            print(sub.deconvolve_dir)
            continue
        print(f"loading sub {sub.name}")

        # load 3dDeconvolve bucket
        sub_fullstats_4d = nib.load(filepath)
        sub_fullstats_4d_data = sub_fullstats_4d.get_fdata()

        beta_task_matrix = np.empty([MDTB_VOXELS, num_tasks])
        tstat_task_matrix = np.empty([MDTB_VOXELS, num_tasks])

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
    beta_matrix = np.empty([MDTB_VOXELS, num_tasks, numsub])
    tstat_matrix = np.empty([MDTB_VOXELS, num_tasks, numsub])

    subject_index = 0
    for sub in subjects:
        filepath = (
            sub.deconvolve_dir + "FIRmodel_MNI_stats" + dataset_key + "+tlrc.BRIK"
        )
        if not os.path.exists(filepath):
            continue
        print(f"loading sub {sub.name}")

        # load 3dDeconvolve bucket
        sub_fullstats_4d = nib.load(filepath)
        sub_fullstats_4d_data = sub_fullstats_4d.get_fdata()

        beta_task_matrix = np.empty([MDTB_VOXELS, num_tasks])
        tstat_task_matrix = np.empty([MDTB_VOXELS, num_tasks])

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


def setup_mdtb(dataset_key, is_setup_block=True):
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

    dir_tree = base.DirectoryTree(MDTB_DIR)
    subjects = base.get_subjects(dir_tree.deconvolve_dir, dir_tree)

    # set number of subjects (four didn't run due to exclusion based on motion)
    # ,standard shape and affine
    numsub = len(subjects)
    std_affine = nib.load(subjects[1].deconvolve_dir + "Go_FIR_MIN.nii.gz").affine

    # Get mask
    mask = nib.load(masks.MOREL_PATH)
    mask = image.math_img("img>0", img=mask)
    # mask = image.resample_img(mask, target_affine=std_affine,
    #                          target_shape=STD_SHAPE, interpolation='nearest')

    masker = input_data.NiftiMasker(
        mask, target_affine=std_affine, target_shape=MDTB_STD_SHAPE
    )
    masker.fit()

    if is_setup_block:
        return setup_blocked(subjects, numsub, masker, std_affine, dataset_key)
    else:
        return setup_many_tasks(subjects, numsub, masker, std_affine, dataset_key)


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


IBC_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/IBC/"


def setup_ibc():
    dir_tree = base.DirectoryTree(IBC_DIR)
    subjects = base.get_subjects(dir_tree.bids_dir, dir_tree)
    numsub = len(subjects)
    masker = masks.get_binary_mask(masks.MOREL_PATH)

    nii_files = sorted(glob.glob(IBC_DIR + "neurovault/*.nii.gz"))

    ibc_data_list = [[] for subject in subjects]
    for file in nii_files:
        print(file)
        sub_name = base.parse_sub_from_file(file)
        index_sub = subjects.get_sub_index(sub_name)
        img = nib.load(file)
        img_masked = masker.fit_transform(img).flatten()
        ibc_data_list[index_sub].append(img_masked)

    print("done")