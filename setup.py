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
MDTB_VOXELS = 2227

BETA_TASK_START = 2
TSTAT_TASK_START = 3
TASK_END = 77
# def setup_cortical(subjects, numsub, masker, std_affine, dataset_key)


def setup_blocked(subjects, numsub, masker, std_affine, dataset_key, voxels):
    # create complete matrix with 3 dimensions: voxels, tasks, and subjects
    num_tasks = 25
    beta_matrix = np.empty([voxels, num_tasks, numsub])
    tstat_matrix = np.empty([voxels, num_tasks, numsub])
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
        sub_fullstats_4d_data = masker.fit_transform(sub_fullstats_4d)

        beta_task_matrix = np.empty([voxels, num_tasks])
        tstat_task_matrix = np.empty([voxels, num_tasks])

        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, stat_index in enumerate(np.arange(2, 77, 3)):
            beta_task_matrix[:, task_index] = sub_fullstats_4d_data[stat_index, :]
        # get tstat matrix
        for task_index, stat_index in enumerate(np.arange(3, 77, 3)):
            tstat_task_matrix[:, task_index] = sub_fullstats_4d_data[stat_index, :]

        beta_matrix[:, :, subject_index] = beta_task_matrix
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
    # mask = nib.load(masks.MOREL_PATH)
    # mask = image.math_img("img>0", img=mask)
    # # mask = image.resample_img(mask, target_affine=std_affine,
    # #                          target_shape=STD_SHAPE, interpolation='nearest')
    # masker = input_data.NiftiMasker(
    #     mask, target_affine=std_affine, target_shape=MDTB_STD_SHAPE
    # )
    # masker.fit()

    masker = masks.get_binary_masker(masks.MOREL_PATH)

    if is_setup_block:
        return setup_blocked(subjects, numsub, masker, std_affine, dataset_key)
    else:
        return setup_many_tasks(subjects, numsub, masker, std_affine, dataset_key)


IBC_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/IBC/"
IBC_CONDITIONS_DF = pd.read_csv(
    "/mnt/nfs/lss/lss_kahwang_hpc/scripts/ibc_authors/ibc_data/conditions.tsv",
    sep="\t",
)
IBC_CONDITIONS_DF.at[253, "contrast"] = "null"
IBC_VOXELS = 2227


def setup_ibc(file_wc, output_file, two_runs=False):
    dir_tree = base.DirectoryTree(IBC_DIR)
    glm_dir = IBC_DIR + "glm/"
    subjects = base.get_subjects(glm_dir, dir_tree)
    numsub = len(subjects) - 2
    masker = masks.get_binary_mask(masks.MOREL_PATH)

    conditions_matrix = np.zeros([numsub, len(IBC_CONDITIONS_DF.index), IBC_VOXELS])
    sub_index = -1
    for subject in subjects:
        if subject.name == "02" or subject.name == "08":
            continue
        sub_index += 1

        sub_effect_sz_files = glob.glob(
            glm_dir + subject.sub_dir + file_wc,
            recursive=True,
        )

        for condition_index, row in IBC_CONDITIONS_DF.iterrows():
            condition = row["contrast"]
            task = row["task"]
            condition_files = sorted(
                [x for x in sub_effect_sz_files if condition and task in x], key=len
            )

            if len(condition_files) >= 2 and len(condition_files[0]) == len(
                condition_files[1]
            ):
                img_one = masker.fit_transform(nib.load(condition_files[0]))
                img_two = masker.fit_transform(nib.load(condition_files[1]))
                fit_imgs = np.empty([img_one.shape[0], img_one.shape[1], 2])
                fit_imgs[:, :, 0] = img_one
                fit_imgs[:, :, 1] = img_two
                masked_img = np.mean(fit_imgs, axis=2)
            elif any(condition_files):
                condition_file = condition_files[0]
                nii_img = nib.load(condition_file)
                masked_img = masker.fit_transform(nii_img)
            else:
                continue

            conditions_matrix[sub_index, condition_index, :] = masked_img

    np.save(dir_tree.analysis_dir + output_file, conditions_matrix)


# dir_tree = base.DirectoryTree(MDTB_DIR)
# subjects = base.get_subjects(dir_tree.deconvolve_dir, dir_tree)
# numsub = len(subjects)
# std_affine = nib.load(subjects[1].deconvolve_dir + "Go_FIR_MIN.nii.gz").affine
# schaefer_mask = masks.get_roi_masker(masks.SCHAEFER_YEO7_PATH)
# cort_beta_matrix, tstat_matrix, masker = setup_blocked(
#     subjects, numsub, schaefer_mask, std_affine, "block", 400
# )
# zscored_cort_matrix = zscore_subject_2d(cort_beta_matrix)
# np.save(dir_tree.analysis_dir + "beta_cortical.npy", zscored_cort_matrix)