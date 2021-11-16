from thalpy.analysis import fc, masks
from thalpy import base
from thalpy.constants import wildcards
import nibabel as nib
import os
import numpy as np

MDTB_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/"
dir_tree = base.DirectoryTree(MDTB_DIR)
subjects = base.get_subjects(dir_tree.deconvolve_dir, dir_tree)

masker = masks.roi_masker(masks.SCHAEFER_900_7N_PATH)
print(dir_tree.deconvolve_dir)
fc_data = fc.FcData(
    MDTB_DIR,
    masker,
    masker,
    "fc_task_residuals_900",
    subjects=subjects,
    censor=False,
    is_denoise=False,
    bold_dir=dir_tree.deconvolve_dir,
    bold_WC="*FIRmodel_errts_block.nii.gz",
    cores=4,
)
fc_data.calc_fc()

# fc_matrix = np.empty([2227, 400, len(fc_data.fc_subjects)])
# for i in range(len(fc_data.fc_subjects)):
#     calculated_fc_subject = fc.try_fc_sub(
#         fc_data.n_masker,
#         fc_data.m_masker,
#         fc_data.n,
#         fc_data.m,
#         fc_data.bold_WC,
#         fc_data.censor,
#         fc_data.censor_WC,
#         fc_data.is_denoise,
#         fc_data.bold_dir,
#         fc_data.fc_subjects[i],
#     )
#     fc_matrix[:, :, i] = calculated_fc_subject.seed_to_voxel_correlations
# np.save(dir_tree.analysis_dir + "fc_task_residuals.npy", fc_matrix)


# fc_matrix = pickle.load(
#     dir_tree.analysis_dir + "fc_task_residuals_900.p"
# )
# for sub_index in range(len(fc_data.fc_subjects)):
#     fc_subject = fc_data.fc_subjects[sub_index]
#     fc_subject.seed_to_voxel_correlations = fc_matrix[:, :, sub_index]
#     fc_data.fc_subjects[sub_index] = fc_subject
# fc_data.save()
